"""LangGraph 多 Agent 工作流

Architecture:
    START → router → [chat | nutrition → expert_review | fitness → expert_review] → END

Nodes:
    - router: 分析用户意图，决定路由到哪个 Agent
    - chat: 闲聊 Agent（无工具调用）
    - nutrition: 营养师 Agent（支持工具调用）
    - fitness: 健身教练 Agent（支持工具调用）
    - expert_review: 专家评审（评分 < 3 时重试）

Edges:
    - START → router
    - router → chat | nutrition | fitness（条件路由）
    - chat → END
    - nutrition → expert_review
    - expert_review → [nutrition | END]（评分 < 3 重试）
    - fitness → expert_review
    - expert_review → [fitness | END]（评分 < 3 重试）
"""

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Sequence
from dataclasses import dataclass, field
import os
import re
from dotenv import load_dotenv

from .base import AGENT_SYSTEM_PROMPTS
from .chat_agent import chat_with_user
from .nutrition_agent import nutrition_tools, nutrition_with_user
from .fitness_agent import fitness_tools, fitness_with_user
from .expert_agent import review_output
from .router import hybrid_route
from ..memory import MemoryManager

load_dotenv()

MAX_RETRIES = 3
MIN_APPROVAL_SCORE = 3

# 快速通道模式：匹配到这些模式的问题属于简单事实查询，跳过专家评审
QUICK_PATTERNS = [
    r".*的热量[是为多少].*",
    r".*多少卡[路里]?.*",
    r".*[每一]百?克.*",
    r".*含[有多]少.*蛋白质.*",
    r".*含[有多]少.*碳水.*",
    r".*含[有多]少.*脂肪.*",
    r"^.{0,15}[是多少].*卡.*$",
]


class AgentState(TypedDict):
    """LangGraph 状态定义"""
    messages: Annotated[List, lambda x, y: x + y]
    user_id: int
    user_profile: Dict[str, Any]
    daily_stats: Dict[str, Any]
    current_agent: str
    retry_count: int
    review_history: List[Dict]
    memory_summary: Dict[str, Any]
    enhanced_prompts: Dict[str, str]
    skip_review: bool


def should_skip_review(state: AgentState) -> bool:
    """判断是否跳过专家评审（快速通道）

    简单事实查询（热量查询、营养成分查询等）直接跳过评审，
    减少 1-2 次不必要的 LLM 调用。

    快速通道条件（满足任一）：
    1. 用户消息匹配简单事实查询模式
    2. Agent 回复长度 < 150 字符（简短事实性回复）
    """
    messages = state.get("messages", [])
    if not messages:
        return False

    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 条件1：用户消息匹配简单查询模式
    for pattern in QUICK_PATTERNS:
        if re.search(pattern, user_message):
            return True

    # 条件2：回复很短（事实性简答）
    last_response = messages[-1].content if messages else ""
    if len(last_response) < 150:
        return True

    return False


def create_llm(temperature: float = 0.7):
    """创建 LLM 实例（通过 LLMManager 复用）"""
    from ..llm_manager import LLMManager
    return LLMManager.get_llm(temperature=temperature)


def router(state: AgentState) -> Dict[str, str]:
    """路由节点 - 分析用户意图，决定路由到哪个 Agent

    使用混合路由策略：
    1. 关键词预筛选：快速路径，明确场景直接返回
    2. LLM 二次确认：模糊场景调用 LLM

    Args:
        state: AgentState

    Returns:
        Dict[str, str]: {"current_agent": "chat" | "nutrition" | "fitness"}
    """
    messages = state["messages"]
    user_message = messages[-1].content if messages else ""

    user_message = user_message.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    user_message = ' '.join(user_message.split())

    if not user_message.strip():
        return {"current_agent": "chat", "retry_count": 0, "review_history": []}

    result = hybrid_route(user_message, require_llm_confirm=True)

    agent = result["agent"]
    if agent not in ["nutrition", "fitness"]:
        agent = "chat"

    return {"current_agent": agent, "retry_count": 0, "review_history": []}


def chat(state: AgentState) -> Dict[str, Any]:
    """闲聊节点 - 直接生成回答，不需要工具调用和评审

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: {"messages": [AIMessage]}
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("chat")

    response = chat_with_user(messages, user_id, memory_summary, enhanced_prompt)

    return {
        "messages": [AIMessage(content=response)],
        "current_agent": "chat"
    }


def nutrition(state: AgentState) -> Dict[str, Any]:
    """营养师节点 - 支持工具调用

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: {"messages": [AIMessage], "current_agent": "nutrition", "skip_review": bool}
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("nutrition")

    response = nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt)

    # 判断是否跳过评审（快速通道）
    state_with_response = {**state, "messages": messages + [AIMessage(content=response)]}
    skip_review = should_skip_review(state_with_response)

    return {
        "messages": [AIMessage(content=response)],
        "current_agent": "nutrition",
        "skip_review": skip_review
    }


def fitness(state: AgentState) -> Dict[str, Any]:
    """健身教练节点 - 支持工具调用

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: {"messages": [AIMessage], "current_agent": "fitness", "skip_review": bool}
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("fitness")

    response = fitness_with_user(messages, user_id, memory_summary, enhanced_prompt)

    # 判断是否跳过评审（快速通道）
    state_with_response = {**state, "messages": messages + [AIMessage(content=response)]}
    skip_review = should_skip_review(state_with_response)

    return {
        "messages": [AIMessage(content=response)],
        "current_agent": "fitness",
        "skip_review": skip_review
    }


def expert_review(state: AgentState) -> Dict[str, Any]:
    """专家评审节点 - 评估 Agent 输出质量，评分 < 3 时重试

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: {
            "retry_count": int,
            "review_history": List[Dict],
            "should_retry": bool
        }
    """
    messages = state["messages"]
    current_agent = state.get("current_agent", "")
    retry_count = state.get("retry_count", 0)
    review_history = state.get("review_history", [])

    last_response = messages[-1].content if messages else ""

    nutrition_output = last_response if current_agent == "nutrition" else ""
    fitness_output = last_response if current_agent == "fitness" else ""

    review = review_output(nutrition_output, fitness_output)

    new_review = {
        "attempt": retry_count + 1,
        "score": review["score"],
        "feedback": review["feedback"]
    }

    review_history.append(new_review)

    should_retry = review["score"] < MIN_APPROVAL_SCORE and retry_count < MAX_RETRIES - 1

    return {
        "retry_count": retry_count + 1,
        "review_history": review_history,
        "should_retry": should_retry
    }


def should_continue_nutrition(state: AgentState) -> str:
    """决定 nutrition 工作流的下一跳

    Args:
        state: AgentState

    Returns:
        str: "nutrition"（重试）或 "__end__"（结束）
    """
    # 快速通道：跳过评审直接结束
    if state.get("skip_review", False):
        return END
    should_retry = state.get("should_retry", False)
    return "nutrition" if should_retry else END


def should_continue_fitness(state: AgentState) -> str:
    """决定 fitness 工作流的下一跳

    Args:
        state: AgentState

    Returns:
        str: "fitness"（重试）或 "__end__"（结束）
    """
    # 快速通道：跳过评审直接结束
    if state.get("skip_review", False):
        return END
    should_retry = state.get("should_retry", False)
    return "fitness" if should_retry else END


def route_after_router(state: AgentState) -> str:
    """路由决策后的下一跳

    Args:
        state: AgentState

    Returns:
        str: Agent 名称
    """
    return state.get("current_agent", "chat")


def build_graph():
    """构建 LangGraph 工作流

    Graph Structure:
        START
          │
          ▼
        router
          │
          ├──────────────────────────────┐
          │                              │
          ▼                              ▼
        chat                          nutrition
          │                              │
          │                              ▼
          │                          expert_review
          │                              │
          │              ┌───────────────┼───────────────┐
          │              │                               │
          │              ▼                               ▼
          │          [retry]                           END
          │              │
          │              ▼
          │          nutrition
          │              │
          └──────────────┤
                         │
                         ▼
                       fitness
                         │
                         ▼
                   expert_review
                         │
                         ▼
                        END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router)
    workflow.add_node("chat", chat)
    workflow.add_node("nutrition", nutrition)
    workflow.add_node("fitness", fitness)
    workflow.add_node("expert_review", expert_review)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chat": "chat",
            "nutrition": "nutrition",
            "fitness": "fitness"
        }
    )

    workflow.add_edge("chat", END)

    workflow.add_edge("nutrition", "expert_review")

    workflow.add_conditional_edges(
        "expert_review",
        should_continue_nutrition,
        {
            "nutrition": "nutrition",
            END: END
        }
    )

    workflow.add_edge("fitness", "expert_review")

    workflow.add_conditional_edges(
        "expert_review",
        should_continue_fitness,
        {
            "fitness": "fitness",
            END: END
        }
    )

    return workflow.compile()


agent_graph = build_graph()


def process_user_message(
    user_message: str,
    user_id: int = 1,
    user_profile: dict = None,
    daily_stats: dict = None,
    session_id: str = None
) -> dict:
    """处理用户消息的入口函数

    使用 LangGraph 工作流处理用户消息。

    Args:
        user_message: 用户输入的消息
        user_id: 用户ID，默认1
        user_profile: 用户信息字典（可选）
        daily_stats: 当日统计数据（可选）
        session_id: 会话 ID（可选）

    Returns:
        dict: {
            "response": str,              # 最终回复内容
            "agent": str,                 # 处理的 Agent 类型
            "expert_review": {            # 专家评审结果
                "score": int,             # 评分 1-5
                "approved": bool,         # 是否通过
                "feedback": str,          # 评审意见
                "retries": int,           # 重试次数
                "review_history": list    # 评审历史
            }
        }
    """
    memory_manager = MemoryManager(user_id=user_id)
    memory_summary = memory_manager.get_memory_summary()

    conversation_history = memory_manager.load_conversation_history(days=7, limit=20)
    memory_summary["conversation_history"] = conversation_history

    messages_for_prompt = [HumanMessage(content=user_message)]
    enhanced_prompts = {
        "chat": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["chat"], "chat", messages_for_prompt
        ),
        "nutrition": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["nutrition"], "nutrition", messages_for_prompt
        ),
        "fitness": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["fitness"], "fitness", messages_for_prompt
        )
    }

    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "user_profile": user_profile or memory_manager.load_profile(),
        "daily_stats": daily_stats or {},
        "current_agent": "chat",
        "retry_count": 0,
        "review_history": [],
        "memory_summary": memory_summary,
        "enhanced_prompts": enhanced_prompts
    }

    final_state = agent_graph.invoke(initial_state)

    messages = final_state.get("messages", [])
    response = messages[-1].content if messages else ""

    current_agent = final_state.get("current_agent", "chat")
    review_history = final_state.get("review_history", [])
    retry_count = final_state.get("retry_count", 0)

    last_review = review_history[-1] if review_history else {}

    memory_manager.save_conversation(
        user_message=user_message,
        agent_response=response,
        agent_type=current_agent,
        session_id=session_id
    )

    return {
        "response": response,
        "agent": current_agent,
        "expert_review": {
            "score": last_review.get("score", 0),
            "approved": last_review.get("score", 0) >= MIN_APPROVAL_SCORE if review_history else True,
            "feedback": last_review.get("feedback", ""),
            "retries": retry_count - 1 if retry_count > 0 else 0,
            "review_history": review_history
        }
    }


def chat_stream(state: AgentState):
    """流式闲聊节点

    Args:
        state: AgentState

    Yields:
        str: 回复片段
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("chat")

    response_generator = chat_with_user(messages, user_id, memory_summary, enhanced_prompt, stream=True)

    for chunk in response_generator:
        yield chunk


def nutrition_stream(state: AgentState):
    """流式营养师节点

    Args:
        state: AgentState

    Yields:
        str: 回复片段
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("nutrition")

    response_generator = nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt, stream=True)

    for chunk in response_generator:
        yield chunk


def fitness_stream(state: AgentState):
    """流式健身教练节点

    Args:
        state: AgentState

    Yields:
        str: 回复片段
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("fitness")

    response_generator = fitness_with_user(messages, user_id, memory_summary, enhanced_prompt, stream=True)

    for chunk in response_generator:
        yield chunk


def stream_user_message(
    user_message: str,
    user_id: int = 1,
    user_profile: dict = None,
    daily_stats: dict = None
):
    """流式处理用户消息

    使用混合路由 + 流式响应，跳过专家评审阶段以支持实时流式输出。

    Args:
        user_message: 用户输入的消息
        user_id: 用户ID，默认1
        user_profile: 用户信息字典（可选）
        daily_stats: 当日统计数据（可选）

    Yields:
        str: 回复片段
    """
    memory_manager = MemoryManager(user_id=user_id)
    memory_summary = memory_manager.get_memory_summary()

    conversation_history = memory_manager.load_conversation_history(days=7, limit=20)
    memory_summary["conversation_history"] = conversation_history

    messages_for_prompt = [HumanMessage(content=user_message)]
    enhanced_prompts = {
        "chat": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["chat"], "chat", messages_for_prompt
        ),
        "nutrition": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["nutrition"], "nutrition", messages_for_prompt
        ),
        "fitness": memory_manager.enhance_system_prompt(
            AGENT_SYSTEM_PROMPTS["fitness"], "fitness", messages_for_prompt
        )
    }

    user_message_clean = user_message.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    user_message_clean = ' '.join(user_message_clean.split())

    if not user_message_clean.strip():
        yield "你好，有什么我可以帮助你的吗？"
        return

    result = hybrid_route(user_message_clean, require_llm_confirm=True)
    agent = result["agent"]
    if agent not in ["nutrition", "fitness"]:
        agent = "chat"

    state = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "user_profile": user_profile or memory_manager.load_profile(),
        "daily_stats": daily_stats or {},
        "current_agent": agent,
        "retry_count": 0,
        "review_history": [],
        "memory_summary": memory_summary,
        "enhanced_prompts": enhanced_prompts
    }

    if agent == "nutrition":
        response_generator = nutrition_stream(state)
    elif agent == "fitness":
        response_generator = fitness_stream(state)
    else:
        response_generator = chat_stream(state)

    # 收集完整回复用于保存对话历史
    full_response = ""
    for chunk in response_generator:
        full_response += chunk
        yield chunk

    # 流式完成后保存对话历史
    memory_manager.save_conversation(
        user_message=user_message,
        agent_response=full_response,
        agent_type=agent,
        session_id=None
    )
