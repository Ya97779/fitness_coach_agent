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
from dotenv import load_dotenv

from .base import AGENT_SYSTEM_PROMPTS
from .chat_agent import chat_with_user
from .nutrition_agent import nutrition_tools, nutrition_with_user
from .fitness_agent import fitness_tools, fitness_with_user
from .expert_agent import review_output
from ..memory import MemoryManager

load_dotenv()

MAX_RETRIES = 3
MIN_APPROVAL_SCORE = 3


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


def create_llm(temperature: float = 0.7):
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=temperature
    )


def router(state: AgentState) -> Dict[str, str]:
    """路由节点 - 分析用户意图，决定路由到哪个 Agent

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

    llm = create_llm(temperature=0.1)

    prompt = f"""用户输入: {user_message}

根据用户输入判断最合适的 Agent 类型：

类型 1 - 闲聊助手：
- 日常问候、情感交流
- 通用知识问答
- 不涉及具体的饮食或运动计划
- 模糊或综合性的健康话题

类型 2 - 营养师：
- 食物热量查询、营养成分计算
- 饮食计划、食谱推荐
- 增肌/减脂/维持的饮食策略
- 每日摄入记录和追踪
- 蛋白质/脂肪/碳水化合物相关问题

类型 3 - 健身教练：
- 运动训练、动作指导
- 健身计划、训练强度
- 增肌/减脂的運動策略
- 运动消耗计算
- 具体运动姿势和发力技巧

只返回一个数字（1、2 或 3），不要其他内容。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    if "2" in text:
        agent = "nutrition"
    elif "3" in text:
        agent = "fitness"
    else:
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
        Dict[str, Any]: {"messages": [AIMessage], "current_agent": "nutrition"}
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("nutrition")

    response = nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt)

    return {
        "messages": [AIMessage(content=response)],
        "current_agent": "nutrition"
    }


def fitness(state: AgentState) -> Dict[str, Any]:
    """健身教练节点 - 支持工具调用

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: {"messages": [AIMessage], "current_agent": "fitness"}
    """
    messages = state["messages"]
    user_id = state.get("user_id", 1)
    memory_summary = state.get("memory_summary", {})
    enhanced_prompt = state.get("enhanced_prompts", {}).get("fitness")

    response = fitness_with_user(messages, user_id, memory_summary, enhanced_prompt)

    return {
        "messages": [AIMessage(content=response)],
        "current_agent": "fitness"
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
    should_retry = state.get("should_retry", False)
    return "nutrition" if should_retry else END


def should_continue_fitness(state: AgentState) -> str:
    """决定 fitness 工作流的下一跳

    Args:
        state: AgentState

    Returns:
        str: "fitness"（重试）或 "__end__"（结束）
    """
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
