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

from .base import AGENT_SYSTEM_PROMPTS
from .chat_agent import chat_with_user, parse_intent
from .nutrition_agent import (
    nutrition_tools,
    nutrition_with_user,
    _extract_food_names,
    _get_food_nutrition,
    log_food_intake,
)
from .fitness_agent import (
    fitness_tools,
    fitness_with_user,
    _extract_exercise_info,
    _extract_training_parameters,
    log_exercise,
)
from ..calorie_calculator import estimate_calories as calculate_exercise_calories
from .expert_agent import review_output
from .router import hybrid_route
from ..memory import MemoryManager
from ..runtime_context import request_context

MAX_RETRIES = 3
MIN_APPROVAL_SCORE = 3


def _collect(gen) -> str:
    """消费 agent generator，收集完整回复字符串"""
    return "".join(gen)


def _record_multi_domain_message(user_message: str, user_id: int) -> str:
    """Execute the existing food/exercise logging tools for a clear multi-record."""

    recorded = []
    for food_name, meal_type in _extract_food_names(user_message):
        if food_name == "食物":
            continue
        nutrition = _get_food_nutrition(food_name)
        calories = nutrition.get("calories", 0)
        result = log_food_intake.invoke({
            "user_id": user_id,
            "food_name": food_name,
            "calories": calories,
            "meal_type": meal_type,
        })
        recorded.append(f"饮食：{food_name} {calories:.0f}kcal")
        print(f"[stream] 多意图饮食记录: {result}", flush=True)

    exercise_name, duration = _extract_exercise_info(user_message)
    if exercise_name != "运动":
        sets, reps, weight = _extract_training_parameters(user_message)
        has_explicit_duration = bool(
            re.search(r"\d+\s*(?:分钟|小时)", user_message)
        )
        if sets and not has_explicit_duration:
            duration = max(5, sets * 5)
        calories = calculate_exercise_calories(
            exercise_name,
            duration=duration,
            sets=sets,
        )
        result = log_exercise.invoke({
            "user_id": user_id,
            "exercise_type": exercise_name,
            "duration": duration,
            "calories": float(calories),
            "sets": sets,
            "reps": reps,
            "weight": weight,
        })
        detail = f"{sets}组" if sets else f"{duration}分钟"
        if reps:
            detail += f"×{reps}次"
        recorded.append(f"运动：{exercise_name} {detail}，约{calories:.0f}kcal")
        print(f"[stream] 多意图运动记录: {result}", flush=True)

    if not recorded:
        raise ValueError("未能从消息中提取可记录的饮食或运动实体")
    return "已为你记录：" + "；".join(recorded) + "。"

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
    route_decision: Dict[str, Any]


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

    result = hybrid_route(
        user_message,
        require_llm_confirm=True,
        context_messages=messages[:-1],
        context_route=state.get("memory_summary", {}).get("last_route"),
    )

    agent = result["agent"]
    if agent not in ["nutrition", "fitness"]:
        agent = "chat"

    return {
        "current_agent": agent,
        "retry_count": 0,
        "review_history": [],
        "route_decision": result,
    }


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

    response = _collect(chat_with_user(messages, user_id, memory_summary, enhanced_prompt))

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

    response = _collect(nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt))

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

    response = _collect(fitness_with_user(messages, user_id, memory_summary, enhanced_prompt))

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
    session_id: str = None,
    request_id: str = None,
) -> dict:
    """Run the existing graph with server-bound request identity."""

    with request_context(user_id, session_id=session_id, request_id=request_id):
        return _process_user_message(
            user_message=user_message,
            user_id=user_id,
            user_profile=user_profile,
            daily_stats=daily_stats,
            session_id=session_id,
            request_id=request_id,
        )


def _process_user_message(
    user_message: str,
    user_id: int = 1,
    user_profile: dict = None,
    daily_stats: dict = None,
    session_id: str = None,
    request_id: str = None,
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
    memory_manager = MemoryManager(user_id=user_id, session_id=session_id)
    memory_manager.load_all_memory()
    memory_summary = memory_manager.get_memory_summary()

    messages_for_prompt = memory_manager.build_recent_messages(
        current_message=user_message, session_id=session_id, limit=8
    )
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
        "messages": messages_for_prompt,
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

    # 解析意图标记
    intent = None
    if current_agent == "chat":
        response, intent = parse_intent(response)

    memory_manager.save_conversation(
        user_message=user_message,
        agent_response=response,
        agent_type=current_agent,
        session_id=session_id,
        request_id=request_id,
    )
    route_decision = final_state.get("route_decision")
    if route_decision:
        memory_manager.save_route_state(
            route_decision,
            working_memory={
                "active_agent": current_agent,
                "intent": route_decision.get("intent"),
                "mode": route_decision.get("mode"),
            },
            session_id=session_id,
            request_id=request_id,
        )

    result = {
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
    if intent:
        result["intent"] = intent
    return result


def chat_stream(state: AgentState):
    """流式闲聊节点（带意图检测）

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

    # 只缓存可能构成意图标记的尾部，正文保持端到端流式输出。
    # 标记通常出现在回复末尾，因此仅在标记前缀尚未完整时短暂保留少量字符。
    pending = ""
    marker_prefixes = ("[INTENT_JSON]", "[INTENT:food]", "[INTENT:exercise]")

    def emit_safe_text(text: str):
        nonlocal pending
        pending += text

        while pending:
            starts = [pending.find(prefix) for prefix in marker_prefixes]
            starts = [index for index in starts if index >= 0]
            marker_start = min(starts) if starts else -1

            if marker_start > 0:
                yield pending[:marker_start]
                pending = pending[marker_start:]
                continue

            if marker_start == 0:
                if pending.startswith("[INTENT_JSON]"):
                    end_tag = "[/INTENT_JSON]"
                    end = pending.find(end_tag)
                    if end < 0:
                        return
                    raw = pending[len("[INTENT_JSON]"):end]
                    pending = pending[end + len(end_tag):]
                    try:
                        import json as _json
                        yield ("intent", _json.loads(raw))
                    except Exception:
                        pass
                    continue

                match = re.match(r"\[INTENT:(food|exercise)\](.*?)(?:\n|$)", pending, re.S)
                if not match:
                    return
                raw_data = match.group(2).strip()
                pending = pending[match.end():]
                parts = raw_data.split("|")
                if len(parts) >= 3:
                    try:
                        from .chat_agent import _parse_calories, _parse_int
                        if match.group(1) == "food":
                            intent = {
                                "type": "food",
                                "data": {
                                    "food_name": parts[0].strip(),
                                    "meal_type": parts[1].strip(),
                                    "calories": _parse_calories(parts[2]),
                                },
                            }
                        else:
                            intent = {
                                "type": "exercise",
                                "data": {
                                    "exercise_name": parts[0].strip(),
                                    "duration": _parse_int(parts[1]),
                                    "calories": _parse_calories(parts[2]),
                                },
                            }
                        yield ("intent", intent)
                    except Exception:
                        pass
                continue

            # 没有完整标记时保留最长可能的标记前缀，避免跨 chunk 的标记
            # 被直接发给客户端；其余字符立即发送。
            keep = 0
            for size in range(1, min(len(pending), 14) + 1):
                suffix = pending[-size:]
                if any(prefix.startswith(suffix) for prefix in marker_prefixes):
                    keep = size
            if len(pending) > keep:
                yield pending[:-keep] if keep else pending
                pending = pending[-keep:] if keep else ""
            return

    for chunk in response_generator:
        if chunk:
            yield from emit_safe_text(chunk)

    if pending:
        # 非法/不完整标记不应阻塞正文；只在没有可识别标记时发出。
        if not pending.startswith(("[INTENT_JSON]", "[INTENT:food]", "[INTENT:exercise]")):
            yield pending


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
    daily_stats: dict = None,
    session_id: str = None,
    request_id: str = None,
):
    """Run the existing streaming workflow with an isolated request context."""

    with request_context(user_id, session_id=session_id, request_id=request_id):
        queue_events = []

        def _on_queue(position):
            queue_events.append(position)
            print(f"[stream] LLM 排队中，前面 {position} 人", flush=True)

        from ..llm_manager import LLMManager
        queue_callback_token = LLMManager.set_queue_callback(_on_queue)
        try:
            yield from _stream_user_message_impl(
                user_message=user_message,
                user_id=user_id,
                user_profile=user_profile,
                daily_stats=daily_stats,
                session_id=session_id,
                request_id=request_id,
                queue_events=queue_events,
            )
        finally:
            LLMManager.reset_queue_callback(queue_callback_token)


def _stream_user_message_impl(
    user_message: str,
    user_id: int = 1,
    user_profile: dict = None,
    daily_stats: dict = None,
    session_id: str = None,
    request_id: str = None,
    queue_events: list = None,
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
    import time

    started_at = time.perf_counter()
    queue_events = queue_events if queue_events is not None else []
    user_message_clean = user_message.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    user_message_clean = ' '.join(user_message_clean.split())

    if not user_message_clean.strip():
        yield ("data", "你好，有什么我可以帮助你的吗？")
        return

    # 先读取已有会话状态，再路由。这样“第二个/换成哑铃”等短输入
    # 能继承上一轮领域，同时仍只为最终 Agent 构建一次 prompt。
    print(f"[stream] 加载用户记忆...", flush=True)
    memory_manager = MemoryManager(user_id=user_id, session_id=session_id)
    memory_manager.load_all_memory()
    memory_summary = memory_manager.get_memory_summary()
    messages_for_prompt = memory_manager.build_recent_messages(
        current_message=user_message,
        session_id=session_id,
        limit=8,
    )
    print(f"[stream] 记忆加载完成", flush=True)

    # 客户端超时重试时，已完成的同一 request_id 直接回放上一轮结果，
    # 避免再次执行记录工具或重复写入业务日志。
    if request_id and memory_summary.get("last_request_id") == request_id:
        cached_response = (memory_summary.get("working_memory") or {}).get(
            "last_agent_response"
        )
        if cached_response:
            yield ("status", "该请求已完成，正在返回原结果...")
            yield ("data", cached_response)
            return

    print(f"[stream] 路由分析中...", flush=True)
    result = hybrid_route(
        user_message_clean,
        require_llm_confirm=True,
        context_messages=messages_for_prompt[:-1],
        context_route=memory_summary.get("last_route"),
    )
    agent = result["agent"]
    if agent not in ["nutrition", "fitness"]:
        agent = "chat"
    memory_summary["route_decision"] = result
    print(
        f"[stream] 路由结果: {agent}, method={result.get('method')}, "
        f"elapsed={time.perf_counter() - started_at:.3f}s",
        flush=True,
    )

    if result.get("reason_code") == "MULTI_RECORD":
        yield ("status", "正在分别记录饮食和运动...")
        try:
            response = _record_multi_domain_message(user_message, user_id)
            yield ("data", response)
            memory_manager.save_conversation(
                user_message=user_message,
                agent_response=response,
                agent_type="chat",
                session_id=session_id,
                request_id=request_id,
            )
            memory_manager.save_route_state(
                result,
                working_memory={"active_agent": "chat", "intent": "record", "completed": True},
                session_id=session_id,
                request_id=request_id,
            )
        except Exception as e:
            print(f"[stream] 多意图记录失败: {e}", flush=True)
            yield ("data", "我识别到了饮食和运动，但记录时遇到问题，请稍后重试。")
        return

    enhanced_prompt = memory_manager.enhance_system_prompt(
        AGENT_SYSTEM_PROMPTS[agent], agent, messages_for_prompt
    )
    if result.get("mode") == "multi_domain":
        enhanced_prompt += """

## 综合问题处理
请以主教练视角同时覆盖用户提到的训练和饮食，不要假装已经执行任何未确认的写入；
先给出清晰的分段建议，并指出需要用户补充的关键信息。
"""
    if result.get("mode") == "safety" or result.get("safety_flags"):
        enhanced_prompt += """

## 安全边界
用户提到疼痛、伤病或特殊情况时，不做诊断，不建议忍痛训练；说明应立即停止的信号，
给出低风险的一般性建议，并在持续、加重或伴随严重症状时建议咨询专业医生。
"""
    enhanced_prompts = {agent: enhanced_prompt}

    state = {
        "messages": messages_for_prompt,
        "user_id": user_id,
        "user_profile": user_profile or memory_manager.load_profile(),
        "daily_stats": daily_stats or {},
        "current_agent": agent,
        "retry_count": 0,
        "review_history": [],
        "memory_summary": memory_summary,
        "enhanced_prompts": enhanced_prompts
    }

    # 统一状态消息，给用户即时反馈降低感知等待
    _status_messages = {
        "nutrition": "Agent正在思考...",
        "fitness": "Agent正在思考...",
        "chat": "Agent正在思考...",
    }
    yield ("status", _status_messages.get(agent, "Agent正在思考..."))

    print(f"[stream] 开始调用 {agent} agent...", flush=True)
    if agent == "nutrition":
        response_generator = nutrition_stream(state)
    elif agent == "fitness":
        response_generator = fitness_stream(state)
    else:
        response_generator = chat_stream(state)

    # 收集完整回复用于保存对话历史，但不阻塞上游 chunk 的发送。
    full_response = ""
    completed = True
    try:
        for chunk in response_generator:
            # 发送排队状态事件
            if queue_events:
                pos = queue_events.pop(0)
                yield ("queue", pos)

            if isinstance(chunk, tuple):
                event_type, event_value = chunk
                if event_type == "intent":
                    yield ("intent", event_value)
                continue

            full_response += chunk
            # 过滤意图标记，不发给前端
            if '[INTENT_JSON]' in chunk or '[INTENT:' in chunk:
                clean_chunk = re.sub(r'\[INTENT_JSON\].*?\[/INTENT_JSON\]', '', chunk)
                clean_chunk = re.sub(r'\n?\[INTENT:(?:food|exercise)\].*?(?:\n|$)', '', clean_chunk)
                clean_chunk = clean_chunk.strip()
                if clean_chunk:
                    yield ("data", clean_chunk)
            else:
                yield ("data", chunk)
    except Exception as e:
        completed = False
        error_msg = f"抱歉，处理时出现问题: {str(e)[:200]}"
        print(f"[stream] 迭代异常: {e}")
        full_response += error_msg
        yield ("data", error_msg)

    # 只有完整成功的回复才进入长期历史，避免错误信息自我强化。
    if completed and full_response:
        memory_manager.save_conversation(
            user_message=user_message,
            agent_response=full_response,
            agent_type=agent,
            session_id=session_id,
            request_id=request_id,
        )
        memory_manager.save_route_state(
            result,
            working_memory={
                "active_agent": agent,
                "intent": result.get("intent"),
                "mode": result.get("mode"),
            },
            session_id=session_id,
            request_id=request_id,
        )
        print(
            f"[stream] request_id={request_id or '-'} completed="
            f"{time.perf_counter() - started_at:.3f}s",
            flush=True,
        )
