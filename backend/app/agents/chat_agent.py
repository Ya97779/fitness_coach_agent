"""闲聊 Agent - 处理日常对话"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any
import os
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS

load_dotenv()


def format_memory_context(memory_summary: Dict[str, Any], agent_type: str = "chat") -> str:
    """格式化记忆上下文

    Args:
        memory_summary: 记忆摘要
        agent_type: Agent 类型

    Returns:
        str: 格式化的记忆上下文
    """
    if not memory_summary:
        return ""

    goal = memory_summary.get("goal", "未知")
    today_intake = memory_summary.get("today_intake", 0)
    today_burn = memory_summary.get("today_burn", 0)
    week_avg = memory_summary.get("week_avg_intake", 0)

    context_parts = []

    if goal:
        context_parts.append(f"用户目标: {goal}")

    if today_intake > 0 or today_burn > 0:
        context_parts.append(f"今日: 摄入{today_intake:.0f}kcal, 消耗{today_burn:.0f}kcal")

    if week_avg > 0:
        context_parts.append(f"本周日均摄入: {week_avg:.0f}kcal")

    if context_parts:
        return "\n\n【用户记忆】" + "\n".join(context_parts)
    return ""


def chat_with_user(messages: list, user_id: int, memory_summary: Dict[str, Any] = None) -> str:
    """直接与用户对话（不需要工具调用）

    Args:
        messages: 消息列表
        user_id: 用户ID
        memory_summary: 记忆摘要（可选）

    Returns:
        str: LLM 生成的回复
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    base_prompt = AGENT_SYSTEM_PROMPTS["chat"]
    if memory_summary:
        memory_context = format_memory_context(memory_summary, "chat")
        if memory_context:
            base_prompt += memory_context

    system_msg = SystemMessage(content=base_prompt)
    try:
        response = llm.invoke([system_msg] + messages)
        return response.content
    except Exception as e:
        error_msg = str(e)
        if "1214" in error_msg or "messages" in error_msg.lower():
            return f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
        return f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"
