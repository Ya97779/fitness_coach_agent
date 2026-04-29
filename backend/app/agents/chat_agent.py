"""闲聊 Agent - 处理日常对话"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, Iterator
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

    parts = []

    goal = memory_summary.get("goal", "未知")
    today_intake = memory_summary.get("today_intake", 0)
    today_burn = memory_summary.get("today_burn", 0)
    week_avg = memory_summary.get("week_avg_intake", 0)

    if goal:
        parts.append(f"用户目标: {goal}")

    if today_intake > 0 or today_burn > 0:
        parts.append(f"今日: 摄入{today_intake:.0f}kcal, 消耗{today_burn:.0f}kcal")

    if week_avg > 0:
        parts.append(f"本周日均摄入: {week_avg:.0f}kcal")

    conversation_history = memory_summary.get("conversation_history", [])
    if conversation_history:
        history_parts = ["【近期对话】"]
        for i, msg in enumerate(conversation_history[-6:]):
            role = "用户" if msg.get("role") == "user" else "AI"
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            history_parts.append(f"{role}: {content}")
        parts.append("\n".join(history_parts))

    if parts:
        return "\n\n【用户记忆】" + "\n".join(parts)
    return ""


def chat_with_user(messages: list, user_id: int, memory_summary: Dict[str, Any] = None, enhanced_prompt: str = None, stream: bool = False):
    """直接与用户对话（不需要工具调用）

    Args:
        messages: 消息列表
        user_id: 用户ID
        memory_summary: 记忆摘要（可选，用于向后兼容）
        enhanced_prompt: 增强后的 system prompt（可选）
        stream: 是否使用流式输出

    Returns:
        str | Iterator[str]: LLM 生成的回复，或回复片段的迭代器
    """
    from ..llm_manager import LLMManager
    llm = LLMManager.get_llm(temperature=0.7)

    if enhanced_prompt:
        system_content = enhanced_prompt
    else:
        base_prompt = AGENT_SYSTEM_PROMPTS["chat"]
        if memory_summary:
            memory_context = format_memory_context(memory_summary, "chat")
            if memory_context:
                base_prompt += memory_context
        system_content = base_prompt

    system_msg = SystemMessage(content=system_content)

    def generate_response():
        try:
            if stream:
                for chunk in llm.stream([system_msg] + messages):
                    if chunk.content:
                        yield chunk.content
            else:
                response = llm.invoke([system_msg] + messages)
                yield response.content
        except Exception as e:
            error_msg = str(e)
            if "1214" in error_msg or "messages" in error_msg.lower():
                yield f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            else:
                yield f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"

    return generate_response()
