"""闲聊 Agent - 处理日常对话，支持记录意图检测"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, Iterator
import re
import os
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS

load_dotenv()

# 意图检测正则
_INTENT_PATTERN = re.compile(r'\n?\[INTENT:(food|exercise)\](.+?)(?:\n|$)')
_INTENT_JSON_PATTERN = re.compile(r'\n?\[INTENT_JSON\](.*?)\[/INTENT_JSON\]')


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

    # 添加意图检测指令
    system_content += """

## 记录意图检测
当你检测到用户有记录饮食或运动的意图时，在回复末尾单独一行输出意图标记：
- 饮食：[INTENT:food]食物名|餐次|估算热量kcal
- 运动：[INTENT:exercise]运动名|时长分钟|估算热量kcal

示例：
用户: "我晚餐吃了一份鸡腿饭"
你的回复: 鸡腿饭是经典快餐，味道不错！
[INTENT:food]鸡腿饭|dinner|650

用户: "今天跑步30分钟"
你的回复: 跑步是很好的有氧运动！
[INTENT:exercise]跑步|30|300

规则：
- 只在用户明确提到"吃了/喝了/做了运动"时才输出标记
- 闲聊、提问、咨询、计划讨论等不输出标记
- 热量用常识估算，不需要精确
- 餐次：早餐→breakfast，午餐→lunch，晚餐→dinner，加餐/零食→snack
- 标记必须在回复末尾单独一行
"""

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


def parse_intent(response_text: str) -> tuple:
    """从 chat agent 回复中解析意图标记

    支持两种格式:
    - [INTENT:food]name|meal|calories
    - [INTENT_JSON]{"type":"food","data":{...}}[/INTENT_JSON]

    Args:
        response_text: LLM 原始回复

    Returns:
        (clean_text, intent_dict): 清理后的文本和意图信息（无意图时 intent_dict 为 None）
    """
    # 优先匹配 JSON 格式
    json_match = _INTENT_JSON_PATTERN.search(response_text)
    if json_match:
        import json as _json
        clean_text = response_text[:json_match.start()].rstrip()
        try:
            intent = _json.loads(json_match.group(1))
            return clean_text, intent
        except Exception:
            return clean_text, None

    # 兜底匹配旧格式
    match = _INTENT_PATTERN.search(response_text)
    if not match:
        return response_text, None

    intent_type = match.group(1)  # "food" or "exercise"
    raw_data = match.group(2).strip()
    clean_text = response_text[:match.start()].rstrip()

    parts = raw_data.split("|")
    intent = {"type": intent_type}

    if intent_type == "food" and len(parts) >= 3:
        intent["data"] = {
            "food_name": parts[0].strip(),
            "meal_type": parts[1].strip(),
            "calories": _parse_calories(parts[2])
        }
    elif intent_type == "exercise" and len(parts) >= 3:
        intent["data"] = {
            "exercise_name": parts[0].strip(),
            "duration": _parse_int(parts[1]),
            "calories": _parse_calories(parts[2])
        }
    else:
        return response_text, None

    return clean_text, intent


def _parse_calories(s: str) -> int:
    """提取热量数值"""
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0


def _parse_int(s: str) -> int:
    """提取整数"""
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0
