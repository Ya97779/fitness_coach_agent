"""营养师 Agent - 负责饮食计划、热量计算"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Dict, Any, Optional, Iterator
import os
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS
from .. import models, database
from ..rag import ModernRAG
from datetime import date

load_dotenv()

_rag_instance = None

def get_rag():
    """获取 RAG 实例（懒加载）"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ModernRAG(enable_agentic=True)
    return _rag_instance


@tool
def get_user_nutrition_info(user_id: int):
    """获取用户的营养相关信息（身高、体重、BMR、TDEE、过敏史）"""
    db = database.SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            return {
                "height": user.height,
                "weight": user.weight,
                "age": user.age,
                "gender": user.gender,
                "bmr": user.bmr,
                "tdee": user.tdee,
                "allergies": user.allergies,
                "target_weight": user.target_weight
            }
        return "未找到用户信息"
    finally:
        db.close()


@tool
def log_food_intake(user_id: int, food_name: str, calories: float, protein: float = 0, fat: float = 0, carbs: float = 0):
    """记录用户摄入的食物及其营养成分到数据库"""
    db = database.SessionLocal()
    try:
        today = date.today()
        log = db.query(models.DailyLog).filter(
            models.DailyLog.user_id == user_id,
            models.DailyLog.date == today
        ).first()

        if not log:
            log = models.DailyLog(user_id=user_id, date=today)
            db.add(log)
            db.commit()
            db.refresh(log)

        food_item = models.FoodItem(log_id=log.id, name=food_name, calories=calories)
        log.intake_calories += calories
        db.add(food_item)
        db.commit()

        return f"已记录: {food_name}, {calories} kcal"
    finally:
        db.close()


@tool
def get_daily_nutrition_summary(user_id: int):
    """获取用户当日的营养摄入总结"""
    db = database.SessionLocal()
    try:
        today = date.today()
        log = db.query(models.DailyLog).filter(
            models.DailyLog.user_id == user_id,
            models.DailyLog.date == today
        ).first()

        user = db.query(models.User).filter(models.User.id == user_id).first()
        tdee = user.tdee if user else 2000

        if log:
            return {
                "intake_calories": log.intake_calories,
                "burn_calories": log.burn_calories,
                "net_calories": log.intake_calories - log.burn_calories,
                "tdee": tdee
            }
        return {"intake_calories": 0, "burn_calories": 0, "net_calories": 0, "tdee": tdee}
    finally:
        db.close()


@tool
def search_food_nutrition(food_name: str):
    """搜索食物营养信息（仅检索，不生成回答）

    从 API 查询食物营养信息，返回检索结果供大模型生成回答。

    Args:
        food_name: 食物名称

    Returns:
        str: API 检索结果（未找到时返回提示信息）
    """
    from ..food_api import search_food_nutrient
    result = search_food_nutrient(food_name)

    if result:
        return f"【API检索】{food_name}: 热量 {result['calories']} kcal, 蛋白质 {result['protein']}g, 脂肪 {result['fat']}g, 碳水 {result['carbs']}g"
    else:
        return f"【API检索】未找到 {food_name} 的营养信息"


@tool
def search_nutrition_knowledge(query: str):
    """搜索营养与饮食专业知识（仅检索，不生成回答）

    从 RAG 知识库检索营养相关的专业知识，包括饮食原理、营养素功能、
    食物搭配、膳食指南、增肌/减脂饮食策略等。

    适用场景：
    - 营养素作用与功能（如蛋白质、碳水、脂肪的作用）
    - 饮食策略与原理（如增肌饮食、减脂饮食、间歇性断食）
    - 膳食搭配与食谱建议
    - 营养补充剂知识
    - 特殊人群饮食（如糖尿病、高血压患者的饮食）

    Args:
        query: 搜索关键词

    Returns:
        str: RAG 检索结果（未找到时返回提示信息）
    """
    rag = get_rag()
    results = rag.search(query, top_k=3, mode="hybrid")

    print(f"[RAG] 营养知识检索: query='{query}', results={len(results)}")

    if not results:
        return f"【RAG检索】未在知识库中找到相关信息"

    content_parts = []
    for i, r in enumerate(results[:3]):
        c = r.get("content", "")
        if c:
            if len(c) > 500:
                c = c[:500] + "..."
            content_parts.append(f"[来源{i+1}] {c}")

    if content_parts:
        return f"【RAG检索】\n" + "\n\n".join(content_parts)
    return f"【RAG检索】未在知识库中找到相关信息"


nutrition_tools = [
    get_user_nutrition_info,
    log_food_intake,
    get_daily_nutrition_summary,
    search_food_nutrition,
    search_nutrition_knowledge
]


def format_nutrition_memory(memory_summary: Dict[str, Any]) -> str:
    """格式化营养相关的记忆上下文

    Args:
        memory_summary: 记忆摘要

    Returns:
        str: 格式化的记忆上下文
    """
    if not memory_summary:
        return ""

    goal = memory_summary.get("goal", "未知")
    today_intake = memory_summary.get("today_intake", 0)
    today_burn = memory_summary.get("today_burn", 0)
    week_avg = memory_summary.get("week_avg_intake", 0)

    context_parts = [f"用户目标: {goal}"]

    if today_intake > 0:
        context_parts.append(f"今日已摄入: {today_intake:.0f} kcal")
        remaining = 2000 - today_intake
        if remaining > 0:
            context_parts.append(f"今日剩余可摄入: ~{remaining:.0f} kcal")
        else:
            context_parts.append("今日已超过目标")

    if week_avg > 0:
        context_parts.append(f"本周日均摄入: {week_avg:.0f} kcal")

    conversation_history = memory_summary.get("conversation_history", [])
    nutrition_history = [msg for msg in conversation_history if msg.get("agent_type") == "nutrition"]
    if nutrition_history:
        history_parts = ["【近期营养咨询】"]
        for msg in nutrition_history[-2:]:
            content = msg.get("content", "")
            if len(content) > 80:
                content = content[:80] + "..."
            history_parts.append(f"- {content}")
        context_parts.append("\n".join(history_parts))

    return "\n\n【用户营养记忆】" + "\n".join(context_parts)


def nutrition_with_user(
    messages: list,
    user_id: int,
    memory_summary: Optional[Dict[str, Any]] = None,
    enhanced_prompt: str = None,
    stream: bool = False
) -> str | Iterator[str]:
    """营养师对话（支持工具调用）

    工作流程：
    1. LLM 判断是否需要调用工具
    2. 执行工具获取检索结果
    3. 将检索结果作为上下文，让 LLM 生成优化后的回答

    Args:
        messages: 消息列表
        user_id: 用户ID
        memory_summary: 记忆摘要（可选）
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
        system_content = AGENT_SYSTEM_PROMPTS["nutrition"]
        if memory_summary:
            system_content += format_nutrition_memory(memory_summary)

    system_content += """

重要：
1. 当用户询问食物热量、营养成分等问题时，使用 search_food_nutrition 工具查询 API
2. 当用户询问营养原理、饮食策略、膳食搭配、营养素功能等专业知识时，使用 search_nutrition_knowledge 工具从知识库检索
3. 两个工具可以配合使用：先用 API 查具体食物数据，再用知识库查饮食建议
4. 工具返回【API检索】或【RAG检索】后，将检索到的信息作为上下文，结合你的专业知识，生成优化后的回答
5. 如果检索工具未找到信息，请基于你自身的营养知识为用户提供专业回答
6. 不要直接返回原始检索结果，要经过你的理解和整理后再回答用户
"""
    system_msg = SystemMessage(content=system_content)
    chat_history = [system_msg] + list(messages)

    def generate_response():
        try:
            response = llm.bind_tools(nutrition_tools).invoke(chat_history)
        except Exception as e:
            error_msg = str(e)
            if "1214" in error_msg or "messages" in error_msg.lower():
                return f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            return f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"

        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            if stream:
                if hasattr(response, 'content') and response.content:
                    for chunk in llm.stream(chat_history + [response]):
                        if chunk.content:
                            yield chunk.content
            else:
                yield response.content if hasattr(response, 'content') else str(response)
            return

        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']

            for t in nutrition_tools:
                if t.name == tool_name:
                    try:
                        tool_result = t.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                    break
            else:
                tool_result = f"未知工具: {tool_name}"

            tool_messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id
            })

        chat_history.append(response)
        chat_history.extend(tool_messages)

        try:
            if stream:
                for chunk in llm.stream(chat_history):
                    if chunk.content:
                        yield chunk.content
            else:
                final_response = llm.invoke(chat_history)
                yield final_response.content if hasattr(final_response, 'content') else str(final_response)
        except Exception as e:
            error_msg = str(e)
            if "1214" in error_msg or "messages" in error_msg.lower():
                yield f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            else:
                yield f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"

    return generate_response()


def nutrition_with_user_stream(
    messages: list,
    user_id: int,
    memory_summary: Dict[str, Any] = None,
    enhanced_prompt: str = None
):
    """流式营养师对话（支持工具调用）

    .. deprecated::
        请使用 nutrition_with_user(..., stream=True) 代替

    工作流程：
    1. 执行工具调用（非流式）
    2. 流式返回最终 LLM 响应

    Args:
        messages: 消息列表
        user_id: 用户ID
        memory_summary: 记忆摘要（可选）
        enhanced_prompt: 增强后的 system prompt（可选）

    Yields:
        str: LLM 生成的回复片段
    """
    return nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt, stream=True)
