"""健身教练 Agent - 负责健身计划、动作指导"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Dict, Any, Optional
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
def get_user_fitness_info(user_id: int):
    """获取用户的健身相关信息（身高、体重、年龄、体能水平）"""
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
                "tdee": user.tdee
            }
        return "未找到用户信息"
    finally:
        db.close()


@tool
def log_exercise(user_id: int, exercise_type: str, duration: int, calories: float, sets: int = None, reps: int = None):
    """记录用户进行的运动及消耗的热量到数据库"""
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

        notes = f"{sets}x{reps}" if sets and reps else f"{duration}分钟"
        exercise_item = models.ExerciseItem(
            log_id=log.id,
            type=exercise_type,
            duration=duration,
            calories=calories,
            notes=notes
        )
        log.burn_calories += calories
        db.add(exercise_item)
        db.commit()

        return f"已记录: {exercise_type}, {notes}, 消耗 {calories} kcal"
    finally:
        db.close()


@tool
def estimate_exercise_calories(exercise_type: str, duration: int, intensity: str = "medium", user_weight: float = 70):
    """估算运动消耗的热量（使用MET值计算）"""
    met_values = {
        "跑步": {"light": 7, "medium": 10, "intense": 14},
        "慢跑": {"light": 6, "medium": 9, "intense": 12},
        "游泳": {"light": 6, "medium": 10, "intense": 14},
        "快走": {"light": 4, "medium": 5, "intense": 7},
        "骑行": {"light": 5, "medium": 8, "intense": 12},
        "跳绳": {"light": 8, "medium": 12, "intense": 15},
        "瑜伽": {"light": 2, "medium": 3, "intense": 5},
        "HIIT": {"light": 8, "medium": 12, "intense": 17},
        "力量训练": {"light": 4, "medium": 6, "intense": 8}
    }

    met = met_values.get(exercise_type, {}).get(intensity, 5)
    calories = met * user_weight * (duration / 60)

    return {
        "exercise_type": exercise_type,
        "duration": duration,
        "calories": round(calories, 1)
    }


@tool
def search_fitness_knowledge(query: str):
    """搜索健身专业知识（仅检索，不生成回答）

    从 RAG 知识库检索相关信息，返回检索结果供大模型生成回答。

    Args:
        query: 搜索关键词

    Returns:
        str: RAG 检索结果（未找到时返回提示信息）
    """
    rag = get_rag()
    results = rag.search(query, top_k=3, mode="hybrid")

    if not results:
        return f"【RAG检索】未在知识库中找到相关信息"

    content = results[0].get("content", "")
    if content:
        return f"【RAG检索】\n{content}"
    return f"【RAG检索】未在知识库中找到相关信息"


fitness_tools = [
    get_user_fitness_info,
    log_exercise,
    estimate_exercise_calories,
    search_fitness_knowledge
]


def format_fitness_memory(memory_summary: Dict[str, Any]) -> str:
    """格式化健身相关的记忆上下文

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

    context_parts = [f"用户目标: {goal}"]

    if today_burn > 0:
        context_parts.append(f"今日已消耗: {today_burn:.0f} kcal")
    else:
        context_parts.append("今日暂无运动记录")

    return "\n\n【用户健身记忆】" + "\n".join(context_parts)


def fitness_with_user(
    messages: list,
    user_id: int,
    memory_summary: Optional[Dict[str, Any]] = None
) -> str:
    """健身教练对话（支持工具调用）

    工作流程：
    1. LLM 判断是否需要调用工具
    2. 执行工具获取检索结果
    3. 将检索结果作为上下文，让 LLM 生成优化后的回答
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    system_content = AGENT_SYSTEM_PROMPTS["fitness"]

    if memory_summary:
        system_content += format_fitness_memory(memory_summary)

    system_content += """

重要：
1. 当用户询问健身动作、训练计划等专业问题时，使用 search_fitness_knowledge 工具
2. 工具返回【RAG检索】后，将检索到的信息作为上下文，结合你的专业知识，生成优化后的回答
3. 如果 RAG 检索未找到信息，请基于你自身的健身知识为用户提供专业回答
4. 不要直接返回原始检索结果，要经过你的理解和整理后再回答用户
"""
    system_msg = SystemMessage(content=system_content)

    chat_history = [system_msg] + list(messages)

    for _ in range(2):
        try:
            response = llm.bind_tools(fitness_tools).invoke(chat_history)
        except Exception as e:
            error_msg = str(e)
            if "1214" in error_msg or "messages" in error_msg.lower():
                return f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            return f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"

        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            return response.content if hasattr(response, 'content') else str(response)

        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']

            for t in fitness_tools:
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

    return response.content if hasattr(response, 'content') else str(response)
