"""健身教练 Agent - 负责健身计划、动作指导"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Dict, Any, Optional, Iterator
import os
import re
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS
from .. import models, database
from ..rag import get_rag_instance
from ..calorie_calculator import estimate_calories as calc_calories, MET_VALUES, STRENGTH_CALORIES_PER_SET
from datetime import date

load_dotenv()

# 记录意图关键词
_EXERCISE_RECORD_PATTERNS = [
    r"帮我记录", r"记录一下", r"记到", r"记一下",
    r"跑了", r"游了", r"骑了", r"练了", r"做了",
    r"今天运动", r"今天训练",
]


def _detect_exercise_record_intent(user_message: str) -> bool:
    """检测用户是否有运动记录意图"""
    for pattern in _EXERCISE_RECORD_PATTERNS:
        if pattern in user_message:
            return True
    return False


def _extract_exercise_info(user_message: str) -> tuple:
    """从用户消息中提取运动名称和时长

    Returns:
        (exercise_name, duration_minutes)
    """
    # 尝试提取时长
    duration = 20  # 默认20分钟
    m = re.search(r'(\d+)\s*分钟', user_message)
    if m:
        duration = int(m.group(1))
    m2 = re.search(r'(\d+)\s*小时', user_message)
    if m2:
        duration = int(m2.group(1)) * 60

    # 匹配已知运动（按长度降序）
    all_exercise_names = set(MET_VALUES.keys()) | set(STRENGTH_CALORIES_PER_SET.keys())
    for name in sorted(all_exercise_names, key=len, reverse=True):
        if name in user_message:
            return name, duration

    # 正则：动词 + 数字组/次 + 运动名（如"练了5组卧推"）
    m3 = re.search(r'[做了练跑游骑]+了?\d*[组次个]?(.+?)(?:\s*\d+[kKgGxX]|[\d ]*[分钟小时]|$)', user_message)
    if m3:
        name = m3.group(1).strip()
        if 1 <= len(name) <= 10:
            return name, duration

    # 正则：简单提取
    m4 = re.search(r'[做了练跑游骑]+了?(?:一下|一会)?(.+?)(?:\d+分钟|[，。,.]|$)', user_message)
    if m4:
        name = m4.group(1).strip()
        if 1 <= len(name) <= 10:
            return name, duration

    return "运动", duration


def get_rag():
    """获取 RAG 实例（使用全局单例）"""
    return get_rag_instance(enable_agentic=True)


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

        exercise_item = models.ExerciseItem(
            log_id=log.id,
            type=exercise_type,
            sets=sets,
            reps=reps,
            duration=duration,
            calories=calories
        )
        log.burn_calories += calories
        db.add(exercise_item)
        db.commit()

        return f"已记录: {exercise_type}, {duration}分钟, 消耗 {calories} kcal"
    finally:
        db.close()


@tool
def estimate_exercise_calories(exercise_type: str, duration: int, intensity: str = "medium", user_weight: float = 70):
    """估算运动消耗的热量（使用MET值计算）"""
    calories = calc_calories(exercise_type, user_weight=user_weight, duration=duration, intensity=intensity)
    return {
        "exercise_type": exercise_type,
        "duration": duration,
        "calories": calories
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

    print(f"[RAG] 健身知识检索: query='{query}', results={len(results)}")

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

    conversation_history = memory_summary.get("conversation_history", [])
    fitness_history = [msg for msg in conversation_history if msg.get("agent_type") == "fitness"]
    if fitness_history:
        history_parts = ["【近期健身咨询】"]
        for msg in fitness_history[-2:]:
            content = msg.get("content", "")
            if len(content) > 80:
                content = content[:80] + "..."
            history_parts.append(f"- {content}")
        context_parts.append("\n".join(history_parts))

    return "\n\n【用户健身记忆】" + "\n".join(context_parts)


def fitness_with_user(
    messages: list,
    user_id: int,
    memory_summary: Optional[Dict[str, Any]] = None,
    enhanced_prompt: str = None,
    stream: bool = False
) -> str | Iterator[str]:
    """健身教练对话（支持工具调用）

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
        system_content = AGENT_SYSTEM_PROMPTS["fitness"]
        if memory_summary:
            system_content += format_fitness_memory(memory_summary)

    system_content += f"""

## 关键规则
- 当前用户 ID = {user_id}，调用工具时必须传入
- 用户要求记录运动时，必须调用 log_exercise（user_id={user_id}）
- 专业问题用 search_fitness_knowledge 检索
- 不要直接返回原始检索结果，整理后回答
"""
    # 获取用户原始消息用于意图检测
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    want_record = _detect_exercise_record_intent(user_message)

    system_msg = SystemMessage(content=system_content)
    chat_history = [system_msg] + list(messages)

    def generate_response():
        called_tools = set()

        try:
            response = llm.bind_tools(fitness_tools).invoke(chat_history)
        except Exception as e:
            error_msg = str(e)
            if "1214" in error_msg or "messages" in error_msg.lower():
                yield f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
                return
            yield f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"
            return

        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            content = response.content if hasattr(response, 'content') else str(response)
            # 兜底：用户要求记录但 LLM 没调用任何工具
            if want_record:
                ex_name, duration = _extract_exercise_info(user_message)
                calories = calc_calories(ex_name, duration=duration)
                print(f"[fitness_agent] 兜底记录: {ex_name}, {duration}分钟, {calories}kcal")
                fallback_result = log_exercise.invoke({
                    "user_id": user_id,
                    "exercise_type": ex_name,
                    "duration": duration,
                    "calories": float(calories)
                })
                print(f"[fitness_agent] 兜底记录结果: {fallback_result}")
                if content:
                    yield content
                    yield f"\n\n已自动记录：{ex_name} {duration}分钟，约消耗 {calories} kcal"
                else:
                    yield f"已为你记录 {ex_name} {duration}分钟，约消耗 {calories} kcal。"
            else:
                if content:
                    yield content
            return

        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            called_tools.add(tool_name)
            print(f"[fitness_agent] 工具调用: {tool_name}({tool_args})")

            for t in fitness_tools:
                if t.name == tool_name:
                    try:
                        tool_result = t.invoke(tool_args)
                        print(f"[fitness_agent] 工具结果: {tool_name} → {tool_result}")
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                        print(f"[fitness_agent] 工具异常: {tool_name} → {e}")
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

        # 兜底：用户要求记录但 LLM 没调用 log_exercise
        if want_record and "log_exercise" not in called_tools:
            ex_name, duration = _extract_exercise_info(user_message)
            calories = calc_calories(ex_name, duration=duration)
            print(f"[fitness_agent] 兜底记录: {ex_name}, {duration}分钟, {calories}kcal")
            fallback_result = log_exercise.invoke({
                "user_id": user_id,
                "exercise_type": ex_name,
                "duration": duration,
                "calories": float(calories)
            })
            print(f"[fitness_agent] 兜底记录结果: {fallback_result}")
            chat_history.append({
                "role": "tool",
                "content": fallback_result,
                "tool_call_id": "fallback_log"
            })

        try:
            if stream:
                chunk_count = 0
                total_content = ""
                print(f"[fitness_agent] 开始 LLM 流式调用, messages={len(chat_history)}", flush=True)
                llm_with_tools = llm.bind_tools(fitness_tools)
                raw_idx = 0
                for chunk in llm_with_tools.stream(chat_history):
                    # 记录 chunk 完整信息用于调试
                    has_content = bool(chunk.content)
                    has_tool_calls = bool(getattr(chunk, 'tool_calls', None))
                    if raw_idx < 3 or (not has_content and raw_idx % 30 == 0):
                        print(f"[fitness_agent] chunk[{raw_idx}]: content={has_content}({len(chunk.content) if chunk.content else 0}), tool_calls={has_tool_calls}", flush=True)
                    raw_idx += 1
                    if has_content:
                        chunk_count += 1
                        total_content += chunk.content
                        if chunk_count == 1:
                            print(f"[fitness_agent] LLM 首个 chunk: {chunk.content[:100]}", flush=True)
                        yield chunk.content
                print(f"[fitness_agent] LLM 流式完成: {chunk_count} chunks, 总长度={len(total_content)}", flush=True)
                if chunk_count == 0:
                    print(f"[fitness_agent] 流式返回空，尝试非流式兜底...", flush=True)
                    try:
                        fallback = llm_with_tools.invoke(chat_history)
                        fb_content = fallback.content if hasattr(fallback, 'content') else str(fallback)
                        fb_tool_calls = getattr(fallback, 'tool_calls', None)
                        print(f"[fitness_agent] 非流式兜底: 长度={len(fb_content)}, tool_calls={bool(fb_tool_calls)}, 前100字={fb_content[:100]}", flush=True)
                        if fb_content:
                            yield fb_content
                        else:
                            yield "抱歉，AI 未能生成回复，请重试。"
                    except Exception as fb_err:
                        print(f"[fitness_agent] 非流式兜底也失败: {fb_err}", flush=True)
                        yield "抱歉，AI 服务暂时不可用，请稍后重试。"
            else:
                final_response = llm.invoke(chat_history)
                content = final_response.content if hasattr(final_response, 'content') else str(final_response)
                print(f"[fitness_agent] LLM 非流式返回: 长度={len(content)}, 前100字={content[:100]}", flush=True)
                yield content
        except Exception as e:
            error_msg = str(e)
            print(f"[fitness_agent] LLM 调用异常: {error_msg[:200]}", flush=True)
            if "1214" in error_msg or "messages" in error_msg.lower():
                yield f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            else:
                yield f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"

    return generate_response()
