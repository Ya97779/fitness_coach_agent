"""营养师 Agent - 负责饮食计划、热量计算"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import Dict, Any, Optional, Iterator
import os
import re
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS
from .. import models, database
from ..rag import ModernRAG
from datetime import date

load_dotenv()

_rag_instance = None

# 记录意图关键词
_FOOD_RECORD_PATTERNS = [
    r"帮我记录", r"记录一下", r"记到", r"记一下",
    r"吃了", r"喝了", r"吃了一个", r"喝了一杯", r"吃了一碗",
    r"早餐吃了", r"午餐吃了", r"晚餐吃了", r"加餐吃了",
]

# 常见食物热量估算（kcal/份）
_FOOD_CALORIE_ESTIMATES = {
    "苹果": 95, "香蕉": 105, "鸡蛋": 78, "牛奶": 150,
    "面包": 120, "米饭": 200, "面条": 220, "馒头": 220,
    "包子": 200, "饺子": 250, "粥": 100, "豆浆": 80,
    "鸡胸肉": 165, "鸡腿": 180, "鸡翅": 100, "鸡排": 350,
    "猪肉": 250, "牛肉": 200, "鱼": 150, "虾": 100,
    "蔬菜沙拉": 80, "炒菜": 200, "汤": 80,
    "酸奶": 120, "橙子": 70, "葡萄": 100,
    "拉面": 550, "牛肉面": 550, "兰州拉面": 550, "兰州牛肉拉面": 600,
    "炒饭": 450, "炒面": 400, "汉堡": 550, "薯条": 400,
    "可乐": 140, "咖啡": 5, "奶茶": 350,
    "鸭腿饭": 650, "鸭腿": 300,  "拌面": 400,
    "盖饭": 600, "便当": 550, "披萨": 700, "沙拉": 150,
}


def _detect_food_record_intent(user_message: str) -> bool:
    """检测用户是否有食物记录意图"""
    for pattern in _FOOD_RECORD_PATTERNS:
        if pattern in user_message:
            return True
    return False


def _extract_food_names(user_message: str) -> list:
    """从用户消息中提取食物名称（支持多种食物）

    Returns:
        list: [(food_name, meal_type), ...]
    """
    results = []

    # 先尝试匹配已知食物（按长度降序，优先匹配长的）
    matched_foods = []
    for food_name in sorted(_FOOD_CALORIE_ESTIMATES.keys(), key=len, reverse=True):
        if food_name in user_message and food_name not in [r[0] for r in matched_foods]:
            matched_foods.append(food_name)

    if matched_foods:
        for food_name in matched_foods:
            # 尝试从上下文推断餐次
            # 找到食物名在原文中的位置，看前面的上下文
            idx = user_message.find(food_name)
            context_before = user_message[max(0, idx-10):idx]
            meal_type = _detect_meal_type(context_before + food_name)
            results.append((food_name, meal_type))
        return results

    # 正则提取：按逗号/句号分割后逐段提取
    segments = re.split(r'[，。,;；]', user_message)
    for seg in segments:
        m = re.search(r'[吃喝]了?(?:一份?|一个|一碗|一杯|一盘|一块|一根)?(.+?)(?:$)', seg)
        if m:
            name = m.group(1).strip()
            if 1 <= len(name) <= 20:
                meal_type = _detect_meal_type(seg)
                results.append((name, meal_type))

    return results if results else [("食物", _detect_meal_type(user_message))]


def _estimate_calories(food_name: str) -> int:
    """估算食物热量（本地查找表）"""
    if food_name in _FOOD_CALORIE_ESTIMATES:
        return _FOOD_CALORIE_ESTIMATES[food_name]
    # 模糊匹配
    for name, cal in _FOOD_CALORIE_ESTIMATES.items():
        if name in food_name or food_name in name:
            return cal
    return 300  # 默认估算


def _get_food_nutrition(food_name: str) -> dict:
    """获取食物营养数据：优先 API，API 查不到再用本地估算

    Returns:
        dict: {"calories": float, "protein": float, "fat": float, "carbs": float}
    """
    from ..food_api import search_food_nutrient
    api_result = search_food_nutrient(food_name)
    if api_result:
        return {
            "calories": float(api_result['calories']),
            "protein": float(api_result.get('protein', 0)),
            "fat": float(api_result.get('fat', 0)),
            "carbs": float(api_result.get('carbs', 0)),
        }
    return {
        "calories": float(_estimate_calories(food_name)),
        "protein": 0, "fat": 0, "carbs": 0,
    }


def _detect_meal_type(user_message: str) -> str:
    """从用户消息中检测餐次"""
    if any(kw in user_message for kw in ["早餐", "早上", "早饭"]):
        return "breakfast"
    if any(kw in user_message for kw in ["午餐", "中午", "午饭"]):
        return "lunch"
    if any(kw in user_message for kw in ["晚餐", "晚上", "晚饭"]):
        return "dinner"
    if any(kw in user_message for kw in ["加餐", "零食", "下午茶"]):
        return "snack"
    return "lunch"  # 默认午餐


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
def log_food_intake(user_id: int, food_name: str, calories: float, meal_type: str = "lunch", protein: float = 0, fat: float = 0, carbs: float = 0):
    """记录用户摄入的食物及其营养成分到数据库。meal_type 可选值: breakfast(早餐), lunch(午餐), dinner(晚餐), snack(加餐)"""
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

        food_item = models.FoodItem(log_id=log.id, name=food_name, calories=calories, meal_type=meal_type)
        log.intake_calories += calories
        db.add(food_item)
        db.commit()

        return f"已记录: {food_name}, {calories} kcal, 餐次: {meal_type}"
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
        tdee = user.tdee if user and user.tdee else None

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
        return f"【API检索】未找到 {food_name} 的营养信息。请根据你的营养知识估算该食物的热量，然后调用 log_food_intake 工具记录到用户日志中。"


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

    system_content += f"""

## 关键规则
- 当前用户 ID = {user_id}，调用工具时必须传入
- 用户要求记录饮食时，必须调用 log_food_intake（user_id={user_id}）
- search_food_nutrition 返回空时，根据营养知识估算热量后调用 log_food_intake 记录
- meal_type：早餐→breakfast，午餐→lunch，晚餐→dinner，加餐→snack
- 专业知识问题用 search_nutrition_knowledge 检索
"""
    system_msg = SystemMessage(content=system_content)
    chat_history = [system_msg] + list(messages)

    # 获取用户原始消息用于意图检测
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    want_record = _detect_food_record_intent(user_message)

    def generate_response():
        called_tools = set()

        try:
            response = llm.bind_tools(nutrition_tools).invoke(chat_history)
        except Exception as e:
            error_msg = str(e)
            print(f"[nutrition_agent] invoke 异常: {error_msg}")
            if "1214" in error_msg or "messages" in error_msg.lower():
                yield f"抱歉，API调用出现问题，请检查API配置是否正确。错误信息: {error_msg[:200]}"
            else:
                yield f"抱歉，处理您的请求时出现问题: {error_msg[:200]}"
            return

        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            content = response.content if hasattr(response, 'content') else str(response)
            # 兜底：用户要求记录但 LLM 没调用任何工具
            if want_record:
                food_items = _extract_food_names(user_message)
                recorded = []
                for food_name, meal_type in food_items:
                    nutrition = _get_food_nutrition(food_name)
                    print(f"[nutrition_agent] 兜底记录: {food_name}, {nutrition['calories']}kcal, {meal_type}")
                    fallback_result = log_food_intake.invoke({
                        "user_id": user_id,
                        "food_name": food_name,
                        "calories": nutrition['calories'],
                        "meal_type": meal_type,
                        "protein": nutrition['protein'],
                        "fat": nutrition['fat'],
                        "carbs": nutrition['carbs'],
                    })
                    print(f"[nutrition_agent] 兜底记录结果: {fallback_result}")
                    recorded.append(f"{food_name} {nutrition['calories']:.0f}kcal({meal_type})")
                if content:
                    yield content
                    yield f"\n\n已自动记录：{', '.join(recorded)}"
                else:
                    yield f"已为你记录：{', '.join(recorded)}。"
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
            print(f"[nutrition_agent] 工具调用: {tool_name}({tool_args})")

            for t in nutrition_tools:
                if t.name == tool_name:
                    try:
                        tool_result = t.invoke(tool_args)
                        print(f"[nutrition_agent] 工具结果: {tool_name} → {tool_result}")
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                        print(f"[nutrition_agent] 工具异常: {tool_name} → {e}")
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

        # 兜底：用户要求记录但 LLM 没调用 log_food_intake
        if want_record and "log_food_intake" not in called_tools:
            food_items = _extract_food_names(user_message)
            recorded = []
            for food_name, meal_type in food_items:
                nutrition = _get_food_nutrition(food_name)
                print(f"[nutrition_agent] 兜底记录: {food_name}, {nutrition['calories']}kcal, {meal_type}")
                fallback_result = log_food_intake.invoke({
                    "user_id": user_id,
                    "food_name": food_name,
                    "calories": nutrition['calories'],
                    "meal_type": meal_type,
                    "protein": nutrition['protein'],
                    "fat": nutrition['fat'],
                    "carbs": nutrition['carbs'],
                })
                print(f"[nutrition_agent] 兜底记录结果: {fallback_result}")
                recorded.append(f"{food_name} {nutrition['calories']:.0f}kcal({meal_type})")
            # 追加工具消息让 LLM 知道已记录
            chat_history.append({
                "role": "tool",
                "content": f"已自动记录：{', '.join(recorded)}",
                "tool_call_id": "fallback_log"
            })

        try:
            if stream:
                chunk_count = 0
                total_content = ""
                print(f"[nutrition_agent] 开始 LLM 流式调用, messages={len(chat_history)}", flush=True)
                llm_with_tools = llm.bind_tools(nutrition_tools)
                for chunk in llm_with_tools.stream(chat_history):
                    # 记录 chunk 完整信息用于调试
                    has_content = bool(chunk.content)
                    has_tool_calls = bool(getattr(chunk, 'tool_calls', None))
                    if chunk_count < 3:
                        print(f"[nutrition_agent] chunk[{chunk_count}]: content={has_content}({len(chunk.content) if chunk.content else 0}), tool_calls={has_tool_calls}, type={type(chunk).__name__}", flush=True)
                    if has_content:
                        chunk_count += 1
                        total_content += chunk.content
                        if chunk_count == 1:
                            print(f"[nutrition_agent] LLM 首个 chunk: {chunk.content[:100]}", flush=True)
                        yield chunk.content
                print(f"[nutrition_agent] LLM 流式完成: {chunk_count} chunks, 总长度={len(total_content)}", flush=True)
                if chunk_count == 0:
                    print(f"[nutrition_agent] 流式返回空，尝试非流式兜底...", flush=True)
                    try:
                        fallback = llm_with_tools.invoke(chat_history)
                        fb_content = fallback.content if hasattr(fallback, 'content') else str(fallback)
                        fb_tool_calls = getattr(fallback, 'tool_calls', None)
                        print(f"[nutrition_agent] 非流式兜底: 长度={len(fb_content)}, tool_calls={bool(fb_tool_calls)}, 前100字={fb_content[:100]}", flush=True)
                        if fb_content:
                            yield fb_content
                        else:
                            yield "抱歉，AI 未能生成回复，请重试。"
                    except Exception as fb_err:
                        print(f"[nutrition_agent] 非流式兜底也失败: {fb_err}", flush=True)
                        yield "抱歉，AI 服务暂时不可用，请稍后重试。"
            else:
                final_response = llm.invoke(chat_history)
                content = final_response.content if hasattr(final_response, 'content') else str(final_response)
                print(f"[nutrition_agent] LLM 非流式返回: 长度={len(content)}, 前100字={content[:100]}", flush=True)
                yield content
        except Exception as e:
            error_msg = str(e)
            print(f"[nutrition_agent] LLM 调用异常: {error_msg[:200]}", flush=True)
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
