"""营养师 Agent - 负责饮食计划、热量计算"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import Dict, Any, Optional, Iterator
import os
import re
from .base import AGENT_SYSTEM_PROMPTS, StreamedToolCall
from .. import models, database
from ..runtime_context import get_effective_user_id
from ..food_api import search_food_nutrient
from ..food_cache import get_cached_total_calories, save_food_calorie_basis
from ..rag import ModernRAG
from datetime import date

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


def _execute_tool_calls(tool_calls: list, tools: list) -> list:
    """执行工具调用列表，返回 tool message 列表"""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', 'unknown')

        if not tool_name:
            continue

        print(f"[nutrition_agent] 执行工具: {tool_name}({tool_args})")

        tool_result = None
        for t in tools:
            if t.name == tool_name:
                try:
                    tool_result = t.invoke(tool_args)
                    print(f"[nutrition_agent] 工具结果: {tool_name} → {tool_result}")
                except Exception as e:
                    tool_result = f"工具执行错误: {e}"
                    print(f"[nutrition_agent] 工具异常: {tool_name} → {e}")
                break

        if tool_result is None:
            tool_result = f"未知工具: {tool_name}"

        results.append({
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_id
        })
    return results


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
    """估算食物热量（本地查找表，兜底用）"""
    if food_name in _FOOD_CALORIE_ESTIMATES:
        return _FOOD_CALORIE_ESTIMATES[food_name]
    for name, cal in _FOOD_CALORIE_ESTIMATES.items():
        if name in food_name or food_name in name:
            return cal
    return 300  # 默认估算


def _get_food_nutrition(food_name: str) -> dict:
    """获取默认一份食物的热量：明确份量缓存 → 本地估算。"""
    db = database.SessionLocal()
    try:
        cached_calories = get_cached_total_calories(
            db, food_name, 1, "份"
        )
        if cached_calories is not None:
            print(f"[nutrition_agent] DB 缓存命中: '{food_name}' → {cached_calories}kcal/份")
            return {"calories": cached_calories, "protein": 0, "fat": 0, "carbs": 0}
    finally:
        db.close()

    # 本地估算兜底
    return {
        "calories": float(_estimate_calories(food_name)),
        "protein": 0, "fat": 0, "carbs": 0,
    }


def _save_food_cache(
    food_name: str,
    calories: float,
    portion_qty: Optional[float] = None,
    portion_unit: Optional[str] = None,
):
    """独立写入明确口径的食物热量缓存。"""

    return save_food_calorie_basis(
        food_name,
        calories,
        portion_qty,
        portion_unit,
        source="llm",
    )


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
    user_id = get_effective_user_id(user_id)
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
def log_food_intake(
    user_id: int,
    food_name: str,
    calories: float,
    meal_type: str = "lunch",
    portion_qty: float = 1,
    portion_unit: str = "份",
):
    """记录用户摄入的食物到数据库，并缓存热量数据。

    calories 为该份食物的总热量（如1个鸡蛋≈72kcal, 1碗兰州拉面≈550kcal）。
    portion_qty 和 portion_unit 描述本次摄入份量。工具会换算成每 100g 或
    每 1 单位的缓存基准，同名食物只有单位一致时才会命中。

    meal_type: breakfast(早餐), lunch(午餐), dinner(晚餐), snack(加餐)
    """
    user_id = get_effective_user_id(user_id)
    if portion_qty <= 0:
        raise ValueError("portion_qty 必须大于 0")
    db = database.SessionLocal()
    try:
        today = date.today()
        log = database.get_or_create_daily_log(
            db, models.DailyLog, user_id, today
        )
        db.flush()

        food_item = models.FoodItem(
            log_id=log.id,
            name=food_name,
            calories=calories,
            meal_type=meal_type,
            portion_qty=portion_qty,
            portion_unit=portion_unit,
        )
        log.intake_calories += calories
        db.add(food_item)
        db.commit()

        # 缓存使用独立事务；缓存失败不能回滚用户的饮食记录。
        _save_food_cache(food_name, calories, portion_qty, portion_unit)

        return f"已记录: {food_name}, {calories} kcal, 餐次: {meal_type}"
    finally:
        db.close()


@tool
def get_daily_nutrition_summary(user_id: int):
    """获取用户当日的营养摄入总结"""
    user_id = get_effective_user_id(user_id)
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
    """查询具体食物的热量和三大营养素（按每 100g 或 API 返回口径）。"""

    result = search_food_nutrient(food_name)
    if not result:
        return f"未找到{food_name}的营养数据"
    calories = result.get("calories", 0)
    portion_qty = result.get("portion_qty")
    portion_unit = result.get("portion_unit")
    if calories and portion_qty and portion_unit:
        save_food_calorie_basis(
            food_name,
            calories,
            portion_qty,
            portion_unit,
            source=result.get("source", "fallback"),
        )
    return {
        "food_name": food_name,
        "calories": calories,
        "protein": result.get("protein", 0),
        "fat": result.get("fat", 0),
        "carbs": result.get("carbs", 0),
        "source": result.get("source", "本地数据"),
        "basis_type": result.get("basis_type"),
        "portion_qty": portion_qty,
        "portion_unit": portion_unit,
    }


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
    results = rag.search(query, top_k=5, mode="hybrid")

    print(f"[RAG] 营养知识检索: query='{query}', results={len(results)}")

    if not results:
        return f"【RAG检索】未在知识库中找到相关信息"

    # 过滤垃圾内容和太短的结果
    spam_patterns = ["加微信", "免费获得", "大礼包", "微信号", "扫码", "关注公众号"]
    content_parts = []
    for i, r in enumerate(results[:5]):
        c = r.get("content", "").strip()
        if not c or len(c) < 20:
            continue
        if any(spam in c for spam in spam_patterns):
            print(f"[RAG] 过滤垃圾内容: {c[:50]}...")
            continue
        if len(c) > 500:
            c = c[:500] + "..."
        heading = r.get("metadata", {}).get("heading_path", "")
        prefix = f"[{heading}] " if heading else ""
        content_parts.append(f"[来源{i+1}] {prefix}{c}")
        if len(content_parts) >= 3:
            break

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
- 用户要求记录饮食时，调用 log_food_intake，提供：
  - food_name: 食物名称（如"鸡蛋"、"兰州拉面"）
  - calories: 该份食物的总热量（如1个鸡蛋≈72, 1碗兰州拉面≈550）
  - meal_type: breakfast/lunch/dinner/snack
- 根据用户描述的份量直接估算总热量，不需要拆分每100g和重量
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
        initial_content_streamed = False

        # 已明确的“记录饮食”属于确定性业务动作，复用现有解析和工具，
        # 不再先让模型完整决策一次。这样记录请求可以立即返回，也避免
        # 模型重复执行/重复记录；复杂饮食咨询仍走原有工具链。
        food_items = _extract_food_names(user_message)
        has_explicit_food = any(
            name != "食物" and name in user_message for name, _ in food_items
        )
        if stream and want_record and has_explicit_food:
            recorded = []
            for food_name, meal_type in food_items:
                nutrition = _get_food_nutrition(food_name)
                calories = nutrition["calories"]
                result = log_food_intake.invoke({
                    "user_id": user_id,
                    "food_name": food_name,
                    "calories": calories,
                    "meal_type": meal_type,
                })
                recorded.append(f"{food_name} {calories:.0f}kcal({meal_type})")
                print(f"[nutrition_agent] 快速记录: {result}", flush=True)
            yield f"已为你记录：{', '.join(recorded)}。"
            return

        try:
            if stream:
                # 工具选择本身也使用流式调用；等工具参数完整后再执行，
                # 但不再用 invoke 阻塞首轮模型响应。
                plan_stream = StreamedToolCall(llm, nutrition_tools, chat_history)
                print("[nutrition_agent] 第一轮 LLM 流式工具决策", flush=True)
                for chunk in plan_stream:
                    if getattr(chunk, "content", None):
                        initial_content_streamed = True
                        yield chunk.content
                response = plan_stream.response or AIMessage(content="")
                print(
                    f"[nutrition_agent] 第一轮 LLM 流式完成: "
                    f"{plan_stream.chunk_count} chunks, "
                    f"tool_calls={len(response.tool_calls or [])}",
                    flush=True,
                )
            else:
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
                    cal = nutrition['calories']
                    print(f"[nutrition_agent] 兜底记录: {food_name}, {cal}kcal, {meal_type}")
                    fallback_result = log_food_intake.invoke({
                        "user_id": user_id,
                        "food_name": food_name,
                        "calories": cal,
                        "meal_type": meal_type,
                    })
                    print(f"[nutrition_agent] 兜底记录结果: {fallback_result}")
                    recorded.append(f"{food_name} {cal:.0f}kcal({meal_type})")
                if content and not initial_content_streamed:
                    yield content
                if content or initial_content_streamed:
                    yield f"\n\n已自动记录：{', '.join(recorded)}"
                else:
                    yield f"已为你记录：{', '.join(recorded)}。"
            else:
                if content and not initial_content_streamed:
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

        # 解析 LLM 回复中的文本工具调用（GLM-4.7 有时把工具调用输出为文本）
        if want_record and "log_food_intake" not in called_tools:
            _response_text = response.content if hasattr(response, 'content') else ''
            import re
            _match = re.search(r'log_food_intake\s*\(([^)]+)\)', _response_text)
            if _match:
                try:
                    args_str = _match.group(1)
                    args = {}
                    for pair in re.findall(r'(\w+)\s*=\s*("([^"]+)"|\'([^\']+)\'|(\d+(?:\.\d+)?))', args_str):
                        key = pair[0]
                        val = pair[2] or pair[3] or pair[4]
                        if val and val.replace('.', '').isdigit():
                            args[key] = float(val)
                        else:
                            args[key] = val
                    args.setdefault('user_id', user_id)
                    if 'food_name' in args and 'calories' in args:
                        result = log_food_intake.invoke(args)
                        print(f"[nutrition_agent] 从文本解析并执行: log_food_intake → {result}")
                        called_tools.add('log_food_intake')
                        chat_history.append({"role": "tool", "content": result, "tool_call_id": "text_parse"})
                except Exception as e:
                    print(f"[nutrition_agent] 文本工具调用解析失败: {e}")

        # 兜底：用户要求记录但 LLM 没调用 log_food_intake
        if want_record and "log_food_intake" not in called_tools:
            food_items = _extract_food_names(user_message)
            recorded = []
            for food_name, meal_type in food_items:
                nutrition = _get_food_nutrition(food_name)
                cal = nutrition['calories']
                print(f"[nutrition_agent] 兜底记录: {food_name}, {cal}kcal, {meal_type}")
                fallback_result = log_food_intake.invoke({
                    "user_id": user_id,
                    "food_name": food_name,
                    "calories": cal,
                    "meal_type": meal_type,
                })
                print(f"[nutrition_agent] 兜底记录结果: {fallback_result}")
                recorded.append(f"{food_name} {cal:.0f}kcal({meal_type})")
            # 追加工具消息让 LLM 知道已记录
            chat_history.append({
                "role": "tool",
                "content": f"已自动记录：{', '.join(recorded)}",
                "tool_call_id": "fallback_log"
            })

        try:
            has_content = False
            streamed_text = ''
            # 第二轮也要绑定工具，否则 LLM 会把工具调用输出为文本
            llm_with_tools = llm.bind_tools(nutrition_tools)
            if stream:
                print(f"[nutrition_agent] 第二轮 LLM 流式调用, messages={len(chat_history)}", flush=True)
                final_stream = StreamedToolCall(llm, nutrition_tools, chat_history)
                for chunk in final_stream:
                    if getattr(chunk, "content", None):
                        has_content = True
                        streamed_text += chunk.content
                        yield chunk.content
                final_response = final_stream.response or AIMessage(content="")
                accumulated_tool_calls = final_response.tool_calls or []
                print(
                    f"[nutrition_agent] 第二轮流式完成: "
                    f"{final_stream.chunk_count} chunks, "
                    f"has_content={has_content}, "
                    f"tool_calls={len(accumulated_tool_calls)}",
                    flush=True,
                )

                # 第二轮返回了 tool_calls → 执行后第三轮调用
                if accumulated_tool_calls and not has_content:
                    print(f"[nutrition_agent] 第二轮返回工具调用，执行后第三轮调用", flush=True)
                    chat_history.append(final_response)
                    _extra_tool_msgs = _execute_tool_calls(accumulated_tool_calls, nutrition_tools)
                    tool_messages.extend(_extra_tool_msgs)
                    chat_history.extend(_extra_tool_msgs)

                    print(f"[nutrition_agent] 第三轮 LLM 流式调用, messages={len(chat_history)}", flush=True)
                    third_stream = StreamedToolCall(llm, nutrition_tools, chat_history)
                    for chunk in third_stream:
                        if getattr(chunk, "content", None):
                            has_content = True
                            streamed_text += chunk.content
                            yield chunk.content
                    print(f"[nutrition_agent] 第三轮 LLM 流式完成, has_content={has_content}", flush=True)
            else:
                final_response = llm_with_tools.invoke(chat_history)
                content = final_response.content if hasattr(final_response, 'content') else str(final_response)
                print(f"[nutrition_agent] 第二轮 invoke 完成, content长度={len(content) if content else 0}", flush=True)

                # 第二轮返回了 tool_calls → 执行后第三轮调用
                if (not content) and hasattr(final_response, 'tool_calls') and final_response.tool_calls:
                    print(f"[nutrition_agent] 第二轮返回工具调用，执行后第三轮调用", flush=True)
                    chat_history.append(final_response)
                    _extra_tool_msgs = _execute_tool_calls(final_response.tool_calls, nutrition_tools)
                    tool_messages.extend(_extra_tool_msgs)
                    chat_history.extend(_extra_tool_msgs)

                    final_response2 = llm_with_tools.invoke(chat_history)
                    content = final_response2.content if hasattr(final_response2, 'content') else str(final_response2)
                    print(f"[nutrition_agent] 第三轮 invoke 完成, content长度={len(content) if content else 0}", flush=True)

                if content:
                    has_content = True
                    streamed_text = content
                    yield content

            # 检测流式输出中的文本工具调用（GLM-4.7 兜底）
            if want_record and 'log_food_intake' in streamed_text and 'log_food_intake' not in called_tools:
                import re
                _match = re.search(r'log_food_intake\s*\(([^)]+)\)', streamed_text)
                if _match:
                    try:
                        args_str = _match.group(1)
                        args = {}
                        for pair in re.findall(r'(\w+)\s*=\s*("([^"]+)"|\'([^\']+)\'|(\d+(?:\.\d+)?))', args_str):
                            key = pair[0]
                            val = pair[2] or pair[3] or pair[4]
                            if val and val.replace('.', '').isdigit():
                                args[key] = float(val)
                            else:
                                args[key] = val
                        args.setdefault('user_id', user_id)
                        if 'food_name' in args and 'calories' in args:
                            result = log_food_intake.invoke(args)
                            print(f"[nutrition_agent] 流式后解析执行: log_food_intake → {result}")
                            # 补充记录结果到输出末尾
                            yield f"\n\n✅ {result}"
                    except Exception as e:
                        print(f"[nutrition_agent] 流式后文本工具调用解析失败: {e}")

            # LLM 未生成内容时，从工具结果构造回复
            if not has_content:
                print(f"[nutrition_agent] LLM 未生成内容, tool_messages={len(tool_messages)}条", flush=True)
                tool_summary = []
                for tm in tool_messages:
                    c = tm.get('content', '') if isinstance(tm, dict) else ''
                    if not c or '工具执行错误' in c or '未知工具' in c:
                        continue
                    # 记录类结果直接包含
                    if '已记录' in c:
                        tool_summary.append(c)
                    # RAG 检索结果包含有效内容时
                    elif '【RAG检索】' in c:
                        rag_content = c.replace('【RAG检索】\n', '').replace('【RAG检索】', '').strip()
                        if rag_content and '未在知识库中找到' not in rag_content and len(rag_content) > 20:
                            tool_summary.append(f"知识库参考: {rag_content[:300]}")
                    # 其他有效结果（热量查询等）
                    elif len(c) > 5 and '未找到' not in c:
                        tool_summary.append(c)
                print(f"[nutrition_agent] 工具摘要: {len(tool_summary)}条", flush=True)
                if tool_summary:
                    yield '；'.join(tool_summary) + '。'
                else:
                    yield '抱歉，暂时无法处理该请求。请稍后再试。'

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
