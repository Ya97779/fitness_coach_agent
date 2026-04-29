"""路由器 Agent - 根据用户输入动态选择合适的 Agent

混合路由策略：
1. 关键词预筛选：快速、轻量，处理明确场景
2. LLM 二次确认：处理模糊/复杂情况

使用示例：
    from app.agents.router import route_with_context, hybrid_route

    # 简单路由
    result = route_with_context("我想减肥吃什么好")

    # 混合路由（关键词 + LLM 确认）
    result = hybrid_route("上斜卧推怎么做", require_llm_confirm=True)
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import re
from dotenv import load_dotenv

load_dotenv()

NUTRITION_KEYWORDS = [
    "吃", "食物", "饮食", "营养", "热量", "卡路里", "蛋白质", "脂肪", "碳水",
    "增肌", "减脂", "饮食计划", "食谱", "早餐", "午餐", "晚餐", "零食",
    "摄入", "消耗", "代谢", "基础代谢", "TDEE", "BMR",
    "碳水化合物", "膳食纤维", "维生素", "矿物质", "补剂", "蛋白粉",
    "鸡胸肉", "鸡蛋", "蔬菜", "水果", "米饭", "面条", "面包",
    "多少", "怎么吃", "吃什么", "多少克", "多少卡"
]

FITNESS_KEYWORDS = [
    "运动", "训练", "健身", "跑步", "卧推", "深蹲", "硬拉", "俯卧撑",
    "增肌", "减脂", "力量", "耐力", "柔韧性", "拉伸", "热身",
    "肌肉", "臂", "腿", "背", "胸", "肩", "腹", "臀",
    "次数", "组数", "重量", "RM", "训练计划", "健身房",
    "有氧", "无氧", "HIIT", "瑜伽", "普拉提", "游泳", "骑行",
    "怎么练", "动作", "姿势", "发力", "肌肉酸痛", "恢复"
]

def _keyword_match(text: str) -> str:
    """基于关键词快速匹配 Agent 类型

    Args:
        text: 用户输入文本

    Returns:
        "nutrition" | "fitness" | "chat" | None (需要 LLM 确认)
    """
    text_lower = text.lower()
    nutrition_score = 0
    fitness_score = 0

    for keyword in NUTRITION_KEYWORDS:
        if keyword in text:
            nutrition_score += 1

    for keyword in FITNESS_KEYWORDS:
        if keyword in text:
            fitness_score += 1

    if nutrition_score >= 2 and fitness_score == 0:
        return "nutrition"
    elif fitness_score >= 2 and nutrition_score == 0:
        return "fitness"
    elif nutrition_score >= 1 and fitness_score >= 1:
        return "mixed"
    elif nutrition_score >= 1:
        return "nutrition"
    elif fitness_score >= 1:
        return "fitness"

    return None

def _llm_route(user_message: str) -> dict:
    """使用 LLM 进行路由判断

    Args:
        user_message: 用户输入消息

    Returns:
        dict: {"agent": str, "reason": str, "confidence": float}
    """
    from ..llm_manager import LLMManager
    llm = LLMManager.get_llm(temperature=0.1)

    prompt = f"""用户输入: {user_message}

判断类型：1=闲聊助手 2=营养师 3=健身教练
只返回一个数字（1、2 或 3），不要包含任何其他文字。"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        # 精确提取独立数字 1/2/3
        match = re.search(r'\b([123])\b', text)
        if match:
            num = match.group(1)
            if num == "2":
                return {"agent": "nutrition", "reason": "饮食/营养相关", "confidence": 0.8}
            elif num == "3":
                return {"agent": "fitness", "reason": "运动/健身相关", "confidence": 0.8}
            else:
                return {"agent": "chat", "reason": "闲聊/通用", "confidence": 0.8}

        # 兜底：语义关键词判断
        if "营养" in text or "饮食" in text:
            return {"agent": "nutrition", "reason": "LLM语义判断：营养", "confidence": 0.6}
        elif "健身" in text or "运动" in text:
            return {"agent": "fitness", "reason": "LLM语义判断：健身", "confidence": 0.6}

        return {"agent": "chat", "reason": "闲聊/通用", "confidence": 0.5}
    except Exception as e:
        print(f"路由出错: {e}")
        return {"agent": "chat", "reason": "默认闲聊", "confidence": 0.3}


def hybrid_route(user_message: str, require_llm_confirm: bool = True) -> dict:
    """混合路由：关键词预筛选 + LLM 二次确认

    工作流程：
    1. 关键词预筛选 → 明确场景直接返回
    2. 模糊场景 → LLM 二次确认

    Args:
        user_message: 用户输入消息
        require_llm_confirm: 是否对模糊场景进行 LLM 确认

    Returns:
        dict: {
            "agent": str,      # 路由目标：chat/nutrition/fitness
            "reason": str,     # 路由原因
            "method": str,     # 路由方法：keyword/llm/mixed
            "confidence": float # 置信度
        }
    """
    if not user_message or not user_message.strip():
        return {"agent": "chat", "reason": "空输入", "method": "keyword", "confidence": 1.0}

    keyword_result = _keyword_match(user_message)

    if keyword_result == "nutrition":
        return {
            "agent": "nutrition",
            "reason": "关键词匹配：饮食/营养",
            "method": "keyword",
            "confidence": 0.85
        }

    if keyword_result == "fitness":
        return {
            "agent": "fitness",
            "reason": "关键词匹配：运动/健身",
            "method": "keyword",
            "confidence": 0.85
        }

    if keyword_result == "mixed":
        return {
            "agent": "chat",
            "reason": "关键词匹配：混合话题，建议闲聊",
            "method": "keyword",
            "confidence": 0.6
        }

    if keyword_result is None and not require_llm_confirm:
        return {
            "agent": "chat",
            "reason": "无关键词匹配，默认闲聊",
            "method": "keyword",
            "confidence": 0.4
        }

    llm_result = _llm_route(user_message)
    llm_result["method"] = "llm"
    llm_result["confidence"] = llm_result.get("confidence", 0.7)

    return llm_result


def route_with_context(user_message: str, user_id: int = None) -> dict:
    """根据用户消息路由到合适的 Agent（兼容旧接口）

    内部使用混合路由策略。

    Args:
        user_message: 用户的输入消息
        user_id: 用户ID（可选）

    Returns:
        dict: {
            "agent": str,   # 路由目标：chat/nutrition/fitness
            "reason": str   # 路由原因
        }
    """
    result = hybrid_route(user_message, require_llm_confirm=True)
    return {
        "agent": result["agent"],
        "reason": result["reason"]
    }
