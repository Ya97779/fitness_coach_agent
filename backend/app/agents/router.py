"""路由器 Agent - 根据用户输入动态选择合适的 Agent"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()


def route_with_context(user_message: str, user_id: int = None) -> dict:
    """根据用户消息路由到合适的 Agent

    路由规则：
    - 包含"吃"、"食物"、"热量"、"饮食"等 → nutrition
    - 包含"运动"、"跑步"、"训练"、"健身"等 → fitness
    - 其他默认为 → chat

    Args:
        user_message: 用户的输入消息
        user_id: 用户ID（可选）

    Returns:
        dict: {
            "agent": str,   # 路由目标：chat/nutrition/fitness
            "reason": str   # 路由原因
        }
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.1  # 低温度保证路由稳定性
    )

    # 简化的路由 prompt
    prompt = f"用户输入: {user_message}\n\n判断类型：1=闲聊助手 2=营养师 3=健身教练\n只返回一个数字。"

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        # 根据 LLM 返回的数字决定路由
        if "2" in text:
            return {"agent": "nutrition", "reason": "饮食/营养相关"}
        elif "3" in text:
            return {"agent": "fitness", "reason": "运动/健身相关"}
        return {"agent": "chat", "reason": "闲聊/通用"}
    except Exception as e:
        print(f"路由出错: {e}")
        return {"agent": "chat", "reason": "默认闲聊"}
