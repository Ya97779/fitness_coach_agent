"""闲聊 Agent - 处理日常对话"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS

load_dotenv()


def chat_with_user(messages: list, user_id: int) -> str:
    """直接与用户对话（不需要工具调用）

    Args:
        messages: 消息列表
        user_id: 用户ID

    Returns:
        str: LLM 生成的回复
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    # 构建提示：系统消息 + 用户消息
    system_msg = SystemMessage(content=AGENT_SYSTEM_PROMPTS["chat"])
    response = llm.invoke([system_msg] + messages)

    return response.content
