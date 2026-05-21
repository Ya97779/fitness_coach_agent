"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
from typing import Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMManager:
    """LLM 单例管理器

    按 temperature 值缓存 ChatOpenAI 实例，相同 temperature 复用同一实例。
    单次请求中可能需要不同 temperature（路由 0.1、摘要 0.3、生成 0.7），
    但相同 temperature 的调用复用同一实例，将 16 个实例降至 4 个。

    使用方式：
        from app.llm_manager import LLMManager

        llm = LLMManager.get_llm(temperature=0.7)
        llm = LLMManager.get_llm(temperature=0.1)  # 复用缓存
    """

    _instances: Dict[float, ChatOpenAI] = {}

    @classmethod
    def get_llm(cls, temperature: float = 0.7) -> ChatOpenAI:
        """获取或创建 ChatOpenAI 实例

        Args:
            temperature: 温度参数，影响输出随机性

        Returns:
            ChatOpenAI 实例（缓存命中时返回已有实例）
        """
        if temperature not in cls._instances:
            cls._instances[temperature] = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4.7"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=temperature
            )
        return cls._instances[temperature]

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        cls._instances.clear()

    @staticmethod
    def stream_with_glm(messages, temperature: float = 0.7):
        """使用 GLM SDK 直接流式调用，绕过 LangChain 的 OpenAI 兼容层缓冲

        Args:
            messages: LangChain 消息列表
            temperature: 温度参数

        Yields:
            str: 响应内容片段
        """
        from zai import ZhipuAiClient

        client = ZhipuAiClient(api_key=os.getenv("OPENAI_API_KEY"))

        glm_messages = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "")

            if msg_type == "system":
                glm_messages.append({"role": "system", "content": content})
            elif msg_type == "human":
                glm_messages.append({"role": "user", "content": content})
            elif msg_type == "ai":
                if content:
                    glm_messages.append({"role": "assistant", "content": content})
            elif msg_type == "tool":
                glm_messages.append({"role": "user", "content": f"[工具返回结果] {content}"})

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "glm-4.7"),
            messages=glm_messages,
            stream=True,
            temperature=temperature,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
