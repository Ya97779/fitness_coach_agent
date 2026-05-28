"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm_api")


class APILogCallback(BaseCallbackHandler):
    """记录 LLM API 请求/响应的回调"""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用完成时记录 token 使用情况"""
        output = response.llm_output or {}
        token_usage = output.get("token_usage", {})
        if token_usage:
            logger.info(
                f"[API] token_usage: prompt={token_usage.get('prompt_tokens', 0)}, "
                f"completion={token_usage.get('completion_tokens', 0)}, "
                f"total={token_usage.get('total_tokens', 0)}"
            )
        # 检查 finish_reason
        finish_reason = output.get("finish_reason")
        if finish_reason:
            logger.info(f"[API] finish_reason: {finish_reason}")
        # 检查生成内容
        generations = response.generations
        if generations and generations[0]:
            gen = generations[0][0]
            if not gen.text and not getattr(gen, "message", None):
                logger.warning("[API] 警告: 响应内容为空!")
            elif hasattr(gen, "message") and gen.message:
                msg = gen.message
                if not msg.content and not getattr(msg, "tool_calls", None):
                    logger.warning("[API] 警告: message.content 和 tool_calls 都为空!")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 调用出错时记录"""
        logger.error(f"[API] 调用异常: {type(error).__name__}: {str(error)[:300]}")


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
                temperature=temperature,
                request_timeout=60,
                max_retries=2,
                model_kwargs={"thinking": {"type": "disabled"}},
                callbacks=[APILogCallback()]
            )
        return cls._instances[temperature]

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        cls._instances.clear()
