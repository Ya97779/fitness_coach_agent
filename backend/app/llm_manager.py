"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
import threading
from typing import Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 并发限制：最多 10 个同时进行的 LLM 调用，防止瞬间打爆 API
_LLM_SEMAPHORE = threading.Semaphore(10)


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
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4.7"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=temperature,
                request_timeout=30,
                max_retries=2,
                extra_body={"thinking": {"type": "disabled"}}
            )
            # 包装 invoke/stream，自动限流
            _orig_invoke = llm.invoke
            _orig_stream = llm.stream

            def _limited_invoke(*args, **kwargs):
                _LLM_SEMAPHORE.acquire()
                try:
                    return _orig_invoke(*args, **kwargs)
                finally:
                    _LLM_SEMAPHORE.release()

            def _limited_stream(*args, **kwargs):
                _LLM_SEMAPHORE.acquire()
                try:
                    return _orig_stream(*args, **kwargs)
                finally:
                    _LLM_SEMAPHORE.release()

            llm.invoke = _limited_invoke
            llm.stream = _limited_stream
            cls._instances[temperature] = llm
        return cls._instances[temperature]

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        cls._instances.clear()

    @staticmethod
    def acquire():
        """获取 LLM 并发许可（阻塞直到有空位）"""
        _LLM_SEMAPHORE.acquire()

    @staticmethod
    def release():
        """释放 LLM 并发许可"""
        _LLM_SEMAPHORE.release()
