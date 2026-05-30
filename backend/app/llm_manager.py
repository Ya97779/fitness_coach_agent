"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
import threading
from typing import Dict, Callable, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class _LLMQueue:
    """LLM 并发队列，追踪等待人数"""
    _semaphore = threading.Semaphore(3)
    _lock = threading.Lock()
    _waiting = 0

    @classmethod
    def acquire(cls, on_queue: Optional[Callable[[int], None]] = None):
        """获取许可，如果需要排队则通过 on_queue 回调通知前面有几人"""
        with cls._lock:
            position = cls._waiting
            cls._waiting += 1
        if position > 0 and on_queue:
            on_queue(position)
        cls._semaphore.acquire()
        with cls._lock:
            cls._waiting -= 1

    @classmethod
    def release(cls):
        cls._semaphore.release()

    @classmethod
    def queue_depth(cls) -> int:
        with cls._lock:
            return cls._waiting


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
    _queue_callback: Optional[Callable[[int], None]] = None

    @classmethod
    def set_queue_callback(cls, callback: Optional[Callable[[int], None]]):
        """设置排队回调，当 LLM 调用需要排队时触发 callback(前面等待人数)"""
        cls._queue_callback = callback

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
            # 包装 invoke/stream，自动限流 + 排队通知
            _orig_invoke = llm.invoke
            _orig_stream = llm.stream

            def _limited_invoke(*args, **kwargs):
                _LLMQueue.acquire(on_queue=cls._queue_callback)
                try:
                    return _orig_invoke(*args, **kwargs)
                finally:
                    _LLMQueue.release()

            def _limited_stream(*args, **kwargs):
                _LLMQueue.acquire(on_queue=cls._queue_callback)
                try:
                    return _orig_stream(*args, **kwargs)
                finally:
                    _LLMQueue.release()

            llm.invoke = _limited_invoke
            llm.stream = _limited_stream
            cls._instances[temperature] = llm
        return cls._instances[temperature]

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        cls._instances.clear()

    @staticmethod
    def queue_depth() -> int:
        """当前排队等待的人数"""
        return _LLMQueue.queue_depth()
