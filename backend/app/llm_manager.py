"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
import threading
from typing import Dict, Callable, Optional, Any
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


class _LLMProxy:
    """ChatOpenAI 代理包装，自动限流 + 排队通知"""

    def __init__(self, llm: ChatOpenAI, queue_callback: Optional[Callable[[int], None]] = None):
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, '_queue_callback', queue_callback)

    def invoke(self, *args, **kwargs):
        _LLMQueue.acquire(on_queue=self._queue_callback)
        try:
            return self._llm.invoke(*args, **kwargs)
        finally:
            _LLMQueue.release()

    def stream(self, *args, **kwargs):
        _LLMQueue.acquire(on_queue=self._queue_callback)
        try:
            return self._llm.stream(*args, **kwargs)
        finally:
            _LLMQueue.release()

    def bind_tools(self, *args, **kwargs):
        return self._llm.bind_tools(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


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
    def get_llm(cls, temperature: float = 0.7):
        """获取 LLM 代理实例（带限流和排队通知）

        Returns:
            _LLMProxy 代理对象，支持 invoke/stream/bind_tools
        """
        if temperature not in cls._instances:
            cls._instances[temperature] = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4.7"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=temperature,
                request_timeout=30,
                max_retries=2,
                extra_body={"thinking": {"type": "disabled"}}
            )
        return _LLMProxy(cls._instances[temperature], cls._queue_callback)

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        cls._instances.clear()

    @staticmethod
    def queue_depth() -> int:
        """当前排队等待的人数"""
        return _LLMQueue.queue_depth()
