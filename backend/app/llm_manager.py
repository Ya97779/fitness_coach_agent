"""LLM 实例管理器 - 按 temperature 分桶缓存，避免重复创建"""

import os
import threading
import time
import logging
import uuid
from contextvars import ContextVar
from typing import Dict, Callable, Optional, Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger("food_estimate.llm")

_GLM_53_FLASH_REASONING_EFFORTS = {"low", "high", "max"}


def get_chunk_reasoning_content(chunk: Any) -> str:
    """Return provider reasoning text preserved on a LangChain message chunk."""

    additional_kwargs = getattr(chunk, "additional_kwargs", None) or {}
    value = additional_kwargs.get("reasoning_content")
    if not value:
        value = getattr(chunk, "reasoning_content", None)
    return value if isinstance(value, str) else ""


class GLMChatOpenAI(ChatOpenAI):
    """Keep GLM's OpenAI-compatible ``reasoning_content`` vendor extension.

    ``langchain-openai`` 1.2.x ignores unknown fields in ``choice.delta``.
    Preserving the field in ``additional_kwargs`` lets the existing agent
    stream and message accumulator handle it without replacing LangChain.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        delta = choices[0].get("delta") if choices else None
        reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
        if reasoning:
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _normalize_reasoning_effort(model: str, configured: Optional[str]) -> str:
    """Return a reasoning level accepted by the configured provider model.

    GLM-5.3-Flash always thinks and its current API accepts only low/high/max.
    Falling back here prevents a bad deployment setting from breaking every
    request with provider error 1210.
    """

    effort = (configured or "low").strip().casefold() or "low"
    if (
        model.strip().casefold().startswith("glm-5.3-flash")
        and effort not in _GLM_53_FLASH_REASONING_EFFORTS
    ):
        logger.warning(
            "unsupported_reasoning_effort model=%s configured=%s fallback=low",
            model,
            effort,
        )
        return "low"
    return effort


class _LLMQueue:
    """LLM 并发队列，追踪等待人数"""
    _semaphore = threading.Semaphore(3)
    _lock = threading.Lock()
    _waiting = 0

    @classmethod
    def acquire(cls, on_queue: Optional[Callable[[int], None]] = None):
        """获取许可，许可覆盖整个 invoke/stream 生命周期。"""
        # 先尝试无阻塞获取；只有确实没有名额时才计入等待队列。
        if cls._semaphore.acquire(blocking=False):
            return

        with cls._lock:
            cls._waiting += 1
            position = cls._waiting
        if on_queue:
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

    def __init__(self, llm: Any, queue_callback: Optional[Callable[[int], None]] = None):
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, '_queue_callback', queue_callback)

    def invoke(self, *args, **kwargs):
        _LLMQueue.acquire(on_queue=self._queue_callback)
        started_at = time.perf_counter()
        request_id = None
        try:
            from .runtime_context import get_request_context
            request_id = get_request_context().request_id
        except Exception:
            pass
        logger.info(
            "llm_request_started request_id=%s mode=invoke",
            request_id or "-",
        )
        try:
            result = self._llm.invoke(*args, **kwargs)
            logger.info(
                "provider_response request_id=%s mode=invoke elapsed=%.3fs",
                request_id or "-", time.perf_counter() - started_at,
            )
            return result
        finally:
            _LLMQueue.release()

    def stream(self, *args, **kwargs):
        _LLMQueue.acquire(on_queue=self._queue_callback)

        def iterate():
            started_at = time.perf_counter()
            first_chunk = False
            first_reasoning = False
            first_content = False
            request_id = None
            call_id = uuid.uuid4().hex[:8]
            try:
                from .runtime_context import get_request_context
                request_id = get_request_context().request_id
            except Exception:
                pass
            logger.info(
                "llm_request_started request_id=%s call_id=%s mode=stream",
                request_id or "-",
                call_id,
            )
            try:
                # ChatOpenAI.stream() 返回迭代器；必须在消费完迭代器后
                # 才释放并发许可，不能在返回迭代器时提前释放。
                for chunk in self._llm.stream(*args, **kwargs):
                    if not first_chunk:
                        first_chunk = True
                        logger.info(
                            "provider_first_chunk request_id=%s call_id=%s elapsed=%.3fs",
                            request_id or "-", call_id, time.perf_counter() - started_at,
                        )
                    reasoning = get_chunk_reasoning_content(chunk)
                    if reasoning and not first_reasoning:
                        first_reasoning = True
                        logger.info(
                            "provider_first_reasoning request_id=%s call_id=%s elapsed=%.3fs",
                            request_id or "-", call_id, time.perf_counter() - started_at,
                        )
                    if getattr(chunk, "content", None) and not first_content:
                        first_content = True
                        logger.info(
                            "provider_first_content request_id=%s call_id=%s elapsed=%.3fs",
                            request_id or "-", call_id, time.perf_counter() - started_at,
                        )
                    yield chunk
            finally:
                logger.info(
                    "provider_stream_completed request_id=%s call_id=%s elapsed=%.3fs",
                    request_id or "-", call_id, time.perf_counter() - started_at,
                )
                _LLMQueue.release()

        return iterate()

    def bind_tools(self, *args, **kwargs):
        # 工具绑定会返回一个新的 Runnable。继续包在代理中，避免
        # bind_tools().invoke/stream 绕过统一限流、超时和队列通知。
        bound_llm = self._llm.bind_tools(*args, **kwargs)
        return _LLMProxy(bound_llm, self._queue_callback)

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
    _instance_lock = threading.Lock()
    _queue_callback_var: ContextVar[Optional[Callable[[int], None]]] = ContextVar(
        "llm_queue_callback", default=None
    )

    @classmethod
    def set_queue_callback(cls, callback: Optional[Callable[[int], None]]):
        """设置当前执行上下文的排队回调并返回可用于恢复的 token。"""
        return cls._queue_callback_var.set(callback)

    @classmethod
    def reset_queue_callback(cls, token):
        """恢复当前执行上下文之前的排队回调。"""
        if token is not None:
            cls._queue_callback_var.reset(token)

    @classmethod
    def get_llm(cls, temperature: float = 0.7):
        """获取 LLM 代理实例（带限流和排队通知）

        Returns:
            _LLMProxy 代理对象，支持 invoke/stream/bind_tools
        """
        if temperature not in cls._instances:
            with cls._instance_lock:
                if temperature not in cls._instances:
                    model = os.getenv("LLM_MODEL", "glm-4.7")
                    reasoning_effort = _normalize_reasoning_effort(
                        model,
                        os.getenv("LLM_REASONING_EFFORT", "low"),
                    )
                    thinking_type = os.getenv(
                        "LLM_THINKING_TYPE", "enabled"
                    ).strip().casefold()
                    extra_body = None
                    if thinking_type:
                        extra_body = {
                            "thinking": {
                                "type": thinking_type,
                                "clear_thinking": _env_bool(
                                    "LLM_CLEAR_THINKING", False
                                ),
                            }
                        }
                    cls._instances[temperature] = GLMChatOpenAI(
                        model=model,
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_API_BASE"),
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        request_timeout=30,
                        max_retries=2,
                        extra_body=extra_body,
                    )
        return _LLMProxy(cls._instances[temperature], cls._queue_callback_var.get())

    @classmethod
    def clear(cls):
        """清空缓存（用于测试或配置变更后）"""
        with cls._instance_lock:
            cls._instances.clear()

    @staticmethod
    def queue_depth() -> int:
        """当前排队等待的人数"""
        return _LLMQueue.queue_depth()
