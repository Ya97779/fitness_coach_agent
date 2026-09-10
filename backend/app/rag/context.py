"""Prepare retrieved documents for an Agent tool response.

Retrieval and prompt construction are separate concerns: ``ModernRAG`` returns
ranked result dictionaries, while this module applies the final safety,
deduplication and size budget shared by nutrition and fitness agents.
"""

from difflib import SequenceMatcher
import hashlib
import logging
import os
import re
from typing import Any, Dict, Iterable, List


logger = logging.getLogger(__name__)

_SPAM_PATTERNS = ("加微信", "免费获得", "大礼包", "微信号", "扫码", "关注公众号")
_ERROR_PREFIXES = ("检索失败", "工具执行错误", "未知工具")


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _normalize_for_dedup(content: str) -> str:
    return re.sub(r"\s+", "", content).casefold()


def _truncate_at_boundary(content: str, max_chars: int) -> str:
    """Truncate near a sentence boundary instead of cutting blindly."""

    if len(content) <= max_chars:
        return content

    candidate = content[:max_chars]
    minimum = max(1, int(max_chars * 0.6))
    boundaries = [candidate.rfind(mark) for mark in "。！？；.!?;\n"]
    boundary = max(boundaries)
    if boundary >= minimum:
        return candidate[: boundary + 1].rstrip()
    return candidate.rstrip() + "..."


def _is_duplicate(content: str, selected: List[str], threshold: float = 0.92) -> bool:
    normalized = _normalize_for_dedup(content)
    for existing in selected:
        existing_normalized = _normalize_for_dedup(existing)
        if normalized == existing_normalized:
            return True
        if SequenceMatcher(None, normalized, existing_normalized).ratio() >= threshold:
            return True
    return False


def format_retrieval_context(
    results: Iterable[Dict[str, Any]],
    label: str = "RAG检索",
) -> str:
    """Return bounded, deduplicated retrieval context for an LLM tool message."""

    max_chunks = _env_positive_int("RAG_FINAL_CHUNKS", 3)
    max_chunk_chars = _env_positive_int("RAG_MAX_CHUNK_CHARS", 500)
    max_context_chars = _env_positive_int("RAG_MAX_CONTEXT_CHARS", 1500)

    selected_contents: List[str] = []
    content_parts: List[str] = []
    selected_ids: List[str] = []
    selected_sources: List[str] = []
    selected_scores: List[str] = []
    used_chars = 0

    for result in results:
        content = str(result.get("content") or "").strip()
        if len(content) < 20:
            continue
        if content.startswith(_ERROR_PREFIXES):
            continue
        if any(spam in content for spam in _SPAM_PATTERNS):
            continue
        if _is_duplicate(content, selected_contents):
            continue

        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        bounded = _truncate_at_boundary(content, min(max_chunk_chars, remaining))
        if len(bounded) < 20:
            continue

        metadata = result.get("metadata") or {}
        heading = str(metadata.get("heading_path") or "").strip()
        prefix = f"[{heading}] " if heading else ""
        content_parts.append(f"[来源{len(content_parts) + 1}] {prefix}{bounded}")
        selected_contents.append(content)
        used_chars += len(bounded)

        stable_id = (
            metadata.get("chunk_id")
            or metadata.get("id")
            or hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        )
        selected_ids.append(str(stable_id))
        selected_sources.append(
            os.path.basename(str(metadata.get("source") or "-"))
        )
        score = result.get("rerank_score", result.get("score"))
        try:
            selected_scores.append("-" if score is None else f"{float(score):.4f}")
        except (TypeError, ValueError):
            selected_scores.append("invalid")
        if len(content_parts) >= max_chunks:
            break

    logger.info(
        "rag_context_selected count=%d chars=%d chunk_ids=%s sources=%s scores=%s",
        len(content_parts),
        used_chars,
        ",".join(selected_ids) or "-",
        ",".join(selected_sources) or "-",
        ",".join(selected_scores) or "-",
    )
    if not content_parts:
        return f"【{label}】未在知识库中找到相关信息"
    return f"【{label}】\n" + "\n\n".join(content_parts)
