"""Reranker 模块 - 检索结果精排

使用 Jina Rerank API 对初排结果进行精排，提升检索准确率。
"""

import os
from typing import List, Dict, Any, Optional
import httpx


class JinaReranker:
    """Jina Reranker - 基于 Jina Rerank API 的精排器

    对初排结果进行 Cross-Encoder 精排，比向量检索的双塔模型更精准。

    使用方式：
        reranker = JinaReranker()
        results = reranker.rerank("深蹲标准姿势", documents, top_n=5)
    """

    API_URL = "https://api.jina.ai/v1/rerank"
    DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0
    ):
        """初始化 Jina Reranker

        Args:
            api_key: Jina API Key，默认从 JINA_API_KEY 环境变量读取
            model: rerank 模型名称
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY 未配置，请在 .env 文件中设置")

        self.model = model
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """对检索结果进行精排

        Args:
            query: 查询文本
            documents: 初排结果列表，每个元素需有 "content" 字段
            top_n: 返回的精排结果数量
            score_threshold: 最低相关性分数阈值（0-1）

        Returns:
            精排后的结果列表，每个元素新增 "rerank_score" 字段
        """
        if not documents:
            return []

        if not query or not query.strip():
            return documents[:top_n]

        # 提取文档内容
        doc_texts = [doc["content"] for doc in documents]

        # 调用 Jina Rerank API
        rerank_results = self._call_api(query, doc_texts, top_n)

        if not rerank_results:
            # API 调用失败时降级返回初排结果
            return documents[:top_n]

        # 组装精排结果
        ranked_results = []
        for item in rerank_results:
            idx = item["index"]
            score = item["relevance_score"]

            if score < score_threshold:
                continue

            if idx < len(documents):
                result = documents[idx].copy()
                result["rerank_score"] = score
                result["original_rank"] = idx
                ranked_results.append(result)

        return ranked_results[:top_n]

    def _call_api(
        self,
        query: str,
        documents: List[str],
        top_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """调用 Jina Rerank API

        Args:
            query: 查询文本
            documents: 文档内容列表
            top_n: 返回数量

        Returns:
            API 返回的 results 列表，失败返回 None
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "return_documents": False
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
        except httpx.TimeoutException:
            print(f"Jina Rerank API 超时 ({self.timeout}s)")
            return None
        except httpx.HTTPStatusError as e:
            print(f"Jina Rerank API HTTP 错误: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Jina Rerank API 调用失败: {e}")
            return None
