"""混合检索器 - 向量检索 + BM25 融合"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from .bm25 import BM25Search


class HybridSearch:
    """混合检索器

    结合向量检索（语义）和 BM25（关键词）的优势：
    - 向量检索：理解语义，处理同义词
    - BM25：精确匹配关键词

    使用 RRF（Reciprocal Rank Fusion）融合分数：
    RRF_score = Σ 1/(k + rank_i)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        """初始化混合检索器

        Args:
            k1: BM25 k1 参数
            b: BM25 b 参数
            rrf_k: RRF 融合参数（通常 60）
            vector_weight: 向量检索权重
            bm25_weight: BM25 权重
        """
        self.bm25 = BM25Search(k1=k1, b=b)
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.vectorstore: Optional[Chroma] = None
        self.embeddings: Optional[Embeddings] = None
        self.documents: List[Document] = []

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float, str]]
    ) -> List[Tuple[int, float]]:
        """RRF 分数融合

        Args:
            vector_results: [(doc_idx, score), ...]
            bm25_results: [(doc_idx, score, content), ...]

        Returns:
            融合后的 [(doc_idx, fused_score), ...]
        """
        fused_scores = {}

        for rank, (doc_idx, score) in enumerate(vector_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + self.vector_weight * rrf_score

        for rank, (doc_idx, score, _) in enumerate(bm25_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + self.bm25_weight * rrf_score

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def index(
        self,
        documents: List[Document],
        vectorstore: Chroma,
        embeddings: Embeddings
    ):
        """构建索引

        Args:
            documents: Document 列表
            vectorstore: Chroma 向量数据库
            embeddings: 嵌入模型
        """
        self.documents = documents
        self.vectorstore = vectorstore
        self.embeddings = embeddings

        texts = [doc.page_content for doc in documents]
        self.bm25.index(texts)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[dict]:
        """混合检索

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            [{
                "index": int,
                "score": float,
                "content": str,
                "metadata": dict
            }, ...]
        """
        vector_results = []
        bm25_results = []

        if self.vectorstore:
            try:
                vector_docs = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
                vector_results = [(i, score) for i, doc in enumerate(vector_docs) if hasattr(doc, 'page_content') for i, doc in enumerate(self.vectorstore.get()["documents"])]

                seen = set()
                vector_results = []
                for doc, score in self.vectorstore.similarity_search_with_score(query, k=top_k * 2):
                    doc_idx = self._find_doc_index(doc.page_content)
                    if doc_idx != -1 and doc_idx not in seen:
                        seen.add(doc_idx)
                        vector_results.append((doc_idx, score))
            except Exception as e:
                print(f"向量检索失败: {e}")
                vector_results = []

        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        if not vector_results and not bm25_results:
            return []

        fused = self._rrf_fusion(vector_results, bm25_results)

        results = []
        for doc_idx, fused_score in fused[:top_k]:
            if doc_idx < len(self.documents):
                doc = self.documents[doc_idx]
                results.append({
                    "index": doc_idx,
                    "score": fused_score,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return results

    def _find_doc_index(self, content: str) -> int:
        """查找文档索引"""
        for i, doc in enumerate(self.documents):
            if doc.page_content == content:
                return i
        return -1


def create_hybrid_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    embeddings: Embeddings,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> HybridSearch:
    """创建混合检索器（工厂函数）

    Args:
        documents: Document 列表
        vectorstore: Chroma 向量数据库
        embeddings: 嵌入模型
        vector_weight: 向量检索权重
        bm25_weight: BM25 权重

    Returns:
        HybridSearch 实例
    """
    retriever = HybridSearch(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight
    )
    retriever.index(documents, vectorstore, embeddings)
    return retriever
