"""Query Expansion - 多查询扩展检索

通过 LLM 生成多个查询变体，提高检索覆盖率。
支持：
- 原始查询保持
- 同义词扩展
- 问题改写
- 中英文扩展
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的查询优化助手。你的任务是根据用户输入，生成多个不同的查询变体，
以帮助从不同角度检索相关信息。

要求：
1. 生成 3-5 个查询变体
2. 每个变体应该从不同角度表达原始查询
3. 可以包括同义词表达、问题改写、不同措辞等
4. 确保变体多样化，覆盖不同检索角度
5. 保持查询简短（不超过 30 字）

输出格式：
直接输出查询列表，每行一个，不要编号。"""),
    ("human", "原始查询：{query}")
])


class QueryExpander:
    """多查询扩展器

    使用 LLM 生成多个查询变体，提高检索召回率。
    适用于：
    - 模糊查询
    - 多义词查询
    - 需要多角度检索的场景
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.7,
        num_variants: int = 4
    ):
        """初始化查询扩展器

        Args:
            llm: 自定义 LLM（可选）
            temperature: 生成温度
            num_variants: 生成变体数量
        """
        if llm is None:
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=temperature
            )
        else:
            self.llm = llm

        self.num_variants = num_variants
        self.prompt = QUERY_EXPANSION_PROMPT

    def expand(self, query: str) -> List[str]:
        """扩展查询

        Args:
            query: 原始查询

        Returns:
            扩展后的查询列表（包含原始查询）
        """
        if not query or not query.strip():
            return [query]

        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query})
            expanded = response.content.strip().split('\n')

            queries = [q.strip() for q in expanded if q.strip()]
            queries = [q for q in queries if len(q) <= 50]

            if query not in queries:
                queries.insert(0, query)

            return queries[:self.num_variants]

        except Exception as e:
            print(f"查询扩展失败: {e}")
            return [query]

    def expand_with_scores(
        self,
        query: str,
        similarity_func: Optional[Callable[[str, str], float]] = None
    ) -> List[Dict[str, Any]]:
        """扩展查询并计算相似度

        Args:
            query: 原始查询
            similarity_func: 相似度函数（可选）

        Returns:
            [{
                "query": str,
                "is_original": bool,
                "similarity": float
            }, ...]
        """
        expanded = self.expand(query)

        if similarity_func is None:
            def similarity_func(a, b):
                return 1.0 if a == b else 0.5

        results = []
        for q in expanded:
            results.append({
                "query": q,
                "is_original": (q == query),
                "similarity": similarity_func(query, q)
            })

        results.sort(key=lambda x: (not x["is_original"], -x["similarity"]))

        return results


class MultiQueryRetriever:
    """多查询检索器

    使用多个查询变体进行检索，结果合并去重。
    """

    def __init__(
        self,
        retriever_func: Callable[[str, int], List],
        expander: Optional[QueryExpander] = None,
        merge_strategy: str = "rrf"
    ):
        """初始化多查询检索器

        Args:
            retriever_func: 检索函数签名 (query, top_k) -> List[Result]
            expander: 查询扩展器
            merge_strategy: 合并策略 ("rrf" | "simple" | "weighted")
        """
        self.expander = expander or QueryExpander()
        self.retriever_func = retriever_func
        self.merge_strategy = merge_strategy

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索

        Args:
            query: 原始查询
            top_k: 返回数量

        Returns:
            合并后的检索结果
        """
        expanded_queries = self.expander.expand(query)

        all_results = []
        for q in expanded_queries:
            results = self.retriever_func(q, top_k * 2)
            all_results.append((q, results))

        merged = self._merge_results(all_results, top_k)
        return merged

    def _merge_results(
        self,
        all_results: List[tuple],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """合并多查询结果

        Args:
            all_results: [(query, results), ...]
            top_k: 返回数量

        Returns:
            合并后的结果
        """
        if self.merge_strategy == "simple":
            return self._simple_merge(all_results, top_k)
        elif self.merge_strategy == "weighted":
            return self._weighted_merge(all_results, top_k)
        else:
            return self._rrf_merge(all_results, top_k)

    def _simple_merge(
        self,
        all_results: List[tuple],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """简单合并（去重）

        Args:
            all_results: [(query, results), ...]
            top_k: 返回数量

        Returns:
            合并后的结果
        """
        seen = set()
        merged = []

        for query, results in all_results:
            for r in results:
                key = r.get("content", "")[:100]
                if key not in seen:
                    seen.add(key)
                    r["query_source"] = query
                    merged.append(r)

        return merged[:top_k]

    def _rrf_merge(
        self,
        all_results: List[tuple],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """RRF 融合（Reciprocal Rank Fusion）

        Args:
            all_results: [(query, results), ...]
            top_k: 返回数量
            k: RRF 参数

        Returns:
            融合后的结果
        """
        fused_scores = {}
        doc_mapping = {}

        for query, results in all_results:
            for rank, r in enumerate(results):
                content_key = r.get("content", "")[:100]
                rrf_score = 1 / (k + rank + 1)

                fused_scores[content_key] = fused_scores.get(content_key, 0) + rrf_score
                doc_mapping[content_key] = r

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for content_key, score in ranked[:top_k]:
            doc = doc_mapping[content_key].copy()
            doc["fused_score"] = score
            doc["num_queries"] = sum(1 for q, r in all_results if any(
                r.get("content", "")[:100] == content_key for r in r
            ))
            merged.append(doc)

        return merged

    def _weighted_merge(
        self,
        all_results: List[tuple],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """加权合并

        Args:
            all_results: [(query, results), ...]
            top_k: 返回数量

        Returns:
            加权合并后的结果
        """
        weights = {}
        doc_mapping = {}

        original_weight = 2.0

        for i, (query, results) in enumerate(all_results):
            for r in results:
                content_key = r.get("content", "")[:100]
                weight = original_weight if i == 0 else 1.0

                if content_key not in weights:
                    weights[content_key] = 0
                    doc_mapping[content_key] = r

                weights[content_key] += weight * (r.get("score", 1.0))

        ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for content_key, weighted_score in ranked[:top_k]:
            doc = doc_mapping[content_key].copy()
            doc["weighted_score"] = weighted_score
            merged.append(doc)

        return merged
