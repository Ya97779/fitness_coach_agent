"""BM25 关键词检索器 - 传统信息检索"""

import math
from typing import List, Tuple, Optional
from collections import Counter
import re


class BM25:
    """BM25 检索算法实现

    BM25 是一种基于词项频率的检索模型，是 Lucene/Elasticsearch 的核心。
    优点：
    - 对词项频率进行饱和处理
    - 考虑文档长度归一化
    - 简单高效
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        avgdl: Optional[float] = None
    ):
        """初始化 BM25

        Args:
            k1: 词项频率饱和参数（通常 1.2-2.0）
            b: 文档长度归一化参数（通常 0.75）
            avgdl: 平均文档长度（自动计算或手动指定）
        """
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl

        self.doc_count = 0
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.idf = {}
        self.corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按空格/中英文标点）

        Args:
            text: 文本

        Returns:
            词项列表
        """
        text = text.lower()
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return tokens

    def _calculate_idf(self) -> dict:
        """计算 IDF（逆文档频率）

        Returns:
            词项 -> IDF 值
        """
        idf = {}
        total_docs = self.doc_count

        for term, doc_freq in self.term_doc_freq.items():
            idf[term] = math.log(
                (total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1
            )

        return idf

    def fit(self, corpus: List[str]):
        """构建 BM25 索引

        Args:
            corpus: 文档列表
        """
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.term_doc_freq = Counter()

        for doc in corpus:
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)

            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(sum(term_freq.values()))

            for term in term_freq.keys():
                self.term_doc_freq[term] += 1

        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0
        self.idf = self._calculate_idf()

    def get_scores(self, query: str) -> List[float]:
        """获取查询对所有文档的 BM25 分数

        Args:
            query: 查询字符串

        Returns:
            每个文档的分数列表
        """
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.doc_count

        for token in query_tokens:
            if token not in self.idf:
                continue

            idf = self.idf[token]

            for i, doc_tf in enumerate(self.doc_term_freqs):
                if token not in doc_tf:
                    continue

                tf = doc_tf[token]
                dl = self.doc_lengths[i]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

                scores[i] += idf * (numerator / denominator)

        return scores

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """搜索最相关的文档

        Args:
            query: 查询字符串
            top_k: 返回前 k 个结果

        Returns:
            [(文档索引, 分数, 文档内容), ...]
        """
        scores = self.get_scores(query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for i, (doc_idx, score) in enumerate(ranked[:top_k]):
            if score > 0:
                results.append((doc_idx, score, self.corpus[doc_idx]))

        return results


class BM25Search:
    """BM25 检索器封装

    提供更简单的接口
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化 BM25 检索器"""
        self.bm25 = BM25(k1=k1, b=b)
        self.documents = []

    def index(self, documents: List[str]):
        """构建索引

        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.bm25.fit(documents)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[dict]:
        """搜索

        Args:
            query: 查询
            top_k: 返回数量

        Returns:
            [{index, score, content}, ...]
        """
        results = self.bm25.search(query, top_k)

        return [
            {
                "index": idx,
                "score": score,
                "content": doc
            }
            for idx, score, doc in results
        ]
