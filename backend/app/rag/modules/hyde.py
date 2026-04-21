"""HyDE - 假设性文档嵌入

核心思想：
1. 让 LLM 根据查询生成一个"假设性答案/文档"
2. 这个假设性文档与真实文档更相似（因为格式相近）
3. 用假设性文档去检索，效果更好

适用场景：
- 原始查询过于简短
- 查询与文档表述方式差异大
- 需要理解查询意图后检索
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的知识库文档撰写助手。根据用户问题，生成一个假设性的高质量答案。

要求：
1. 生成的答案应该像一个真实从知识库中检索出来的文档
2. 内容要专业、准确、详细
3. 可以包含具体的数据、步骤、注意事项等
4. 格式规范，像正式文档
5. 长度适中（100-300字）

如果无法生成有意义的答案，请输出"[无法生成]"。"""),
    ("human", "用户问题：{query}")
])


class HyDEGenerator:
    """假设性文档生成器

    使用 LLM 生成假设性答案/文档，用于增强检索效果。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        """初始化 HyDE 生成器

        Args:
            llm: LLM 模型（用于生成假设性文档）
            embeddings: 嵌入模型（用于向量化假设性文档）
        """
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.8
        )

        self.embeddings = embeddings or OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "embedding-2"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

        self.prompt = HYDE_PROMPT

    def generate(self, query: str) -> Optional[str]:
        """生成假设性文档

        Args:
            query: 用户查询

        Returns:
            假设性文档内容，失败返回 None
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query})
            content = response.content.strip()

            if content == "[无法生成]" or not content:
                return None

            return content

        except Exception as e:
            print(f"HyDE 生成失败: {e}")
            return None

    def generate_with_retry(
        self,
        query: str,
        max_retries: int = 2
    ) -> Optional[str]:
        """带重试的假设性文档生成

        Args:
            query: 用户查询
            max_retries: 最大重试次数

        Returns:
            假设性文档内容
        """
        for attempt in range(max_retries):
            content = self.generate(query)
            if content:
                return content
        return None

    def embed_query(self, query: str) -> List[float]:
        """直接嵌入查询

        Args:
            query: 查询文本

        Returns:
            嵌入向量
        """
        return self.embeddings.embed_query(query)


class HyDERetriever:
    """基于 HyDE 的检索器

    流程：
    1. 根据查询生成假设性文档
    2. 使用假设性文档进行检索
    3. 可选：用原始查询补充检索
    """

    def __init__(
        self,
        vectorstore: Any,
       hyde_generator: Optional[HyDEGenerator] = None,
        use_original: bool = True,
        hyde_weight: float = 0.7,
        original_weight: float = 0.3
    ):
        """初始化 HyDE 检索器

        Args:
            vectorstore: 向量数据库
            hyde_generator: HyDE 生成器
            use_original: 是否同时用原始查询检索
            hyde_weight: HyDE 检索结果权重
            original_weight: 原始查询检索结果权重
        """
        self.vectorstore = vectorstore
        self.hyde = hyde_generator or HyDEGenerator()
        self.use_original = use_original
        self.hyde_weight = hyde_weight
        self.original_weight = original_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """检索

        Args:
            query: 用户查询
            top_k: 返回数量

        Returns:
            {
                "hyde_doc": Optional[str],  # 假设性文档
                "results": List[Dict],       # 检索结果
                "method": str                 # 使用的检索方法
            }
        """
        hyde_doc = self.hyde.generate(query)

        results = []

        if hyde_doc:
            hyde_results = self.vectorstore.similarity_search_with_score(
                hyde_doc, k=top_k
            )
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "source": "hyde"
                }
                for doc, score in hyde_results
            ]

        if self.use_original:
            original_results = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )

            for doc, score in original_results:
                existing = False
                for r in results:
                    if r["content"] == doc.page_content:
                        r["score"] = (
                            r["score"] * self.hyde_weight +
                            score * self.original_weight
                        )
                        r["source"] = "hybrid"
                        existing = True
                        break

                if not existing:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score * self.original_weight,
                        "source": "original"
                    })

        results.sort(key=lambda x: x["score"])

        return {
            "hyde_doc": hyde_doc,
            "results": results[:top_k],
            "method": "hyde" if hyde_doc else "original"
        }

    def retrieve_with_alternatives(
        self,
        query: str,
        top_k: int = 5,
        num_alternatives: int = 3
    ) -> List[Dict[str, Any]]:
        """生成多个假设性答案并检索

        用于获取更多样化的检索结果。

        Args:
            query: 用户查询
            top_k: 每个假设答案的检索数量
            num_alternatives: 生成的假设答案数量

        Returns:
            合并后的检索结果
        """
        all_results = []

        hyde_doc = self.hyde.generate(query)
        if hyde_doc:
            results = self.vectorstore.similarity_search_with_score(
                hyde_doc, k=top_k
            )
            for doc, score in results:
                all_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "hyde_doc": hyde_doc,
                    "source": "hyde"
                })

        original_results = self.vectorstore.similarity_search_with_score(
            query, k=top_k
        )
        for doc, score in original_results:
            all_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "source": "original"
            })

        unique = {}
        for r in all_results:
            key = r["content"][:100]
            if key not in unique:
                unique[key] = r

        merged = list(unique.values())
        merged.sort(key=lambda x: x["score"])

        return merged[:top_k]
