"""RAG 系统 - 现代化升级版

目录结构：
├── __init__.py          # 主入口
├── modules/             # 核心模块
│   ├── __init__.py     # 模块导出
│   ├── loader.py       # 文档加载器（带重试）
│   ├── splitter.py     # 智能文本分割器
│   ├── preprocessor.py # 文本预处理器
│   ├── bm25.py         # BM25 关键词检索
│   ├── hybrid_search.py # 混合检索（向量+BM25）
│   ├── query_expansion.py # 多查询扩展
│   ├── hyde.py         # 假设性文档嵌入
│   ├── cot.py          # 思维链推理
│   └── self_rag.py     # 自我反思纠正
└── rag_utils.py        # 兼容旧接口

使用方式：
    from app.rag import ModernRAG

    rag = ModernRAG()
    results = rag.search("上斜卧推怎么做")

高级用法：
    # 启用 Self-RAG 模式
    rag = ModernRAG(enable_self_rag=True)
    result = rag.query("如何科学增肌？")

    # 启用 HyDE 模式
    rag = ModernRAG(enable_hyde=True)
    results = rag.search("腿部训练计划")
"""

import os
import shutil
from typing import List, Optional, Dict, Any, Callable
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

from .modules import (
    DocumentLoader,
    IntelligentSplitter,
    TextPreprocessor,
    HybridSearch,
    QueryExpander,
    HyDEGenerator,
    CoTReasoner,
    SelfRAG as SelfRAGModule,
    RouterAgent,
    AgenticRAG,
    AutoRAG,
    AdvancedDocumentProcessor
)

load_dotenv()

CHROMA_DIR = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class ModernRAG:
    """现代化 RAG 系统

    支持：
    - 基础增强：文档加载、智能分割、预处理、BM25、混合检索
    - 高级检索：Query Expansion、HyDE
    - 生成增强：CoT 思维链、Self-RAG 自我反思

    使用示例：
        # 基础检索
        rag = ModernRAG()
        results = rag.search("上斜卧推")

        # HyDE 检索（假设性文档嵌入）
        rag = ModernRAG(enable_hyde=True)
        results = rag.search("减脂饮食")

        # Self-RAG（自我反思）
        rag = ModernRAG(enable_self_rag=True)
        result = rag.query("增肌饮食计划")

        # CoT 思维链推理
        rag = ModernRAG(enable_cot=True)
        result = rag.query("为什么深蹲膝盖内扣")
    """

    def __init__(
        self,
        knowledge_base_dir: str = KNOWLEDGE_BASE_DIR,
        chroma_dir: str = CHROMA_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        force_rebuild: bool = False,
        enable_query_expansion: bool = False,
        enable_hyde: bool = False,
        enable_cot: bool = False,
        enable_self_rag: bool = False,
        enable_agentic: bool = False,
        llm_model: str = None,
        embedding_model: str = None
    ):
        """初始化 ModernRAG

        Args:
            knowledge_base_dir: 知识库目录
            chroma_dir: Chroma 向量库目录
            chunk_size: 文本分割块大小
            chunk_overlap: 块重叠大小
            vector_weight: 向量检索权重
            bm25_weight: BM25 权重
            force_rebuild: 是否强制重建索引
            enable_query_expansion: 启用多查询扩展
            enable_hyde: 启用假设性文档嵌入
            enable_cot: 启用思维链推理
            enable_self_rag: 启用自我反思纠正
            llm_model: LLM 模型名称
            embedding_model: 嵌入模型名称
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.chroma_dir = chroma_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.enable_query_expansion = enable_query_expansion
        self.enable_hyde = enable_hyde
        self.enable_cot = enable_cot
        self.enable_self_rag = enable_self_rag
        self.enable_agentic = enable_agentic

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or os.getenv("EMBEDDING_MODEL", "embedding-2"),
            api_key=api_key,
            base_url=api_base
        )

        self.llm = ChatOpenAI(
            model=llm_model or os.getenv("LLM_MODEL", "glm-4"),
            api_key=api_key,
            base_url=api_base,
            temperature=0.7
        )

        self.loader = DocumentLoader(knowledge_base_dir)
        self.splitter = IntelligentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.preprocessor = TextPreprocessor(
            remove_duplicates=True,
            similarity_threshold=0.85
        )

        self.hybrid_search: Optional[HybridSearch] = None
        self.vectorstore: Optional[Chroma] = None
        self.documents: List[Document] = []

        self._init_components()

        if enable_query_expansion:
            self.query_expander = QueryExpander(llm=self.llm)
        else:
            self.query_expander = None

        if enable_hyde:
            self.hyde_generator = HyDEGenerator(llm=self.llm, embeddings=self.embeddings)
        else:
            self.hyde_generator = None

        if enable_cot:
            self.cot_reasoner = CoTReasoner(llm=self.llm, mode="chain")
        else:
            self.cot_reasoner = None

        if enable_self_rag:
            self._setup_self_rag()
        else:
            self.self_rag = None

        if enable_agentic:
            self._setup_agentic_rag()
        else:
            self.agentic_rag = None

    def _setup_agentic_rag(self):
        """设置 Agentic RAG"""
        from .modules import AgenticRAG as AgenticRAGModule

        self.agentic_rag = AgenticRAGModule(
            modern_rag=self,
            llm=self.llm
        )

    def _init_components(self, force_rebuild: bool = False):
        """初始化核心组件

        Args:
            force_rebuild: 是否强制重建
        """
        if force_rebuild and os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

        if os.path.exists(self.chroma_dir) and os.listdir(self.chroma_dir):
            print(f"加载现有向量库: {self.chroma_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
            self._load_existing_docs()
        else:
            print("构建新向量库...")
            self._build_index()

        self._setup_hybrid_search()

    def _load_existing_docs(self):
        """加载现有文档"""
        if self.vectorstore:
            all_docs = self.vectorstore.get()
            self.documents = [
                Document(page_content=doc, metadata={})
                for doc in all_docs["documents"]
            ]

    def _build_index(self):
        """构建向量库索引"""
        docs = self.loader.load_directory()

        if not docs:
            print("知识库为空，使用默认知识")
            docs = [Document(
                page_content="欢迎使用健身教练系统。您可以向我询问健身、营养相关的问题。",
                metadata={"source": "system"}
            )]

        print(f"加载文档: {len(docs)} 个")

        processor = AdvancedDocumentProcessor(
            embeddings=self.embeddings,
            use_semantic_chunking=True,
            detect_tables=True,
            detect_code=True,
            analyze_structure=True
        )
        processed_docs = processor.process_documents(docs)
        docs = processed_docs
        print(f"高级处理后: {len(docs)} 个块")

        docs = self.splitter.split_documents(docs)
        print(f"分割后: {len(docs)} 个块")

        docs = self.preprocessor.preprocess_documents(docs)
        print(f"预处理后: {len(docs)} 个块")

        self.documents = docs

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.chroma_dir
        )

        print(f"向量库已构建: {self.vectorstore._collection.count()} 条")

    def _setup_hybrid_search(self):
        """设置混合检索"""
        if not self.documents and self.vectorstore:
            self._load_existing_docs()

        if self.documents:
            self.hybrid_search = HybridSearch(
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight
            )
            self.hybrid_search.index(
                self.documents,
                self.vectorstore,
                self.embeddings
            )

    def _setup_self_rag(self):
        """设置 Self-RAG"""
        def retriever_func(query: str, top_k: int) -> List[Dict]:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            return [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ]

        self.self_rag = SelfRAGModule(
            retriever=retriever_func,
            llm=self.llm
        )

    def _basic_retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """基础检索

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        try:
            if self.hybrid_search:
                results = self.hybrid_search.search(query, top_k)
            else:
                docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
                results = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in docs
                ]
            return results
        except Exception as e:
            print(f"检索错误: {e}")
            return []

    def _query_expansion_retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """多查询扩展检索

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if not self.query_expander:
            return self._basic_retrieve(query, top_k)

        expanded_queries = self.query_expander.expand(query)
        print(f"扩展查询: {expanded_queries}")

        all_results = []
        seen = set()

        for q in expanded_queries:
            results = self._basic_retrieve(q, top_k * 2)
            for r in results:
                key = r["content"][:100]
                if key not in seen:
                    seen.add(key)
                    r["query_source"] = q
                    all_results.append(r)

        all_results.sort(key=lambda x: x.get("score", 1))
        return all_results[:top_k]

    def _hyde_retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """HyDE 检索

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            {
                "hyde_doc": str,    # 假设性文档
                "results": List,    # 检索结果
                "method": str       # 使用的检索方法
            }
        """
        hyde_doc = None
        results = []

        if self.hyde_generator:
            hyde_doc = self.hyde_generator.generate(query)
            if hyde_doc:
                print(f"HyDE 生成: {hyde_doc[:50]}...")
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

        original_results = self.vectorstore.similarity_search_with_score(
            query, k=top_k
        )

        for doc, score in original_results:
            existing = False
            for r in results:
                if r["content"] == doc.page_content:
                    r["score"] = 0.5 * r["score"] + 0.5 * score
                    r["source"] = "hybrid"
                    existing = True
                    break
            if not existing:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "source": "original"
                })

        results.sort(key=lambda x: x["score"])

        return {
            "hyde_doc": hyde_doc,
            "results": results[:top_k],
            "method": "hyde" if hyde_doc else "original"
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """检索

        Args:
            query: 查询字符串
            top_k: 返回数量
            mode: 检索模式
                - "hybrid": 混合检索（向量+BM25）
                - "vector": 仅向量检索
                - "bm25": 仅 BM25 检索
                - "query_expansion": 多查询扩展
                - "hyde": 假设性文档嵌入

        Returns:
            [{
                "content": str,
                "metadata": dict,
                "score": float,
                "source": str  # (hyde/ hybrid/ original)
            }, ...]
        """
        if not query or not query.strip():
            return [{"content": "请提供有效的查询内容。", "metadata": {}, "score": 0}]

        try:
            if mode == "query_expansion" or self.enable_query_expansion:
                return self._query_expansion_retrieve(query, top_k)

            if mode == "hyde" or self.enable_hyde:
                hyde_result = self._hyde_retrieve(query, top_k)
                return hyde_result["results"]

            if mode == "bm25":
                from .modules import BM25Search
                bm25 = BM25Search()
                bm25.index([doc.page_content for doc in self.documents])
                raw_results = bm25.search(query, top_k)
                return [
                    {
                        "content": r["content"],
                        "metadata": self.documents[r["index"]].metadata,
                        "score": r["score"],
                        "source": "bm25"
                    }
                    for r in raw_results
                ]

            if mode == "vector":
                docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
                return [
                    {"content": doc.page_content, "metadata": doc.metadata, "score": score, "source": "vector"}
                    for doc, score in docs
                ]

            return self._basic_retrieve(query, top_k)

        except Exception as e:
            print(f"检索错误: {e}")
            return [{"content": f"检索失败: {str(e)}", "metadata": {}, "score": 0}]

    def query(
        self,
        query: str,
        top_k: int = 5,
        show_reasoning: bool = True
    ) -> Dict[str, Any]:
        """查询（带生成）

        整合检索和生成，支持 CoT 思维链和 Self-RAG 自我反思。
        如果启用了 agentic 模式，由大模型自主决定策略。

        Args:
            query: 用户问题
            top_k: 检索数量
            show_reasoning: 是否显示推理过程（CoT 模式）

        Returns:
            {
                "answer": str,           # 最终回答
                "reasoning": str,       # 推理过程（CoT 模式）
                "sources": List[Dict],   # 参考来源
                "reflection": str,      # 自我反思标记（Self-RAG 模式）
                "confidence": float     # 可信度
            }
        """
        if self.enable_agentic and self.agentic_rag:
            return self.agentic_rag.query(query)

        if self.enable_self_rag and self.self_rag:
            result = self.self_rag.query(query, top_k)
            return {
                "answer": result["answer"],
                "reasoning": "",
                "sources": result["retrieval_used"],
                "reflection": result["reflection"],
                "confidence": result["utility_score"] / 4.0
            }

        results = self.search(query, top_k)

        context = "\n\n".join([
            f"[来源] {r['content']}"
            for r in results
        ]) if results else "无相关检索结果"

        if self.enable_cot and self.cot_reasoner:
            cot_result = self.cot_reasoner.reason_with_sources(
                query, context, results
            )
            return {
                "answer": cot_result["answer"],
                "reasoning": cot_result["reasoning"] if show_reasoning else "",
                "sources": cot_result["sources"],
                "reflection": "",
                "confidence": cot_result.get("confidence", 0.5)
            }

        answer = self._generate_answer(query, context)
        return {
            "answer": answer,
            "reasoning": "",
            "sources": results,
            "reflection": "",
            "confidence": 0.5
        }

    def _generate_answer(self, query: str, context: str) -> str:
        """生成回答

        Args:
            query: 用户问题
            context: 检索上下文

        Returns:
            生成的回答
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的健身与营养顾问。基于检索信息回答用户问题。

要求：
1. 优先使用检索信息
2. 如果检索信息不足，可以结合自身知识
3. 保持回答准确、清晰、有条理
4. 如果不确定，明确说明"""),
            ("human", "问题：{query}\n\n检索内容：\n{context}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            return chain.invoke({
                "query": query,
                "context": context
            })
        except Exception as e:
            print(f"回答生成失败: {e}")
            return f"生成回答时出错: {str(e)}"

    def add_document(self, content: str, metadata: dict = None):
        """添加单个文档

        Args:
            content: 文档内容
            metadata: 元数据
        """
        chunks = self.splitter.split_text(content, metadata)
        chunks = self.preprocessor.preprocess_documents(chunks)

        for chunk in chunks:
            self.vectorstore.add_documents([chunk])
            self.documents.append(chunk)

        if self.hybrid_search:
            self.hybrid_search.documents = self.documents
            self.hybrid_search.bm25.index(
                [doc.page_content for doc in self.documents]
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "chroma_count": self.vectorstore._collection.count() if self.vectorstore else 0,
            "chroma_dir": self.chroma_dir,
            "knowledge_base_dir": self.knowledge_base_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_hyde": self.enable_hyde,
            "enable_cot": self.enable_cot,
            "enable_self_rag": self.enable_self_rag,
            "enable_agentic": self.enable_agentic
        }


modern_rag_instance: Optional[ModernRAG] = None


def get_rag_instance(**kwargs) -> ModernRAG:
    """获取全局 RAG 实例（单例）"""
    global modern_rag_instance
    if modern_rag_instance is None:
        modern_rag_instance = ModernRAG(**kwargs)
    return modern_rag_instance


def rag_medical_search(query: str, top_k: int = 3) -> str:
    """兼容旧接口的检索函数

    Args:
        query: 查询字符串
        top_k: 返回数量

    Returns:
        检索结果文本
    """
    rag = get_rag_instance()
    results = rag.search(query, top_k, mode="hybrid")

    if not results:
        return "在知识库中未找到相关信息。"

    context = "\n\n".join([r["content"] for r in results])
    return context
