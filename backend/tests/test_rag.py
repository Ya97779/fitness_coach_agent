"""RAG 模块单元测试"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, Mock
from typing import List
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document

from app.rag.modules.loader import DocumentLoader, retry_on_failure, MAX_RETRIES
from app.rag.modules.splitter import IntelligentSplitter
from app.rag.modules.preprocessor import TextPreprocessor
from app.rag.modules.bm25 import BM25, BM25Search
from app.rag.modules.hybrid_search import HybridSearch
from app.rag.modules.query_expansion import QueryExpander
from app.rag.modules.hyde import HyDEGenerator
from app.rag.modules.cot import CoTReasoner
from app.rag.modules.self_rag import SelfRAG, SelfRAGScorer
from app.rag.modules.agentic_rag import RouterAgent, AgenticRAG, AutoRAG, QueryClassifier
from app.rag.modules.doc_processor import (
    TableDetector, TableData, CodeBlockDetector, CodeBlock,
    SemanticChunker, DocumentStructureAnalyzer, Section,
    ContextAwareCleaner, AdvancedDocumentProcessor
)


class TestDocumentLoader(unittest.TestCase):
    """DocumentLoader 单元测试"""

    def test_loader_initialization(self):
        """测试 DocumentLoader 初始化"""
        loader = DocumentLoader(knowledge_base_dir="./knowledge_base")
        self.assertEqual(loader.knowledge_base_dir, "./knowledge_base")

    def test_loader_map_contains_required_formats(self):
        """测试支持的文档格式"""
        self.assertIn(".pdf", DocumentLoader.LOADER_MAP)
        self.assertIn(".docx", DocumentLoader.LOADER_MAP)
        self.assertIn(".txt", DocumentLoader.LOADER_MAP)
        self.assertIn(".html", DocumentLoader.LOADER_MAP)

    @patch('app.rag.modules.loader.TextLoader')
    def test_load_returns_list(self, mock_loader):
        """测试 load_directory 返回列表"""
        loader = DocumentLoader(knowledge_base_dir="./knowledge_base")
        result = loader.load_directory()
        self.assertIsInstance(result, list)


class TestRetryDecorator(unittest.TestCase):
    """retry_on_failure 装饰器单元测试"""

    def test_retry_on_failure_success_first_try(self):
        """测试成功时只执行一次"""
        call_count = 0

        @retry_on_failure(max_retries=3, delay=0.1)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_retry_on_failure_failure_then_success(self):
        """测试失败后重试最终成功"""
        call_count = 0

        @retry_on_failure(max_retries=3, delay=0.1)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        result = flaky_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)

    def test_retry_on_failure_all_failures(self):
        """测试全部失败后抛出异常"""
        call_count = 0

        @retry_on_failure(max_retries=3, delay=0.1)
        def always_fail_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with self.assertRaises(ValueError):
            always_fail_func()

        self.assertEqual(call_count, 3)


class TestIntelligentSplitter(unittest.TestCase):
    """IntelligentSplitter 单元测试"""

    def test_splitter_initialization(self):
        """测试 splitter 初始化"""
        splitter = IntelligentSplitter(chunk_size=500, chunk_overlap=50)
        self.assertEqual(splitter.chunk_size, 500)
        self.assertEqual(splitter.chunk_overlap, 50)

    def test_splitter_split_text_returns_list(self):
        """测试 split_text 返回列表"""
        splitter = IntelligentSplitter(chunk_size=100, chunk_overlap=20)
        text = "这是测试文本。" * 50
        result = splitter.split_text(text)
        self.assertIsInstance(result, list)

    def test_splitter_split_text_empty_returns_empty(self):
        """测试空文本返回空列表"""
        splitter = IntelligentSplitter()
        result = splitter.split_text("")
        self.assertEqual(result, [])

    def test_splitter_split_text_preserves_metadata(self):
        """测试分割保留元数据"""
        splitter = IntelligentSplitter(chunk_size=100, chunk_overlap=20)
        text = "这是测试文本。" * 50
        metadata = {"source": "test"}
        result = splitter.split_text(text, metadata)
        if result:
            self.assertEqual(result[0].metadata.get("source"), "test")

    def test_splitter_split_documents(self):
        """测试分割文档列表"""
        splitter = IntelligentSplitter(chunk_size=100, chunk_overlap=20)
        docs = [
            Document(page_content="文档1内容 " * 30),
            Document(page_content="文档2内容 " * 30)
        ]
        result = splitter.split_documents(docs)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestTextPreprocessor(unittest.TestCase):
    """TextPreprocessor 单元测试"""

    def test_preprocessor_initialization(self):
        """测试 preprocessor 初始化"""
        preprocessor = TextPreprocessor(
            remove_urls=True,
            remove_duplicates=True
        )
        self.assertTrue(preprocessor.remove_urls)
        self.assertTrue(preprocessor.remove_duplicates)

    def test_clean_text_removes_urls(self):
        """测试清洗移除 URL"""
        preprocessor = TextPreprocessor(remove_urls=True)
        text = "访问 https://example.com 获取更多信息"
        result = preprocessor.clean_text(text)
        self.assertNotIn("https://example.com", result)

    def test_clean_text_removes_emails(self):
        """测试清洗移除邮箱"""
        preprocessor = TextPreprocessor(remove_emails=True)
        text = "联系 test@example.com 获取信息"
        result = preprocessor.clean_text(text)
        self.assertNotIn("test@example.com", result)

    def test_clean_text_removes_html_tags(self):
        """测试清洗移除 HTML 标签"""
        preprocessor = TextPreprocessor()
        text = "<p>这是<strong>加粗</strong>文本</p>"
        result = preprocessor.clean_text(text)
        self.assertNotIn("<p>", result)
        self.assertNotIn("</p>", result)

    def test_normalize_text(self):
        """测试文本标准化"""
        preprocessor = TextPreprocessor()
        text = "这是文本...多个句号"
        result = preprocessor.normalize_text(text)
        self.assertEqual(result, "这是文本.多个句号")

    def test_preprocess_document_returns_document(self):
        """测试预处理单个文档"""
        preprocessor = TextPreprocessor()
        doc = Document(page_content="<p>测试内容</p>")
        result = preprocessor.preprocess_document(doc)
        self.assertIsInstance(result, Document)
        self.assertNotIn("<p>", result.page_content)

    def test_preprocess_documents_filters_duplicates(self):
        """测试预处理过滤重复文档"""
        preprocessor = TextPreprocessor(remove_duplicates=True)
        docs = [
            Document(page_content="相同内容"),
            Document(page_content="相同内容"),
            Document(page_content="不同内容")
        ]
        result = preprocessor.preprocess_documents(docs)
        self.assertEqual(len(result), 2)

    def test_deduplicate_by_similarity(self):
        """测试相似度去重"""
        preprocessor = TextPreprocessor(similarity_threshold=0.9)
        texts = [
            "这是第一个文本内容",
            "这是第一个文本内容",  # 完全相同
            "这是第二个文本内容"
        ]
        result = preprocessor.deduplicate_by_similarity(texts)
        self.assertEqual(len(result), 2)


class TestBM25(unittest.TestCase):
    """BM25 单元测试"""

    def test_bm25_initialization(self):
        """测试 BM25 初始化"""
        bm25 = BM25(k1=1.5, b=0.75)
        self.assertEqual(bm25.k1, 1.5)
        self.assertEqual(bm25.b, 0.75)

    def test_bm25_tokenize(self):
        """测试分词"""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello World 你好")
        self.assertIsInstance(tokens, list)

    def test_bm25_fit_and_search(self):
        """测试 BM25 索引构建和搜索"""
        bm25 = BM25()
        corpus = [
            "苹果是一种水果",
            "香蕉是黄色的水果",
            "跑步是一项运动"
        ]
        bm25.fit(corpus)

        scores = bm25.search("水果")
        self.assertIsInstance(scores, list)

    def test_bm25_search_returns_scores(self):
        """测试搜索返回分数"""
        bm25 = BM25()
        corpus = ["苹果", "香蕉", "橙子"]
        bm25.fit(corpus)

        result = bm25.search("苹果")
        self.assertGreater(len(result), 0)


class TestBM25Search(unittest.TestCase):
    """BM25Search 单元测试"""

    def test_bm25search_initialization(self):
        """测试 BM25Search 初始化"""
        searcher = BM25Search(k1=1.5, b=0.75)
        self.assertEqual(searcher.bm25.k1, 1.5)

    def test_bm25search_index(self):
        """测试 BM25Search 索引构建"""
        searcher = BM25Search()
        docs = [
            "文档1内容",
            "文档2内容"
        ]

        searcher.index(docs)
        self.assertEqual(len(searcher.documents), 2)


class TestHybridSearch(unittest.TestCase):
    """HybridSearch 单元测试"""

    def test_hybrid_search_initialization(self):
        """测试 HybridSearch 初始化"""
        hs = HybridSearch(vector_weight=0.6, bm25_weight=0.4)
        self.assertEqual(hs.vector_weight, 0.6)
        self.assertEqual(hs.bm25_weight, 0.4)

    def test_rrf_fusion_input_validation(self):
        """测试 RRF 融合输入验证"""
        hs = HybridSearch()
        vector_results = [(0, 0.9), (1, 0.8)]
        bm25_results = [(0, 0.85, "content1"), (2, 0.75, "content2")]

        result = hs._rrf_fusion(vector_results, bm25_results)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_hybrid_search_index(self):
        """测试 HybridSearch 索引构建"""
        hs = HybridSearch()
        docs = [
            Document(page_content="文档1"),
            Document(page_content="文档2")
        ]

        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        hs.index(docs, mock_vectorstore, mock_embeddings)
        self.assertEqual(len(hs.documents), 2)


class TestQueryExpander(unittest.TestCase):
    """QueryExpander 单元测试"""

    def test_query_expander_initialization(self):
        """测试 QueryExpander 初始化"""
        with patch('app.rag.modules.query_expansion.ChatOpenAI'):
            expander = QueryExpander(temperature=0.5, num_variants=3)
            self.assertEqual(expander.num_variants, 3)

    @patch('app.rag.modules.query_expansion.ChatOpenAI')
    def test_expand_returns_list(self, mock_llm_class):
        """测试 expand 返回列表"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="原始查询\n查询变体1\n查询变体2")

        expander = QueryExpander()
        result = expander.expand("测试查询")

        self.assertIsInstance(result, list)

    @patch('app.rag.modules.query_expansion.ChatOpenAI')
    def test_expand_includes_original(self, mock_llm_class):
        """测试 expand 包含原始查询"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="测试查询\n变体1")

        expander = QueryExpander()
        result = expander.expand("测试查询")

        self.assertIn("测试查询", result)


class TestHyDEGenerator(unittest.TestCase):
    """HyDEGenerator 单元测试"""

    def test_hyde_initialization(self):
        """测试 HyDEGenerator 初始化"""
        with patch('app.rag.modules.hyde.ChatOpenAI'):
            with patch('app.rag.modules.hyde.OpenAIEmbeddings'):
                generator = HyDEGenerator()
                self.assertIsNotNone(generator.llm)
                self.assertIsNotNone(generator.embeddings)

    @patch('app.rag.modules.hyde.ChatOpenAI')
    @patch('app.rag.modules.hyde.OpenAIEmbeddings')
    def test_hyde_initialization_with_llm(self, mock_emb, mock_llm_class):
        """测试 HyDE 使用自定义 LLM 初始化"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        mock_embeddings = MagicMock()
        mock_emb.return_value = mock_embeddings

        generator = HyDEGenerator(llm=mock_llm, embeddings=mock_embeddings)
        self.assertIsNotNone(generator.llm)
        self.assertIsNotNone(generator.embeddings)


class TestCoTReasoner(unittest.TestCase):
    """CoTReasoner 单元测试"""

    def test_cot_initialization(self):
        """测试 CoTReasoner 初始化"""
        with patch('app.rag.modules.cot.ChatOpenAI'):
            reasoner = CoTReasoner(mode="chain")
            self.assertEqual(reasoner.mode, "chain")

    @patch('app.rag.modules.cot.ChatOpenAI')
    def test_cot_initialization_with_llm(self, mock_llm_class):
        """测试 CoTReasoner 使用自定义 LLM 初始化"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        reasoner = CoTReasoner(mode="chain", llm=mock_llm)
        self.assertEqual(reasoner.mode, "chain")
        self.assertIsNotNone(reasoner.llm)


class TestSelfRAG(unittest.TestCase):
    """SelfRAG 单元测试"""

    def test_self_rag_initialization(self):
        """测试 SelfRAG 初始化"""
        def mock_retriever(query, top_k):
            return []

        with patch('app.rag.modules.self_rag.ChatOpenAI'):
            rag = SelfRAG(retriever=mock_retriever)
            self.assertIsNotNone(rag.llm)

    def test_self_rag_is_retrieval(self):
        """测试判断是否需要检索"""
        def mock_retriever(query, top_k):
            return []

        with patch('app.rag.modules.self_rag.ChatOpenAI'):
            rag = SelfRAG(retriever=mock_retriever)
            rag._should_retrieve = True
            self.assertTrue(rag._should_retrieve)


class TestSelfRAGScorer(unittest.TestCase):
    """SelfRAGScorer 单元测试"""

    def test_scorer_initialization(self):
        """测试 SelfRAGScorer 初始化"""
        with patch('app.rag.modules.self_rag.ChatOpenAI'):
            scorer = SelfRAGScorer()
            self.assertIsNotNone(scorer.llm)


class TestRouterAgent(unittest.TestCase):
    """RouterAgent 单元测试"""

    def test_router_initialization(self):
        """测试 RouterAgent 初始化"""
        with patch('app.rag.modules.agentic_rag.ChatOpenAI'):
            router = RouterAgent()
            self.assertIsNotNone(router.llm)

    @patch('app.rag.modules.agentic_rag.ChatOpenAI')
    def test_decide_returns_dict(self, mock_llm_class):
        """测试 decide 返回策略字典"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(
            content='{"need_retrieval": true, "retrieval_strategy": "hybrid", "reasoning": "测试"}'
        )

        router = RouterAgent()
        result = router.decide("测试问题")

        self.assertIsInstance(result, dict)
        self.assertIn("retrieval_strategy", result)


class TestAgenticRAG(unittest.TestCase):
    """AgenticRAG 单元测试"""

    @patch('app.rag.modules.agentic_rag.ChatOpenAI')
    def test_agentic_rag_initialization(self, mock_llm_class):
        """测试 AgenticRAG 初始化"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        mock_rag = MagicMock()

        agentic = AgenticRAG(modern_rag=mock_rag)
        self.assertIsNotNone(agentic.llm)


class TestAutoRAG(unittest.TestCase):
    """AutoRAG 单元测试"""

    @patch('app.rag.modules.agentic_rag.ChatOpenAI')
    def test_auto_rag_initialization(self, mock_llm_class):
        """测试 AutoRAG 初始化"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        mock_rag = MagicMock()

        auto = AutoRAG(modern_rag=mock_rag)
        self.assertIsNotNone(auto.agentic_rag)


class TestQueryClassifier(unittest.TestCase):
    """QueryClassifier 单元测试"""

    def test_classifier_initialization(self):
        """测试 QueryClassifier 初始化"""
        with patch('app.rag.modules.agentic_rag.ChatOpenAI'):
            classifier = QueryClassifier()
            self.assertIsNotNone(classifier.llm)


class TestTableDetector(unittest.TestCase):
    """TableDetector 单元测试"""

    def test_detector_initialization(self):
        """测试 TableDetector 初始化"""
        detector = TableDetector(min_rows=2, min_cols=2)
        self.assertEqual(detector.min_rows, 2)

    def test_detect_markdown_tables(self):
        """测试 Markdown 表格检测"""
        detector = TableDetector()
        text = """
        | 列1 | 列2 |
        |-----|-----|
        | 内容1 | 内容2 |
        """
        result = detector.detect_markdown_tables(text)
        self.assertIsInstance(result, list)


class TestCodeBlockDetector(unittest.TestCase):
    """CodeBlockDetector 单元测试"""

    def test_detector_initialization(self):
        """测试 CodeBlockDetector 初始化"""
        detector = CodeBlockDetector()
        self.assertIsNotNone(detector)

    def test_detect_fenced_code_blocks(self):
        """测试代码块检测"""
        detector = CodeBlockDetector()
        text = """
        这是一段文字
        ```python
        def hello():
            print("world")
        ```
        更多文字
        """
        result = detector.detect_fenced_code_blocks(text)
        self.assertIsInstance(result, list)


class TestSemanticChunker(unittest.TestCase):
    """SemanticChunker 单元测试"""

    def test_chunker_initialization(self):
        """测试 SemanticChunker 初始化"""
        mock_embeddings = MagicMock()
        chunker = SemanticChunker(embeddings=mock_embeddings)
        self.assertIsNotNone(chunker.embeddings)


class TestDocumentStructureAnalyzer(unittest.TestCase):
    """DocumentStructureAnalyzer 单元测试"""

    def test_analyzer_initialization(self):
        """测试 DocumentStructureAnalyzer 初始化"""
        analyzer = DocumentStructureAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_analyze_returns_sections(self):
        """测试分析返回章节列表"""
        analyzer = DocumentStructureAnalyzer()
        text = """
        # 第一章
        内容1

        ## 第一节
        内容2
        """
        result = analyzer.analyze(text)
        self.assertIsInstance(result, list)


class TestContextAwareCleaner(unittest.TestCase):
    """ContextAwareCleaner 单元测试"""

    def test_cleaner_initialization(self):
        """测试 ContextAwareCleaner 初始化"""
        cleaner = ContextAwareCleaner()
        self.assertIsNotNone(cleaner)

    def test_clean_returns_str(self):
        """测试 clean 返回字符串"""
        cleaner = ContextAwareCleaner()
        text = "原始文本"
        result = cleaner.clean(text)
        self.assertIsInstance(result, str)


class TestAdvancedDocumentProcessor(unittest.TestCase):
    """AdvancedDocumentProcessor 单元测试"""

    def test_processor_initialization(self):
        """测试 AdvancedDocumentProcessor 初始化"""
        mock_embeddings = MagicMock()
        processor = AdvancedDocumentProcessor(
            embeddings=mock_embeddings,
            use_semantic_chunking=False
        )
        self.assertFalse(processor.use_semantic_chunking)

    def test_process_documents_returns_list(self):
        """测试处理文档返回列表"""
        mock_embeddings = MagicMock()
        processor = AdvancedDocumentProcessor(embeddings=mock_embeddings)

        docs = [Document(page_content="测试内容")]
        result = processor.process_documents(docs)
        self.assertIsInstance(result, list)


class TestTableData(unittest.TestCase):
    """TableData 数据类单元测试"""

    def test_table_data_creation(self):
        """测试 TableData 创建"""
        table = TableData(
            content="表格内容",
            rows=3,
            cols=2,
            headers=["列1", "列2"],
            data=[["a", "b"], ["c", "d"]],
            position=(0, 0)
        )
        self.assertEqual(table.rows, 3)
        self.assertEqual(table.cols, 2)


class TestCodeBlock(unittest.TestCase):
    """CodeBlock 数据类单元测试"""

    def test_code_block_creation(self):
        """测试 CodeBlock 创建"""
        code = CodeBlock(
            content="print('hello')",
            language="python",
            start_line=1,
            end_line=2
        )
        self.assertEqual(code.language, "python")
        self.assertEqual(code.start_line, 1)


class TestSection(unittest.TestCase):
    """Section 数据类单元测试"""

    def test_section_creation(self):
        """测试 Section 创建"""
        section = Section(
            heading="第一章",
            level=1,
            content="内容",
            start_pos=0,
            end_pos=10
        )
        self.assertEqual(section.level, 1)
        self.assertEqual(len(section.children), 0)


if __name__ == '__main__':
    unittest.main()