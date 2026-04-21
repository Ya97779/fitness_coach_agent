"""文本预处理器 - 清洗、标准化、去重"""

import re
from typing import List, Set, Tuple
from langchain_core.documents import Document


class TextPreprocessor:
    """文本预处理器

    功能：
    - 清洗：去除噪声字符、HTML 标签
    - 标准化：统一格式、标点
    - 去重：相似内容检测
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False,
        remove_duplicates: bool = True,
        similarity_threshold: float = 0.85
    ):
        """初始化预处理器

        Args:
            remove_urls: 是否移除 URL
            remove_emails: 是否移除邮箱
            remove_special_chars: 是否移除特殊字符
            lowercase: 是否转为小写
            remove_duplicates: 是否去重
            similarity_threshold: 相似度阈值
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_duplicates = remove_duplicates
        self.similarity_threshold = similarity_threshold

        self.seen_texts: Set[str] = set()

    def clean_text(self, text: str) -> str:
        """清洗单个文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        cleaned = text

        if self.remove_urls:
            cleaned = re.sub(
                r'https?://\S+|www\.\S+',
                '',
                cleaned
            )

        if self.remove_emails:
            cleaned = re.sub(
                r'\S+@\S+\.\S+',
                '',
                cleaned
            )

        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        cleaned = re.sub(r'\s+', ' ', cleaned)

        if self.remove_special_chars:
            cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?。，！？]', '', cleaned)

        if self.lowercase:
            cleaned = cleaned.lower()

        cleaned = cleaned.strip()

        return cleaned

    def normalize_text(self, text: str) -> str:
        """标准化文本格式

        Args:
            text: 文本

        Returns:
            标准化后的文本
        """
        text = re.sub(r'[Ff]ull\s*[Mm]oon', '满月', text)
        text = re.sub(r'[Hh]eart\s*[Rr]ate', '心率', text)
        text = re.sub(r'[Bb]lood\s*[Pp]ressure', '血压', text)

        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'。+', '。', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'！+', '！', text)

        text = re.sub(r'\s+([.,!?，。！？])', r'\1', text)
        text = re.sub(r'([.,!?，。！？])\s+', r'\1', text)

        return text

    def preprocess_document(self, doc: Document) -> Document:
        """预处理单个 Document

        Args:
            doc: 原始 Document

        Returns:
            预处理后的 Document（如果去重则可能返回 None）
        """
        content = self.clean_text(doc.page_content)
        content = self.normalize_text(content)

        if not content:
            return None

        if self.remove_duplicates:
            text_hash = hash(content)
            if text_hash in self.seen_texts:
                return None
            self.seen_texts.add(text_hash)

        doc.page_content = content
        return doc

    def preprocess_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """预处理文档列表

        Args:
            documents: Document 列表

        Returns:
            预处理后的 Document 列表
        """
        self.seen_texts.clear()

        cleaned_docs = []
        for doc in documents:
            processed = self.preprocess_document(doc)
            if processed:
                cleaned_docs.append(processed)

        return cleaned_docs

    def deduplicate_by_similarity(
        self,
        texts: List[str],
        threshold: float = None
    ) -> List[Tuple[int, str]]:
        """基于相似度去重

        Args:
            texts: 文本列表
            threshold: 相似度阈值（默认使用 self.similarity_threshold）

        Returns:
            去重后的 (索引, 文本) 列表
        """
        threshold = threshold or self.similarity_threshold
        unique_texts = []
        unique_indices = []

        def jaccard_similarity(a: str, b: str) -> float:
            a_words = set(a.split())
            b_words = set(b.split())
            if not a_words and not b_words:
                return 1.0
            intersection = len(a_words & b_words)
            union = len(a_words | b_words)
            return intersection / union if union > 0 else 0

        for i, text in enumerate(texts):
            is_duplicate = False
            for unique_text in unique_texts:
                if jaccard_similarity(text, unique_text) >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_texts.append(text)
                unique_indices.append(i)

        return list(zip(unique_indices, unique_texts))

    def reset(self):
        """重置去重记录"""
        self.seen_texts.clear()
