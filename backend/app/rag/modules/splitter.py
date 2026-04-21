"""智能文本分割器 - 考虑文档结构的语义分割"""

import re
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IntelligentSplitter:
    """智能文本分割器

    特性：
    - 识别并保留标题结构
    - 动态调整分割尺寸
    - 句子边界保护
    - 元数据保留
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """初始化分割器

        Args:
            chunk_size: 默认块大小（字符数）
            chunk_overlap: 块重叠大小
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        )

    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """分割文本为块

        Args:
            text: 待分割文本
            metadata: 附加元数据

        Returns:
            Document 列表
        """
        if not text or not text.strip():
            return []

        chunks = self.base_splitter.split_text(text)

        docs = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "chunk_index": i,
                "chunk_count": len(chunks),
                **(metadata or {})
            }
            docs.append(Document(page_content=chunk, metadata=chunk_meta))

        return docs

    def split_documents(
        self,
        documents: List[Document],
        metadata_prefix: str = ""
    ) -> List[Document]:
        """分割文档列表

        Args:
            documents: Document 列表
            metadata_prefix: 元数据键前缀

        Returns:
            分割后的 Document 列表
        """
        all_chunks = []

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)

            chunks = self.split_text(doc.page_content, {
                f"{metadata_prefix}source" if metadata_prefix else "source": source,
                f"{metadata_prefix}page" if metadata_prefix else "page": page,
                "original_length": len(doc.page_content)
            })

            all_chunks.extend(chunks)

        return all_chunks

    def split_by_headings(
        self,
        text: str,
        metadata: dict = None
    ) -> List[Document]:
        """按标题分割（保留标题作为上下文）

        适用于结构化文档如 Markdown、HTML

        Args:
            text: 待分割文本
            metadata: 附加元数据

        Returns:
            Document 列表
        """
        heading_pattern = r'^#{1,6}\s+.+$|^\[.*\]\(.*\)|^【.*】$'

        lines = text.split('\n')
        sections = []
        current_section = []
        current_heading = ""

        for line in lines:
            is_heading = bool(re.match(heading_pattern, line.strip()))

            if is_heading and current_section:
                section_text = '\n'.join(current_section)
                if section_text.strip():
                    sections.append((current_heading, section_text))
                current_section = []

            if is_heading:
                current_heading = line.strip()
            else:
                current_section.append(line)

        if current_section:
            section_text = '\n'.join(current_section)
            if section_text.strip():
                sections.append((current_heading, section_text))

        docs = []
        for i, (heading, content) in enumerate(sections):
            doc_meta = {
                "chunk_index": i,
                "heading": heading,
                "section_count": len(sections),
                **(metadata or {})
            }
            full_content = f"{heading}\n{content}" if heading else content
            docs.append(Document(page_content=full_content, metadata=doc_meta))

        return docs

    def split_by_paragraphs(
        self,
        text: str,
        metadata: dict = None
    ) -> List[Document]:
        """按段落分割（保持段落完整性）

        Args:
            text: 待分割文本
            metadata: 附加元数据

        Returns:
            Document 列表
        """
        paragraphs = re.split(r'\n\s*\n', text)

        docs = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            if para_length < self.min_chunk_size:
                docs.append(Document(page_content=para, metadata={
                    "chunk_index": i,
                    "is_small": True,
                    **(metadata or {})
                }))
            elif para_length > self.max_chunk_size:
                sub_chunks = self.split_text(para, metadata)
                docs.extend(sub_chunks)
            else:
                docs.append(Document(page_content=para, metadata={
                    "chunk_index": i,
                    "is_small": False,
                    **(metadata or {})
                }))

        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_count"] = len(docs)

        return docs
