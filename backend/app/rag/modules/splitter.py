"""智能文本分割器 - 考虑文档结构的语义分割"""

import re
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MarkdownSplitter:
    """Markdown 标题层级感知分割器

    以 markdown 的 # 标题结构作为分块边界，替代按字符数硬切的方式。
    每个 chunk 携带标题路径元数据，用于检索时提供上下文。

    特性：
    - 按标题层级切分，保留完整知识点
    - 标题路径元数据（如 "力量训练 > 深蹲 > 力学原理"）
    - 过长 chunk 按子标题拆分，过短 chunk 合并
    - 无标题内容作为独立 chunk
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._heading_re = re.compile(r'^(#{1,6})\s+(.+)$')

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割 markdown 文档列表

        Args:
            documents: 由 TextLoader 加载的 Document 列表

        Returns:
            按标题分块的 Document 列表
        """
        all_chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunks = self._split_markdown(doc.page_content, source)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_markdown(self, text: str, source: str) -> List[Document]:
        """将 markdown 文本按标题层级切分

        Args:
            text: markdown 原文
            source: 文件来源路径

        Returns:
            Document 列表
        """
        lines = text.split('\n')
        filename = source.split('/')[-1].split('\\')[-1]
        default_title = re.sub(r'\.md$', '', filename)

        # 第一步：按标题解析出段落
        sections = []  # [(heading_level, heading_text, content_lines), ...]
        current_level = 0
        current_heading = ""
        current_lines = []

        for line in lines:
            m = self._heading_re.match(line.strip())
            if m:
                if current_lines or current_heading:
                    sections.append((current_level, current_heading, current_lines))
                current_level = len(m.group(1))
                current_heading = m.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines or current_heading:
            sections.append((current_level, current_heading, current_lines))

        # 没有标题的文件，整体作为一个 chunk
        if not sections:
            text_stripped = text.strip()
            if text_stripped:
                return [Document(
                    page_content=text_stripped,
                    metadata={
                        "source": source,
                        "heading_path": default_title,
                        "heading_level": 0,
                        "chunk_index": 0,
                        "chunk_count": 1,
                    }
                )]
            return []

        # 第二步：构建标题路径，生成 chunk
        chunks = []
        heading_stack = {}  # {level: heading_text}

        for level, heading, content_lines in sections:
            content = '\n'.join(content_lines).strip()

            # 更新标题栈：清除更深层级的标题
            for lv in list(heading_stack.keys()):
                if lv >= level:
                    del heading_stack[lv]
            if heading:
                heading_stack[level] = heading

            # 构建标题路径
            path_parts = [default_title]
            for lv in sorted(heading_stack.keys()):
                if heading_stack[lv]:
                    path_parts.append(heading_stack[lv])
            heading_path = ' > '.join(path_parts)

            if not content:
                continue

            if heading:
                full_content = f"{'#' * level} {heading}\n{content}"
            else:
                full_content = content

            chunks.append(Document(
                page_content=full_content,
                metadata={
                    "source": source,
                    "heading_path": heading_path,
                    "heading_level": level,
                    "chunk_index": 0,
                    "chunk_count": 0,
                }
            ))

        # 第三步：合并过短 chunk，拆分过长 chunk
        chunks = self._merge_short_chunks(chunks)
        chunks = self._split_long_chunks(chunks)

        # 第四步：统一编号
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(chunks)

        return chunks

    def _merge_short_chunks(self, chunks: List[Document]) -> List[Document]:
        """合并过短的 chunk 到相邻 chunk"""
        if not chunks:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk.page_content) < self.min_chunk_size:
                prev = merged[-1]
                prev.page_content = prev.page_content + '\n\n' + chunk.page_content
            else:
                merged.append(chunk)

        if len(merged) > 1 and len(merged[0].page_content) < self.min_chunk_size:
            merged[1].page_content = merged[0].page_content + '\n\n' + merged[1].page_content
            merged.pop(0)

        return merged

    def _split_long_chunks(self, chunks: List[Document]) -> List[Document]:
        """对过长的 chunk 按段落进一步拆分"""
        result = []
        for chunk in chunks:
            if len(chunk.page_content) <= self.max_chunk_size:
                result.append(chunk)
                continue

            paragraphs = re.split(r'\n\s*\n', chunk.page_content)
            current_part = ""

            for para in paragraphs:
                if len(current_part) + len(para) + 2 > self.max_chunk_size and current_part:
                    result.append(Document(
                        page_content=current_part.strip(),
                        metadata={**chunk.metadata}
                    ))
                    current_part = para
                else:
                    if current_part:
                        current_part = current_part + '\n\n' + para
                    else:
                        current_part = para

            if current_part.strip():
                result.append(Document(
                    page_content=current_part.strip(),
                    metadata={**chunk.metadata}
                ))

        return result


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
