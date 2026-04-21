"""高级文档处理器 - 语义分割、表格提取、结构分析

功能：
- 语义分割：基于嵌入相似度识别主题边界
- 表格检测：识别并单独提取表格内容
- 代码块保留：检测并保护代码块不被错误分割
- 结构分析：识别标题层级，构建文档树
- 上下文感知清洗：保留关键格式信息
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class TableData:
    """表格数据结构"""
    content: str
    rows: int
    cols: int
    headers: List[str]
    data: List[List[str]]
    position: Tuple[int, int]


@dataclass
class CodeBlock:
    """代码块数据结构"""
    content: str
    language: str
    start_line: int
    end_line: int


@dataclass
class Section:
    """文档章节"""
    heading: str
    level: int
    content: str
    start_pos: int
    end_pos: int
    children: List['Section'] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)


class TableDetector:
    """表格检测器

    支持格式：
    - Markdown 表格
    - CSV 格式
    - 空格分隔的表格
    - HTML 表格
    """

    MARKDOWN_TABLE_PATTERN = re.compile(
        r'^\|.*\|\s*$|^\|[-:\s|]+\|\s*$',
        re.MULTILINE
    )

    CSV_TABLE_PATTERN = re.compile(
        r'^[^,\n]+,(?:[^,\n]+,)*[^,\n]+\n'
    )

    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """初始化表格检测器

        Args:
            min_rows: 最小行数
            min_cols: 最小列数
        """
        self.min_rows = min_rows
        self.min_cols = min_cols

    def detect_markdown_tables(self, text: str) -> List[TableData]:
        """检测 Markdown 表格

        Args:
            text: 文本内容

        Returns:
            表格列表
        """
        tables = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if self._is_markdown_table_row(line):
                table_lines = [line]
                header_line = i

                i += 1
                if i < len(lines) and self._is_markdown_separator(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                while i < len(lines) and self._is_markdown_table_row(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                table_text = '\n'.join(table_lines)
                parsed = self._parse_markdown_table(table_lines)

                if parsed:
                    headers, data = parsed
                    if len(headers) >= self.min_cols and len(data) >= self.min_rows:
                        tables.append(TableData(
                            content=table_text,
                            rows=len(data) + 1,
                            cols=len(headers),
                            headers=headers,
                            data=data,
                            position=(header_line, i - 1)
                        ))

                continue

            i += 1

        return tables

    def _is_markdown_table_row(self, line: str) -> bool:
        """检查是否为 Markdown 表格行"""
        line = line.strip()
        return bool(line) and line.startswith('|') and line.endswith('|')

    def _is_markdown_separator(self, line: str) -> bool:
        """检查是否为 Markdown 表格分隔行"""
        line = line.strip()
        return bool(line) and all(
            c in '|:- ' for c in line
        ) and ('-' in line or ':' in line)

    def _parse_markdown_table(self, lines: List[str]) -> Optional[Tuple[List[str], List[List[str]]]]:
        """解析 Markdown 表格"""
        if len(lines) < 2:
            return None

        header_line = lines[0]
        separator_line = lines[1]

        headers = [h.strip() for h in header_line.strip('|').split('|')]

        if not headers or not any(headers):
            return None

        data = []
        for line in lines[2:]:
            if self._is_markdown_table_row(line):
                row = [cell.strip() for cell in line.strip('|').split('|')]
                if len(row) == len(headers):
                    data.append(row)

        return headers, data

    def detect_all_tables(self, text: str) -> List[TableData]:
        """检测所有类型的表格

        Args:
            text: 文本内容

        Returns:
            表格列表
        """
        tables = []
        tables.extend(self.detect_markdown_tables(text))
        return tables

    def table_to_markdown(self, table: TableData) -> str:
        """将表格转换为 Markdown 格式

        Args:
            table: 表格数据

        Returns:
            Markdown 格式字符串
        """
        lines = []

        header_line = '| ' + ' | '.join(table.headers) + ' |'
        lines.append(header_line)

        separator_line = '| ' + ' | '.join(['---'] * len(table.headers)) + ' |'
        lines.append(separator_line)

        for row in table.data:
            row_line = '| ' + ' | '.join(row) + ' |'
            lines.append(row_line)

        return '\n'.join(lines)


class CodeBlockDetector:
    """代码块检测器

    支持格式：
    - Markdown 代码块 (```language)
    - 行内代码 (`code`)
    - 缩进代码块
    """

    FENCED_CODE_PATTERN = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )

    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')

    INDENTED_CODE_PATTERN = re.compile(
        r'^( {4,}|\t+)(\S.*)$',
        re.MULTILINE
    )

    def __init__(self, min_lines: int = 2):
        """初始化代码块检测器

        Args:
            min_lines: 最小行数（缩进代码块）
        """
        self.min_lines = min_lines

    def detect_fenced_code_blocks(self, text: str) -> List[CodeBlock]:
        """检测围栏代码块

        Args:
            text: 文本内容

        Returns:
            代码块列表
        """
        blocks = []
        lines = text.split('\n')

        in_block = False
        block_start = 0
        language = ""
        content_lines = []

        for i, line in enumerate(lines):
            fenced_match = re.match(r'^```(\w*)$', line.strip())

            if fenced_match and not in_block:
                in_block = True
                block_start = i
                language = fenced_match.group(1)
                content_lines = []
            elif fenced_match and in_block:
                content = '\n'.join(content_lines)
                blocks.append(CodeBlock(
                    content=content,
                    language=language,
                    start_line=block_start,
                    end_line=i
                ))
                in_block = False
                content_lines = []
            elif in_block:
                content_lines.append(line)

        return blocks

    def detect_indented_code_blocks(self, text: str) -> List[CodeBlock]:
        """检测缩进代码块

        Args:
            text: 文本内容

        Returns:
            代码块列表
        """
        blocks = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            indented_match = self.INDENTED_CODE_PATTERN.match(lines[i])

            if indented_match:
                code_lines = [indented_match.group(2)]
                start_line = i

                i += 1
                while i < len(lines):
                    next_match = self.INDENTED_CODE_PATTERN.match(lines[i])
                    if next_match:
                        code_lines.append(next_match.group(2))
                        i += 1
                    elif lines[i].strip() == '':
                        code_lines.append('')
                        i += 1
                    else:
                        break

                if len(code_lines) >= self.min_lines:
                    blocks.append(CodeBlock(
                        content='\n'.join(code_lines),
                        language='',
                        start_line=start_line,
                        end_line=i - 1
                    ))

                continue

            i += 1

        return blocks

    def remove_inline_code(self, text: str) -> Tuple[str, List[str]]:
        """移除行内代码，保留占位符

        Args:
            text: 文本内容

        Returns:
            (处理后的文本, 提取的代码列表)
        """
        inline_codes = []

        def replace_func(match):
            inline_codes.append(match.group(1))
            return f'__INLINE_CODE_{len(inline_codes) - 1}__'

        processed = self.INLINE_CODE_PATTERN.sub(replace_func, text)

        return processed, inline_codes

    def restore_inline_code(self, text: str, codes: List[str]) -> str:
        """恢复行内代码

        Args:
            text: 包含占位符的文本
            codes: 代码列表

        Returns:
            恢复后的文本
        """
        for i, code in enumerate(codes):
            text = text.replace(f'__INLINE_CODE_{i}__', f'`{code}`')

        return text

    def detect_all(self, text: str) -> Tuple[str, List[CodeBlock], List[str]]:
        """检测所有代码块

        Args:
            text: 文本内容

        Returns:
            (处理后文本, 代码块列表, 行内代码列表)
        """
        fenced_blocks = self.detect_fenced_code_blocks(text)
        indented_blocks = self.detect_indented_code_blocks(text)

        all_blocks = fenced_blocks + indented_blocks
        all_blocks.sort(key=lambda x: x.start_line)

        text_without_inline, inline_codes = self.remove_inline_code(text)

        return text_without_inline, all_blocks, inline_codes


class SemanticChunker:
    """语义分割器

    基于嵌入相似度识别主题边界，将文档分割成语义连贯的块。

    算法：
    1. 将文本分割成句子
    2. 计算相邻句子的嵌入相似度
    3. 相似度低于阈值处作为分割点
    4. 合并小片段形成语义块
    """

    def __init__(
        self,
        embeddings: Embeddings,
        threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        overlap: int = 50
    ):
        """初始化语义分割器

        Args:
            embeddings: 嵌入模型
            threshold: 相似度阈值（低于此值分割）
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
            overlap: 重叠大小
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """分割文本

        Args:
            text: 待分割文本

        Returns:
            文本块列表
        """
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings_list = []
        for sent in sentences:
            emb = self.embeddings.embed_query(sent)
            embeddings_list.append(emb)

        similarities = []
        for i in range(len(embeddings_list) - 1):
            sim = self._cosine_similarity(embeddings_list[i], embeddings_list[i + 1])
            similarities.append(sim)

        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                boundaries.append(i + 1)

        boundaries.append(len(sentences))

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = ''.join(chunk_sentences)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)

        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentence_endings = re.compile(r'[。！？.!?\n]+')
        parts = sentence_endings.split(text)

        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part + ('。' if not part.endswith(('。', '.', '!', '?')) else ''))

        return sentences

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块"""
        if not chunks:
            return []

        merged = [chunks[0]]

        for chunk in chunks[1:]:
            if len(merged[-1]) < self.min_chunk_size:
                merged[-1] += chunk
            else:
                merged.append(chunk)

        return merged

    def split_documents(
        self,
        documents: List[Document],
        embeddings_model: Optional[Embeddings] = None
    ) -> List[Document]:
        """分割文档列表

        Args:
            documents: Document 列表
            embeddings_model: 嵌入模型（覆盖初始化时的模型）

        Returns:
            分割后的 Document 列表
        """
        emb_model = embeddings_model or self.embeddings

        all_chunks = []
        for doc in documents:
            text = doc.page_content

            chunks = self.split_text(text)

            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                all_chunks.append(chunk_doc)

        return all_chunks


class DocumentStructureAnalyzer:
    """文档结构分析器

    识别标题层级，构建文档树
    """

    HEADING_PATTERNS = [
        (re.compile(r'^#{1}\s+(.+)$'), 1),
        (re.compile(r'^#{2}\s+(.+)$'), 2),
        (re.compile(r'^#{3}\s+(.+)$'), 3),
        (re.compile(r'^#{4}\s+(.+)$'), 4),
        (re.compile(r'^#{5,6}\s+(.+)$'), 5),
        (re.compile(r'^【(.+?)】$'), 1),
        (re.compile(r'^\[(.+?)\]$'), 1),
    ]

    def __init__(self):
        pass

    def analyze(self, text: str) -> List[Section]:
        """分析文档结构

        Args:
            text: 文档文本

        Returns:
            章节列表
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        current_pos = 0

        for i, line in enumerate(lines):
            heading_info = self._match_heading(line)

            if heading_info:
                if current_section is not None:
                    current_section.content = '\n'.join(current_content)
                    current_section.end_pos = current_pos
                    sections.append(current_section)

                heading_text, level = heading_info
                current_section = Section(
                    heading=heading_text,
                    level=level,
                    content='',
                    start_pos=i,
                    end_pos=i
                )
                current_content = []
                current_pos = i
            else:
                current_content.append(line)

        if current_section is not None:
            current_section.content = '\n'.join(current_content)
            current_section.end_pos = len(lines) - 1
            sections.append(current_section)
        elif current_content:
            section = Section(
                heading='',
                level=0,
                content='\n'.join(current_content),
                start_pos=0,
                end_pos=len(lines) - 1
            )
            sections.append(section)

        return sections

    def _match_heading(self, line: str) -> Optional[Tuple[str, int]]:
        """匹配标题

        Returns:
            (标题文本, 级别) 或 None
        """
        line = line.strip()

        for pattern, level in self.HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                return match.group(1), level

        return None

    def extract_heading_tree(self, sections: List[Section]) -> Dict[str, Any]:
        """提取标题树结构

        Args:
            sections: 章节列表

        Returns:
            树形结构字典
        """
        root = {'children': []}
        stack = [root]

        for section in sections:
            while len(stack) > section.level + 1:
                stack.pop()

            node = {
                'heading': section.heading,
                'level': section.level,
                'content_preview': section.content[:100] if section.content else '',
                'children': []
            }

            stack[-1]['children'].append(node)
            stack.append(node)

        return root


class ContextAwareCleaner:
    """上下文感知清洗器

    特点：
    - 保留列表结构（有序/无序）
    - 保留强调标记（粗体、斜体）
    - 保留引用块
    - 智能去除冗余空白
    - 保留关键标点
    """

    def __init__(
        self,
        preserve_formatting: bool = True,
        remove_urls: bool = False,
        normalize_whitespace: bool = True
    ):
        """初始化清洗器

        Args:
            preserve_formatting: 保留格式标记
            remove_urls: 移除 URL
            normalize_whitespace: 规范化空白
        """
        self.preserve_formatting = preserve_formatting
        self.remove_urls = remove_urls
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> str:
        """清洗文本

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
                '[链接已移除]',
                cleaned
            )

        if self.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)

        cleaned = self._fix_common_issues(cleaned)

        return cleaned.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白"""
        text = re.sub(r'\r\n', '\n', text)

        text = re.sub(r'[ \t]+\n', '\n', text)

        text = re.sub(r'\n{3,}', '\n\n', text)

        text = re.sub(r' +', ' ', text)

        return text

    def _fix_common_issues(self, text: str) -> str:
        """修复常见问题"""
        text = re.sub(r'\n +', '\n', text)

        text = re.sub(r' +\n', '\n', text)

        text = re.sub(r'。+', '。', text)
        text = re.sub(r'，+', '，', text)
        text = re.sub(r'、+', '、', text)

        text = re.sub(r'\[ +', '[', text)
        text = re.sub(r' +\]', ']', text)

        return text

    def preserve_list_structures(self, text: str) -> str:
        """保留列表结构

        Args:
            text: 文本

        Returns:
            处理后的文本
        """
        lines = text.split('\n')
        result = []

        in_list = False
        list_indent = 0

        for line in lines:
            list_match = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', line)

            if list_match:
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                content = list_match.group(3)

                if not in_list or indent != list_indent:
                    result.append('')

                in_list = True
                list_indent = indent
                result.append(line)
            else:
                in_list = False
                result.append(line)

        return '\n'.join(result)


class AdvancedDocumentProcessor:
    """高级文档处理器

    整合所有文档处理组件，提供端到端的处理流程。
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        use_semantic_chunking: bool = True,
        detect_tables: bool = True,
        detect_code: bool = True,
        analyze_structure: bool = True
    ):
        """初始化高级文档处理器

        Args:
            embeddings: 嵌入模型（用于语义分割）
            use_semantic_chunking: 使用语义分割
            detect_tables: 检测表格
            detect_code: 检测代码块
            analyze_structure: 分析文档结构
        """
        self.embeddings = embeddings
        self.use_semantic_chunking = use_semantic_chunking

        self.table_detector = TableDetector() if detect_tables else None
        self.code_detector = CodeBlockDetector() if detect_code else None
        self.structure_analyzer = DocumentStructureAnalyzer() if analyze_structure else None
        self.cleaner = ContextAwareCleaner()

        if use_semantic_chunking and embeddings:
            self.semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                threshold=0.6,
                min_chunk_size=200,
                max_chunk_size=800
            )
        else:
            self.semantic_chunker = None

    def process(self, text: str) -> Dict[str, Any]:
        """处理文档

        Args:
            text: 文档文本

        Returns:
            {
                "chunks": List[str],           # 分割后的文本块
                "tables": List[TableData],     # 检测到的表格
                "code_blocks": List[CodeBlock], # 代码块
                "structure": Dict,             # 文档结构
                "cleaned_text": str           # 清洗后的完整文本
            }
        """
        result = {
            "chunks": [],
            "tables": [],
            "code_blocks": [],
            "structure": {},
            "cleaned_text": ""
        }

        text, code_blocks, inline_codes = self._preprocess(text)
        result["code_blocks"] = code_blocks

        if self.table_detector:
            tables = self.table_detector.detect_all_tables(text)
            result["tables"] = tables
            text = self._remove_tables_from_text(text, tables)

        if self.structure_analyzer:
            sections = self.structure_analyzer.analyze(text)
            result["structure"] = self.structure_analyzer.extract_heading_tree(sections)

        if self.use_semantic_chunking and self.semantic_chunker:
            chunks = self.semantic_chunker.split_text(text)
            result["chunks"] = chunks
        else:
            result["chunks"] = [text]

        result["cleaned_text"] = text

        return result

    def _preprocess(
        self,
        text: str
    ) -> Tuple[str, List[CodeBlock], List[str]]:
        """预处理"""
        if self.code_detector:
            return self.code_detector.detect_all(text)

        return text, [], []

    def _remove_tables_from_text(
        self,
        text: str,
        tables: List[TableData]
    ) -> str:
        """从文本中移除表格"""
        for table in tables:
            text = text.replace(table.content, f'\n[表格: {len(table.headers)}列 x {len(table.data)}行]\n')

        return text

    def process_document(
        self,
        document: Document
    ) -> List[Document]:
        """处理单个文档

        Args:
            document: 原始 Document

        Returns:
            处理后的 Document 列表
        """
        result = self.process(document.page_content)

        chunks = []
        for i, chunk_text in enumerate(result["chunks"]):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(result["chunks"]),
                    "has_tables": len(result["tables"]) > 0,
                    "has_code": len(result["code_blocks"]) > 0
                }
            )
            chunks.append(chunk_doc)

        return chunks

    def process_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """处理文档列表

        Args:
            documents: Document 列表

        Returns:
            处理后的 Document 列表
        """
        all_chunks = []

        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)

        return all_chunks
