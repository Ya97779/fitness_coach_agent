# Markdown RAG 优化 — 实现计划

基于设计文档 `2026-05-31-markdown-rag-optimization-design.md`，按以下步骤实施。

## 步骤 1：`loader.py` — 添加 `.md` 支持

**文件**：`backend/app/rag/modules/loader.py`

**改动**：`LOADER_MAP` 新增 `.md: TextLoader` 条目。

## 步骤 2：`splitter.py` — 新增 `MarkdownSplitter` 类

**文件**：`backend/app/rag/modules/splitter.py`

**改动**：在 `IntelligentSplitter` 类之前新增 `MarkdownSplitter` 类（约 120 行）。

**类设计**：

```python
class MarkdownSplitter:
    def __init__(self, min_chunk_size=100, max_chunk_size=1000)
    def split_documents(self, documents: List[Document]) -> List[Document]
    def _split_markdown(self, text: str, source: str) -> List[Document]
    def _merge_short_chunks(self, chunks: List[Document]) -> List[Document]
    def _split_long_chunks(self, chunks: List[Document]) -> List[Document]
```

**分块逻辑**：
1. 正则匹配 `#{1,6}` 标题行，按标题切分段落
2. 维护标题栈构建 `heading_path`（如 `力量训练 > 深蹲 > 力学原理`）
3. 无标题文件以文件名为默认标题
4. 合并过短 chunk（< min_chunk_size），拆分过长 chunk（> max_chunk_size）
5. 统一编号 `chunk_index` 和 `chunk_count`

**元数据**：
- `source`：文件路径
- `heading_path`：标题层级路径
- `heading_level`：标题层级数
- `chunk_index`：chunk 序号
- `chunk_count`：总 chunk 数

## 步骤 3：`modules/__init__.py` — 导出 `MarkdownSplitter`

**文件**：`backend/app/rag/modules/__init__.py`

**改动**：
- `from .splitter import IntelligentSplitter` 改为 `from .splitter import IntelligentSplitter, MarkdownSplitter`
- `__all__` 列表新增 `"MarkdownSplitter"`

## 步骤 4：`rag/__init__.py` — 索引流程分流

**文件**：`backend/app/rag/__init__.py`

**改动**：

### 4a. 导入
- 从 `.modules` 导入中新增 `MarkdownSplitter`

### 4b. 构造函数
- `self.splitter` 初始化之后新增 `self.markdown_splitter = MarkdownSplitter(min_chunk_size=100, max_chunk_size=1000)`

### 4c. `_build_index()` 方法
改造为按文件类型分流：

```python
def _build_index(self):
    docs = self.loader.load_directory()
    if not docs:
        # 默认知识 fallback
        ...

    print(f"加载文档: {len(docs)} 个")

    # 按文件类型分流
    md_docs = [d for d in docs if d.metadata.get("source", "").endswith(".md")]
    other_docs = [d for d in docs if not d.metadata.get("source", "").endswith(".md")]

    all_chunks = []

    # .md 文件：MarkdownSplitter（跳过 AdvancedDocumentProcessor 和 IntelligentSplitter）
    if md_docs:
        md_chunks = self.markdown_splitter.split_documents(md_docs)
        all_chunks.extend(md_chunks)
        print(f"Markdown 分块: {len(md_chunks)} 个块")

    # 其他文件：原有流程
    if other_docs:
        processor = AdvancedDocumentProcessor(...)
        processed = processor.process_documents(other_docs)
        chunks = self.splitter.split_documents(processed)
        all_chunks.extend(chunks)
        print(f"其他文件分块: {len(chunks)} 个块")

    docs = self.preprocessor.preprocess_documents(all_chunks)
    print(f"预处理后: {len(docs)} 个块")

    self.documents = docs
    self.vectorstore = Chroma.from_documents(...)
```

### 4d. `_index_single_file()` 方法
同理分流：`.md` 文件走 `self.markdown_splitter`，其他走原流程。

## 步骤 5：Agent 工具 — 检索结果格式化增加标题路径

**文件**：
- `backend/app/agents/nutrition_agent.py`
- `backend/app/agents/fitness_agent.py`

**改动**：两个 `search_*_knowledge` 工具函数中，格式化结果时读取 `heading_path` 元数据，拼到内容前面。

改动前：`content_parts.append(f"[来源{i+1}] {c}")`

改动后：
```python
heading = r.get("heading_path", "")
prefix = f"[{heading}] " if heading else ""
content_parts.append(f"[来源{i+1}] {prefix}{c}")
```

## 步骤 6：提交 + 推送

- `git add` 所有改动文件
- `git commit`
- `git push origin main:deploy --force-with-lease`
