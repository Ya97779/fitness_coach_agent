# Markdown RAG 优化设计

## 背景

知识库中的 PDF 扫描件已转换为 markdown 格式。当前 RAG 系统的分块流程不适应结构化 markdown：
- `SemanticChunker` + `IntelligentSplitter` 双重分块，第二次按字符数硬切破坏了第一次的语义边界
- 不利用 markdown 的标题层级结构作为天然分块边界
- chunk 缺少标题路径元数据，检索到的片段脱离上下文

## 目标

1. 用 markdown 标题结构替代双重分块，一次切分到位
2. 每个 chunk 携带标题路径元数据，检索结果自带上下文
3. 精准率和完整性兼顾

## 设计

### 模块 1：MarkdownSplitter

新增 `backend/app/rag/modules/splitter.py` 中的 `MarkdownSplitter` 类，替代 `SemanticChunker` + `IntelligentSplitter` 对 `.md` 文件的处理。

**分块逻辑：**

1. 按 `#` ~ `######` 标题层级解析 markdown
2. 每个 chunk = 标题行 + 该标题下的正文内容
3. 标题路径作为元数据，如 `力量训练基础 > 深蹲 > 力学原理`
4. 过长 chunk（> 1000 字符）：按子标题进一步拆分
5. 过短 chunk（< 100 字符）：合并到同级相邻标题下
6. 无标题的开头内容（前言等）：作为独立 chunk，来源标记为文件名

### 模块 2：元数据增强

每个 chunk 携带元数据：

```python
{
    "source": "力量训练基础1.md",
    "heading_path": "力量训练 > 深蹲 > 力学原理",
    "heading_level": 3,
    "chunk_index": 5,
    "chunk_count": 42,
}
```

检索时将 `heading_path` 拼到 chunk 内容前面，如 `[力量训练 > 深蹲 > 力学原理] 实际内容...`，Agent 收到的结果自带上下文。

### 模块 3：索引构建流程改造

**改造前：**
```
load_directory() → AdvancedDocumentProcessor → IntelligentSplitter → TextPreprocessor → Chroma
```

**改造后：**
```
load_directory() → 按文件类型分流：
  .md 文件 → MarkdownSplitter（跳过 AdvancedDocumentProcessor 和 IntelligentSplitter）
  其他文件 → 保留原有流程
→ TextPreprocessor → Chroma
```

`_build_index()` 和 `_index_single_file()` 均需改造，按 `source` 扩展名分流。

### 模块 4：Loader 支持 .md

`loader.py` 的 `LOADER_MAP` 新增 `.md: TextLoader`。

## 改动文件清单

| 文件 | 改动 |
|------|------|
| `backend/app/rag/modules/loader.py` | LOADER_MAP 新增 `.md: TextLoader` |
| `backend/app/rag/modules/splitter.py` | 新增 `MarkdownSplitter` 类 |
| `backend/app/rag/__init__.py` | `_build_index()` 和 `_index_single_file()` 按文件类型分流 |
| `backend/app/agents/nutrition_agent.py` | 检索结果格式化时读取 `heading_path` |
| `backend/app/agents/fitness_agent.py` | 检索结果格式化时读取 `heading_path` |

## 验证

1. 重启服务，检查日志中 `.md` 文件是否走 `MarkdownSplitter` 分块
2. 检查 ChromaDB 中 chunk 的 `heading_path` 元数据是否正确
3. 通过聊天提问验证检索效果（具体知识点、跨主题综合问题、计划推荐）
