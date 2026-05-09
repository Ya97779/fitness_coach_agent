# Claude 改动记录

**日期**：2026-04-28

**改动范围**：项目说明书 337-351 行「改进建议 > 架构层面」的三个问题

---

## 改动 1：路由 LLM 解析优化

**文件**：`backend/app/agents/router.py`

**问题**：LLM 解析用 `if "2" in text` 子串匹配，返回"这是一个2号类型的请求"也会命中，容易误判。

**改动**：
- 优化 prompt：`"只返回一个数字。"` → `"只返回一个数字（1、2 或 3），不要包含任何其他文字。"`
- 解析逻辑：`"2" in text` → `re.search(r'\b([123])\b', text)` 精确匹配独立数字
- 添加语义兜底：正则失败时检查"营养/饮食"或"健身/运动"关键词

---

## 改动 2：专家评审快速通道

**文件**：`backend/app/agents/graph.py`

**问题**：所有 nutrition/fitness 回复都经过专家评审，简单事实查询（如"苹果热量"）也多 1-2 次 LLM 调用。

**改动**：
- `AgentState` 添加 `skip_review: bool` 字段
- 添加 `QUICK_PATTERNS` 列表（7 个正则模式）：热量查询、营养成分查询等
- 添加 `should_skip_review()` 函数：匹配模式 OR 回复长度 < 150 字符时返回 True
- `nutrition()` 和 `fitness()` 节点返回值添加 `skip_review` 标记
- `should_continue_nutrition()` 和 `should_continue_fitness()` 检查 `skip_review`，为 True 直接跳过评审

---

## 改动 3：流式响应修复

### 3A：前端 SSE 解析修复

**文件**：`frontend/app.py`

**问题**：后端发送 `data: {chunk}\n\n` SSE 格式，前端用 `iter_content()` 直接拼接，会把 `data: ` 前缀也显示出来。

**改动**：
- `iter_content()` → `iter_lines(decode_unicode=True)` 按行读取
- 添加 `data: ` 前缀剥离逻辑
- 添加 `[DONE]` 结束标记处理

### 3B：流式模式保存对话

**文件**：`backend/app/agents/graph.py` 的 `stream_user_message` 函数

**问题**：流式模式只 yield chunk，不保存对话历史。

**改动**：
- 在 yield 循环中用 `full_response` 收集完整回复
- 流式结束后调用 `memory_manager.save_conversation()`

### 3C：后端 SSE 结束标记

**文件**：`backend/app/main.py` 的 `event_generator` 函数

**改动**：yield 循环后追加 `yield "data: [DONE]\n\n"`

---

## 测试结果

```
Agent 测试：  61 passed
Memory 测试： 32 passed
RAG 测试：    57 passed
```

全部通过，无回归。

---

**日期**：2026-04-29

**改动范围**：项目说明书 395-409 行「改进建议 > 性能优化」的三个问题

---

## 改动 4：LLM 实例复用

**新增文件**：`backend/app/llm_manager.py`

**问题**：每次请求都创建新的 `ChatOpenAI` 实例，单次对话可能创建 16+ 个实例，浪费资源。

**改动**：
- 新建 `LLMManager` 类，按 temperature 值缓存 `ChatOpenAI` 实例（单例模式）
- `get_llm(temperature)` 方法：首次调用时创建实例并缓存，后续直接返回
- 修改 8 个文件统一使用 `LLMManager.get_llm()`：
  - `agents/graph.py` → `create_llm()`
  - `agents/router.py` → `_llm_route()`
  - `agents/chat_agent.py` → `chat_with_user()`
  - `agents/nutrition_agent.py` → `nutrition_with_user()`
  - `agents/fitness_agent.py` → `fitness_with_user()`
  - `agents/expert_agent.py` → `review_output()`
  - `memory/conversation_summary.py` → `_generate_summary()`
  - `rag/__init__.py` → `ModernRAG.__init__()`

---

## 改动 5：数据库查询缓存

**文件**：`backend/app/memory/memory_manager.py`

**问题**：`MemoryManager` 多个方法重复查询同一天/周的统计数据（`get_today_stats()`、`get_week_stats()` 被多次调用）。

**改动**：
- `__init__` 添加 `_today_stats` 和 `_week_stats` 缓存字段
- 新增 `get_today_stats()` 和 `get_week_stats()` 方法，首次查询后缓存结果
- 修改 5 个受益方法使用缓存版本：
  - `get_memory_summary()`
  - `format_today_stats_for_agent()`
  - `format_week_stats_for_agent()`
  - `get_nutrition_context()`
  - `get_fitness_context()`

---

## 改动 6：RAG 检索缓存

**文件**：`backend/app/rag/__init__.py`, `backend/app/agents/fitness_agent.py`

**问题**：相同查询重复执行向量检索和 BM25 检索，浪费计算资源。

**改动**：
- `ModernRAG.__init__` 添加缓存字段：`_query_cache`（字典）、`_cache_max_size`（128）、`_cache_ttl`（300秒）
- 新增 `_get_cache_key()`、`_get_from_cache()`、`_put_to_cache()` 方法
- `search()` 方法加入缓存逻辑：查询前检查缓存，命中直接返回；未命中则执行检索并缓存结果
- LRU 淘汰：缓存满时删除最旧条目
- `fitness_agent.py` 的 `get_rag()` 改用全局 `get_rag_instance()` 单例，消除重复实例

---

## 测试结果

```
Agent 测试：  61 passed
Memory 测试： 32 passed
RAG 测试：    57 passed
```

全部通过，无回归。测试 mock 目标从 `ChatOpenAI` 更新为 `LLMManager.get_llm`。

---

**日期**：2026-05-09

**改动范围**：RAG 系统 P0 级别问题修复（基于 RAG 系统深度分析）

---

## 改动 7：BM25 中文分词优化

**文件**：`backend/app/rag/modules/bm25.py`、`requirements.txt`

**问题**：BM25 使用 `re.findall(r'[\w一-鿿]+', text)` 做分词，对中文是按单字切分（"蛋白质摄入" → "蛋/白/质/摄/入"），导致 BM25 的 IDF 和词频计算对中文基本失效。Hybrid Search 的 BM25 这条腿处于"瘸腿"状态。

**改动**：
- `bm25.py` 添加 `import jieba`
- `_tokenize()` 方法从正则切分改为 `jieba.lcut(text)` 分词
- 返回结果过滤空字符串：`[t for t in tokens if t.strip()]`
- `requirements.txt` 添加 `jieba` 依赖

**预期效果**：BM25 检索召回率提升 20-30%，中文关键词匹配从单字级别提升到词语级别。

---

## 改动 8：SelfRAG 模型名修正

**文件**：`backend/app/rag/modules/self_rag.py`

**问题**：`SelfRAGScorer`（第 112 行）和 `SelfRAG`（第 297 行）的 LLM fallback 模型名为 `glm-4`，但项目实际使用的生成模型是 `glm-4.7`（`.env` 中 `LLM_MODEL=glm-4.7`）。当环境变量未设置时，Self-RAG 的评分和纠正会使用旧模型，与生成模型不一致，评估质量不可控。

**改动**：
- 第 112 行：`os.getenv("LLM_MODEL", "glm-4")` → `os.getenv("LLM_MODEL", "glm-4.7")`
- 第 297 行：`os.getenv("LLM_MODEL", "glm-4")` → `os.getenv("LLM_MODEL", "glm-4.7")`

**预期效果**：Self-RAG 的 relevance scoring 和 answer correction 使用与生成一致的模型，评估结果更可靠。

---

## 测试结果

```
RAG 测试：    57 passed
```

全部通过，无回归。

---

**日期**：2026-05-09

**改动范围**：RAG 系统 P2 级别优化（性能 + 评估体系）

---

## 改动 9：HybridSearch._find_doc_index O(1) 哈希映射

**文件**：`backend/app/rag/modules/hybrid_search.py`

**问题**：`_find_doc_index()` 对每次向量检索结果都遍历全部文档做字符串匹配（O(n)），当文档量增大时检索性能线性下降。同时 `search()` 方法中有一行死代码（第 119-124 行），调用了 `self.vectorstore.get()` 拉取全部文档但结果被后续代码覆盖。

**改动**：
- `__init__` 添加 `_content_to_index: dict` 字段
- `index()` 方法中预建 `{page_content: doc_index}` 字典映射
- `_find_doc_index()` 从线性扫描改为 `dict.get()` O(1) 查找
- 删除 `search()` 中的死代码（无用的 `vectorstore.get()` 调用和重复的 `similarity_search_with_score` 调用）

**预期效果**：向量检索结果匹配从 O(n) 降到 O(1)，消除冗余的全量文档查询。

---

## 改动 10：RAG 评估测试集 + RAGAS 自动化评估脚本

**新增文件**：
- `backend/tests/eval_dataset.json` — 15 条健身领域评估用例
- `backend/tests/test_rag_evaluation.py` — RAGAS 自动化评估脚本
- `requirements.txt` 添加 `ragas`、`datasets` 依赖

**评估数据集设计**：
- 15 条覆盖健身/营养核心场景的中文问题
- 每条包含 `question`（问题）、`ground_truth`（标准答案）、`expected_topics`（预期关键词）
- 覆盖类型：事实性问题、操作指导、概念解释、数据查询

**评估脚本功能**：
- `RAGEvaluator` 类封装完整评估流程
- 自动运行 RAG 检索 + 生成，收集 answer 和 contexts
- 使用 RAGAS 框架计算 4 项核心指标：
  - Faithfulness（忠实度）：回答是否忠于检索内容
  - Answer Relevancy（回答相关性）：回答与问题的匹配度
  - Context Precision（上下文精确度）：检索排序质量
  - Context Recall（上下文召回率）：正确信息被检索到的比例
- 控制台可视化报告（带进度条和分数柱状图）
- 支持 JSON 报告导出

**使用方式**：
```bash
cd backend
python -m tests.test_rag_evaluation                     # 完整评估（15 条）
python -m tests.test_rag_evaluation --quick             # 快速模式（5 条）
python -m tests.test_rag_evaluation --output report.json  # 导出报告
python -m tests.test_rag_evaluation --metrics faithfulness answer_relevancy  # 指定指标
```

---

## 测试结果

```
RAG 测试：    57 passed
```

全部通过，无回归。评估脚本语法检查通过。
