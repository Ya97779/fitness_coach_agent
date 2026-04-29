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
