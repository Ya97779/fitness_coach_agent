# Agent 回复首字延迟优化分析

## 项目架构概览

FitCoach AI 是一个多 Agent 架构的健身营养顾问系统，后端使用 FastAPI + LangGraph 编排多个 Agent，前端为微信小程序，通过 SSE（Server-Sent Events）实现流式对话。

### Agent 类型

| Agent | 职责 | 工具调用 | 评审 |
|-------|------|----------|------|
| chat | 闲聊助手 | 无 | 否 |
| nutrition | 营养师 | search_food_nutrition, search_nutrition_knowledge, get_user_nutrition_info, log_food_intake, get_daily_nutrition_summary | 是 |
| fitness | 健身教练 | search_fitness_knowledge, get_user_fitness_info, log_exercise, estimate_exercise_calories | 是 |
| expert | 专家评审 | 评分 1-5，< 3 重试 | - |

### 工作流图

```
START → router → [chat | nutrition → expert_review | fitness → expert_review] → END
```

## 当前流式请求链路

从用户发送消息到产生第一个 chunk，完整链路如下：

```
用户发送消息
  │
  ▼
FastAPI SSE endpoint (/chat/stream) ── 鉴权、构建 user context
  │
  ▼
stream_user_message()
  ├─ 1. hybrid_route()           ← 关键词匹配，无 LLM 调用（require_llm_confirm=False）
  ├─ 2. MemoryManager()          ← 实例化
  ├─ 3. load_all_memory()        ← 1 次 DB 连接，查 User + 本周 DailyLog
  ├─ 4. load_conversation_history() ← 又一次 DB 连接，查 ConversationLog
  ├─ 5. enhance_system_prompt()  ← 构建增强后的 system prompt（仅目标 Agent）
  ├─ 6. yield ("status", "正在查询营养数据...")  ← 首字反馈
  │
  ▼
  nutrition_stream() / fitness_stream() / chat_stream()
    ├─ 闲聊：llm.stream() 直接逐字输出
    └─ 营养/健身：llm.bind_tools().invoke() → 执行工具 → llm.stream() 输出
```

## 已完成的优化

| 优化项 | 原理 | 效果 |
|--------|------|------|
| DB 连接合并 | load_all_memory() 用单次 DB 连接一次性查完 User 表和本周 DailyLog，替代了 8-10 次独立连接 | 省掉 6-8 次 TCP 握手 + 认证 |
| 路由前置 | 先通过关键词确定 Agent 类型，再只为该 Agent 构建 prompt（省掉另外 2 份） | 省掉 2/3 prompt 构建开销 |
| 状态消息 | 工具调用型 Agent 在进入 LLM 决策前 yield 一条即时状态消息（如"正在查询营养数据..."） | 感知等待从 2-5s 无响应 → 即刻有反馈 |
| SSE 心跳保活 | thread + queue 架构，30 秒无数据发心跳注释 | 避免 LLM 长决策期间超时断连 |
| System Prompt 精简 | 去掉未填写数据的不必要默认值 | 减少 prompt token 数 |

---

## 进一步优化方向

### 优先级 1：收益高、改动小

#### ① 合并 ConversationHistory 查询到 load_all_memory 中

**现状**：`load_all_memory()` 和 `load_conversation_history()` 各开一次 DB 连接。

```python
# 改前：2 次 DB 连接
memory_manager.load_all_memory()                            # DB 连接 1
conversation_history = memory_manager.load_conversation_history(days=7, limit=20)  # DB 连接 2

# 改后：1 次 DB 连接
memory_manager.load_all_memory(include_history=True)  # 一次查完 User + DailyLog + ConversationLog
```

**改造位置**：`stream_user_message()` 和 MemoryManager

**预计节省**：30-80ms

#### ② 裁剪 System Prompt 长度

**现状**：三个 Agent 的 system prompt 非常长（每个 100+ 行，约 1200+ tokens）。System prompt 越长，LLM prefill 阶段 token 数越多，首字延迟越大。

**具体做法**：
- 把静态专业知识说明（如"专业知识要求"、"评估维度"）移到 RAG 知识库，按需检索注入
- 精简"回复格式"、"安全边界"等规则为核心 2-3 条
- 去掉从未被触发的冗余规则

```python
# 改前（~1200 tokens）
system_content = AGENT_SYSTEM_PROMPTS["nutrition"]  # 完整长篇

# 改后（~400 tokens）
system_content = AGENT_SYSTEM_PROMPTS_MINIMAL["nutrition"]  # 核心规则
# 专业知识从 RAG 检索注入（只在需要时）
```

**改造位置**：`base.py` 中 AGENT_SYSTEM_PROMPTS

**预计节省**：200-500ms

#### ③ Agent 内部 format_memory 逻辑统一

**现状**：`chat_agent.py` 的 `format_memory_context`、`nutrition_agent.py` 的 `format_nutrition_memory`、`fitness_agent.py` 的 `format_fitness_memory` 各有一套对话历史格式化逻辑，且当 `enhanced_prompt` 已传入时走短路——但非流式路径 `process_user_message()` 仍然不传 `enhanced_prompt`，导致重复格式化。

**建议**：非流式路径也先路由再构建 prompt，或者统一收敛到 MemoryManager.enhance_system_prompt()。

**预计节省**：10-30ms

---

### 优先级 2：收益中、改动中

#### ④ LLM 请求连接预热

**现状**：每次 `llm.invoke()` 或 `llm.stream()` 时的首个请求需要走 TCP/TLS 握手。可在 LLMManager 的 `get_llm()` 时预建连接池。

#### ⑤ RAG 启动预热 & 统一 RAG 实例

**现状**：
- `nutrition_agent.py` 和 `fitness_agent.py` 各自维护了一个 `_rag_instance` 懒加载变量，可能创建两份独立实例
- 首次工具调用才触发 RAG 懒加载，导致 Chroma 向量库冷启动

**建议**：
1. 统一使用 `from ..rag import get_rag_instance`（fitness_agent 已正确使用）
2. 在 `startup_event` 中预热 RAG

```python
@app.on_event("startup")
async def startup_event():
    from .rag import get_rag_instance
    get_rag_instance(enable_agentic=True)  # 预热
```

**预计节省**：100-500ms（仅首次请求）

#### ⑥ System Prompt 会话级缓存

**原理**：同一会话内用户 profile/stats 变化很慢，可以在 MemoryManager 层面给 `enhance_system_prompt()` 加 TTL 缓存（如 30 秒）。

```python
self._prompt_cache: Dict[str, tuple[str, float]] = {}  # agent_type → (prompt, timestamp)

def enhance_system_prompt(self, base_prompt, agent_type, messages=None):
    cache_key = agent_type
    if cache_key in self._prompt_cache:
        cached, ts = self._prompt_cache[cache_key]
        if time.time() - ts < 30:
            return cached
    # ... 构建 ...
    self._prompt_cache[cache_key] = (result, time.time())
    return result
```

**预计节省**：50-150ms（多轮对话场景）

---

### 优先级 3：收益取决于场景、改动较大

#### ⑦ 工具决策 LLM 与生成 LLM 分离优化

**现状**：工具决策调用和最终生成使用同一个 temperature=0.7 的 LLM 实例。

**优化**：
- 工具决策用更低 temperature（0.1）和更低 max_tokens（512），减少输出 token 数
- 最终生成保持正常 temperature（0.7）

需改造 `LLMManager.get_llm()` 将 `max_tokens` 纳入缓存 key。

**预计节省**：100-300ms

#### ⑧ 小程序端感知优化

**现状**：发送消息后推送 `aiMsg` 带 `loading: true`，后台状态消息（"正在查询营养数据..."）也会推送。

**建议**：
- loading 状态给 UI 加三点闪烁打字机动画
- 已做的状态消息（"正在查询营养数据..." / "正在搜索训练方案..."）在小程序端作为独立行显示

**预计节省**：0（纯感知优化，但对用户体验影响显著）

---

## 优化效果总览

| 优化项 | 预计节省 | 改动量 | 风险 |
|--------|----------|--------|------|
| ① 合并 DB 查询 | 30-80ms | 小 | 低 |
| ② System Prompt 裁剪 | 200-500ms | 中 | 中（需保留核心规则） |
| ③ 统一格式化逻辑 | 10-30ms | 小 | 低 |
| ④ LLM 连接预热 | 50-200ms（仅首次） | 小 | 低 |
| ⑤ RAG 启动预热 + 统一实例 | 100-500ms（仅首次） | 小 | 低 |
| ⑥ Prompt 会话缓存 | 50-150ms（后续请求） | 中 | 低 |
| ⑦ 工具决策分离 | 100-300ms | 中 | 中 |
| ⑧ 前端动画 | 0（纯感知优化） | 小 | 低 |

## 建议实施顺序

1. **① + ③** — 改动最小，立即见效
2. **⑥** — 多轮对话场景收益大
3. **②** — 收益最大但需仔细裁剪 prompt，建议先 A/B 测试
4. **⑤** — 解决首次查询卡顿
5. **④ + ⑦** — 根据实际压测数据决定是否需要

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `backend/app/agents/graph.py` | LangGraph 工作流编排，流式/非流式入口 |
| `backend/app/agents/base.py` | Agent system prompt 定义 |
| `backend/app/agents/nutrition_agent.py` | 营养师 Agent，5 个工具 |
| `backend/app/agents/fitness_agent.py` | 健身教练 Agent，4 个工具 |
| `backend/app/agents/chat_agent.py` | 闲聊 Agent，无工具 |
| `backend/app/agents/router.py` | 混合路由：关键词 + LLM |
| `backend/app/agents/expert_agent.py` | 专家评审，1-5 分评分 |
| `backend/app/main.py` | FastAPI 入口，SSE 端点 |
| `backend/app/llm_manager.py` | LLM 单例管理，按 temperature 缓存 |
| `backend/app/memory/memory_manager.py` | 记忆管理器（画像/统计/对话历史） |
| `backend/app/memory/conversation_summary.py` | 对话历史摘要 |
| `miniprogram/pages/chat/chat.js` | 小程序聊天页，SSE 解析 |
| `miniprogram/utils/request.js` | wx.request 封装 + SSE 流式 |