# 流式响应首字延迟优化

## 遇到了什么问题

FitCoach AI 是一个基于多 Agent 架构的健身营养顾问系统，后端使用 FastAPI + LangGraph 编排多个 Agent（闲聊、营养师、健身教练），前端是微信小程序，通过 SSE（Server-Sent Events）实现流式对话。

小程序端的流式输出已经跑通，但实测发现**首字延迟过高**——用户发送消息后要等好几秒才看到第一个字出现，而一旦首字出现，后续的流式输出就很顺畅。这意味着瓶颈集中在"收到请求 → 第一个字符 yield 出来"这段路径上。

## 如何排查的

没有直接猜原因，而是沿着请求的完整数据链路逐步追踪，从 API 入口到第一个 chunk yield 出来的每一步都列出来，标注哪些是同步阻塞操作。

`stream_user_message()` 在 yield 第一个 chunk 之前，依次做了这些事：

1. **MemoryManager 初始化** — 加载用户画像、目标、今日统计、本周统计
2. **加载对话历史** — 从数据库查最近 7 天的对话记录
3. **构建 System Prompt ×3** — 为 chat、nutrition、fitness 三个 Agent 各构建一份增强后的 system prompt
4. **路由决策** — 关键词匹配决定用哪个 Agent
5. **Agent 流式调用** — 对于 nutrition/fitness，还有一次非流式的工具决策 LLM 调用在前面

逐一排查后定位到三个根因。

## 具体问题与解决方案

### 问题一：数据库连接风暴

**现象**：MemoryManager 的每个方法（`load_profile`、`get_goal`、`get_today_stats`、`get_week_stats`、`load_conversation_history`）都各自开一个数据库连接、查完再关闭。一个请求下来要开关 8-10 次 PostgreSQL 连接。

**根因**：模块设计时各方法独立封装，没有考虑会被同一个请求串联调用。每次 `SessionLocal()` 都会创建一个新的 TCP 连接到 PostgreSQL，包含 TCP 三次握手 + 认证开销。

**解决**：在 MemoryManager 中新增 `load_all_memory()` 方法，用单个数据库 session 一次性查完 User 表和 DailyLog 表，将结果填充到内部缓存（`_profile`、`_goal`、`_today_stats`、`_week_stats`）。后续调用 `get_memory_summary()`、`enhance_system_prompt()` 时直接命中缓存，不再触发数据库查询。

```
改前：8-10 次独立 DB 连接（每次开/关）
改后：2 次 DB 连接（load_all_memory + load_conversation_history）
```

关键优化点：将 User 表查询从 4-5 次合并为 1 次，DailyLog 表查询从 3-4 次合并为 1 次，用一次查本周所有记录的结果同时服务 today_stats 和 week_stats。

### 问题二：为三个 Agent 都构建了 Prompt，但只用一个

**现象**：`stream_user_message()` 的执行顺序是先为 chat、nutrition、fitness 三个 Agent 各调用一次 `enhance_system_prompt()` 构建完整 prompt，然后才做路由决策。但路由只会选中其中一个 Agent，另外两份 prompt 白构建了。

**根因**：代码结构上，记忆加载和 prompt 构建是一块、路由是另一块，没有从"哪些数据会被实际使用"的角度去组织执行顺序。

**解决**：将路由决策前置到 prompt 构建之前。先通过关键词匹配确定 Agent 类型，再只为该 Agent 构建 prompt。

```python
# 改前：构建 3 个 prompt → 路由 → 用 1 个
enhanced_prompts = {
    "chat": enhance_system_prompt(...),
    "nutrition": enhance_system_prompt(...),
    "fitness": enhance_system_prompt(...),
}
agent = hybrid_route(...)  # 只用了其中一个

# 改后：路由 → 只构建 1 个 prompt
agent = hybrid_route(...)
enhanced_prompt = enhance_system_prompt(AGENT_SYSTEM_PROMPTS[agent], agent, ...)
enhanced_prompts = {agent: enhanced_prompt}
```

每次 `enhance_system_prompt()` 内部要做用户画像格式化、统计格式化、可能的对话摘要（涉及 LLM 调用），省掉 2 次等于省掉了 2/3 的 prompt 构建开销。

### 问题三：工具调用型 Agent 的双重 LLM 调用导致用户无反馈等待

**现象**：营养师和健身教练 Agent 的工作流是"LLM 决策工具 → 执行工具 → LLM 生成回复"。第一次 LLM 调用（工具决策）是非流式的 `invoke`，用户在这段时间（通常 2-5 秒）看不到任何输出。

**根因**：这是架构层面的设计——LLM 需要先决定调用什么工具（如搜索食物 API、检索 RAG 知识库），拿到工具结果后才能生成有依据的回复。第一次调用必须等完整响应才能解析 `tool_calls`，无法流式化。

**解决**：不能消除双重调用（否则 LLM 无法使用工具），但在进入 Agent 流式循环之前，先 yield 一条状态消息给用户：

- 营养师 Agent → `正在查询营养数据...`
- 健身教练 Agent → `正在搜索训练方案...`
- 闲聊 Agent → 不发（无需工具调用，LLM 直接流式输出）

状态消息通过 SSE 立即推送到小程序，用户发完消息后瞬间看到系统在工作，感知等待时间从"好几秒无反馈"变为"立刻有响应"。

## 优化效果

| 环节 | 优化前 | 优化后 |
|------|--------|--------|
| 数据库连接数 | 8-10 次 | 2 次 |
| User 表查询 | 4-5 次 | 1 次 |
| DailyLog 表查询 | 3-4 次 | 1 次 |
| System Prompt 构建 | 3 次 | 1 次 |
| 用户首字反馈 | 等待 2-5 秒无响应 | 立即显示状态消息 |

## 收获

1. **排查性能问题要先画数据链路图**：不是直接猜"可能是数据库慢"或"可能是 LLM 慢"，而是把从请求到响应的每一步都列出来，标注哪些是同步阻塞、哪些可以延迟执行、哪些可以并行化。这次三个问题都不是单点故障，而是多处低效叠加。

2. **模块独立封装不等于请求级高效**：MemoryManager 的每个方法各自管理数据库连接，单独看没问题（职责清晰、连接及时释放），但串联到一个请求里就变成了连接风暴。设计模块时要考虑它在请求链路中的调用模式，必要时提供"批量加载"入口。

3. **执行顺序决定用户体验**：路由和 prompt 构建的顺序调换，代码量几乎没变，但省掉了 2/3 的无效工作。类似的"先判断再做事"模式在很多场景都适用——先确定需要什么，再只做需要的部分。

4. **不能优化的瓶颈可以用感知设计补偿**：工具调用的双重 LLM 调用是架构决定的，消除不了。但一条即时的状态消息就能把用户从"焦虑等待"变成"知道系统在工作"。有时候最好的技术方案不是消除延迟，而是管理用户对延迟的感知。
