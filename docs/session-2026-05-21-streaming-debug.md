# 流式响应调试与优化会话记录

日期：2026-05-21

## 问题背景

小程序与 AI 对话时出现多个问题：
1. 查询食物热量报错"抱歉，发生了错误，请稍后重试"
2. 请求训练计划报错
3. AI 回复不是流式输出，而是一坨一起出现
4. 首页显示默认 TDEE 2000

## 已完成的修复（已推送到 main + deploy）

### 1. generator 中 return vs yield 修复
- **文件**: `backend/app/agents/nutrition_agent.py`
- **问题**: `generate_response()` 内部的错误处理用 `return "错误信息"` 而非 `yield "错误信息"`。Python generator 中 `return` 带值不会传递给调用方，导致错误信息静默丢失，前端收到空响应。
- **修复**: 所有错误路径改为 `yield` + `return`

### 2. 非流式节点 generator 消费修复
- **文件**: `backend/app/agents/graph.py`
- **问题**: `chat()`、`nutrition()`、`fitness()` 节点调用 `*_with_user()` 时不传 `stream=True`，拿到的是 generator 对象而非字符串。`AIMessage(content=generator)` 导致后续 `len()` TypeError 和序列化失败。
- **修复**: 新增 `_collect()` 辅助函数，消费 generator 收集完整字符串

### 3. SSE 心跳保活
- **文件**: `backend/app/main.py`
- **问题**: 第一次 LLM 调用（工具决策）+ RAG 搜索期间零 SSE 数据，超过 60 秒小程序超时断连
- **修复**: 用 thread + queue 重写 `event_generator`，30 秒无数据发 `": heartbeat\n\n"`（SSE 注释，客户端忽略但连接不丢）

### 4. 小程序流式超时增加
- **文件**: `miniprogram/utils/request.js`
- **修复**: 流式请求超时从 60s 提到 120s

### 5. 本地调试 JWT 放行
- **文件**: `backend/app/auth.py`
- **修复**: `127.0.0.1` 请求无 token 时放行 user_id=1，方便 curl 调试。`HTTPBearer(auto_error=False)` + `get_current_user` 内判断 client IP。
- **提醒**: 上线前去掉这个逻辑

### 6. 默认值不注入 prompt
- **文件**: `backend/app/memory/user_profile.py`、`backend/app/memory/stats_summary.py`
- **修复**: 用户未填写身体数据时（height/weight/age 为 0 或 None），不注入默认值到 Agent prompt。TDEE 为 None 时跳过热量平衡计算。

### 7. 首页 TDEE 显示
- **文件**: `miniprogram/pages/home/home.js`、`home.wxml`
- **修复**: TDEE 为 null 时显示"--"和"未设目标"

### 8. 反馈功能
- **文件**: `miniprogram/pages/feedback/`（4 个新文件）、`miniprogram/pages/home/home.wxml`、`backend/app/main.py`
- **功能**: 首页今日记录下方添加"帮助改进 FitCoach"入口 → 反馈页面 → POST /api/v1/feedback → 保存为 `backend/static/feedback/{date}_{user_id}_{timestamp}.md`

### 9. 运动 GIF 图片修复
- **文件**: 7 个 `miniprogram/data/exercises/*.js`
- **问题**: 代码中用英文文件名（如 `incline-bench-press.gif`），服务器上是中文文件名（`上斜杠铃卧推.gif`）
- **修复**: 全部改为中文文件名

## 当前状态：流式输出调试中

### 已验证
- GLM SDK 直接调用流式正常（逐字输出）
- LangChain ChatOpenAI + GLM OpenAI 兼容接口流式正常（逐字输出）
- SSE 首字时间 0.005s，总时间 42s，说明数据在 42 秒内逐渐到达
- curl 终端看起来"一起显示"可能是终端行缓冲问题

### 待确认
- **小程序端是否逐字显示？** — 需要在小程序上实际测试"你好"，观察 AI 回复是否逐字出现
- **训练计划是否还报错？** — 测试"帮我制定一个三分化的训练计划"

### 可能的下一步（如果小程序不是逐字的）
1. **Nginx SSE 配置** — 确认服务器 Nginx 有 `proxy_buffering off;`，否则 `X-Accel-Buffering: no` header 会被忽略
2. **前端打字机动画** — 发送后立刻显示"AI 正在思考中..."三点闪烁动画，用户感知等待时间缩短 70%
3. **裁剪 System Prompt** — 精简不常用规则到 RAG，减少 prefill 阶段延迟

### 不建议做的优化
- 流式工具调用 Pre-fetch（过度工程，风险大于收益）
- 全面 Async 化（当前架构够用，单用户场景收益有限）
- HTTP/2（对 SSE 单连接流式无实际帮助）

## 分支状态

- `main` — 最新提交: `8be397d` (Revert LangChain bypass)
- `deploy` — 与 main 同步
- 服务器从 deploy 拉取

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `backend/app/agents/graph.py` | LangGraph 工作流编排，流式/非流式入口 |
| `backend/app/agents/nutrition_agent.py` | 营养师 Agent，5 个工具 |
| `backend/app/agents/fitness_agent.py` | 健身教练 Agent，4 个工具 |
| `backend/app/agents/chat_agent.py` | 闲聊 Agent，无工具 |
| `backend/app/agents/router.py` | 混合路由：关键词 + LLM |
| `backend/app/agents/expert_agent.py` | 专家评审，1-5 分评分 |
| `backend/app/main.py` | FastAPI 入口，SSE 端点 |
| `backend/app/auth.py` | JWT 鉴权（当前含 localhost 放行） |
| `backend/app/llm_manager.py` | LLM 单例管理 |
| `backend/app/memory/` | 记忆系统（画像/统计/对话历史） |
| `miniprogram/pages/chat/chat.js` | 小程序聊天页，SSE 解析 |
| `miniprogram/utils/request.js` | wx.request 封装 + SSE 流式 |
