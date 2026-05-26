# 记忆系统增强 + 聊天记录持久化设计

## 概述

解决5个关联问题：对话历史分层压缩、跨Agent记忆增强、聊天记录本地缓存、流式中断恢复、状态消息统一。

## 1. 后端 — 对话历史分层压缩

**改造 `ConversationSummarizer`**

滑动窗口策略（单用户对话历史）：

| 层级 | 范围 | 处理方式 | 压缩后长度 |
|------|------|----------|-----------|
| 完整保留 | 最近 10 条 | 原文不动 | — |
| 简要摘要 | 11-20 条 | LLM 每 5 条→1 条摘要 | ~50 字/条 |
| 精简摘要 | 20 条以上 | LLM 每 10 条→1 条摘要 | ~30 字/条 |

Agent 看到的顺序：`[精简摘要] → [简要摘要] → [最近10条原文]`

**改动文件**：`backend/app/memory/conversation_summary.py`

- `summarize_messages()` 改为分层处理，不再简单地"保留后N条 + 摘要前面所有"
- 新增 `_generate_layered_summary()` 方法，按层级生成不同精度的摘要
- `should_summarize()` 逻辑不变（>10 条时触发）

## 2. 后端 — 跨 Agent 历史注入增强

**改造 `MemoryManager.format_conversation_history_for_agent()`**

- 加载最近 10 条跨 Agent 对话（当前是 6 条）
- 超过 10 条的也用同样的渐进压缩策略
- 每条记录明确标注 agent_type

注入到 Agent system prompt 的格式：

```
【近期对话历史】
[2026-05-26 14:30] 用户: 今天吃了什么好？
→ 营养师回复: 你今天摄入了1200kcal，建议晚餐...
[2026-05-26 14:35] 用户: 帮我安排胸部训练
→ 健身教练回复: 建议以下动作...
[历史摘要] 用户之前讨论过减脂目标，每日目标热量...
```

**改动文件**：`backend/app/memory/memory_manager.py`

## 3. 小程序 — 聊天记录本地缓存

**改造 `chat.js`**

数据流：

```
onShow 触发
  ├─① 立即从 wx.getStorageSync('chat_messages') 加载 → 渲染（< 50ms）
  └─② 同时 GET /api/v1/chat/history?limit=20（不阻塞）
       ├─ 一致 → 不更新
       └─ 不一致 → 覆盖缓存 + 刷新UI
       └─ 失败 → 忽略，本地缓存继续显示

每次收发消息后 → wx.setStorageSync('chat_messages', messages.slice(-20))
```

缓存结构：`[{id, role, content, agent_type, timestamp}]`

**新增后端接口**：`GET /api/v1/chat/history`

- 从 `ConversationLog` 表查询最近 N 条
- 返回格式与缓存结构一致
- 用于首次打开或缓存丢失时的数据恢复

**改动文件**：
- `miniprogram/pages/chat/chat.js`
- `backend/app/main.py`（新增接口）

## 4. 小程序 — 流式中断恢复

**改造 `chat.js` + `app.js`**

流式状态存储在 `app.globalData.chatStream`：

```javascript
{
  active: true,              // 是否正在流式接收
  requestTask: null,         // wx.request 句柄
  messages: [],              // 当前消息列表
  pendingContent: "...",     // 已收到但未渲染的内容
  aiMsgId: "m5"             // AI 消息 ID
}
```

流程：

| 事件 | 处理 |
|------|------|
| sendMessage() | 创建 chatStream，开始接收 |
| onUnload（切走） | 不取消 requestTask，数据追加到 pendingContent |
| onShow（切回） | 检测 active=true，恢复 messages，补全 pendingContent |
| 流式完成 | active=false，清理状态 |

**改动文件**：
- `miniprogram/app.js`（新增 chatStream 初始化）
- `miniprogram/pages/chat/chat.js`

## 5. 状态消息统一

将 nutrition/fitness agent 的状态消息统一为 "Agent正在思考..."。

**后端 `graph.py`**：

```python
_status_messages = {
    "nutrition": "Agent正在思考...",
    "fitness": "Agent正在思考...",
}
```

chat agent 也加上相同的状态消息。

**小程序 `chat.js`**：status 事件处理逻辑不变，仅显示内容统一。

**改动文件**：
- `backend/app/agents/graph.py`
- `miniprogram/pages/chat/chat.js`

## 改动文件汇总

| 文件 | 改动内容 |
|------|---------|
| `backend/app/memory/conversation_summary.py` | 分层压缩逻辑 |
| `backend/app/memory/memory_manager.py` | 跨Agent历史增加到10条 |
| `backend/app/main.py` | 新增 `/chat/history` 接口 |
| `backend/app/agents/graph.py` | 状态消息统一 |
| `miniprogram/app.js` | 新增 chatStream 全局状态 |
| `miniprogram/pages/chat/chat.js` | 缓存、流式恢复、状态消息 |

## 验证方式

1. 对话超过10条后，Agent 仍能理解早期上下文（分层压缩生效）
2. 用户告诉营养师"不吃辣"，健身教练也知道（跨Agent记忆）
3. 关闭聊天页再打开，历史消息仍在（本地缓存）
4. 流式接收中切走再切回，内容不丢失（流式恢复）
5. 所有 Agent 回复前统一显示"Agent正在思考..."
