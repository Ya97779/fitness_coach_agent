# 记忆系统增强 + 聊天记录持久化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现对话历史分层压缩、跨Agent记忆增强、聊天记录本地缓存、流式中断恢复、状态消息统一

**Architecture:** 后端改造 ConversationSummarizer 实现滑动窗口分层压缩，MemoryManager 增加跨Agent历史注入；小程序侧用 wx.setStorageSync 缓存消息，app.globalData 维护流式状态

**Tech Stack:** Python, LangChain, FastAPI, 微信小程序原生框架

---

### Task 1: 后端 — 分层压缩改造 ConversationSummarizer

**Files:**
- Modify: `backend/app/memory/conversation_summary.py:55-92`
- Test: `backend/tests/test_memory.py:79-127`

- [ ] **Step 1: 写失败测试 — 分层压缩保留最近10条原文**

```python
# 在 test_memory.py 的 TestConversationSummarizer 类中添加：

def test_summarize_messages_preserves_last_10_raw(self):
    """测试分层压缩保留最近10条消息原文"""
    summarizer = ConversationSummarizer(max_messages=10)
    # 创建25条消息：15条旧 + 10条新
    old_messages = [HumanMessage(content=f"旧消息 {i}") for i in range(15)]
    recent_messages = [AIMessage(content=f"新消息 {i}") for i in range(10)]
    all_messages = old_messages + recent_messages

    with patch.object(summarizer, '_generate_summary', return_value="摘要内容"):
        result = summarizer.summarize_messages(all_messages)

    # 最近10条应该完整保留
    recent_contents = [m.content for m in result[-10:]]
    for i in range(10):
        self.assertIn(f"新消息 {i}", recent_contents)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd D:/fitness_coach && python -m pytest backend/tests/test_memory.py::TestConversationSummarizer::test_summarize_messages_preserves_last_10_raw -v`
Expected: FAIL（方法还没改）

- [ ] **Step 3: 实现分层压缩逻辑**

改造 `conversation_summary.py` 的 `summarize_messages()` 方法：

```python
def summarize_messages(
    self,
    messages: List[BaseMessage],
    user_profile: Optional[Dict[str, Any]] = None
) -> List[BaseMessage]:
    """对早期消息进行分层摘要，保留关键信息

    分层策略：
    - 最近 10 条：完整保留
    - 11-20 条：简要摘要（每5条→1条，~50字）
    - 20 条以上：精简摘要（每10条→1条，~30字）
    """
    if not self.should_summarize(messages):
        return messages

    keep_recent = self.MAX_MESSAGES_BEFORE_SUMMARY  # 10
    recent_messages = messages[-keep_recent:]
    older_messages = messages[:-keep_recent]

    # 分层处理
    system_messages = [m for m in older_messages if isinstance(m, SystemMessage)]
    non_system = [m for m in older_messages if not isinstance(m, SystemMessage)]

    summary_parts = []
    if len(non_system) > 10:
        # 精简摘要层：20条以上的部分
        deep_old = non_system[:-10]
        summary_parts.append(self._generate_layered_summary(deep_old, "compact"))
        # 简要摘要层：11-20条的部分
        mid_old = non_system[-10:]
        summary_parts.append(self._generate_layered_summary(mid_old, "brief"))
    else:
        # 只有简要摘要层
        summary_parts.append(self._generate_layered_summary(non_system, "brief"))

    result = []
    result.extend(system_messages)
    result.append(AIMessage(content="\n".join(summary_parts)))
    result.extend(recent_messages)
    return result
```

新增 `_generate_layered_summary()` 方法：

```python
def _generate_layered_summary(
    self,
    messages: List[BaseMessage],
    level: str = "brief"
) -> str:
    """按层级生成摘要

    Args:
        messages: 需要摘要的消息
        level: "brief"（简要，~50字）或 "compact"（精简，~30字）

    Returns:
        str: 生成的摘要
    """
    if not messages:
        return ""

    message_texts = []
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        message_texts.append(f"{role}: {content}")

    conversation_text = "\n".join(message_texts)

    if level == "compact":
        prompt = f"用30字以内精简概括以下对话的核心结论和用户偏好：\n{conversation_text}"
    else:
        prompt = f"用50字以内概括以下对话的主要话题和结论：\n{conversation_text}"

    try:
        from ..llm_manager import LLMManager
        llm = LLMManager.get_llm(temperature=0.3)
        response = llm.invoke([HumanMessage(content=prompt)])
        return f"【历史摘要】{response.content}"
    except Exception as e:
        return f"【历史摘要】（摘要生成失败）"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd D:/fitness_coach && python -m pytest backend/tests/test_memory.py::TestConversationSummarizer -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/memory/conversation_summary.py backend/tests/test_memory.py
git commit -m "feat: ConversationSummarizer 分层压缩改造"
```

---

### Task 2: 后端 — 跨Agent历史注入增强

**Files:**
- Modify: `backend/app/memory/memory_manager.py:444-475`
- Test: `backend/tests/test_memory.py`

- [ ] **Step 1: 写失败测试 — 跨Agent历史加载10条**

```python
# 在 test_memory.py 的 TestMemoryManager 类中添加：

@patch('app.memory.memory_manager.database.SessionLocal')
def test_format_conversation_history_for_agent_loads_10(self, mock_db):
    """测试跨Agent历史加载最近10条"""
    mock_session = MagicMock()
    mock_db.return_value = mock_session

    # 模拟12条对话记录
    mock_logs = []
    for i in range(12):
        log = MagicMock()
        log.user_message = f"用户消息 {i}"
        log.agent_response = f"AI回复 {i}"
        log.agent_type = "nutrition" if i % 2 == 0 else "fitness"
        log.created_at = datetime(2026, 5, 27, 14, i, 0)
        mock_logs.append(log)

    mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_logs

    mm = MemoryManager(user_id=1)
    result = mm.format_conversation_history_for_agent(days=7, limit=10)

    self.assertIsInstance(result, str)
    self.assertIn("近期对话历史", result)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd D:/fitness_coach && python -m pytest backend/tests/test_memory.py::TestMemoryManager::test_format_conversation_history_for_agent_loads_10 -v`
Expected: FAIL

- [ ] **Step 3: 改造 format_conversation_history_for_agent()**

将 `memory_manager.py` 中的 `format_conversation_history_for_agent()` 的 limit 默认值从 6 改为 10，并调整格式化逻辑：

```python
def format_conversation_history_for_agent(
    self,
    days: int = 7,
    limit: int = 10
) -> str:
    """格式化对话历史为 Agent 可读格式

    Args:
        days: 加载最近 N 天的对话
        limit: 最多加载 N 轮对话（默认10）

    Returns:
        str: 格式化的对话历史
    """
    history = self.load_conversation_history(days=days, limit=limit)

    if not history:
        return "（无历史对话）"

    agent_names = {
        "chat": "闲聊",
        "nutrition": "营养师",
        "fitness": "健身教练"
    }

    parts = ["【近期对话历史】"]
    for i, msg in enumerate(history):
        role = "用户" if msg["role"] == "user" else "AI"
        agent = msg.get("agent_type", "")
        agent_label = agent_names.get(agent, agent)
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        time = msg.get("created_at", "")[:16] if msg.get("created_at") else ""

        if i % 2 == 0:
            parts.append(f"\n[{time}] {role}: {content}")
        else:
            parts.append(f"→ {agent_label}回复: {content}")

    return "\n".join(parts)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd D:/fitness_coach && python -m pytest backend/tests/test_memory.py::TestMemoryManager -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/memory/memory_manager.py backend/tests/test_memory.py
git commit -m "feat: 跨Agent历史注入增加到10条，优化agent标记"
```

---

### Task 3: 后端 — 新增 /chat/history 接口

**Files:**
- Modify: `backend/app/main.py`
- Test: 手动 curl 测试

- [ ] **Step 1: 添加 Pydantic 模型和接口**

在 `main.py` 中添加：

```python
# 在现有 Pydantic 模型附近添加
class ChatHistoryItem(BaseModel):
    id: int
    role: str  # "user" 或 "assistant"
    content: str
    agent_type: str
    timestamp: Optional[str] = None

# 在现有路由附近添加
@router.get("/chat/history", response_model=List[ChatHistoryItem])
def get_chat_history(
    limit: int = 20,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    """获取最近的对话历史"""
    logs = db.query(models.ConversationLog).filter(
        models.ConversationLog.user_id == current_user.id
    ).order_by(
        models.ConversationLog.created_at.desc()
    ).limit(limit).all()

    result = []
    for log in reversed(logs):
        result.append(ChatHistoryItem(
            id=log.id,
            role="user",
            content=log.user_message,
            agent_type=log.agent_type,
            timestamp=log.created_at.isoformat() if log.created_at else None
        ))
        result.append(ChatHistoryItem(
            id=log.id,
            role="assistant",
            content=log.agent_response,
            agent_type=log.agent_type,
            timestamp=log.created_at.isoformat() if log.created_at else None
        ))

    return result
```

- [ ] **Step 2: 启动后端测试接口**

Run: `cd D:/fitness_coach && uvicorn backend.app.main:app --reload --port 8000`

在另一个终端测试：
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/chat/history?limit=5
```

Expected: 返回 JSON 数组

- [ ] **Step 3: 提交**

```bash
git add backend/app/main.py
git commit -m "feat: 新增 GET /chat/history 接口"
```

---

### Task 4: 后端 — 状态消息统一

**Files:**
- Modify: `backend/app/agents/graph.py:633-638`

- [ ] **Step 1: 修改状态消息**

将 `graph.py` 中的 `_status_messages` 改为：

```python
_status_messages = {
    "nutrition": "Agent正在思考...",
    "fitness": "Agent正在思考...",
    "chat": "Agent正在思考...",
}
```

同时修改条件判断，让 chat agent 也发送状态消息：

```python
# 原来
if agent in _status_messages:
    yield ("status", _status_messages[agent])

# 改为（所有 agent 都发送）
yield ("status", _status_messages.get(agent, "Agent正在思考..."))
```

- [ ] **Step 2: 提交**

```bash
git add backend/app/agents/graph.py
git commit -m "feat: 统一所有Agent状态消息为'Agent正在思考...'"
```

---

### Task 5: 小程序 — app.js 新增 chatStream 全局状态

**Files:**
- Modify: `miniprogram/app.js:1-18`

- [ ] **Step 1: 在 globalData 中添加 chatStream**

在 `app.js` 的 `globalData` 中添加：

```javascript
globalData: {
    userInfo: null,
    token: null,
    chatStream: {
      active: false,
      requestTask: null,
      messages: [],
      pendingContent: '',
      aiMsgId: ''
    },
    training: {
      // ... 保持不变
    }
}
```

- [ ] **Step 2: 提交**

```bash
git add miniprogram/app.js
git commit -m "feat: app.js 新增 chatStream 全局状态"
```

---

### Task 6: 小程序 — chat.js 缓存 + 流式恢复 + 状态消息

**Files:**
- Modify: `miniprogram/pages/chat/chat.js`

- [ ] **Step 1: 添加本地缓存方法**

在 `chat.js` 中添加缓存相关方法：

```javascript
// 保存消息到本地缓存
saveMessagesToCache() {
  const messages = this.data.messages.slice(-20).map(m => ({
    id: m.id,
    role: m.role,
    content: m.content,
    agent_type: m.agent_type || '',
    timestamp: m.timestamp || Date.now()
  }))
  wx.setStorageSync('chat_messages', messages)
},

// 从本地缓存加载消息
loadMessagesFromCache() {
  const cached = wx.getStorageSync('chat_messages')
  if (cached && cached.length > 0) {
    this.setData({ messages: cached })
    return true
  }
  return false
},

// 从后端同步最新消息
syncMessagesFromServer() {
  request({ url: '/api/v1/chat/history?limit=20' }).then(serverMessages => {
    if (!serverMessages || serverMessages.length === 0) return
    const cached = this.data.messages
    // 简单对比最后一条消息内容
    if (cached.length === 0 || 
        cached[cached.length - 1].content !== serverMessages[serverMessages.length - 1].content) {
      const formatted = serverMessages.map((m, i) => ({
        id: `sync_${i}`,
        role: m.role,
        content: m.content,
        agent_type: m.agent_type,
        timestamp: m.timestamp
      }))
      this.setData({ messages: formatted })
      wx.setStorageSync('chat_messages', formatted)
    }
  }).catch(() => {})
}
```

- [ ] **Step 2: 改造 onShow 加载缓存和同步**

```javascript
onShow() {
  // 加载用户头像
  if (isLoggedIn() && !this.data.userAvatar) {
    request({ url: '/api/v1/user/me' }).then(user => {
      if (user.avatar_url) {
        this.setData({ userAvatar: user.avatar_url })
      }
    }).catch(() => {})
  }

  // 加载本地缓存
  this.loadMessagesFromCache()

  // 后台同步最新数据
  if (isLoggedIn()) {
    this.syncMessagesFromServer()
  }

  // 检查是否有进行中的流式请求
  const app = getApp()
  if (app.globalData.chatStream.active) {
    this.restoreChatStream()
  }
}
```

- [ ] **Step 3: 添加流式恢复方法**

```javascript
restoreChatStream() {
  const app = getApp()
  const stream = app.globalData.chatStream
  
  // 恢复消息列表
  if (stream.messages.length > 0) {
    this.setData({ messages: stream.messages })
  }

  // 补全 pendingContent
  if (stream.pendingContent && stream.aiMsgId) {
    this.updateAiMessage(stream.aiMsgId, stream.pendingContent)
    stream.pendingContent = ''
  }

  // 重新绑定 onChunkReceived 回调
  if (stream.requestTask) {
    stream.requestTask.onChunkReceived((response) => {
      try {
        const { decodeChunk } = require('../../utils/request')
        const text = decodeChunk(response.data)
        this.handleStreamChunk(text, stream.aiMsgId)
      } catch (e) {
        console.warn('流式接收异常:', e)
      }
    })
  }
}
```

- [ ] **Step 4: 改造 sendMessage 使用 chatStream**

```javascript
sendMessage() {
  const text = this.data.inputValue.trim()
  if (!text || this.data.sending) return

  if (!isLoggedIn()) {
    showLoginPrompt()
    return
  }

  this.setData({ pendingIntent: null, intentButtonText: '' })

  const userMsg = { id: `m${++msgId}`, role: 'user', content: text }
  const aiMsg = { id: `m${++msgId}`, role: 'ai', content: '', loading: true }

  const messages = [...this.data.messages, userMsg, aiMsg]
  this.setData({
    messages,
    inputValue: '',
    sending: true,
    scrollToId: `msg-${aiMsg.id}`
  })

  // 保存到全局状态
  const app = getApp()
  app.globalData.chatStream.active = true
  app.globalData.chatStream.messages = messages
  app.globalData.chatStream.aiMsgId = aiMsg.id
  app.globalData.chatStream.pendingContent = ''

  let fullContent = ''
  let lineBuffer = ''
  let currentEventType = 'data'
  
  const requestTask = streamRequest(
    { url: '/api/v1/chat/stream', data: { message: text } },
    (chunk) => {
      lineBuffer += chunk
      const parts = lineBuffer.split('\n')
      lineBuffer = parts.pop() || ''
      for (const line of parts) {
        const trimmed = line.trim()
        if (trimmed.startsWith('event: ')) {
          currentEventType = trimmed.slice(7).trim()
          continue
        }
        if (trimmed.startsWith('data: ')) {
          const data = trimmed.slice(6)
          if (data === '[DONE]') continue
          if (data.startsWith('Error:')) {
            fullContent = data
            this.updateAiMessage(aiMsg.id, fullContent)
            continue
          }
          if (currentEventType === 'status') {
            this.updateAiMessage(aiMsg.id, data)
            currentEventType = 'data'
            continue
          }
          if (currentEventType === 'intent') {
            try {
              const intentData = JSON.parse(data)
              const btnText = intentData.type === 'food'
                ? `记录${intentData.data.food_name}到饮食日志 (${intentData.data.calories} kcal)`
                : `记录${intentData.data.exercise_name}运动 (${intentData.data.duration}分钟)`
              this.setData({ pendingIntent: intentData, intentButtonText: btnText })
            } catch (e) {}
            currentEventType = 'data'
            continue
          }
          currentEventType = 'data'
          fullContent += data
          this.updateAiMessage(aiMsg.id, fullContent)
          // 更新全局 pendingContent
          app.globalData.chatStream.pendingContent = fullContent
        }
      }
    },
    () => {
      this.finishAiMessage(aiMsg.id)
      app.globalData.chatStream.active = false
      this.saveMessagesToCache()
    },
    (err) => {
      this.updateAiMessage(aiMsg.id, fullContent || '抱歉，发生了错误，请稍后重试。')
      this.finishAiMessage(aiMsg.id)
      app.globalData.chatStream.active = false
      this.saveMessagesToCache()
    }
  )

  app.globalData.chatStream.requestTask = requestTask
}
```

- [ ] **Step 5: 添加 updateAiMessage 保存缓存**

在 `updateAiMessage` 末尾添加缓存保存：

```javascript
updateAiMessage(msgId, content) {
  const messages = this.data.messages.map(m => {
    if (m.id === msgId) {
      return { ...m, content }
    }
    return m
  })
  this.setData({
    messages,
    scrollToId: 'msg-bottom'
  })
  // 流式过程中也定期保存
  this.saveMessagesToCache()
}
```

- [ ] **Step 6: 提交**

```bash
git add miniprogram/pages/chat/chat.js
git commit -m "feat: 聊天记录本地缓存 + 流式中断恢复"
```

---

### Task 7: 运行全部测试 + 同步部署

- [ ] **Step 1: 运行后端测试**

Run: `cd D:/fitness_coach && python -m pytest backend/tests/ -v --tb=short`
Expected: 全部 PASS

- [ ] **Step 2: 同步到 deploy 分支**

```bash
git push origin main
git checkout deploy && git merge main && git push origin deploy && git checkout main
```
