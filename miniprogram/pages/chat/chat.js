const { request, streamRequest } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')
const { parse: parseMarkdown } = require('../../utils/markdown')

let msgId = 0

function createMessageId(prefix) {
  msgId += 1
  return `${prefix || 'msg'}_${Date.now().toString(36)}_${msgId}_${Math.random().toString(36).slice(2, 8)}`
}

function getChatSessionId() {
  let sessionId = wx.getStorageSync('chat_session_id')
  if (!sessionId) {
    sessionId = createMessageId('session')
    wx.setStorageSync('chat_session_id', sessionId)
  }
  return sessionId
}

function ensureUniqueMessageIds(messages) {
  const seen = new Set()
  return (messages || []).map(message => {
    let id = message.id
    if (!id || seen.has(id)) {
      id = createMessageId(message.role || 'msg')
    }
    seen.add(id)
    return id === message.id ? message : { ...message, id }
  })
}

function messageContentSignature(messages) {
  return JSON.stringify((messages || []).map(message => [message.role, message.content]))
}

function formatTime(ts) {
  const d = new Date(ts)
  const M = String(d.getMonth() + 1).padStart(2, '0')
  const D = String(d.getDate()).padStart(2, '0')
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  return `${M}/${D} ${h}:${m}`
}

Page({
  data: {
    messages: [],
    inputValue: '',
    scrollToId: '',
    sending: false,
    userAvatar: '',
    pendingIntent: null,
    intentButtonText: '',
    shortcuts: [
      { icon: '🍚', text: '记录早餐' },
      { icon: '🏋️', text: '记录运动' },
      { icon: '🔍', text: '查询热量' },
      { icon: '💪', text: '训练建议' },
      { icon: '🥦', text: '饮食计划' }
    ]
  },

  onLoad() {
    const app = getApp()
    if (app.globalData.chatStream.active) {
      this.restoreChatStream()
    } else {
      // 冷启动：加载缓存 + 服务端同步
      this.loadMessagesFromCache()
      if (isLoggedIn()) {
        this.syncMessagesFromServer()
      }
    }
  },

  onShow() {
    // 加载用户头像
    if (isLoggedIn() && !this.data.userAvatar) {
      request({ url: '/api/v1/user/me' }).then(user => {
        if (user.avatar_url) {
          this.setData({ userAvatar: user.avatar_url })
        }
      }).catch(() => {})
    }

    const app = getApp()
    if (app.globalData.chatStream.active) {
      this.restoreChatStream()
    } else {
      this.loadMessagesFromCache()
    }

    // 首次进入小程序：滚动到底部；切换 tab 回来：保持原位
    if (app.globalData.appLaunched) {
      app.globalData.appLaunched = false
      setTimeout(() => {
        this.setData({ scrollToId: 'msg-bottom' })
        setTimeout(() => {
          this.setData({ scrollToId: '' })
        }, 500)
      }, 300)
    }
  },

  onPullDownRefresh() {
    if (isLoggedIn()) {
      this.syncMessagesFromServer().then(() => {
        wx.stopPullDownRefresh()
      }).catch(() => {
        wx.stopPullDownRefresh()
      })
    } else {
      wx.stopPullDownRefresh()
    }
  },

  onInput(e) {
    this.setData({ inputValue: e.detail.value })
  },

  useShortcut(e) {
    this.setData({ inputValue: e.currentTarget.dataset.text })
  },

  sendMessage() {
    const text = this.data.inputValue.trim()
    const app = getApp()
    if (!text || this.data.sending || app.globalData.chatStream.active) return

    if (!isLoggedIn()) {
      showLoginPrompt()
      return
    }

    this.setData({ pendingIntent: null, intentButtonText: '' })

    const timestamp = Date.now()
    const userMsg = {
      id: createMessageId('user'),
      role: 'user',
      content: text,
      timeStr: formatTime(timestamp),
      timestamp
    }
    const aiMsg = {
      id: createMessageId('ai'),
      role: 'ai',
      content: '',
      statusText: '',
      reasoningContent: '',
      reasoningExpanded: true,
      reasoningComplete: false,
      loading: true,
      _streaming: true,
      timestamp
    }

    const messages = [...this.data.messages, userMsg, aiMsg]
    this.setData({
      messages,
      inputValue: '',
      sending: true,
      scrollToId: `msg-${aiMsg.id}`
    })
    this.saveMessagesToCache()
    setTimeout(() => {
      if (this.data.scrollToId === `msg-${aiMsg.id}`) {
        this.setData({ scrollToId: '' })
      }
    }, 500)

    // 保存到全局状态
    const requestId = createMessageId('stream')
    app.globalData.chatStream.active = true
    app.globalData.chatStream.requestId = requestId
    app.globalData.chatStream.messages = messages
    app.globalData.chatStream.aiMsgId = aiMsg.id
    app.globalData.chatStream.pendingContent = ''
    app.globalData.chatStream.pendingReasoning = ''

    let fullContent = ''
    let fullReasoning = ''
    let lineBuffer = ''
    let currentEventType = 'data'
    const requestTask = streamRequest(
      {
        url: '/api/v1/chat/stream',
        data: {
          message: text,
          session_id: getChatSessionId(),
          request_id: requestId
        }
      },
      (chunk) => {
        if (!this.isCurrentChatStream(requestId)) return
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
            const data = trimmed.slice(6).replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
            if (data === '[DONE]') continue
            if (data.startsWith('Error:')) {
              fullContent = data
              this.updateAiMessage(aiMsg.id, fullContent)
              continue
            }
            if (currentEventType === 'status') {
              this.updateAiMessage(aiMsg.id, data, true)
              currentEventType = 'data'
              continue
            }
            if (currentEventType === 'thinking') {
              fullReasoning += data
              this.updateAiReasoning(aiMsg.id, fullReasoning)
              app.globalData.chatStream.pendingReasoning = fullReasoning
              currentEventType = 'data'
              continue
            }
            if (currentEventType === 'queue') {
              this.updateAiMessage(aiMsg.id, `排队中，前面还有 ${data} 位...`, true)
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
              } catch (e) {
                console.warn('解析意图数据失败:', e)
              }
              currentEventType = 'data'
              continue
            }
            currentEventType = 'data'
            fullContent += data
            this.updateAiMessage(aiMsg.id, fullContent, false)
            // 更新全局 pendingContent
            app.globalData.chatStream.pendingContent = fullContent
          }
        }
      },
      () => {
        if (!this.isCurrentChatStream(requestId)) return
        this.finishAiMessage(aiMsg.id)
        this.completeChatStream(requestId)
      },
      (err) => {
        if (!this.isCurrentChatStream(requestId)) return
        this.updateAiMessage(aiMsg.id, fullContent || '抱歉，发生了错误，请稍后重试。')
        this.finishAiMessage(aiMsg.id)
        this.completeChatStream(requestId)
      }
    )

    if (this.isCurrentChatStream(requestId)) {
      app.globalData.chatStream.requestTask = requestTask
    }
  },

  isCurrentChatStream(requestId) {
    const stream = getApp().globalData.chatStream
    return !!stream.active && stream.requestId === requestId
  },

  completeChatStream(requestId) {
    const stream = getApp().globalData.chatStream
    if (stream.requestId !== requestId) return
    stream.active = false
    stream.requestTask = null
    stream.requestId = ''
    stream.messages = this.data.messages
    stream.pendingContent = ''
    stream.pendingReasoning = ''
  },

  updateAiMessage(msgId, content, isStatus) {
    // 流式过程中只更新纯文本，不解析 markdown（避免高频 setData 导致 mp-html 不刷新）
    // isStatus=true 表示状态消息（如"Agent正在思考..."），不是最终内容
    const index = this.data.messages.findIndex(message => message.id === msgId)
    if (index < 0) return
    const messages = [...this.data.messages]
    const message = messages[index]
    messages[index] = isStatus
      ? {
          ...message,
          statusText: content,
          _streaming: true,
          _isStatus: true
        }
      : {
          ...message,
          content,
          statusText: '',
          reasoningExpanded: message.reasoningExpanded !== false,
          reasoningComplete: !!message.reasoningContent,
          _streaming: true,
          _isStatus: false,
          _hasRealContent: true
        }
    this.setData({ messages })
    const stream = getApp().globalData.chatStream
    if (stream.active && stream.aiMsgId === msgId) {
      stream.messages = messages
    }
    this.saveMessagesToCache()
  },

  updateAiReasoning(msgId, reasoningContent) {
    const index = this.data.messages.findIndex(message => message.id === msgId)
    if (index < 0) return
    const messages = [...this.data.messages]
    const message = messages[index]
    messages[index] = {
      ...message,
      reasoningContent,
      reasoningExpanded: message.reasoningExpanded !== false,
      reasoningComplete: !!message._hasRealContent,
      statusText: '',
      _streaming: true
    }
    this.setData({ messages })
    const stream = getApp().globalData.chatStream
    if (stream.active && stream.aiMsgId === msgId) {
      stream.messages = messages
    }
  },

  toggleReasoning(e) {
    const msgId = e.currentTarget.dataset.id
    const index = this.data.messages.findIndex(message => message.id === msgId)
    if (index < 0) return
    this.setData({
      [`messages[${index}].reasoningExpanded`]: !this.data.messages[index].reasoningExpanded
    })
  },

  finishAiMessage(msgId) {
    // 流式完成后一次性解析 markdown 并渲染
    const index = this.data.messages.findIndex(message => message.id === msgId)
    if (index < 0) return
    const messages = [...this.data.messages]
    const message = messages[index]
    // 如果最后仍是 status 消息（LLM 没返回真实内容），显示错误提示
    if (message._isStatus && !message._hasRealContent) {
      messages[index] = {
        ...message,
        loading: false,
        _streaming: false,
        statusText: '',
        content: '抱歉，未能获取回复，请重试。',
        html: '<p>抱歉，未能获取回复，请重试。</p>'
      }
    } else {
      messages[index] = {
        ...message,
        loading: false,
        _streaming: false,
        _isStatus: false,
        statusText: '',
        reasoningExpanded: message.reasoningExpanded !== false,
        reasoningComplete: !!message.reasoningContent,
        html: parseMarkdown(message.content)
      }
    }
    this.setData({ messages, sending: false, scrollToId: '' })
    const stream = getApp().globalData.chatStream
    if (stream.active && stream.aiMsgId === msgId) {
      stream.messages = messages
    }
    this.saveMessagesToCache()
  },

  recordFromIntent() {
    const intent = this.data.pendingIntent
    if (!intent) return

    if (intent.type === 'food') {
      const d = intent.data
      request({
        url: '/api/v1/food-log',
        method: 'POST',
        data: {
          name: d.food_name,
          calories: d.calories,
          meal_type: d.meal_type || 'dinner'
        }
      }).then(() => {
        wx.showToast({ title: '已记录饮食', icon: 'success' })
        this.setData({ pendingIntent: null, intentButtonText: '' })
      }).catch(() => {
        wx.showToast({ title: '记录失败', icon: 'none' })
      })
    } else if (intent.type === 'exercise') {
      const d = intent.data
      request({
        url: '/api/v1/exercise-log',
        method: 'POST',
        data: {
          name: d.exercise_name,
          type: d.exercise_name,
          duration: d.duration || 0,
          calories: d.calories
        }
      }).then(() => {
        wx.showToast({ title: '已记录运动', icon: 'success' })
        this.setData({ pendingIntent: null, intentButtonText: '' })
      }).catch(() => {
        wx.showToast({ title: '记录失败', icon: 'none' })
      })
    }
  },

  clearIntent() {
    this.setData({ pendingIntent: null, intentButtonText: '' })
  },

  // ========== 本地缓存 ==========

  // 保存消息到本地缓存（只保存已完成的消息，跳过正在流式的）
  saveMessagesToCache() {
    const messages = this.data.messages
      .filter(m => m.role === 'user' || (!m._streaming && !m.loading))
      .slice(-20)
      .map(m => ({
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
      const messages = ensureUniqueMessageIds(cached.map(m => ({
        ...m,
        html: m.role !== 'user' ? parseMarkdown(m.content) : '',
        timeStr: m.role === 'user' ? (m.timeStr || formatTime(m.timestamp || Date.now())) : '',
        _streaming: false
      })))
      this.setData({ messages })
      this.saveMessagesToCache()
      return true
    }
    return false
  },

  // 从后端同步最新消息
  syncMessagesFromServer() {
    // 有进行中的流式请求时跳过同步，避免覆盖恢复的消息
    const app = getApp()
    if (app.globalData.chatStream.active) return Promise.resolve()

    const sessionId = getChatSessionId()
    return request({
      url: `/api/v1/chat/history?limit=20&session_id=${encodeURIComponent(sessionId)}`
    }).then(serverMessages => {
      if (!serverMessages || serverMessages.length === 0) return
      // 再次检查，因为异步返回时状态可能已变
      if (app.globalData.chatStream.active) return
      const cached = this.data.messages.filter(message => !message._streaming)
      const formatted = ensureUniqueMessageIds(serverMessages.map((m, i) => {
          const role = m.role === 'assistant' ? 'ai' : m.role
          return {
            id: `server_${m.id || i}_${role}_${i}`,
            role,
            content: m.content,
            agent_type: m.agent_type,
            timestamp: m.timestamp,
            html: role !== 'user' ? parseMarkdown(m.content) : '',
            timeStr: role === 'user' ? formatTime(m.timestamp || Date.now()) : '',
            _streaming: false
          }
        }))
      // 比较同一时间窗口的内容。消息数量相同也可能是本地缓存已被错误覆盖。
      const comparableServer = cached.length > 0 ? formatted.slice(-cached.length) : []
      if (cached.length === 0 ||
          messageContentSignature(cached) !== messageContentSignature(comparableServer)) {
        this.setData({ messages: formatted })
        this.saveMessagesToCache()
      }
    }).catch(() => {})
  },

  // 恢复进行中的流式请求状态
  restoreChatStream() {
    const app = getApp()
    const stream = app.globalData.chatStream

    let messages = ensureUniqueMessageIds(stream.messages)
    if (stream.pendingContent && stream.aiMsgId) {
      const index = messages.findIndex(message => message.id === stream.aiMsgId)
      if (index >= 0) {
        messages = [...messages]
        messages[index] = {
          ...messages[index],
          content: stream.pendingContent,
          _streaming: true,
          _isStatus: false,
          _hasRealContent: true
        }
      }
    }
    if (stream.pendingReasoning && stream.aiMsgId) {
      const index = messages.findIndex(message => message.id === stream.aiMsgId)
      if (index >= 0) {
        messages = [...messages]
        messages[index] = {
          ...messages[index],
          reasoningContent: stream.pendingReasoning,
          reasoningExpanded: messages[index].reasoningExpanded !== false,
          reasoningComplete: !!messages[index]._hasRealContent,
          _streaming: true
        }
      }
    }
    stream.messages = messages
    stream.pendingContent = ''
    stream.pendingReasoning = ''
    if (messages.length > 0) {
      this.setData({ messages, sending: true })
    }
  },

  onShareAppMessage() {
    return {
      title: '健身助手Agent - 智能健身营养问答',
      path: '/pages/chat/chat'
    }
  },

  onShareTimeline() {
    return {
      title: '健身助手Agent - 智能健身营养问答'
    }
  }
})
