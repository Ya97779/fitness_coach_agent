const { request, streamRequest } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')
const { parse: parseMarkdown } = require('../../utils/markdown')

let msgId = 0

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
  },

  onInput(e) {
    this.setData({ inputValue: e.detail.value })
  },

  useShortcut(e) {
    this.setData({ inputValue: e.currentTarget.dataset.text })
  },

  sendMessage() {
    const text = this.data.inputValue.trim()
    if (!text || this.data.sending) return

    if (!isLoggedIn()) {
      showLoginPrompt()
      return
    }

    this.setData({ pendingIntent: null, intentButtonText: '' })
    this._refreshTimer = null
    this._refreshPending = false

    const userMsg = { id: `m${++msgId}`, role: 'user', content: text, timeStr: formatTime(Date.now()) }
    const aiMsg = { id: `m${++msgId}`, role: 'ai', content: '', loading: true }

    const messages = [...this.data.messages, userMsg, aiMsg]
    this.setData({
      messages,
      inputValue: '',
      sending: true,
      scrollToId: `msg-${aiMsg.id}`
    })
    this.saveMessagesToCache()

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
              } catch (e) {
                console.warn('解析意图数据失败:', e)
              }
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
  },

  updateAiMessage(msgId, content) {
    const html = parseMarkdown(content)
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) {
        return { ...m, content, html }
      }
      return m
    })
    this.setData({
      messages,
      scrollToId: 'msg-bottom'
    })
    this.saveMessagesToCache()

    // 流式输出过程中定期强制刷新 mp-html 组件（解决 wx:for 中 observer 不触发的问题）
    if (!this._refreshPending) {
      this._refreshPending = true
      clearTimeout(this._refreshTimer)
      this._refreshTimer = setTimeout(() => {
        this._refreshPending = false
        const idx = this.data.messages.findIndex(m => m.id === msgId)
        if (idx === -1) return
        // 先隐藏组件
        this.setData({ [`messages[${idx}]._refresh`]: false })
        // 下一帧重新显示，强制组件重建
        setTimeout(() => {
          this.setData({ [`messages[${idx}]._refresh`]: true })
        }, 50)
      }, 500)
    }
  },

  finishAiMessage(msgId) {
    // 清理流式刷新定时器
    clearTimeout(this._refreshTimer)
    this._refreshPending = false

    const messages = this.data.messages.map(m => {
      if (m.id === msgId) return { ...m, loading: false }
      return m
    })
    this.setData({ messages, sending: false })
    this.saveMessagesToCache()
    // 通过 toggle _visible 强制 mp-html 重建，触发 observer 渲染 markdown
    setTimeout(() => {
      const hideMsgs = this.data.messages.map(m => {
        if (m.id === msgId) return { ...m, _visible: false }
        return m
      })
      this.setData({ messages: hideMsgs })
      setTimeout(() => {
        const showMsgs = this.data.messages.map(m => {
          if (m.id === msgId) return { ...m, _visible: true }
          return m
        })
        this.setData({ messages: showMsgs })
      }, 50)
    }, 50)
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
      const messages = cached.map(m => ({
        ...m,
        html: m.role !== 'user' ? parseMarkdown(m.content) : '',
        timeStr: m.role === 'user' ? (m.timeStr || formatTime(m.timestamp || Date.now())) : ''
      }))
      this.setData({ messages, scrollToId: 'msg-bottom' })
      return true
    }
    return false
  },

  // 从后端同步最新消息
  syncMessagesFromServer() {
    // 有进行中的流式请求时跳过同步，避免覆盖恢复的消息
    const app = getApp()
    if (app.globalData.chatStream.active) return

    request({ url: '/api/v1/chat/history?limit=20' }).then(serverMessages => {
      if (!serverMessages || serverMessages.length === 0) return
      // 再次检查，因为异步返回时状态可能已变
      if (app.globalData.chatStream.active) return
      const cached = this.data.messages
      // 简单对比最后一条消息内容
      if (cached.length === 0 ||
          cached[cached.length - 1].content !== serverMessages[serverMessages.length - 1].content) {
        const formatted = serverMessages.map((m, i) => {
          const role = m.role === 'assistant' ? 'ai' : m.role
          return {
            id: `sync_${i}`,
            role,
            content: m.content,
            agent_type: m.agent_type,
            timestamp: m.timestamp,
            html: role !== 'user' ? parseMarkdown(m.content) : '',
            timeStr: role === 'user' ? formatTime(m.timestamp || Date.now()) : ''
          }
        })
        this.setData({ messages: formatted, scrollToId: 'msg-bottom' })
        wx.setStorageSync('chat_messages', formatted)
      }
    }).catch(() => {})
  },

  // 恢复进行中的流式请求状态
  restoreChatStream() {
    const app = getApp()
    const stream = app.globalData.chatStream

    // 恢复消息列表
    if (stream.messages.length > 0) {
      this.setData({ messages: stream.messages, scrollToId: 'msg-bottom' })
    }

    // 补全 pendingContent
    if (stream.pendingContent && stream.aiMsgId) {
      this.updateAiMessage(stream.aiMsgId, stream.pendingContent)
      stream.pendingContent = ''
    }
  }
})
