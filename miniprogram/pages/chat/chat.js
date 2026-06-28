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

  onLoad() {
    // 冷启动：加载缓存 + 服务端同步
    this.loadMessagesFromCache()
    if (isLoggedIn()) {
      this.syncMessagesFromServer()
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

    // 加载缓存消息（缓存在流式过程中被持续更新，始终是最新的）
    this.loadMessagesFromCache()

    // 流式进行中：恢复实时内容
    const app = getApp()
    if (app.globalData.chatStream.active) {
      this.restoreChatStream()
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
    if (!text || this.data.sending) return

    if (!isLoggedIn()) {
      showLoginPrompt()
      return
    }

    this.setData({ pendingIntent: null, intentButtonText: '' })

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

  updateAiMessage(msgId, content, isStatus) {
    // 流式过程中只更新纯文本，不解析 markdown（避免高频 setData 导致 mp-html 不刷新）
    // isStatus=true 表示状态消息（如"Agent正在思考..."），不是最终内容
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) {
        return { ...m, content, _streaming: true, _isStatus: !!isStatus, _hasRealContent: m._hasRealContent || !isStatus }
      }
      return m
    })
    this.setData({ messages })
    this.saveMessagesToCache()
  },

  finishAiMessage(msgId) {
    // 流式完成后一次性解析 markdown 并渲染
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) {
        // 如果最后仍是 status 消息（LLM 没返回真实内容），显示错误提示
        if (m._isStatus && !m._hasRealContent) {
          return { ...m, loading: false, _streaming: false, content: '抱歉，未能获取回复，请重试。', html: '<p>抱歉，未能获取回复，请重试。</p>' }
        }
        return { ...m, loading: false, _streaming: false, _isStatus: false, html: parseMarkdown(m.content) }
      }
      return m
    })
    this.setData({ messages, sending: false })
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
      .filter(m => m.role === 'user' || !m._streaming)
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
      const messages = cached.map(m => ({
        ...m,
        html: m.role !== 'user' ? parseMarkdown(m.content) : '',
        timeStr: m.role === 'user' ? (m.timeStr || formatTime(m.timestamp || Date.now())) : '',
        _streaming: false
      }))
      this.setData({ messages })
      return true
    }
    return false
  },

  // 从后端同步最新消息
  syncMessagesFromServer() {
    // 有进行中的流式请求时跳过同步，避免覆盖恢复的消息
    const app = getApp()
    if (app.globalData.chatStream.active) return Promise.resolve()

    return request({ url: '/api/v1/chat/history?limit=20' }).then(serverMessages => {
      if (!serverMessages || serverMessages.length === 0) return
      // 再次检查，因为异步返回时状态可能已变
      if (app.globalData.chatStream.active) return
      const cached = this.data.messages
      // 缓存消息比服务器多，说明服务器还没同步，不覆盖
      if (cached.length >= serverMessages.length) return
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
            timeStr: role === 'user' ? formatTime(m.timestamp || Date.now()) : '',
            _streaming: false
          }
        })
        this.setData({ messages: formatted })
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
      this.setData({ messages: stream.messages })
    }

    // 补全 pendingContent
    if (stream.pendingContent && stream.aiMsgId) {
      this.updateAiMessage(stream.aiMsgId, stream.pendingContent)
      stream.pendingContent = ''
    }
  }
})
