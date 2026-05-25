const { request, streamRequest } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')
const { parse: parseMarkdown } = require('../../utils/markdown')

let msgId = 0

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
    if (isLoggedIn() && !this.data.userAvatar) {
      request({ url: '/api/v1/user/me' }).then(user => {
        if (user.avatar_url) {
          this.setData({ userAvatar: user.avatar_url })
        }
      }).catch(() => {})
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

    const userMsg = { id: `m${++msgId}`, role: 'user', content: text }
    const aiMsg = { id: `m${++msgId}`, role: 'ai', content: '', loading: true }

    this.setData({
      messages: [...this.data.messages, userMsg, aiMsg],
      inputValue: '',
      sending: true,
      scrollToId: `msg-${aiMsg.id}`
    })

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
          }
        }
      },
      () => {
        this.finishAiMessage(aiMsg.id)
      },
      (err) => {
        this.updateAiMessage(aiMsg.id, fullContent || '抱歉，发生了错误，请稍后重试。')
        this.finishAiMessage(aiMsg.id)
      }
    )
  },

  updateAiMessage(msgId, content) {
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) {
        return { ...m, content, html: parseMarkdown(content) }
      }
      return m
    })
    this.setData({
      messages,
      scrollToId: 'msg-bottom'
    })
  },

  finishAiMessage(msgId) {
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) return { ...m, loading: false }
      return m
    })
    this.setData({ messages, sending: false })
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
  }
})
