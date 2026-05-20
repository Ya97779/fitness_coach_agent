const { streamRequest } = require('../../utils/request')

let msgId = 0

Page({
  data: {
    messages: [],
    inputValue: '',
    scrollToId: '',
    sending: false,
    disconnected: false,
    lastUserMessage: '',
    shortcuts: [
      { icon: '🍚', text: '记录早餐' },
      { icon: '🏋️', text: '记录运动' },
      { icon: '🔍', text: '查询热量' },
      { icon: '💪', text: '训练建议' },
      { icon: '🥦', text: '饮食计划' }
    ]
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

    const userMsg = { id: `m${++msgId}`, role: 'user', content: text }
    const aiMsg = { id: `m${++msgId}`, role: 'ai', content: '', loading: true }

    this.setData({
      messages: [...this.data.messages, userMsg, aiMsg],
      inputValue: '',
      sending: true,
      disconnected: false,
      lastUserMessage: text,
      scrollToId: `msg-${aiMsg.id}`
    })

    this._doStream(text, aiMsg.id)
  },

  retryLastMessage() {
    if (!this.data.lastUserMessage || this.data.sending) return
    // 删除最后一条空的或错误的 AI 消息，重新创建
    const messages = this.data.messages.filter(m => {
      if (m.role === 'ai' && m.id === `m${msgId}`) return false
      return true
    })
    const aiMsg = { id: `m${++msgId}`, role: 'ai', content: '', loading: true }
    this.setData({
      messages: [...messages, aiMsg],
      sending: true,
      disconnected: false,
      scrollToId: `msg-${aiMsg.id}`
    })
    this._doStream(this.data.lastUserMessage, aiMsg.id)
  },

  _doStream(text, aiMsgId) {
    // 构建上下文：最近 10 条消息
    const history = this.data.messages
      .slice(-10)
      .filter(m => !m.loading)
      .map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content }))

    let fullContent = ''
    const requestTask = streamRequest(
      { url: '/api/v1/chat/stream', data: { message: text, history } },
      (chunk) => {
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') continue
            fullContent += data
            this.updateAiMessage(aiMsgId, fullContent)
          }
        }
      },
      () => {
        this.finishAiMessage(aiMsgId)
      },
      (err) => {
        if (!fullContent) {
          this.setData({ disconnected: true })
        }
        this.updateAiMessage(aiMsgId, fullContent || '连接中断，请点击重试')
        this.setData({ sending: false })
        this.finishAiMessage(aiMsgId)
      }
    )
  },

  updateAiMessage(msgId, content) {
    const messages = this.data.messages.map(m => {
      if (m.id === msgId) return { ...m, content }
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
  }
})
