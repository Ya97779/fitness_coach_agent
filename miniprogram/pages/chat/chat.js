const { streamRequest } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')
const showdown = require('../../utils/showdown')

const converter = new showdown.Converter({
  simplifiedAutoLink: true,
  literalMidWordUnderscores: true,
  tables: true,
  tasklists: false,
  simpleLineBreaks: true
})

let msgId = 0

Page({
  data: {
    messages: [],
    inputValue: '',
    scrollToId: '',
    sending: false,
    shortcuts: [
      { icon: '🍚', text: '记录早餐' },
      { icon: '🏋️', text: '记录运动' },
      { icon: '🔍', text: '查询热量' },
      { icon: '💪', text: '训练建议' },
      { icon: '🥦', text: '饮食计划' }
    ],
    tagStyle: {
      p: 'margin-bottom: 16rpx; line-height: 1.8;',
      ul: 'margin-bottom: 16rpx; padding-left: 32rpx;',
      ol: 'margin-bottom: 16rpx; padding-left: 32rpx;',
      li: 'margin-bottom: 8rpx; line-height: 1.7;',
      h2: 'margin-top: 24rpx; margin-bottom: 12rpx; font-weight: 700;',
      h3: 'margin-top: 20rpx; margin-bottom: 10rpx; font-weight: 700;'
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
    let currentEventType = 'data'  // 默认 event type
    const requestTask = streamRequest(
      { url: '/api/v1/chat/stream', data: { message: text } },
      (chunk) => {
        // 行缓冲：TCP 分包可能截断 SSE 行，需要拼接后再解析
        lineBuffer += chunk
        const parts = lineBuffer.split('\n')
        // 最后一个元素可能是不完整的行，保留在 buffer 中
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
            // status 事件只显示提示，不计入最终内容
            if (currentEventType === 'status') {
              this.updateAiMessage(aiMsg.id, data)
              currentEventType = 'data'  // 重置
              continue
            }
            currentEventType = 'data'  // 重置
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
        const htmlContent = converter.makeHtml(content)
        return { ...m, content, htmlContent }
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
  }
})
