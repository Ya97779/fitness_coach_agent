const { request } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')

Page({
  data: {
    content: '',
    contact: '',
    submitting: false
  },

  onLoad() {
    if (!isLoggedIn()) {
      showLoginPrompt().then(loggedIn => {
        if (!loggedIn) wx.navigateBack()
      })
    }
  },

  onContentInput(e) {
    this.setData({ content: e.detail.value })
  },

  onContactInput(e) {
    this.setData({ contact: e.detail.value })
  },

  submit() {
    if (this.data.submitting) return
    const content = this.data.content.trim()
    if (!content) {
      wx.showToast({ title: '请输入反馈内容', icon: 'none' })
      return
    }

    this.setData({ submitting: true })
    request({
      url: '/api/v1/feedback',
      method: 'POST',
      data: {
        content,
        contact: this.data.contact.trim() || null
      }
    }).then(() => {
      wx.showToast({ title: '感谢你的反馈', icon: 'success' })
      setTimeout(() => wx.navigateBack(), 800)
    }).catch(err => {
      wx.showToast({ title: err.message || '提交失败', icon: 'none' })
    }).finally(() => {
      this.setData({ submitting: false })
    })
  }
})
