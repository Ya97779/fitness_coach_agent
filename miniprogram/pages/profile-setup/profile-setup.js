const { request } = require('../../utils/request')

Page({
  data: {
    nickname: ''
  },

  onNicknameInput(e) {
    this.setData({ nickname: e.detail.value })
  },

  onNicknameChange(e) {
    if (e.detail.value) {
      this.setData({ nickname: e.detail.value })
    }
  },

  saveProfile() {
    const nickname = this.data.nickname.trim()
    if (!nickname) {
      wx.showToast({ title: '请输入昵称', icon: 'none' })
      return
    }

    wx.showLoading({ title: '保存中...' })
    request({
      url: '/api/v1/user/profile',
      method: 'POST',
      data: { nickname }
    }).then(() => {
      const app = getApp()
      if (app && app.globalData.userInfo) {
        app.globalData.userInfo.nickname = nickname
      }
      wx.hideLoading()
      wx.showToast({ title: '设置成功', icon: 'success' })
      setTimeout(() => {
        wx.switchTab({ url: '/pages/home/home' })
      }, 800)
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '保存失败', icon: 'none' })
    })
  },

  skip() {
    wx.switchTab({ url: '/pages/home/home' })
  },

  onShareAppMessage() {
    return {
      title: '健身助手Agent - 你的智能健身营养顾问',
      path: '/pages/home/home'
    }
  },

  onShareTimeline() {
    return {
      title: '健身助手Agent - 你的智能健身营养顾问'
    }
  }
})
