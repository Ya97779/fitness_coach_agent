const { request } = require('../../utils/request')

Page({
  data: {
    avatarUrl: '',
    nickname: ''
  },

  onChooseAvatar(e) {
    this.setData({ avatarUrl: e.detail.avatarUrl })
  },

  onNicknameInput(e) {
    this.setData({ nickname: e.detail.value })
  },

  saveProfile() {
    const { avatarUrl, nickname } = this.data
    if (!nickname.trim()) {
      wx.showToast({ title: '请输入昵称', icon: 'none' })
      return
    }

    wx.showLoading({ title: '保存中...' })

    // 如果有头像，先上传再保存
    if (avatarUrl) {
      wx.uploadFile({
        url: require('../../utils/config').API_BASE_URL + '/api/v1/user/avatar',
        filePath: avatarUrl,
        name: 'file',
        header: {
          Authorization: 'Bearer ' + wx.getStorageSync('token')
        },
        success: (res) => {
          const data = JSON.parse(res.data)
          this.saveToBackend(nickname, data.avatar_url)
        },
        fail: () => {
          // 上传失败，只保存昵称
          this.saveToBackend(nickname, '')
        }
      })
    } else {
      this.saveToBackend(nickname, '')
    }
  },

  saveToBackend(nickname, avatarUrl) {
    request({
      url: '/api/v1/user/profile',
      method: 'POST',
      data: { nickname, avatar_url: avatarUrl }
    }).then(() => {
      const app = getApp()
      if (app && app.globalData.userInfo) {
        app.globalData.userInfo.nickname = nickname
        if (avatarUrl) app.globalData.userInfo.avatar_url = avatarUrl
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
  }
})
