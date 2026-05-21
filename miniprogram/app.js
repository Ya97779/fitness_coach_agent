App({
  globalData: {
    userInfo: null,
    token: null
  },

  onLaunch() {
    const token = wx.getStorageSync('token')
    if (token) {
      this.globalData.token = token
      this.loadUserInfo()
    }
  },

  loadUserInfo() {
    const { request } = require('./utils/request')
    request({ url: '/api/v1/user/me' }).then(user => {
      this.globalData.userInfo = user
    }).catch(() => {})
  }
})
