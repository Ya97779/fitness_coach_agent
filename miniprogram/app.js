const { login } = require('./utils/auth')

App({
  globalData: {
    userInfo: null,
    token: null
  },

  onLaunch() {
    this.autoLogin()
  },

  autoLogin() {
    const token = wx.getStorageSync('token')
    if (token) {
      this.globalData.token = token
      this.loadUserInfo()
      return
    }
    login().then(({ token, user }) => {
      this.globalData.token = token
      this.globalData.userInfo = user
    }).catch(err => {
      console.error('自动登录失败:', err)
    })
  },

  loadUserInfo() {
    const { request } = require('./utils/request')
    request({ url: '/api/v1/user/me' }).then(user => {
      this.globalData.userInfo = user
    }).catch(() => {})
  }
})
