const { request } = require('./request')

function login() {
  return new Promise((resolve, reject) => {
    wx.login({
      success(res) {
        if (!res.code) {
          reject(new Error('wx.login 获取 code 失败'))
          return
        }
        request({
          url: '/api/v1/auth/wx-login',
          method: 'POST',
          data: { code: res.code }
        }).then(data => {
          wx.setStorageSync('token', data.token)
          const app = getApp()
          if (app) {
            app.globalData.token = data.token
            app.globalData.userInfo = data.user
          }
          resolve(data)
        }).catch(reject)
      },
      fail(err) {
        reject(new Error(err.errMsg || 'wx.login 调用失败'))
      }
    })
  })
}

function logout() {
  wx.removeStorageSync('token')
  const app = getApp()
  if (app) {
    app.globalData.token = null
    app.globalData.userInfo = null
  }
}

function getToken() {
  return wx.getStorageSync('token') || ''
}

function isLoggedIn() {
  return !!wx.getStorageSync('token')
}

module.exports = { login, logout, getToken, isLoggedIn }
