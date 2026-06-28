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

function showLoginPrompt() {
  return new Promise((resolve) => {
    wx.showModal({
      title: '登录提示',
      content: '登录后可使用知识库查询、记录训练、查看统计等功能',
      confirmText: '立即登录',
      cancelText: '稍后再说',
      success(res) {
        if (res.confirm) {
          login().then((data) => {
            // 登录成功，检查是否需要设置头像昵称
            const user = data.user
            if (!user.nickname) {
              wx.navigateTo({ url: '/pages/profile-setup/profile-setup' })
            }
            resolve(true)
          }).catch(() => {
            wx.showToast({ title: '登录失败，请重试', icon: 'none' })
            resolve(false)
          })
        } else {
          resolve(false)
        }
      }
    })
  })
}

module.exports = { login, logout, getToken, isLoggedIn, showLoginPrompt }
