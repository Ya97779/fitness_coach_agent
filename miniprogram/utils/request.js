const { API_BASE_URL } = require('./config')

let loginModalShowing = false

function request(options) {
  return new Promise((resolve, reject) => {
    const token = wx.getStorageSync('token')
    const header = {
      'Content-Type': 'application/json',
      ...(options.header || {})
    }
    if (token) {
      header['Authorization'] = `Bearer ${token}`
    }

    wx.request({
      url: `${API_BASE_URL}${options.url}`,
      method: options.method || 'GET',
      data: options.data,
      header,
      timeout: 10000,
      success(res) {
        if (res.statusCode === 200) {
          resolve(res.data)
        } else if (res.statusCode === 401) {
          wx.removeStorageSync('token')
          const app = getApp()
          if (app) { app.globalData.token = null }
          if (!loginModalShowing) {
            loginModalShowing = true
            const { login } = require('./auth')
            wx.showModal({
              title: '登录已过期',
              content: '是否重新登录？',
              confirmText: '重新登录',
              cancelText: '取消',
              success(modalRes) {
                if (modalRes.confirm) {
                  login().then(() => {
                    wx.showToast({ title: '登录成功', icon: 'success' })
                  }).catch(() => {
                    wx.showToast({ title: '登录失败', icon: 'none' })
                  })
                }
              },
              complete() { loginModalShowing = false }
            })
          }
          reject(new Error('登录已过期，请重新登录'))
        } else {
          const msg = (res.data && res.data.message) || '请求失败'
          reject(new Error(msg))
        }
      },
      fail(err) {
        if ((err.errMsg || '').includes('timeout')) {
          reject(new Error('请求超时，请检查网络'))
        } else {
          reject(new Error(err.errMsg || '网络错误'))
        }
      }
    })
  })
}

function streamRequest(options, onChunk, onDone, onError) {
  const token = wx.getStorageSync('token')
  const header = {
    'Content-Type': 'application/json',
    ...(options.header || {})
  }
  if (token) {
    header['Authorization'] = `Bearer ${token}`
  }

  const requestTask = wx.request({
    url: `${API_BASE_URL}${options.url}`,
    method: 'POST',
    data: options.data,
    header,
    enableChunked: true,
    timeout: 120000,
    success(res) {
      if (onDone) onDone(res.data)
    },
    fail(err) {
      if (onError) onError(new Error(err.errMsg || '网络错误'))
    }
  })

  requestTask.onChunkReceived(function(response) {
    try {
      const text = decodeChunk(response.data)
      if (onChunk) onChunk(text)
    } catch (e) {
      if (onError) onError(e)
    }
  })

  return requestTask
}

function decodeChunk(buffer) {
  if (typeof buffer === 'string') return buffer
  const uint8 = new Uint8Array(buffer)
  let result = ''
  for (let i = 0; i < uint8.length; i++) {
    result += String.fromCharCode(uint8[i])
  }
  try {
    return decodeURIComponent(escape(result))
  } catch (e) {
    return result
  }
}

module.exports = { request, streamRequest }
