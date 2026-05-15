const { request } = require('../../../utils/request')

Page({
  data: {
    exercises: [],
    totalSets: 0,
    completedSets: 0,
    duration: 0,
    estimatedCalories: 0,
    saved: false
  },

  onLoad() {
    const eventChannel = this.getOpenerEventChannel()
    eventChannel.on('trainingResult', (data) => {
      this.setData({
        exercises: data.exercises,
        totalSets: data.totalSets,
        completedSets: data.completedSets,
        duration: data.duration,
        estimatedCalories: data.estimatedCalories
      })
    })
  },

  saveRecord() {
    if (this.data.saved) {
      wx.showToast({ title: '已保存过', icon: 'none' })
      return
    }

    const { duration } = this.data
    wx.showLoading({ title: '保存中...' })
    request({
      url: '/api/v1/exercise-log',
      method: 'POST',
      data: { type: '力量训练', duration: duration || 1 }
    }).then(() => {
      wx.hideLoading()
      wx.showToast({ title: '保存成功', icon: 'success' })
      this.setData({ saved: true })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '保存失败', icon: 'none' })
    })
  },

  trainAgain() {
    wx.redirectTo({ url: '/pages/timer/timer-setup/timer-setup' })
  },

  goHome() {
    wx.switchTab({ url: '/pages/home/home' })
  }
})
