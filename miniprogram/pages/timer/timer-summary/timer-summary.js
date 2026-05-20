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

    const { exercises, duration } = this.data
    const perExerciseDuration = Math.max(1, Math.round((duration || 1) / exercises.length))

    wx.showLoading({ title: '保存中...' })
    const requests = exercises.map(ex => {
      const data = {
        type: ex.name || '力量训练',
        name: ex.name,
        sets: ex.sets,
        duration: perExerciseDuration
      }
      if (ex.weight) data.weight = ex.weight
      return request({ url: '/api/v1/exercise-log', method: 'POST', data })
    })

    Promise.allSettled(requests).then(results => {
      wx.hideLoading()
      const failed = results.filter(r => r.status === 'rejected')
      if (failed.length === 0) {
        wx.showToast({ title: '保存成功', icon: 'success' })
      } else if (failed.length < results.length) {
        wx.showToast({ title: `部分保存失败（${failed.length}/${results.length}）`, icon: 'none' })
      } else {
        wx.showToast({ title: '保存失败', icon: 'none' })
      }
      if (failed.length < results.length) {
        this.setData({ saved: true })
      }
    })
  },

  trainAgain() {
    wx.redirectTo({ url: '/pages/timer/timer-setup/timer-setup' })
  },

  goHome() {
    wx.switchTab({ url: '/pages/home/home' })
  }
})
