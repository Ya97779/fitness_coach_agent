const { request } = require('../../../utils/request')
const app = getApp()

Page({
  data: {
    exercises: [],
    totalSets: 0,
    completedSets: 0,
    durationSeconds: 0,
    durationText: '00:00',
    estimatedCalories: 0,
    calorieDetails: [],
    saved: false
  },

  onLoad() {
    const data = app.globalData.trainingResult
    if (data) {
      const secs = data.durationSeconds || 0
      const mins = Math.floor(secs / 60)
      const remainSecs = secs % 60
      this.setData({
        exercises: data.exercises,
        totalSets: data.totalSets,
        completedSets: data.completedSets,
        durationSeconds: secs,
        durationText: `${String(mins).padStart(2, '0')}:${String(remainSecs).padStart(2, '0')}`,
        estimatedCalories: data.estimatedCalories,
        calorieDetails: data.calorieDetails || []
      })
      app.globalData.trainingResult = null
    }
  },

  saveRecord() {
    if (this.data.saved) {
      wx.showToast({ title: '已保存过', icon: 'none' })
      return
    }

    const { exercises, durationSeconds, calorieDetails } = this.data
    const durationMin = Math.max(1, Math.round(durationSeconds / 60))
    const perExerciseDuration = Math.max(1, Math.round(durationMin / exercises.length))

    // 构建动作热量映射
    const calMap = {}
    if (calorieDetails) {
      calorieDetails.forEach(d => { calMap[d.name] = d.calories })
    }

    // API 未返回详情时，按组数比例分配总热量
    const hasDetails = Object.keys(calMap).length > 0
    const totalSets = exercises.reduce((s, ex) => s + (ex.sets || 1), 0) || 1

    wx.showLoading({ title: '保存中...' })
    const requests = exercises.map(ex => {
      let cal = calMap[ex.name] || 0
      // API 未返回详情时，按组数比例分配总热量
      if (!hasDetails && this.data.estimatedCalories) {
        cal = Math.round(this.data.estimatedCalories * (ex.sets || 1) / totalSets)
      }
      const data = {
        type: ex.name || '力量训练',
        name: ex.name,
        sets: ex.sets,
        duration: perExerciseDuration,
        calories: cal
      }
      if (ex.weight) data.weight = ex.weight
      return request({ url: '/api/v1/exercise-log', method: 'POST', data })
    })

    Promise.all(requests).then(() => {
      wx.hideLoading()
      wx.showToast({ title: '保存成功', icon: 'success' })
      this.setData({ saved: true })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '保存失败', icon: 'none' })
    })
  },

  trainAgain() {
    wx.switchTab({ url: '/pages/timer/timer-setup/timer-setup' })
  },

  goHome() {
    wx.switchTab({ url: '/pages/home/home' })
  },

  onShareAppMessage() {
    return {
      title: '健身助手Agent - 完成训练！',
      path: '/pages/timer/timer-setup/timer-setup'
    }
  },

  onShareTimeline() {
    return {
      title: '健身助手Agent - 完成训练！'
    }
  }
})
