const app = getApp()
const { request } = require('../../../utils/request')

Page({
  data: {
    exercises: [],
    currentExIndex: 0,
    currentEx: {},
    currentSet: 1,
    nextEx: null,
    isResting: false,
    restRemaining: 0,
    restTotal: 0,
    overallProgress: 0,
    elapsed: '00:00',
    completedSets: 0
  },

  onLoad(options) {
    const t = app.globalData.training
    if (t.active) {
      // 从悬浮组件进入，训练已在进行中
      this._syncFromGlobal()
      this._subscribe()
    } else {
      // 从 timer-setup 进入，通过 eventChannel 接收训练计划
      const eventChannel = this.getOpenerEventChannel()
      eventChannel.on('trainingPlan', (data) => {
        const exercises = data.exercises
        app.startTraining(exercises)
        const ex = exercises[0]
        const nextEx = exercises.length > 1 ? exercises[1] : null
        this.setData({
          exercises,
          currentEx: ex,
          currentSet: 1,
          nextEx,
          overallProgress: 0,
          elapsed: '00:00',
          completedSets: 0,
          isResting: false
        })
        this._subscribe()
      })
    }
  },

  onShow() {
    // 从其他页面返回时重新订阅
    if (app.globalData.training.active) {
      this._syncFromGlobal()
      this._subscribe()
    }
  },

  onHide() {
    this._unsubscribe()
  },

  onUnload() {
    this._unsubscribe()
  },

  _subscribe() {
    this._unsubscribe()
    this._onTick = (t) => this._handleTick(t)
    app.onTimerTick(this._onTick)
  },

  _unsubscribe() {
    if (this._onTick) {
      app.offTimerTick(this._onTick)
      this._onTick = null
    }
  },

  _handleTick(t) {
    const ex = t.exercises[t.currentExIndex]
    const nextEx = t.currentExIndex + 1 < t.exercises.length ? t.exercises[t.currentExIndex + 1] : null
    const mins = Math.floor(t.elapsedSeconds / 60)
    const secs = t.elapsedSeconds % 60
    const progress = t.totalSets > 0 ? Math.round((t.completedSets / t.totalSets) * 100) : 0

    const update = {
      currentExIndex: t.currentExIndex,
      currentEx: ex,
      currentSet: t.currentSet,
      nextEx,
      isResting: t.isResting,
      restRemaining: t.restRemaining,
      restTotal: t.restTotal,
      elapsed: `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`,
      overallProgress: progress,
      completedSets: t.completedSets
    }
    this.setData(update)

    if (t.isResting) {
      this.drawRestRing(t.restTotal, t.restRemaining)
    }

    // 休息结束后检测到全部完成
    if (t.finished) {
      t.finished = false
      this._unsubscribe()
      this._goToSummary()
    }
  },

  _syncFromGlobal() {
    const t = app.globalData.training
    const ex = t.exercises[t.currentExIndex]
    const nextEx = t.currentExIndex + 1 < t.exercises.length ? t.exercises[t.currentExIndex + 1] : null
    const mins = Math.floor(t.elapsedSeconds / 60)
    const secs = t.elapsedSeconds % 60
    const progress = t.totalSets > 0 ? Math.round((t.completedSets / t.totalSets) * 100) : 0

    this.setData({
      exercises: t.exercises,
      currentExIndex: t.currentExIndex,
      currentEx: ex,
      currentSet: t.currentSet,
      nextEx,
      isResting: t.isResting,
      restRemaining: t.restRemaining,
      restTotal: t.restTotal,
      elapsed: `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`,
      overallProgress: progress,
      completedSets: t.completedSets
    })
  },

  finishSet() {
    if (app.completeSet()) {
      this._unsubscribe()
      this._goToSummary()
    }
  },

  skipRest() {
    if (app.skipRest()) {
      this._unsubscribe()
      this._goToSummary()
    }
  },

  prevExercise() {
    const t = app.globalData.training
    if (t.currentExIndex <= 0) return
    t.currentExIndex--
    t.currentSet = 1
    t.isResting = false
    t.restRemaining = 0
    t.restTotal = 0
    this._syncFromGlobal()
  },

  nextExercise() {
    const t = app.globalData.training
    if (t.currentExIndex + 1 >= t.exercises.length) return
    t.currentExIndex++
    t.currentSet = 1
    t.isResting = false
    t.restRemaining = 0
    t.restTotal = 0
    wx.vibrateShort({ type: 'medium' })
    this._syncFromGlobal()
  },

  drawRestRing(total, remaining) {
    if (remaining === undefined) remaining = total
    const query = wx.createSelectorQuery()
    query.select('#restRing').boundingClientRect()
    query.exec(res => {
      if (!res || !res[0]) return
      const { width, height } = res[0]
      const ctx = wx.createCanvasContext('restRing', this)
      const cx = width / 2
      const cy = height / 2
      const radius = Math.min(cx, cy) - 10
      const lineWidth = 6

      ctx.setLineWidth(lineWidth)
      ctx.setStrokeStyle('#e8e8e8')
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI)
      ctx.stroke()

      const progress = remaining / total
      ctx.setLineWidth(lineWidth)
      ctx.setStrokeStyle('#1a1a1a')
      ctx.setLineCap('butt')
      ctx.beginPath()
      ctx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + progress * 2 * Math.PI)
      ctx.stroke()

      ctx.draw()
    })
  },

  endTraining() {
    wx.showModal({
      title: '结束训练',
      content: '确定要结束当前训练吗？',
      success: res => {
        if (res.confirm) {
          this._unsubscribe()
          this._goToSummary()
        }
      }
    })
  },

  _goToSummary() {
    const result = app.finishTraining()

    // 构造热量估算请求
    const exercises = result.exercises.map(ex => ({
      name: ex.name,
      sets: ex.sets,
      weight: ex.weight || 0,
      duration: Math.max(1, Math.round((result.durationSeconds / 60) / result.exercises.length))
    }))

    request({
      url: '/api/v1/estimate-calories',
      method: 'POST',
      data: { exercises }
    }).then(res => {
      result.estimatedCalories = res.total_calories
      result.calorieDetails = res.details
      app.globalData.trainingResult = result
      wx.redirectTo({ url: '/pages/timer/timer-summary/timer-summary' })
    }).catch(() => {
      // 接口失败时使用原有公式
      app.globalData.trainingResult = result
      wx.redirectTo({ url: '/pages/timer/timer-summary/timer-summary' })
    })
  },

  onShareAppMessage() {
    return {
      title: '健身助手Agent - 正在训练中',
      path: '/pages/timer/timer-setup/timer-setup'
    }
  },

  onShareTimeline() {
    return {
      title: '健身助手Agent - 训练计时器'
    }
  }
})
