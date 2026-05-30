App({
  globalData: {
    userInfo: null,
    token: null,
    appLaunched: true,  // 首次进入标记，chat 页面用：首次滚底部，切换 tab 不滚
    chatStream: {
      active: false,
      requestTask: null,
      messages: [],
      pendingContent: '',
      aiMsgId: ''
    },
    training: {
      active: false,
      elapsedSeconds: 0,
      exercises: [],
      currentExIndex: 0,
      currentSet: 1,
      totalSets: 0,
      completedSets: 0,
      isResting: false,
      restRemaining: 0,
      restTotal: 0,
      finished: false
    }
  },

  _tickTimer: null,
  _tickCallbacks: [],

  onLaunch() {
    const token = wx.getStorageSync('token')
    if (token) {
      this.globalData.token = token
      this.loadUserInfo()
    }
  },

  onPageNotFound(res) {
    wx.reLaunch({ url: '/pages/home/home' })
  },

  loadUserInfo() {
    const { request } = require('./utils/request')
    request({ url: '/api/v1/user/me' }).then(user => {
      this.globalData.userInfo = user
    }).catch(() => {})
  },

  // ========== 训练计时器 ==========

  startTraining(exercises) {
    const totalSets = exercises.reduce((sum, ex) => sum + ex.sets, 0)
    const t = this.globalData.training
    t.active = true
    t.elapsedSeconds = 0
    t.exercises = exercises
    t.currentExIndex = 0
    t.currentSet = 1
    t.totalSets = totalSets
    t.completedSets = 0
    t.isResting = false
    t.restRemaining = 0
    t.restTotal = 0
    t.finished = false
    this._startTick()
  },

  completeSet() {
    const t = this.globalData.training
    if (!t.active) return false
    const ex = t.exercises[t.currentExIndex]
    t.completedSets++

    if (t.currentSet < ex.sets) {
      // 还有剩余组，开始休息
      t.isResting = true
      t.restRemaining = ex.rest
      t.restTotal = ex.rest
      return false
    } else if (t.currentExIndex >= t.exercises.length - 1) {
      // 最后一个动作的最后一组，训练完成
      return true
    } else {
      // 当前动作完成，跳到下一个
      t.isResting = false
      t.restRemaining = 0
      t.restTotal = 0
      this._goToNextExercise()
      return false
    }
  },

  skipRest() {
    const t = this.globalData.training
    if (!t.active || !t.isResting) return false
    t.isResting = false
    t.restRemaining = 0
    t.restTotal = 0
    t.currentSet++
    // 检查是否全部完成（休息结束后的组数可能超出）
    if (t.currentExIndex >= t.exercises.length - 1 && t.currentSet > t.exercises[t.currentExIndex].sets) {
      return true
    }
    return false
  },

  finishTraining() {
    const t = this.globalData.training
    this._stopTick()
    const durationMin = t.elapsedSeconds / 60
    const result = {
      exercises: t.exercises.map(ex => ({ name: ex.name, sets: ex.sets, weight: ex.weight || 0 })),
      totalSets: t.totalSets,
      completedSets: t.completedSets,
      durationSeconds: t.elapsedSeconds,
      estimatedCalories: Math.round(durationMin * 6)
    }
    t.active = false
    t.elapsedSeconds = 0
    t.exercises = []
    t.isResting = false
    t.restRemaining = 0
    t.restTotal = 0
    this._tickCallbacks.forEach(cb => cb(t))
    return result
  },

  onTimerTick(callback) {
    this._tickCallbacks.push(callback)
  },

  offTimerTick(callback) {
    this._tickCallbacks = this._tickCallbacks.filter(cb => cb !== callback)
  },

  _startTick() {
    this._stopTick()
    this._tickTimer = setInterval(() => {
      const t = this.globalData.training
      if (!t.active) return
      t.elapsedSeconds++
      if (t.isResting) {
        t.restRemaining--
        if (t.restRemaining <= 0) {
          t.isResting = false
          t.restRemaining = 0
          t.restTotal = 0
          t.currentSet++
          // 休息结束后检查是否全部完成
          const ex = t.exercises[t.currentExIndex]
          if (t.currentExIndex >= t.exercises.length - 1 && t.currentSet > ex.sets) {
            t.finished = true
          }
          wx.vibrateShort({ type: 'heavy' })
        }
      }
      this._tickCallbacks.forEach(cb => cb(t))
    }, 1000)
  },

  _stopTick() {
    if (this._tickTimer) {
      clearInterval(this._tickTimer)
      this._tickTimer = null
    }
  },

  _goToNextExercise() {
    const t = this.globalData.training
    if (t.currentExIndex + 1 >= t.exercises.length) {
      // 全部完成，不做自动跳转，由页面处理
      return
    }
    t.currentExIndex++
    t.currentSet = 1
    wx.vibrateShort({ type: 'medium' })
  }
})
