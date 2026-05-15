Page({
  data: {
    exercises: [],
    currentExIndex: 0,
    currentEx: {},
    currentSet: 1,
    nextEx: null,
    isResting: false,
    restRemaining: 0,
    overallProgress: 0,
    // 训练统计
    startTime: 0,
    completedSets: 0
  },

  restTimer: null,
  totalSets: 0,

  onLoad(options) {
    const eventChannel = this.getOpenerEventChannel()
    eventChannel.on('trainingPlan', (data) => {
      const exercises = data.exercises
      const totalSets = exercises.reduce((sum, ex) => sum + ex.sets, 0)
      this.totalSets = totalSets
      this.setData({
        exercises,
        currentEx: exercises[0],
        currentSet: 1,
        nextEx: exercises.length > 1 ? exercises[1] : null,
        startTime: Date.now(),
        overallProgress: 0
      })
    })
  },

  onUnload() {
    this.clearRestTimer()
  },

  finishSet() {
    const { exercises, currentExIndex, currentSet, currentEx, completedSets } = this.data
    const newCompleted = completedSets + 1

    if (currentSet < currentEx.sets) {
      // 还有剩余组，开始休息
      this.setData({
        isResting: true,
        restRemaining: currentEx.rest,
        completedSets: newCompleted
      })
      this.drawRestRing(currentEx.rest)
      this.startRestTimer(currentEx.rest)
    } else {
      // 当前动作完成，跳到下一个
      this.setData({ completedSets: newCompleted })
      this.goToNextExercise()
    }
  },

  startRestTimer(totalSeconds) {
    this.clearRestTimer()
    let remaining = totalSeconds
    this.restTimer = setInterval(() => {
      remaining--
      this.setData({ restRemaining: remaining })
      this.drawRestRing(totalSeconds, remaining)
      if (remaining <= 0) {
        this.clearRestTimer()
        this.endRest()
      }
    }, 1000)
  },

  clearRestTimer() {
    if (this.restTimer) {
      clearInterval(this.restTimer)
      this.restTimer = null
    }
  },

  endRest() {
    const { currentSet, currentEx } = this.data
    this.setData({
      isResting: false,
      currentSet: currentSet + 1
    })
    this.updateProgress()
    wx.vibrateShort({ type: 'heavy' })
  },

  skipRest() {
    this.clearRestTimer()
    this.endRest()
  },

  goToNextExercise() {
    const { exercises, currentExIndex } = this.data
    const nextIndex = currentExIndex + 1

    if (nextIndex >= exercises.length) {
      // 全部完成
      this.finishTraining()
      return
    }

    const nextNext = nextIndex + 1 < exercises.length ? exercises[nextIndex + 1] : null
    this.setData({
      currentExIndex: nextIndex,
      currentEx: exercises[nextIndex],
      currentSet: 1,
      nextEx: nextNext,
      isResting: false
    })
    this.updateProgress()
    wx.vibrateShort({ type: 'medium' })
  },

  prevExercise() {
    const { exercises, currentExIndex } = this.data
    if (currentExIndex <= 0) return
    const prevIndex = currentExIndex - 1
    this.setData({
      currentExIndex: prevIndex,
      currentEx: exercises[prevIndex],
      currentSet: 1,
      nextEx: exercises[prevIndex + 1] || null,
      isResting: false
    })
    this.updateProgress()
  },

  nextExercise() {
    this.goToNextExercise()
  },

  updateProgress() {
    const { completedSets } = this.data
    const progress = this.totalSets > 0 ? Math.round((completedSets / this.totalSets) * 100) : 0
    this.setData({ overallProgress: progress })
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

      // 背景环
      ctx.setLineWidth(lineWidth)
      ctx.setStrokeStyle('rgba(255,255,255,0.08)')
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI)
      ctx.stroke()

      // 进度环
      const progress = remaining / total
      ctx.setLineWidth(lineWidth)
      ctx.setStrokeStyle('rgba(255,255,255,0.6)')
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
          this.finishTraining()
        }
      }
    })
  },

  finishTraining() {
    this.clearRestTimer()
    const { exercises, startTime, completedSets } = this.data
    const duration = Math.round((Date.now() - startTime) / 1000 / 60) // 分钟

    wx.redirectTo({
      url: '/pages/timer/timer-summary/timer-summary',
      success: (res) => {
        res.eventChannel.emit('trainingResult', {
          exercises: exercises.map(ex => ({ name: ex.name, sets: ex.sets })),
          totalSets: this.totalSets,
          completedSets,
          duration,
          estimatedCalories: Math.round(duration * 6) // 粗略估算
        })
      }
    })
  }
})
