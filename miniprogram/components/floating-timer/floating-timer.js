const app = getApp()

Component({
  data: {
    visible: false,
    exName: '',
    setInfo: '',
    isResting: false,
    restRemaining: 0,
    restTotal: 0,
    progress: 0
  },

  lifetimes: {
    attached() {
      this._syncState()
      this._onTick = () => this._syncState()
      app.onTimerTick(this._onTick)
    },
    detached() {
      if (this._onTick) {
        app.offTimerTick(this._onTick)
      }
    }
  },

  methods: {
    _syncState() {
      const t = app.globalData.training
      if (!t.active) {
        if (this.data.visible) this.setData({ visible: false })
        return
      }
      const ex = t.exercises[t.currentExIndex]
      const progress = t.totalSets > 0 ? Math.round((t.completedSets / t.totalSets) * 100) : 0

      this.setData({
        visible: true,
        exName: ex ? ex.name : '',
        setInfo: ex ? `${t.currentSet}/${ex.sets}` : '',
        isResting: t.isResting,
        restRemaining: t.restRemaining,
        restTotal: t.restTotal,
        progress
      })
    },

    onTap() {
      wx.navigateTo({ url: '/pages/timer/timer-training/timer-training' })
    },

    onSkipRest() {
      app.skipRest()
    }
  }
})
