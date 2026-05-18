const { TEMPLATES } = require('../../../data/templates')

const DAYS = [
  { key: 'mon', label: '周一' },
  { key: 'tue', label: '周二' },
  { key: 'wed', label: '周三' },
  { key: 'thu', label: '周四' },
  { key: 'fri', label: '周五' },
  { key: 'sat', label: '周六' },
  { key: 'sun', label: '周日' }
]

Page({
  data: {
    templates: TEMPLATES,
    activeTemplate: '',
    defaultRest: 60,
    exercises: [],
    totalSets: 0,
    todayLabel: '',
    isRestDay: false
  },

  onShow() {
    const dayIndex = new Date().getDay()
    const today = DAYS[dayIndex === 0 ? 6 : dayIndex - 1]
    this.setData({ todayLabel: today.label })

    // 优先从周计划加载今日训练
    const weeklyPlan = wx.getStorageSync('weekly_plan')
    if (weeklyPlan && weeklyPlan.days && weeklyPlan.days[today.key]) {
      const dayPlan = weeklyPlan.days[today.key]
      if (dayPlan.isRest) {
        this.setData({ exercises: [], isRestDay: true, activeTemplate: '' })
        this.calcTotalSets()
        return
      }
      if (dayPlan.exercises && dayPlan.exercises.length > 0) {
        this.setData({
          exercises: dayPlan.exercises.map(ex => ({ ...ex })),
          isRestDay: false,
          activeTemplate: ''
        })
        this.calcTotalSets()
        return
      }
    }

    // 回退到上次手动保存的计划
    const saved = wx.getStorageSync('training_plan')
    if (saved) {
      this.setData({
        exercises: saved.exercises || [],
        defaultRest: saved.defaultRest || 60,
        isRestDay: false
      })
      this.calcTotalSets()
    } else {
      this.setData({ isRestDay: false })
    }
  },

  useTemplate(e) {
    const id = e.currentTarget.dataset.id
    const tpl = TEMPLATES.find(t => t.id === id)
    if (!tpl) return
    this.setData({
      exercises: tpl.exercises.map(ex => ({ ...ex })),
      activeTemplate: id,
      isRestDay: false
    })
    this.calcTotalSets()
    this.syncToWeeklyPlan()
  },

  adjustDefaultRest(e) {
    const delta = parseInt(e.currentTarget.dataset.delta)
    const val = Math.max(15, Math.min(300, this.data.defaultRest + delta))
    this.setData({ defaultRest: val })
  },

  onExInput(e) {
    const { index, field } = e.currentTarget.dataset
    const value = e.detail.value
    const exercises = [...this.data.exercises]
    exercises[index][field] = field === 'name' ? value : (parseInt(value) || 0)
    this.setData({ exercises })
  },

  adjustSets(e) {
    const index = parseInt(e.currentTarget.dataset.index)
    const delta = parseInt(e.currentTarget.dataset.delta)
    const exercises = [...this.data.exercises]
    exercises[index].sets = Math.max(1, Math.min(20, exercises[index].sets + delta))
    this.setData({ exercises })
    this.calcTotalSets()
    this.syncToWeeklyPlan()
  },

  deleteExercise(e) {
    const index = parseInt(e.currentTarget.dataset.index)
    const exercises = this.data.exercises.filter((_, i) => i !== index)
    this.setData({ exercises, activeTemplate: '' })
    this.calcTotalSets()
    this.syncToWeeklyPlan()
  },

  addExercise() {
    const exercises = [...this.data.exercises, { name: '', sets: 3, rest: this.data.defaultRest }]
    this.setData({ exercises, activeTemplate: '', isRestDay: false })
    this.calcTotalSets()
    this.syncToWeeklyPlan()
  },

  calcTotalSets() {
    const total = this.data.exercises.reduce((sum, ex) => sum + ex.sets, 0)
    this.setData({ totalSets: total })
  },

  syncToWeeklyPlan() {
    const dayIndex = new Date().getDay()
    const todayKey = DAYS[dayIndex === 0 ? 6 : dayIndex - 1].key
    const weeklyPlan = wx.getStorageSync('weekly_plan') || { days: {} }
    if (!weeklyPlan.days) weeklyPlan.days = {}
    weeklyPlan.days[todayKey] = {
      exercises: this.data.exercises.map(ex => ({ ...ex })),
      isRest: false
    }
    wx.setStorageSync('weekly_plan', weeklyPlan)
  },

  saveToWeeklyPlan() {
    this.syncToWeeklyPlan()
    wx.showToast({ title: '已保存至' + this.data.todayLabel + '计划', icon: 'none' })
  },

  goPlan() {
    wx.navigateTo({ url: '/pages/timer/training-plan/training-plan' })
  },

  startTraining() {
    const { exercises, defaultRest } = this.data
    if (exercises.length === 0) return
    if (exercises.some(ex => !ex.name)) {
      wx.showToast({ title: '请填写所有动作名称', icon: 'none' })
      return
    }

    // 同步到周计划并保存
    this.syncToWeeklyPlan()
    wx.setStorageSync('training_plan', { exercises, defaultRest })

    wx.navigateTo({
      url: '/pages/timer/timer-training/timer-training',
      events: { trainingDone: () => {} },
      success: (res) => {
        res.eventChannel.emit('trainingPlan', { exercises, defaultRest })
      }
    })
  }
})
