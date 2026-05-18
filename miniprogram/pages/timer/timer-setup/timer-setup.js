const { TEMPLATES } = require('../../../data/templates')

Page({
  data: {
    templates: TEMPLATES,
    activeTemplate: '',
    defaultRest: 60,
    exercises: [],
    totalSets: 0
  },

  onLoad() {
    // 加载上次的计划
    const saved = wx.getStorageSync('training_plan')
    if (saved) {
      this.setData({
        exercises: saved.exercises || [],
        defaultRest: saved.defaultRest || 60
      })
      this.calcTotalSets()
    }
  },

  useTemplate(e) {
    const id = e.currentTarget.dataset.id
    const tpl = TEMPLATES.find(t => t.id === id)
    if (!tpl) return
    this.setData({
      exercises: tpl.exercises.map(ex => ({ ...ex })),
      activeTemplate: id
    })
    this.calcTotalSets()
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
  },

  deleteExercise(e) {
    const index = parseInt(e.currentTarget.dataset.index)
    const exercises = this.data.exercises.filter((_, i) => i !== index)
    this.setData({ exercises, activeTemplate: '' })
    this.calcTotalSets()
  },

  addExercise() {
    const exercises = [...this.data.exercises, { name: '', sets: 3, rest: this.data.defaultRest }]
    this.setData({ exercises, activeTemplate: '' })
    this.calcTotalSets()
  },

  calcTotalSets() {
    const total = this.data.exercises.reduce((sum, ex) => sum + ex.sets, 0)
    this.setData({ totalSets: total })
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

    // 保存计划
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
