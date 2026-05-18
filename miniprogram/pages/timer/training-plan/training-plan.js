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

function emptyWeek() {
  const days = {}
  DAYS.forEach(d => { days[d.key] = { exercises: [], isRest: false } })
  return days
}

Page({
  data: {
    days: DAYS,
    plan: emptyWeek(),
    todayKey: '',
    expandedDay: '',
    showTemplatePicker: false,
    templates: TEMPLATES
  },

  onLoad() {
    const dayIndex = new Date().getDay()
    const todayKey = DAYS[dayIndex === 0 ? 6 : dayIndex - 1].key
    this.setData({ todayKey })

    const saved = wx.getStorageSync('weekly_plan')
    if (saved && saved.days) {
      this.setData({ plan: saved.days })
    }
  },

  toggleExpand(e) {
    const key = e.currentTarget.dataset.key
    this.setData({ expandedDay: this.data.expandedDay === key ? '' : key })
  },

  toggleRest(e) {
    const key = e.currentTarget.dataset.key
    const plan = { ...this.data.plan }
    const day = { ...plan[key] }
    day.isRest = !day.isRest
    if (day.isRest) day.exercises = []
    plan[key] = day
    this.setData({ plan })
    this.save(plan)
  },

  applyTemplate(e) {
    const tplId = e.currentTarget.dataset.id
    const tpl = TEMPLATES.find(t => t.id === tplId)
    if (!tpl) return
    const key = this.data.expandedDay
    const plan = { ...this.data.plan }
    plan[key] = {
      exercises: tpl.exercises.map(ex => ({ ...ex })),
      isRest: false
    }
    this.setData({ plan, showTemplatePicker: false })
    this.save(plan)
  },

  openTemplatePicker() {
    this.setData({ showTemplatePicker: true })
  },

  closeTemplatePicker() {
    this.setData({ showTemplatePicker: false })
  },

  copyFromDay(e) {
    const index = parseInt(e.detail.value)
    const sourceKey = this.data.days[index].key
    const targetKey = this.data.expandedDay
    if (!sourceKey || !targetKey || sourceKey === targetKey) return
    const plan = { ...this.data.plan }
    plan[targetKey] = {
      exercises: plan[sourceKey].exercises.map(ex => ({ ...ex })),
      isRest: plan[sourceKey].isRest
    }
    this.setData({ plan })
    this.save(plan)
  },

  deleteExercise(e) {
    const { daykey, index } = e.currentTarget.dataset
    const plan = { ...this.data.plan }
    const day = { ...plan[daykey] }
    day.exercises = day.exercises.filter((_, i) => i !== index)
    plan[daykey] = day
    this.setData({ plan })
    this.save(plan)
  },

  addExerciseToDay() {
    const key = this.data.expandedDay
    const plan = { ...this.data.plan }
    const day = { ...plan[key] }
    day.exercises = [...day.exercises, { name: '', sets: 3, rest: 60 }]
    day.isRest = false
    plan[key] = day
    this.setData({ plan })
  },

  onExInput(e) {
    const { daykey, index, field } = e.currentTarget.dataset
    const value = e.detail.value
    const plan = { ...this.data.plan }
    const day = { ...plan[daykey] }
    const exercises = [...day.exercises]
    exercises[index] = { ...exercises[index] }
    exercises[index][field] = field === 'name' ? value : (parseInt(value) || 0)
    day.exercises = exercises
    plan[daykey] = day
    this.setData({ plan })
  },

  onExBlur() {
    this.save(this.data.plan)
  },

  adjustSets(e) {
    const { daykey, index, delta } = e.currentTarget.dataset
    const plan = { ...this.data.plan }
    const day = { ...plan[daykey] }
    const exercises = [...day.exercises]
    exercises[index] = { ...exercises[index] }
    exercises[index].sets = Math.max(1, Math.min(20, exercises[index].sets + parseInt(delta)))
    day.exercises = exercises
    plan[daykey] = day
    this.setData({ plan })
    this.save(plan)
  },

  save(plan) {
    wx.setStorageSync('weekly_plan', { days: plan })
  },

  getSummary(day) {
    if (day.isRest) return '休息日'
    if (!day.exercises || day.exercises.length === 0) return '未安排'
    const names = day.exercises.filter(ex => ex.name).map(ex => ex.name)
    if (names.length === 0) return '未安排'
    const totalSets = day.exercises.reduce((s, ex) => s + ex.sets, 0)
    return names.join('、') + ' · ' + totalSets + ' 组'
  },

  startTodayTraining() {
    const { plan, todayKey } = this.data
    const today = plan[todayKey]
    if (!today || today.isRest || !today.exercises || today.exercises.length === 0) {
      wx.showToast({ title: '今日无训练安排', icon: 'none' })
      return
    }
    if (today.exercises.some(ex => !ex.name)) {
      wx.showToast({ title: '请填写所有动作名称', icon: 'none' })
      return
    }
    wx.setStorageSync('training_plan', { exercises: today.exercises, defaultRest: 60 })
    wx.navigateTo({
      url: '/pages/timer/timer-training/timer-training',
      events: { trainingDone: () => {} },
      success: (res) => {
        res.eventChannel.emit('trainingPlan', { exercises: today.exercises, defaultRest: 60 })
      }
    })
  }
})
