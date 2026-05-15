const TEMPLATES = [
  {
    id: 'chest', icon: '💪', name: '胸部训练',
    exercises: [
      { name: '杠铃卧推', sets: 4, rest: 90 },
      { name: '上斜哑铃卧推', sets: 3, rest: 75 },
      { name: '龙门架夹胸', sets: 3, rest: 60 },
      { name: '双杠臂屈伸', sets: 3, rest: 90 }
    ]
  },
  {
    id: 'back', icon: '🔙', name: '背部训练',
    exercises: [
      { name: '引体向上', sets: 4, rest: 90 },
      { name: '杠铃划船', sets: 4, rest: 90 },
      { name: '高位下拉', sets: 3, rest: 60 },
      { name: '坐姿划船', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'shoulder', icon: '🏋️', name: '肩部训练',
    exercises: [
      { name: '站姿推举', sets: 4, rest: 90 },
      { name: '侧平举', sets: 3, rest: 60 },
      { name: '俯身飞鸟', sets: 3, rest: 60 },
      { name: '面拉', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'arms', icon: '💪', name: '手臂训练',
    exercises: [
      { name: '杠铃弯举', sets: 3, rest: 60 },
      { name: '锤式弯举', sets: 3, rest: 60 },
      { name: '窄距卧推', sets: 3, rest: 75 },
      { name: '绳索下压', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'legs', icon: '🦵', name: '腿部训练',
    exercises: [
      { name: '深蹲', sets: 5, rest: 120 },
      { name: '腿举', sets: 4, rest: 90 },
      { name: '腿屈伸', sets: 3, rest: 60 },
      { name: '腿弯举', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'core', icon: '🎯', name: '核心训练',
    exercises: [
      { name: '卷腹', sets: 3, rest: 45 },
      { name: '平板支撑', sets: 3, rest: 45 },
      { name: '俄罗斯转体', sets: 3, rest: 45 },
      { name: '悬垂举腿', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'cardio', icon: '🏃', name: '有氧减脂',
    exercises: [
      { name: '波比跳', sets: 4, rest: 60 },
      { name: '开合跳', sets: 4, rest: 45 },
      { name: '高抬腿', sets: 4, rest: 45 },
      { name: '登山跑', sets: 4, rest: 45 }
    ]
  }
]

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
