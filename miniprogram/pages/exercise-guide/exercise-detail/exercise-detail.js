const { exerciseData } = require('../../../data/exercises')

const DIFF_MAP = {
  beginner: { text: '初级', color: '#666', bg: '#f0f0f0' },
  intermediate: { text: '中级', color: '#999', bg: '#f0f0f0' },
  advanced: { text: '高级', color: '#c47a6c', bg: 'rgba(196,122,108,0.08)' }
}

Page({
  data: {
    exercise: {},
    activeTab: 'steps',
    diffText: '',
    diffColor: '',
    diffBg: ''
  },

  onLoad(options) {
    const { id, group } = options
    const groupData = exerciseData[group]
    if (!groupData) return

    const exercise = groupData.exercises.find(ex => ex.id === id)
    if (!exercise) return

    const diff = DIFF_MAP[exercise.difficulty] || DIFF_MAP.beginner
    wx.setNavigationBarTitle({ title: exercise.name })
    this.setData({
      exercise,
      diffText: diff.text,
      diffColor: diff.color,
      diffBg: diff.bg
    })
  },

  switchTab(e) {
    this.setData({ activeTab: e.currentTarget.dataset.tab })
  },

  addToPlan() {
    const plan = wx.getStorageSync('training_plan') || { exercises: [], defaultRest: 60 }
    plan.exercises.push({
      name: this.data.exercise.name,
      sets: 3,
      rest: 60
    })
    wx.setStorageSync('training_plan', plan)
    wx.showToast({ title: '已加入训练计划', icon: 'success' })
  },

  askAI() {
    const name = this.data.exercise.name
    wx.switchTab({
      url: '/pages/chat/chat',
      success: () => {
        const app = getApp()
        if (app) {
          app.globalData.pendingMessage = `请详细讲解 ${name} 的动作要领和注意事项`
        }
      }
    })
  },

  goVariation(e) {
    const varId = e.currentTarget.dataset.id
    // 在当前肌群中查找变体
    const groupId = getCurrentPages().pop().__route__.split('?')[1] // fallback
    // 简单处理：遍历所有肌群找动作
    for (const [gId, gData] of Object.entries(exerciseData)) {
      const found = gData.exercises.find(ex => ex.id === varId)
      if (found) {
        wx.redirectTo({
          url: `/pages/exercise-guide/exercise-detail/exercise-detail?id=${varId}&group=${gId}`
        })
        return
      }
    }
    wx.showToast({ title: '暂无该变体详情', icon: 'none' })
  }
})
