const { request } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')

const MEAL_TYPE_MAP = {
  breakfast: '早餐',
  lunch: '午餐',
  dinner: '晚餐',
  snack: '加餐'
}

Page({
  data: {
    greeting: '',
    todayStr: '',
    intake: 0,
    burn: 0,
    remaining: 0,
    tdee: null,
    hasTdee: false,
    foodItems: [],
    exerciseItems: [],
    loading: true,
    loggedIn: false,
    // 编辑弹窗
    editModalVisible: false,
    editType: '', // 'food' or 'exercise'
    editItem: null,
    editForm: {},
    // 左滑
    swipeIndex: -1,
    touchStartX: 0
  },

  onShow() {
    this.setGreeting()
    if (!isLoggedIn()) {
      this.setData({ loggedIn: false, loading: false })
      return
    }
    this.setData({ loggedIn: true })
    this.loadData()
  },

  setGreeting() {
    const hour = new Date().getHours()
    let greeting = '晚上好'
    if (hour < 6) greeting = '夜深了'
    else if (hour < 12) greeting = '早上好'
    else if (hour < 14) greeting = '中午好'
    else if (hour < 18) greeting = '下午好'

    const now = new Date()
    const month = now.getMonth() + 1
    const day = now.getDate()
    const weekdays = ['日', '一', '二', '三', '四', '五', '六']
    const todayStr = `${month}月${day}日 周${weekdays[now.getDay()]}`

    this.setData({ greeting, todayStr })
  },

  loadData() {
    this.setData({ loading: true })
    Promise.all([
      request({ url: '/api/v1/user/me' }),
      request({ url: '/api/v1/user/me/today' })
    ]).then(([user, today]) => {
      const tdee = user.tdee || null
      const hasTdee = !!tdee
      const intake = today.intake_calories || 0
      const burn = today.burn_calories || 0
      const remaining = tdee ? Math.round(tdee - intake + burn) : 0

      const foodItems = (today.food_items || []).map(item => ({
        ...item,
        meal_type_text: MEAL_TYPE_MAP[item.meal_type] || item.meal_type
      }))
      const exerciseItems = today.exercise_items || []

      this.setData({
        tdee, hasTdee, intake: Math.round(intake), burn: Math.round(burn),
        remaining: remaining > 0 ? remaining : 0,
        foodItems, exerciseItems,
        loading: false
      })
      this.drawRing(intake, burn, tdee || 1)
    }).catch(() => {
      this.setData({ loading: false })
    })
  },

  drawRing(intake, burn, tdee) {
    const query = wx.createSelectorQuery()
    query.select('#calorieRing').boundingClientRect()
    query.exec(res => {
      if (!res || !res[0]) return
      const { width, height } = res[0]
      const ctx = wx.createCanvasContext('calorieRing', this)
      const cx = width / 2
      const cy = height / 2
      const radius = Math.min(cx, cy) - 12
      const lineWidth = 14

      // 背景环
      ctx.setLineWidth(lineWidth)
      ctx.setStrokeStyle('#e8e8e8')
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI)
      ctx.stroke()

      // 摄入环
      const intakeAngle = Math.min(intake / tdee, 1) * 2 * Math.PI
      if (intakeAngle > 0) {
        ctx.setLineWidth(lineWidth)
        ctx.setStrokeStyle('#1a1a1a')
        ctx.setLineCap('butt')
        ctx.beginPath()
        ctx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + intakeAngle)
        ctx.stroke()
      }

      // 消耗环（外圈）
      const burnAngle = Math.min(burn / tdee, 1) * 2 * Math.PI
      if (burnAngle > 0) {
        const outerRadius = radius + lineWidth + 4
        ctx.setLineWidth(6)
        ctx.setStrokeStyle('#999')
        ctx.setLineCap('butt')
        ctx.beginPath()
        ctx.arc(cx, cy, outerRadius, -Math.PI / 2, -Math.PI / 2 + burnAngle)
        ctx.stroke()
      }

      ctx.draw()
    })
  },

  goTimer() {
    wx.switchTab({ url: '/pages/timer/timer-setup/timer-setup' })
  },

  goLog() {
    wx.navigateTo({ url: '/pages/log/log' })
  },

  goStats() {
    wx.navigateTo({ url: '/pages/stats/stats' })
  },

  goFeedback() {
    wx.navigateTo({ url: '/pages/feedback/feedback' })
  },

  handleLogin() {
    showLoginPrompt().then(loggedIn => {
      if (loggedIn) {
        this.setData({ loggedIn: true })
        this.loadData()
      }
    })
  },

  // === 编辑 ===
  onItemTap(e) {
    const { type, item } = e.currentTarget.dataset
    const editForm = type === 'food'
      ? { name: item.name, calories: item.calories, meal_type: item.meal_type }
      : { type: item.type, name: item.name || '', sets: item.sets || 1, weight: item.weight || '', duration: item.duration, calories: item.calories }
    this.setData({ editModalVisible: true, editType: type, editItem: item, editForm, swipeIndex: -1 })
  },

  closeEditModal() {
    this.setData({ editModalVisible: false, editItem: null })
  },

  preventBubble() {},

  onEditInput(e) {
    const field = e.currentTarget.dataset.field
    let value = e.detail.value
    if (['calories', 'sets', 'weight', 'duration'].includes(field)) {
      value = parseFloat(value) || 0
    }
    this.setData({ [`editForm.${field}`]: value })
  },

  selectEditMeal(e) {
    this.setData({ 'editForm.meal_type': e.currentTarget.dataset.meal })
  },

  adjustEditSets(e) {
    const delta = parseInt(e.currentTarget.dataset.delta)
    let sets = (this.data.editForm.sets || 1) + delta
    if (sets < 1) sets = 1
    this.setData({ 'editForm.sets': sets })
  },

  saveEdit() {
    const { editType, editItem, editForm } = this.data
    const url = editType === 'food'
      ? `/api/v1/food-log/${editItem.id}`
      : `/api/v1/exercise-log/${editItem.id}`

    wx.showLoading({ title: '保存中...' })
    request({ url, method: 'PATCH', data: editForm }).then(() => {
      wx.hideLoading()
      wx.showToast({ title: '已更新', icon: 'success' })
      this.closeEditModal()
      this.loadData()
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '更新失败', icon: 'none' })
    })
  },

  deleteFromEdit() {
    const { editType, editItem } = this.data
    wx.showModal({
      title: '确认删除',
      content: '删除后不可恢复',
      confirmColor: '#c47a6c',
      success: (res) => {
        if (res.confirm) {
          const url = editType === 'food'
            ? `/api/v1/food-log/${editItem.id}`
            : `/api/v1/exercise-log/${editItem.id}`
          request({ url, method: 'DELETE' }).then(() => {
            wx.showToast({ title: '已删除', icon: 'success' })
            this.closeEditModal()
            this.loadData()
          })
        }
      }
    })
  },

  // === 左滑删除 ===
  onItemTouchStart(e) {
    this.setData({ touchStartX: e.touches[0].clientX })
  },

  onItemTouchEnd(e) {
    const startX = this.data.touchStartX || 0
    const endX = e.changedTouches[0].clientX
    const { type, index } = e.currentTarget.dataset
    if (startX - endX > 60) {
      this.setData({ swipeIndex: `${type}-${index}` })
    } else {
      this.setData({ swipeIndex: -1 })
    }
  },

  resetSwipe() {
    this.setData({ swipeIndex: -1 })
  },

  deleteItem(e) {
    const { type, item } = e.currentTarget.dataset
    wx.showModal({
      title: '确认删除',
      content: '删除后不可恢复',
      confirmColor: '#c47a6c',
      success: (res) => {
        if (res.confirm) {
          const url = type === 'food'
            ? `/api/v1/food-log/${item.id}`
            : `/api/v1/exercise-log/${item.id}`
          request({ url, method: 'DELETE' }).then(() => {
            wx.showToast({ title: '已删除', icon: 'success' })
            this.loadData()
          })
        }
      }
    })
  }
})
