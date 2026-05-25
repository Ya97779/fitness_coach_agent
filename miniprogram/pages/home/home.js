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
    // 份量
    portionUnits: ['克', '份', '碗', '个', '杯', '盘', '块', '片', '条'],
    editUnitIndex: 1,
    // 左滑
    swipeIndex: -1,
    touchStartX: 0,
    _pollTimer: null
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

  onHide() {
    this._stopPoll()
  },

  onUnload() {
    this._stopPoll()
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
      const adj = user.calorie_adjustment || 0
      const intake = today.intake_calories || 0
      const burn = today.burn_calories || 0
      const target = tdee ? Math.round(tdee + adj + burn) : null
      const hasTdee = !!target

      const foodItems = (today.food_items || []).map(item => ({
        ...item,
        meal_type_text: MEAL_TYPE_MAP[item.meal_type] || item.meal_type
      }))
      const exerciseItems = today.exercise_items || []

      // 进度条和热量差
      const barPercent = target ? Math.min(Math.round(intake / target * 100), 100) : 0
      const gap = tdee ? Math.round(tdee + burn - intake) : 0
      const isDeficit = gap >= 0
      const gapText = isDeficit ? gap : '+' + Math.abs(gap)

      this.setData({
        tdee, target, hasTdee, intake: Math.round(intake), burn: Math.round(burn),
        calorie_adjustment: adj,
        barPercent, gapText, isDeficit,
        foodItems, exerciseItems,
        loading: false
      })

      // 有"计算中..."的食物时启动轮询
      const hasEstimating = foodItems.some(item => !item.calories || item.calories <= 0)
      if (hasEstimating) {
        this._startPoll()
      } else {
        this._stopPoll()
      }
    }).catch(() => {
      this.setData({ loading: false })
    })
  },

  _startPoll() {
    this._stopPoll()
    this._pollCount = 0
    this._pollTimer = setInterval(() => {
      this._pollCount++
      if (this._pollCount > 10) {
        this._stopPoll()
        return
      }
      request({ url: '/api/v1/user/me/today' }).then(today => {
        const foodItems = (today.food_items || []).map(item => ({
          ...item,
          meal_type_text: MEAL_TYPE_MAP[item.meal_type] || item.meal_type
        }))
        const hasEstimating = foodItems.some(item => !item.calories || item.calories <= 0)
        if (!hasEstimating || this._pollCount > 10) {
          this._stopPoll()
        }
        // 重新计算汇总数据
        const tdee = this.data.tdee
        const adj = this.data.calorie_adjustment || 0
        const intake = today.intake_calories || 0
        const burn = today.burn_calories || 0
        const target = tdee ? Math.round(tdee + adj + burn) : null
        const barPercent = target ? Math.min(Math.round(intake / target * 100), 100) : 0
        const gap = tdee ? Math.round(tdee + burn - intake) : 0
        const isDeficit = gap >= 0
        const gapText = isDeficit ? gap : '+' + Math.abs(gap)
        const exerciseItems = today.exercise_items || []
        this.setData({
          foodItems, exerciseItems,
          intake: Math.round(intake), burn: Math.round(burn),
          barPercent, gapText, isDeficit
        })
      }).catch(() => {})
    }, 3000)
  },

  _stopPoll() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer)
      this._pollTimer = null
    }
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
    let editForm
    if (type === 'food') {
      const unitIdx = this.data.portionUnits.indexOf(item.portion_unit)
      editForm = {
        name: item.name,
        calories: item.calories,
        meal_type: item.meal_type,
        portion_qty: item.portion_qty || 1,
        portion_unit: item.portion_unit || '份'
      }
      this.setData({ editUnitIndex: unitIdx >= 0 ? unitIdx : 1 })
    } else {
      editForm = { name: item.name || item.type || '', sets: item.sets || 1, reps: item.reps || '', weight: item.weight || '', duration: item.duration, calories: item.calories }
    }
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

  adjustEditReps(e) {
    const delta = parseInt(e.currentTarget.dataset.delta)
    let reps = (this.data.editForm.reps || 0) + delta
    if (reps < 0) reps = 0
    this.setData({ 'editForm.reps': reps })
  },

  adjustEditQty(e) {
    const delta = parseFloat(e.currentTarget.dataset.delta)
    const isGram = this.data.editForm.portion_unit === '克'
    const step = isGram ? 50 : 1
    let qty = (this.data.editForm.portion_qty || 1) + delta * step
    if (isGram) {
      qty = Math.max(50, Math.round(qty / 50) * 50)
    } else {
      qty = Math.max(1, Math.round(qty))
    }
    this.setData({ 'editForm.portion_qty': qty })
  },

  onEditUnitChange(e) {
    const idx = e.detail.value
    const unit = this.data.portionUnits[idx]
    const prevUnit = this.data.editForm.portion_unit
    let qty = this.data.editForm.portion_qty
    if (unit === '克' && prevUnit !== '克') {
      qty = 100
    } else if (unit !== '克' && prevUnit === '克') {
      qty = 1
    }
    this.setData({ editUnitIndex: idx, 'editForm.portion_unit': unit, 'editForm.portion_qty': qty })
  },

  saveEdit() {
    const { editType, editItem, editForm } = this.data
    const url = editType === 'food'
      ? `/api/v1/food-log/${editItem.id}`
      : `/api/v1/exercise-log/${editItem.id}`

    // 清理空字符串，避免后端 Pydantic 校验失败
    const cleaned = {}
    Object.keys(editForm).forEach(k => {
      const v = editForm[k]
      if (v !== '' && v !== null && v !== undefined) cleaned[k] = v
    })

    wx.showLoading({ title: '保存中...' })
    request({ url, method: 'PATCH', data: cleaned }).then(() => {
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
