const { request } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')

Page({
  data: {
    activeTab: 'food',
    loggedIn: false,
    // 饮食
    foodName: '',
    mealType: '',
    // 份量选择
    portionQty: 1,
    portionUnit: '份',
    portions: ['克', '份', '碗', '个', '杯', '盘', '块', '片', '条'],
    // 运动
    exerciseType: '',
    exerciseDuration: '',
    exercisePresets: ['跑步', '游泳', '骑行', '跳绳', '力量训练', 'HIIT', '瑜伽', '快走']
  },

  onLoad() {
    if (!isLoggedIn()) {
      showLoginPrompt().then(loggedIn => {
        if (loggedIn) this.setData({ loggedIn: true })
      })
      return
    }
    this.setData({ loggedIn: true })
  },

  switchTab(e) {
    this.setData({ activeTab: e.currentTarget.dataset.tab })
  },

  onFoodNameInput(e) { this.setData({ foodName: e.detail.value }) },
  selectMeal(e) { this.setData({ mealType: e.currentTarget.dataset.meal }) },

  // 份量步进
  adjustQty(e) {
    const delta = parseFloat(e.currentTarget.dataset.delta)
    const isGram = this.data.portionUnit === '克'
    const step = isGram ? 50 : 1
    let qty = this.data.portionQty + delta * step
    if (isGram) {
      qty = Math.max(50, Math.round(qty / 50) * 50)
    } else {
      qty = Math.max(1, Math.round(qty))
    }
    this.setData({ portionQty: qty })
  },

  onQtyInput(e) {
    let val = parseFloat(e.detail.value)
    if (!val || val <= 0) val = this.data.portionUnit === '克' ? 100 : 1
    this.setData({ portionQty: val })
  },

  // 单位选择
  bindUnitChange(e) {
    const unit = this.data.portions[e.detail.value]
    const prevUnit = this.data.portionUnit
    let qty = this.data.portionQty
    // 切换到克：默认 100；切出克：默认 1
    if (unit === '克' && prevUnit !== '克') {
      qty = 100
    } else if (unit !== '克' && prevUnit === '克') {
      qty = 1
    }
    this.setData({ portionUnit: unit, portionQty: qty })
  },

  onExerciseTypeInput(e) { this.setData({ exerciseType: e.detail.value }) },
  onDurationInput(e) { this.setData({ exerciseDuration: e.detail.value }) },
  selectExercise(e) { this.setData({ exerciseType: e.currentTarget.dataset.type }) },

  submitFood() {
    const { foodName, mealType, portionQty, portionUnit } = this.data
    if (!foodName || !mealType) return

    const data = {
      name: foodName,
      meal_type: mealType,
      portion_qty: portionQty,
      portion_unit: portionUnit
    }

    wx.showLoading({ title: '记录中...' })
    request({ url: '/api/v1/food-log', method: 'POST', data }).then((res) => {
      wx.hideLoading()
      if (res.estimating) {
        wx.showToast({ title: 'AI 正在估算热量，稍后刷新查看', icon: 'none', duration: 2500 })
      } else {
        wx.showToast({ title: '记录成功', icon: 'success' })
      }
      this.setData({ foodName: '', portionQty: 1, portionUnit: '份', mealType: '' })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '记录失败', icon: 'none' })
    })
  },

  submitExercise() {
    const { exerciseType, exerciseDuration } = this.data
    if (!exerciseType || !exerciseDuration) return

    wx.showLoading({ title: '记录中...' })
    request({
      url: '/api/v1/exercise-log',
      method: 'POST',
      data: { type: exerciseType, duration: parseInt(exerciseDuration) }
    }).then(() => {
      wx.hideLoading()
      wx.showToast({ title: '记录成功', icon: 'success' })
      this.setData({ exerciseType: '', exerciseDuration: '' })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '记录失败', icon: 'none' })
    })
  },

  handleLogin() {
    showLoginPrompt().then(loggedIn => {
      if (loggedIn) {
        this.setData({ loggedIn: true })
      }
    })
  }
})
