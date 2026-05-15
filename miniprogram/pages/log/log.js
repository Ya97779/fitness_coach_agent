const { request } = require('../../utils/request')

Page({
  data: {
    activeTab: 'food',
    // 饮食
    foodName: '',
    foodCalories: '',
    mealType: '',
    // 运动
    exerciseType: '',
    exerciseDuration: '',
    exercisePresets: ['跑步', '游泳', '骑行', '跳绳', '力量训练', 'HIIT', '瑜伽', '快走']
  },

  switchTab(e) {
    this.setData({ activeTab: e.currentTarget.dataset.tab })
  },

  onFoodNameInput(e) { this.setData({ foodName: e.detail.value }) },
  onFoodCaloriesInput(e) { this.setData({ foodCalories: e.detail.value }) },
  selectMeal(e) { this.setData({ mealType: e.currentTarget.dataset.meal }) },
  onExerciseTypeInput(e) { this.setData({ exerciseType: e.detail.value }) },
  onDurationInput(e) { this.setData({ exerciseDuration: e.detail.value }) },
  selectExercise(e) { this.setData({ exerciseType: e.currentTarget.dataset.type }) },

  submitFood() {
    const { foodName, foodCalories, mealType } = this.data
    if (!foodName || !mealType) return

    const data = { name: foodName, meal_type: mealType }
    if (foodCalories) data.calories = parseFloat(foodCalories)

    wx.showLoading({ title: '记录中...' })
    request({ url: '/api/v1/food-log', method: 'POST', data }).then(() => {
      wx.hideLoading()
      wx.showToast({ title: '记录成功', icon: 'success' })
      this.setData({ foodName: '', foodCalories: '', mealType: '' })
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
  }
})
