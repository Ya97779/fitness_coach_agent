const { request } = require('../../utils/request')
const { logout: authLogout, isLoggedIn, showLoginPrompt } = require('../../utils/auth')

Page({
  data: {
    userInfo: {},
    loggedIn: false,
    showEditModal: false,
    editForm: {
      height: '', weight: '', age: '', gender: '男',
      target_weight: '', allergies: '', goal: ''
    },
    goalOptions: ['增肌', '减脂', '塑形', '改善体态', '提升体能', '保持健康']
  },

  onShow() {
    if (!isLoggedIn()) {
      this.setData({ loggedIn: false })
      return
    }
    this.setData({ loggedIn: true })
    this.loadProfile()
  },

  loadProfile() {
    request({ url: '/api/v1/user/me' }).then(user => {
      user.bmrRounded = user.bmr ? Math.round(user.bmr) : '--'
      user.tdeeRounded = user.tdee ? Math.round(user.tdee) : '--'
      this._calcBmi(user)
      this.setData({ userInfo: user })
    }).catch(() => {})
  },

  _calcBmi(user) {
    if (user.height && user.weight) {
      const h = user.height / 100
      const bmi = user.weight / (h * h)
      user.bmiVal = bmi.toFixed(1)
      if (bmi < 18.5) {
        user.bmiCategory = '偏瘦'
        user.bmiLevel = 'low'
      } else if (bmi < 24) {
        user.bmiCategory = '正常'
        user.bmiLevel = 'normal'
      } else if (bmi < 28) {
        user.bmiCategory = '超重'
        user.bmiLevel = 'high'
      } else {
        user.bmiCategory = '肥胖'
        user.bmiLevel = 'danger'
      }
    } else {
      user.bmiVal = '--'
      user.bmiCategory = ''
      user.bmiLevel = ''
    }
  },

  showEdit() {
    const { userInfo } = this.data
    this.setData({
      showEditModal: true,
      editForm: {
        height: userInfo.height ? String(userInfo.height) : '',
        weight: userInfo.weight ? String(userInfo.weight) : '',
        age: userInfo.age ? String(userInfo.age) : '',
        gender: userInfo.gender || '男',
        target_weight: userInfo.target_weight ? String(userInfo.target_weight) : '',
        allergies: userInfo.allergies || '',
        goal: userInfo.goal || '',
        calorie_adjustment: userInfo.calorie_adjustment ? String(userInfo.calorie_adjustment) : '0'
      }
    })
  },

  preventBubble() {},

  hideEdit() {
    this.setData({ showEditModal: false })
  },

  onEditInput(e) {
    const field = e.currentTarget.dataset.field
    this.setData({ [`editForm.${field}`]: e.detail.value })
  },

  selectGender(e) {
    this.setData({ 'editForm.gender': e.currentTarget.dataset.gender })
  },

  selectGoal(e) {
    this.setData({ 'editForm.goal': e.currentTarget.dataset.goal })
  },

  saveProfile() {
    const form = this.data.editForm
    const data = {
      height: parseFloat(form.height) || 0,
      weight: parseFloat(form.weight) || 0,
      age: parseInt(form.age) || 0,
      gender: form.gender,
      target_weight: form.target_weight ? parseFloat(form.target_weight) : null,
      allergies: form.allergies || null,
      goal: form.goal || null,
      calorie_adjustment: form.calorie_adjustment ? parseFloat(form.calorie_adjustment) : 0
    }

    wx.showLoading({ title: '保存中...' })
    request({ url: '/api/v1/user/', method: 'POST', data }).then(user => {
      wx.hideLoading()
      wx.showToast({ title: '保存成功', icon: 'success' })
      user.bmrRounded = user.bmr ? Math.round(user.bmr) : '--'
      user.tdeeRounded = user.tdee ? Math.round(user.tdee) : '--'
      this._calcBmi(user)
      this.setData({ userInfo: user, showEditModal: false })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '保存失败', icon: 'none' })
    })
  },

  clearAllData() {
    wx.showModal({
      title: '清除全部数据',
      content: '将删除所有饮食记录、运动记录和对话历史，账户信息保留。此操作不可恢复。',
      confirmColor: '#c47a6c',
      success: (res) => {
        if (res.confirm) {
          wx.showLoading({ title: '清除中...' })
          request({ url: '/api/v1/user/me/data', method: 'DELETE' }).then(() => {
            wx.hideLoading()
            wx.showToast({ title: '已清除', icon: 'success' })
            setTimeout(() => {
              wx.switchTab({ url: '/pages/home/home' })
            }, 1500)
          }).catch(err => {
            wx.hideLoading()
            wx.showToast({ title: err.message || '清除失败', icon: 'none' })
          })
        }
      }
    })
  },

  logout() {
    wx.showModal({
      title: '确认退出',
      content: '退出后需要重新登录',
      success: res => {
        if (res.confirm) {
          authLogout()
          this.setData({ loggedIn: false, userInfo: {} })
        }
      }
    })
  },

  handleLogin() {
    showLoginPrompt().then(loggedIn => {
      if (loggedIn) {
        this.setData({ loggedIn: true })
        this.loadProfile()
      }
    })
  }
})
