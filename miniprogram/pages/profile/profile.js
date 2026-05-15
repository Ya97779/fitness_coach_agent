const { request } = require('../../utils/request')
const { logout: authLogout } = require('../../utils/auth')

Page({
  data: {
    userInfo: {},
    showEditModal: false,
    editForm: {
      height: '', weight: '', age: '', gender: '男',
      target_weight: '', allergies: ''
    }
  },

  onShow() {
    this.loadProfile()
  },

  loadProfile() {
    request({ url: '/api/v1/user/me' }).then(user => {
      this.setData({ userInfo: user })
    }).catch(() => {})
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
        allergies: userInfo.allergies || ''
      }
    })
  },

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

  saveProfile() {
    const form = this.data.editForm
    const data = {
      height: parseFloat(form.height) || 0,
      weight: parseFloat(form.weight) || 0,
      age: parseInt(form.age) || 0,
      gender: form.gender,
      target_weight: form.target_weight ? parseFloat(form.target_weight) : null,
      allergies: form.allergies || null
    }

    wx.showLoading({ title: '保存中...' })
    request({ url: '/api/v1/user/', method: 'POST', data }).then(user => {
      wx.hideLoading()
      wx.showToast({ title: '保存成功', icon: 'success' })
      this.setData({ userInfo: user, showEditModal: false })
    }).catch(err => {
      wx.hideLoading()
      wx.showToast({ title: err.message || '保存失败', icon: 'none' })
    })
  },

  logout() {
    wx.showModal({
      title: '确认退出',
      content: '退出后需要重新登录',
      success: res => {
        if (res.confirm) {
          authLogout()
          wx.reLaunch({ url: '/pages/home/home' })
        }
      }
    })
  }
})
