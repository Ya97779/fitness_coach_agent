const { request } = require('../../utils/request')

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
    tdee: 2000,
    foodItems: [],
    exerciseItems: [],
    loading: true
  },

  onShow() {
    this.setGreeting()
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
      const tdee = user.tdee || 2000
      const intake = today.intake_calories || 0
      const burn = today.burn_calories || 0
      const remaining = Math.round(tdee - intake + burn)

      const foodItems = (today.food_items || []).map(item => ({
        ...item,
        meal_type_text: MEAL_TYPE_MAP[item.meal_type] || item.meal_type
      }))
      const exerciseItems = today.exercise_items || []

      this.setData({
        tdee, intake: Math.round(intake), burn: Math.round(burn),
        remaining: remaining > 0 ? remaining : 0,
        foodItems, exerciseItems,
        loading: false
      })
      this.drawRing(intake, burn, tdee)
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
  }
})
