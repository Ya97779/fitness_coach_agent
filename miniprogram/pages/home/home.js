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

      this.setData({
        tdee, intake: Math.round(intake), burn: Math.round(burn),
        remaining: remaining > 0 ? remaining : 0,
        loading: false
      })
      this.drawRing(intake, burn, tdee)

      // 加载今日详细记录
      this.loadTodayItems(today.id)
    }).catch(() => {
      this.setData({ loading: false })
    })
  },

  loadTodayItems(logId) {
    // 通过 today 接口的 food_items / exercise_items 获取
    // 后端 DailyLog 需要返回关联数据，这里先用空数组
    // 实际项目中需后端在 today 接口返回 items 列表
    this.setData({ foodItems: [], exerciseItems: [] })
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
      ctx.setStrokeStyle('rgba(255,255,255,0.06)')
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI)
      ctx.stroke()

      // 摄入环
      const intakeAngle = Math.min(intake / tdee, 1) * 2 * Math.PI
      if (intakeAngle > 0) {
        ctx.setLineWidth(lineWidth)
        ctx.setStrokeStyle('#a8b5a0')
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
        ctx.setStrokeStyle('rgba(168,181,160,0.4)')
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
