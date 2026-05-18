const { request } = require('../../utils/request')

Page({
  data: {
    range: 7,
    logs: [],
    loading: true
  },

  onLoad() {
    this.loadLogs()
  },

  setRange(e) {
    this.setData({ range: parseInt(e.currentTarget.dataset.range) })
    this.loadLogs()
  },

  loadLogs() {
    this.setData({ loading: true })
    request({ url: '/api/v1/user/me/logs' }).then(logs => {
      const range = this.data.range
      const cutoff = new Date()
      cutoff.setDate(cutoff.getDate() - range)
      const filtered = logs
        .filter(l => new Date(l.date) >= cutoff)
        .sort((a, b) => new Date(a.date) - new Date(b.date))

      this.setData({ logs: filtered, loading: false })
      this.drawCharts(filtered)
    }).catch(() => {
      this.setData({ loading: false })
    })
  },

  drawCharts(logs) {
    if (logs.length === 0) return
    this.drawLineChart(logs)
    this.drawBarChart(logs)
  },

  drawLineChart(logs) {
    const query = wx.createSelectorQuery()
    query.select('#lineChart').boundingClientRect()
    query.exec(res => {
      if (!res || !res[0]) return
      const { width, height } = res[0]
      const ctx = wx.createCanvasContext('lineChart', this)
      const padding = { top: 30, right: 20, bottom: 40, left: 50 }
      const chartW = width - padding.left - padding.right
      const chartH = height - padding.top - padding.bottom

      const intakes = logs.map(l => l.intake_calories || 0)
      const burns = logs.map(l => l.burn_calories || 0)
      const maxVal = Math.max(...intakes, ...burns, 100)

      // 网格线
      ctx.setStrokeStyle('rgba(255,255,255,0.06)')
      ctx.setLineWidth(1)
      for (let i = 0; i <= 4; i++) {
        const y = padding.top + chartH * (1 - i / 4)
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(width - padding.right, y)
        ctx.stroke()

        ctx.setFillStyle('rgba(255,255,255,0.2)')
        ctx.setFontSize(18)
        ctx.fillText(Math.round(maxVal * i / 4), 4, y + 5)
      }

      // X 轴标签
      ctx.setFillStyle('rgba(255,255,255,0.2)')
      ctx.setFontSize(16)
      ctx.setTextAlign('center')
      logs.forEach((l, i) => {
        const x = padding.left + chartW * i / (logs.length - 1 || 1)
        const day = l.date.slice(5) // MM-DD
        ctx.fillText(day, x, height - 8)
      })

      // 绘制摄入线
      this.drawLine(ctx, intakes, maxVal, logs.length, padding, chartW, chartH, '#a8b5a0')
      // 绘制消耗线
      this.drawLine(ctx, burns, maxVal, logs.length, padding, chartW, chartH, 'rgba(168,181,160,0.4)')

      ctx.draw()
    })
  },

  drawLine(ctx, values, maxVal, count, padding, chartW, chartH, color) {
    if (count < 2) return
    ctx.setStrokeStyle(color)
    ctx.setLineWidth(3)
    ctx.setLineCap('round')
    ctx.setLineJoin('round')
    ctx.beginPath()
    values.forEach((v, i) => {
      const x = padding.left + chartW * i / (count - 1)
      const y = padding.top + chartH * (1 - v / maxVal)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // 数据点
    values.forEach((v, i) => {
      const x = padding.left + chartW * i / (count - 1)
      const y = padding.top + chartH * (1 - v / maxVal)
      ctx.setFillStyle('#111113')
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, 2 * Math.PI)
      ctx.fill()
      ctx.setStrokeStyle(color)
      ctx.setLineWidth(2)
      ctx.stroke()
    })
  },

  drawBarChart(logs) {
    const query = wx.createSelectorQuery()
    query.select('#barChart').boundingClientRect()
    query.exec(res => {
      if (!res || !res[0]) return
      const { width, height } = res[0]
      const ctx = wx.createCanvasContext('barChart', this)
      const padding = { top: 30, right: 20, bottom: 40, left: 50 }
      const chartW = width - padding.left - padding.right
      const chartH = height - padding.top - padding.bottom

      const nets = logs.map(l => (l.intake_calories || 0) - (l.burn_calories || 0))
      const maxAbs = Math.max(Math.abs(Math.min(...nets)), Math.abs(Math.max(...nets)), 100)

      // 零线
      const zeroY = padding.top + chartH / 2
      ctx.setStrokeStyle('rgba(255,255,255,0.08)')
      ctx.setLineWidth(1)
      ctx.beginPath()
      ctx.moveTo(padding.left, zeroY)
      ctx.lineTo(width - padding.right, zeroY)
      ctx.stroke()

      // Y 轴标签
      ctx.setFillStyle('rgba(255,255,255,0.2)')
      ctx.setFontSize(18)
      ctx.setTextAlign('right')
      ctx.fillText(`+${Math.round(maxAbs)}`, padding.left - 6, padding.top + 14)
      ctx.fillText('0', padding.left - 6, zeroY + 5)
      ctx.fillText(`-${Math.round(maxAbs)}`, padding.left - 6, padding.top + chartH + 4)

      // 柱状
      const barW = Math.max(chartW / logs.length * 0.6, 8)
      const gap = chartW / logs.length
      logs.forEach((l, i) => {
        const net = nets[i]
        const barH = (Math.abs(net) / maxAbs) * (chartH / 2)
        const x = padding.left + gap * i + (gap - barW) / 2
        const y = net >= 0 ? zeroY - barH : zeroY
        ctx.setFillStyle(net >= 0 ? '#a8b5a0' : 'rgba(168,181,160,0.35)')
        ctx.fillRect(x, y, barW, barH)
      })

      // X 轴标签
      ctx.setFillStyle('rgba(255,255,255,0.2)')
      ctx.setFontSize(16)
      ctx.setTextAlign('center')
      logs.forEach((l, i) => {
        const x = padding.left + gap * i + gap / 2
        ctx.fillText(l.date.slice(5), x, height - 8)
      })

      ctx.draw()
    })
  }
})
