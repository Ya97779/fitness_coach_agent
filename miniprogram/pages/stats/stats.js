const { request } = require('../../utils/request')
const { isLoggedIn, showLoginPrompt } = require('../../utils/auth')

const CACHE_KEY = 'history_logs'
const CACHE_DAYS = 90
const WEEKDAYS = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
const MEAL_LABELS = { breakfast: '早餐', lunch: '午餐', dinner: '晚餐', snack: '加餐' }

Page({
  data: {
    loggedIn: false,
    loading: true,
    activeTab: 0, // 0=历史记录, 1=统计图表

    // 日历
    calendarYear: 0,
    calendarMonth: 0, // 0-indexed
    calendarDays: [],
    selectedDate: '',
    hasRecordDates: {},

    // 记录列表
    allLogs: [],
    displayLogs: [],
    displayCount: 7,
    hasMore: true,

    // 图表
    chartRange: 7,
    chartLogs: []
  },

  onLoad() {
    const today = new Date()
    this.setData({
      calendarYear: today.getFullYear(),
      calendarMonth: today.getMonth(),
      selectedDate: this._formatDate(today)
    })

    if (!isLoggedIn()) {
      showLoginPrompt().then(loggedIn => {
        if (loggedIn) {
          this.setData({ loggedIn: true })
          this._initData()
        } else {
          this.setData({ loading: false })
        }
        this._pageReady = true
      })
      return
    }
    this.setData({ loggedIn: true })
    this._initData()
    this._pageReady = true
  },

  onShow() {
    // 从其他页面返回时，如果有缓存则后台同步（跳过首次加载）
    if (this.data.loggedIn && this._pageReady) {
      this._syncFromServer()
    }
  },

  // --- Tab 切换 ---
  switchTab(e) {
    const tab = parseInt(e.currentTarget.dataset.tab)
    this.setData({ activeTab: tab })
    if (tab === 1) {
      this._prepareChartData()
      this._drawNetCalorieChart()
    }
  },

  // --- 数据初始化（缓存优先 + 后台同步） ---
  _initData() {
    const cached = wx.getStorageSync(CACHE_KEY)
    if (cached && cached.length > 0) {
      this._processLogs(cached)
      this.setData({ loading: false })
      this._syncFromServer()
    } else {
      this._loadFromServer()
    }
  },

  _loadFromServer() {
    this.setData({ loading: true })
    request({ url: '/api/v1/user/me/logs' }).then(logs => {
      if (!logs) logs = []
      this._processLogs(logs)
      this._saveToCache(logs)
      this.setData({ loading: false })
    }).catch(() => {
      this.setData({ loading: false })
    })
  },

  _syncFromServer() {
    request({ url: '/api/v1/user/me/logs' }).then(logs => {
      if (!logs) logs = []
      const cached = wx.getStorageSync(CACHE_KEY)
      if (JSON.stringify(logs) !== JSON.stringify(cached)) {
        this._processLogs(logs)
        this._saveToCache(logs)
      }
    }).catch(() => {})
  },

  _saveToCache(logs) {
    // 只保留最近 90 天
    const cutoff = new Date()
    cutoff.setDate(cutoff.getDate() - CACHE_DAYS)
    const filtered = logs.filter(l => new Date(l.date) >= cutoff)
    wx.setStorageSync(CACHE_KEY, filtered)
  },

  _processLogs(logs) {
    // 按日期倒序排列
    const sorted = logs.slice().sort((a, b) => new Date(b.date) - new Date(a.date))

    // 构建有记录的日期集合
    const hasRecordDates = {}
    sorted.forEach(l => {
      hasRecordDates[l.date] = true
    })

    this.setData({ allLogs: sorted, hasRecordDates })
    this._buildCalendar()
    this._filterDisplayLogs()
  },

  // --- 日历 ---
  _buildCalendar() {
    const { calendarYear, calendarMonth } = this.data
    const firstDay = new Date(calendarYear, calendarMonth, 1)
    const lastDay = new Date(calendarYear, calendarMonth + 1, 0)
    const startWeekday = firstDay.getDay() // 0=周日

    const days = []

    // 上月补位
    const prevLastDay = new Date(calendarYear, calendarMonth, 0).getDate()
    for (let i = startWeekday - 1; i >= 0; i--) {
      days.push({
        day: prevLastDay - i,
        date: '',
        isCurrentMonth: false,
        hasRecord: false,
        isSelected: false
      })
    }

    // 本月
    for (let d = 1; d <= lastDay.getDate(); d++) {
      const dateStr = `${calendarYear}-${String(calendarMonth + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
      days.push({
        day: d,
        date: dateStr,
        isCurrentMonth: true,
        hasRecord: !!this.data.hasRecordDates[dateStr],
        isSelected: dateStr === this.data.selectedDate
      })
    }

    // 下月补位（补齐到 6 行 = 42 天）
    const remaining = 42 - days.length
    for (let d = 1; d <= remaining; d++) {
      days.push({
        day: d,
        date: '',
        isCurrentMonth: false,
        hasRecord: false,
        isSelected: false
      })
    }

    this.setData({ calendarDays: days })
  },

  prevMonth() {
    let { calendarYear, calendarMonth } = this.data
    calendarMonth--
    if (calendarMonth < 0) { calendarMonth = 11; calendarYear-- }
    this.setData({ calendarYear, calendarMonth })
    this._buildCalendar()
  },

  nextMonth() {
    let { calendarYear, calendarMonth } = this.data
    calendarMonth++
    if (calendarMonth > 11) { calendarMonth = 0; calendarYear++ }
    this.setData({ calendarYear, calendarMonth })
    this._buildCalendar()
  },

  selectDate(e) {
    const date = e.currentTarget.dataset.date
    if (!date) return
    this.setData({ selectedDate: date })
    this._buildCalendar()

    // 检查该日期是否有记录
    if (this.data.hasRecordDates[date]) {
      // 延迟一帧确保 DOM 更新
      setTimeout(() => {
        wx.pageScrollTo({ selector: `#date-${date}`, duration: 300 })
      }, 100)
    }
  },

  // --- 记录列表 ---
  _filterDisplayLogs() {
    const { allLogs, displayCount } = this.data
    const displayLogs = allLogs.slice(0, displayCount).map(log => {
      const d = new Date(log.date)
      return {
        ...log,
        weekday: WEEKDAYS[d.getDay()],
        month: d.getMonth() + 1,
        day: d.getDate(),
        intakeCal: log.intake_calories || 0,
        burnCal: log.burn_calories || 0,
        foodByMeal: this._groupByMeal(log.food_items || []),
        exerciseItems: log.exercise_items || []
      }
    })
    this.setData({
      displayLogs,
      hasMore: allLogs.length > displayCount
    })
  },

  _groupByMeal(items) {
    const groups = {}
    items.forEach(item => {
      const meal = item.meal_type || 'other'
      if (!groups[meal]) groups[meal] = []
      groups[meal].push(item)
    })
    // 按餐次顺序排列
    const order = ['breakfast', 'lunch', 'dinner', 'snack']
    const result = []
    order.forEach(key => {
      if (groups[key]) {
        result.push({ meal: key, label: MEAL_LABELS[key] || key, items: groups[key] })
      }
    })
    return result
  },

  loadMore() {
    if (!this.data.hasMore) return
    const newCount = this.data.displayCount + 7
    this.setData({ displayCount: newCount })
    this._filterDisplayLogs()
  },

  // --- 图表 ---
  _prepareChartData() {
    const { chartRange } = this.data
    const today = new Date()
    today.setHours(0, 0, 0, 0)

    // 构建日期→log 的映射
    const logMap = {}
    this.data.allLogs.forEach(l => {
      logMap[l.date] = l
    })

    // 补全每一天，没有记录的填 0
    const chartLogs = []
    for (let i = chartRange - 1; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(d.getDate() - i)
      const dateStr = this._formatDate(d)
      const log = logMap[dateStr]
      chartLogs.push({
        date: dateStr,
        intake_calories: log ? (log.intake_calories || 0) : 0,
        burn_calories: log ? (log.burn_calories || 0) : 0
      })
    }
    this.setData({ chartLogs })
  },

  setChartRange(e) {
    this.setData({ chartRange: parseInt(e.currentTarget.dataset.range) })
    this._prepareChartData()
    this._drawNetCalorieChart()
  },

  _drawNetCalorieChart() {
    const { chartLogs } = this.data
    if (chartLogs.length === 0) return

    const query = wx.createSelectorQuery()
    query.select('#netCalorieChart').boundingClientRect()
    query.exec(res => {
      if (!res || !res[0]) return
      const { width, height } = res[0]
      const ctx = wx.createCanvasContext('netCalorieChart', this)
      const padding = { top: 40, right: 20, bottom: 40, left: 60 }
      const chartW = width - padding.left - padding.right
      const chartH = height - padding.top - padding.bottom

      // 计算净热量
      const nets = chartLogs.map(l => (l.intake_calories || 0) - (l.burn_calories || 0))
      const minVal = Math.min(...nets, 0)
      const maxVal = Math.max(...nets, 0)

      // 自动计算纵轴范围和间距
      const range = maxVal - minVal || 200
      const rawStep = range / 4
      // 取整到"好看"的步长（10/20/50/100/200/500 的倍数）
      const niceSteps = [10, 20, 50, 100, 200, 500, 1000]
      let step = niceSteps.find(s => s >= rawStep) || Math.ceil(rawStep / 100) * 100
      const yMin = Math.floor(minVal / step) * step
      const yMax = Math.ceil(maxVal / step) * step
      const yRange = yMax - yMin || step

      // 零线 Y 坐标
      const zeroY = padding.top + chartH * (1 - (0 - yMin) / yRange)

      // 绘制网格线和 Y 轴标签
      ctx.setTextAlign('right')
      for (let v = yMin; v <= yMax; v += step) {
        const y = padding.top + chartH * (1 - (v - yMin) / yRange)
        ctx.setStrokeStyle(v === 0 ? '#bbb' : '#f0f0f0')
        ctx.setLineWidth(v === 0 ? 1 : 0.5)
        ctx.setLineDash(v === 0 ? [4, 4] : [])
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(width - padding.right, y)
        ctx.stroke()
        ctx.setLineDash([])

        // Y 轴数值标签
        ctx.setFillStyle('#999')
        ctx.setFontSize(18)
        ctx.fillText(v, padding.left - 8, y + 5)
      }

      // 绘制区域填充 + 折线
      if (chartLogs.length >= 2) {
        // 计算各点坐标
        const points = nets.map((v, i) => ({
          x: padding.left + chartW * i / (chartLogs.length - 1),
          y: padding.top + chartH * (1 - (v - yMin) / yRange)
        }))

        // Helper: 计算两点之间与零线的交点 x 坐标
        const getZeroCrossingX = (p1, p2) => {
          if (p1.y === p2.y) return p1.x
          const t = (zeroY - p1.y) / (p2.y - p1.y)
          return p1.x + t * (p2.x - p1.x)
        }

        // 构建带零线交点的路径
        const buildFillPath = (isAbove) => {
          const path = []
          for (let i = 0; i < points.length; i++) {
            const p = points[i]
            const val = nets[i]
            const isPositive = val >= 0

            if (i > 0) {
              const prev = points[i - 1]
              const prevPositive = nets[i - 1] >= 0
              // 跨越零线时插入交点
              if (prevPositive !== isPositive) {
                const crossX = getZeroCrossingX(prev, p)
                path.push({ x: crossX, y: zeroY })
              }
            }

            if (isAbove) {
              path.push({ x: p.x, y: Math.min(p.y, zeroY) })
            } else {
              path.push({ x: p.x, y: Math.max(p.y, zeroY) })
            }
          }
          return path
        }

        // 盈余区域填充（零线上方）
        const surplusPath = buildFillPath(true)
        if (surplusPath.length > 1) {
          ctx.setFillStyle('rgba(255, 107, 107, 0.15)')
          ctx.beginPath()
          ctx.moveTo(surplusPath[0].x, zeroY)
          surplusPath.forEach(p => ctx.lineTo(p.x, p.y))
          ctx.lineTo(surplusPath[surplusPath.length - 1].x, zeroY)
          ctx.closePath()
          ctx.fill()
        }

        // 缺口区域填充（零线下方）
        const deficitPath = buildFillPath(false)
        if (deficitPath.length > 1) {
          ctx.setFillStyle('rgba(76, 175, 80, 0.15)')
          ctx.beginPath()
          ctx.moveTo(deficitPath[0].x, zeroY)
          deficitPath.forEach(p => ctx.lineTo(p.x, p.y))
          ctx.lineTo(deficitPath[deficitPath.length - 1].x, zeroY)
          ctx.closePath()
          ctx.fill()
        }

        // 绘制折线
        ctx.setStrokeStyle('#1a1a1a')
        ctx.setLineWidth(2.5)
        ctx.setLineCap('round')
        ctx.setLineJoin('round')
        ctx.beginPath()
        points.forEach((p, i) => {
          if (i === 0) ctx.moveTo(p.x, p.y)
          else ctx.lineTo(p.x, p.y)
        })
        ctx.stroke()

        // 数据点 + 数值标签
        points.forEach((p, i) => {
          ctx.setFillStyle('#ffffff')
          ctx.beginPath()
          ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI)
          ctx.fill()
          ctx.setStrokeStyle('#1a1a1a')
          ctx.setLineWidth(2)
          ctx.stroke()

          // 数值标签（30天模式跳过 0 值，避免拥挤）
          const val = nets[i]
          if (chartLogs.length <= 7 || val !== 0) {
            ctx.setFillStyle('#333')
            ctx.setFontSize(chartLogs.length > 7 ? 14 : 16)
            ctx.setTextAlign('center')
            const labelY = val >= 0 ? p.y - 12 : p.y + 20
            ctx.fillText(val, p.x, labelY)
          }
        })
      }

      // X 轴日期标签
      ctx.setFillStyle('#999')
      ctx.setFontSize(16)
      ctx.setTextAlign('center')
      const total = chartLogs.length
      if (total <= 7) {
        // 近7天全部显示
        chartLogs.forEach((l, i) => {
          const x = padding.left + chartW * i / (total - 1 || 1)
          ctx.fillText(l.date.slice(5), x, height - 8)
        })
      } else {
        // 近30天固定5个刻度：首日、1/4、中间、3/4、末日
        const ticks = [0, Math.round(total * 0.25), Math.round(total * 0.5), Math.round(total * 0.75), total - 1]
        ticks.forEach(i => {
          const x = padding.left + chartW * i / (total - 1)
          ctx.fillText(chartLogs[i].date.slice(5), x, height - 8)
        })
      }

      ctx.draw()
    })
  },

  // --- 工具函数 ---
  _formatDate(date) {
    const y = date.getFullYear()
    const m = String(date.getMonth() + 1).padStart(2, '0')
    const d = String(date.getDate()).padStart(2, '0')
    return `${y}-${m}-${d}`
  },

  handleLogin() {
    showLoginPrompt().then(loggedIn => {
      if (loggedIn) {
        this.setData({ loggedIn: true })
        this._initData()
      }
    })
  }
})
