# 历史记录页面实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将小程序 stats 页面改造为历史记录 + 统计图表的双 Tab 页面，支持日历选择、按日期分组的记录列表、本地缓存，以及重写的净热量趋势图。

**Architecture:** 改造现有 `pages/stats/stats` 页面，顶部加 Tab 切换。Tab 1 为历史记录（日历 + 记录列表），Tab 2 为统计图表（单线净热量趋势）。数据层引入 `wx.setStorageSync` 本地缓存，参考聊天页面的缓存优先 + 后台同步模式。

**Tech Stack:** 微信小程序原生开发、Canvas API（图表）、wx.setStorageSync（本地缓存）

---

### Task 1: 首页按钮文案修改

**Files:**
- Modify: `miniprogram/pages/home/home.wxml:55-58`

- [ ] **Step 1: 修改按钮文案和图标**

将"历史趋势"改为"历史记录"，图标从 `~` 改为日历图标。

```xml
<!-- home.wxml 第 55-58 行 -->
<view class="shortcut" bindtap="goStats">
  <text class="shortcut-icon">📅</text>
  <text class="shortcut-text">历史记录</text>
</view>
```

- [ ] **Step 2: 在微信开发者工具中验证**

打开首页，确认按钮显示"历史记录"，点击可正常跳转到 stats 页面。

- [ ] **Step 3: 提交**

```bash
git add miniprogram/pages/home/home.wxml
git commit -m "feat: 首页历史趋势按钮改为历史记录"
```

---

### Task 2: Stats 页面 Tab 结构 + 数据加载重构 + 本地缓存

**Files:**
- Rewrite: `miniprogram/pages/stats/stats.js`
- Modify: `miniprogram/pages/stats/stats.json`
- Modify: `miniprogram/pages/stats/stats.wxml`
- Modify: `miniprogram/pages/stats/stats.wxss`

- [ ] **Step 1: 重写 stats.json，更新导航栏标题**

```json
{
  "navigationBarTitleText": "历史记录",
  "usingComponents": {
    "floating-timer": "/components/floating-timer/floating-timer"
  }
}
```

- [ ] **Step 2: 重写 stats.js — 数据层 + Tab 切换 + 缓存逻辑**

完整替换 stats.js，包含：Tab 切换、本地缓存（读/写/同步）、日历数据计算、记录分组、图表数据准备。

```js
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
      })
      return
    }
    this.setData({ loggedIn: true })
    this._initData()
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
    const today = this._formatDate(new Date())
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
    this._scrollToDate(date)
  },

  _scrollToDate(date) {
    // 滚动到对应日期的锚点
    wx.pageScrollTo({ selector: `#date-${date}`, duration: 300 })
  },

  // --- 记录列表 ---
  _filterDisplayLogs() {
    const { allLogs, displayCount } = this.data
    const displayLogs = allLogs.slice(0, displayCount).map(log => ({
      ...log,
      weekday: WEEKDAYS[new Date(log.date).getDay()],
      month: new Date(log.date).getMonth() + 1,
      day: new Date(log.date).getDate(),
      intakeCal: log.intake_calories || 0,
      burnCal: log.burn_calories || 0,
      foodByMeal: this._groupByMeal(log.food_items || []),
      exerciseItems: log.exercise_items || []
    }))
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
    const cutoff = new Date()
    cutoff.setDate(cutoff.getDate() - chartRange)
    const chartLogs = this.data.allLogs
      .filter(l => new Date(l.date) >= cutoff)
      .sort((a, b) => new Date(a.date) - new Date(b.date))
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
      const padding = { top: 30, right: 20, bottom: 40, left: 20 }
      const chartW = width - padding.left - padding.right
      const chartH = height - padding.top - padding.bottom

      // 计算净热量
      const nets = chartLogs.map(l => (l.intake_calories || 0) - (l.burn_calories || 0))
      const maxAbs = Math.max(Math.abs(Math.min(...nets)), Math.abs(Math.max(...nets)), 100)

      // 零线 Y 坐标
      const zeroY = padding.top + chartH / 2

      // 绘制零线虚线
      ctx.setStrokeStyle('#e0e0e0')
      ctx.setLineWidth(1)
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(padding.left, zeroY)
      ctx.lineTo(width - padding.right, zeroY)
      ctx.stroke()
      ctx.setLineDash([])

      // 绘制区域填充 + 折线
      if (chartLogs.length >= 2) {
        // 计算各点坐标
        const points = nets.map((v, i) => ({
          x: padding.left + chartW * i / (chartLogs.length - 1),
          y: zeroY - (v / maxAbs) * (chartH / 2)
        }))

        // 盈余区域填充（零线上方）
        ctx.setFillStyle('rgba(255, 107, 107, 0.15)')
        ctx.beginPath()
        ctx.moveTo(points[0].x, zeroY)
        points.forEach(p => {
          const y = Math.min(p.y, zeroY)
          ctx.lineTo(p.x, y)
        })
        ctx.lineTo(points[points.length - 1].x, zeroY)
        ctx.closePath()
        ctx.fill()

        // 缺口区域填充（零线下方）
        ctx.setFillStyle('rgba(76, 175, 80, 0.15)')
        ctx.beginPath()
        ctx.moveTo(points[0].x, zeroY)
        points.forEach(p => {
          const y = Math.max(p.y, zeroY)
          ctx.lineTo(p.x, y)
        })
        ctx.lineTo(points[points.length - 1].x, zeroY)
        ctx.closePath()
        ctx.fill()

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

        // 数据点
        points.forEach(p => {
          ctx.setFillStyle('#ffffff')
          ctx.beginPath()
          ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI)
          ctx.fill()
          ctx.setStrokeStyle('#1a1a1a')
          ctx.setLineWidth(2)
          ctx.stroke()
        })
      }

      // X 轴日期标签
      ctx.setFillStyle('#999')
      ctx.setFontSize(16)
      ctx.setTextAlign('center')
      chartLogs.forEach((l, i) => {
        const x = padding.left + chartW * i / (chartLogs.length - 1 || 1)
        ctx.fillText(l.date.slice(5), x, height - 8)
      })

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
```

- [ ] **Step 3: 重写 stats.wxml — Tab + 日历 + 记录列表 + 图表**

完整替换 stats.wxml。

```xml
<view class="page">
  <floating-timer />

  <!-- 未登录提示 -->
  <view class="login-card" wx:if="{{!loggedIn}}">
    <text class="login-card-title">登录后可查看历史记录</text>
    <view class="login-card-btn" bindtap="handleLogin">
      <text class="login-card-btn-text">立即登录</text>
    </view>
  </view>

  <!-- Tab 切换 -->
  <view class="tabs" wx:if="{{loggedIn}}">
    <view class="tab {{activeTab === 0 ? 'tab-active' : ''}}" bindtap="switchTab" data-tab="0">历史记录</view>
    <view class="tab {{activeTab === 1 ? 'tab-active' : ''}}" bindtap="switchTab" data-tab="1">统计图表</view>
  </view>

  <!-- Tab 1: 历史记录 -->
  <view wx:if="{{loggedIn && activeTab === 0}}">

    <!-- 日历 -->
    <view class="calendar">
      <view class="calendar-header">
        <view class="calendar-nav" bindtap="prevMonth">&lt;</view>
        <text class="calendar-title">{{calendarYear}}年{{calendarMonth + 1}}月</text>
        <view class="calendar-nav" bindtap="nextMonth">&gt;</view>
      </view>
      <view class="calendar-weekdays">
        <text class="calendar-weekday">日</text>
        <text class="calendar-weekday">一</text>
        <text class="calendar-weekday">二</text>
        <text class="calendar-weekday">三</text>
        <text class="calendar-weekday">四</text>
        <text class="calendar-weekday">五</text>
        <text class="calendar-weekday">六</text>
      </view>
      <view class="calendar-grid">
        <view
          wx:for="{{calendarDays}}"
          wx:key="index"
          class="calendar-day {{item.isCurrentMonth ? '' : 'calendar-day-other'}} {{item.isSelected ? 'calendar-day-selected' : ''}}"
          bindtap="selectDate"
          data-date="{{item.date}}"
        >
          <text class="calendar-day-num">{{item.day}}</text>
          <view class="calendar-dot" wx:if="{{item.hasRecord}}"></view>
        </view>
      </view>
    </view>

    <!-- 记录列表 -->
    <view class="records">
      <view wx:if="{{displayLogs.length === 0 && !loading}}" class="empty">
        <text class="empty-text">暂无历史记录</text>
      </view>

      <view wx:for="{{displayLogs}}" wx:key="date" class="day-card" id="date-{{item.date}}">
        <!-- 日期标题 -->
        <view class="day-header">
          <text class="day-title">{{item.month}}月{{item.day}}日 {{item.weekday}}</text>
          <text class="day-summary">摄入 {{item.intakeCal}} kcal · 消耗 {{item.burnCal}} kcal</text>
        </view>

        <!-- 饮食记录 -->
        <view wx:for="{{item.foodByMeal}}" wx:for-item="meal" wx:key="meal" class="meal-group">
          <text class="meal-label">{{meal.label}}</text>
          <view wx:for="{{meal.items}}" wx:for-item="food" wx:key="id" class="record-item">
            <text class="record-name">{{food.name}}</text>
            <text class="record-value">{{food.calories}} kcal</text>
          </view>
        </view>

        <!-- 运动记录 -->
        <view wx:if="{{item.exerciseItems.length > 0}}" class="meal-group">
          <text class="meal-label">运动</text>
          <view wx:for="{{item.exerciseItems}}" wx:for-item="ex" wx:key="id" class="record-item">
            <text class="record-name">{{ex.name || ex.type}}</text>
            <text class="record-value">{{ex.duration}}分钟  {{ex.calories}} kcal</text>
          </view>
        </view>
      </view>

      <!-- 加载更多 -->
      <view wx:if="{{hasMore}}" class="load-more" bindtap="loadMore">
        <text class="load-more-text">加载更多</text>
      </view>
    </view>
  </view>

  <!-- Tab 2: 统计图表 -->
  <view wx:if="{{loggedIn && activeTab === 1}}">
    <view class="range-tabs">
      <view class="range-tab {{chartRange === 7 ? 'range-on' : ''}}" bindtap="setChartRange" data-range="7">近 7 天</view>
      <view class="range-tab {{chartRange === 30 ? 'range-on' : ''}}" bindtap="setChartRange" data-range="30">近 30 天</view>
    </view>

    <view class="card" wx:if="{{chartLogs.length > 0}}">
      <text class="chart-label">每日净热量趋势</text>
      <canvas canvas-id="netCalorieChart" id="netCalorieChart" class="chart-canvas" />
    </view>

    <view class="empty" wx:if="{{!loading && chartLogs.length === 0}}">
      <text class="empty-text">暂无历史数据</text>
    </view>
  </view>

  <view class="empty" wx:if="{{loggedIn && loading}}">
    <text class="empty-text">加载中...</text>
  </view>
</view>
```

- [ ] **Step 4: 重写 stats.wxss — Tab + 日历 + 记录卡片 + 图表样式**

完整替换 stats.wxss。

```css
.page {
  padding: 0 28rpx 40rpx;
}

/* 登录提示 */
.login-card {
  padding: 36rpx;
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-md);
  text-align: center;
  margin-top: 40rpx;
}

.login-card-title {
  font-size: 30rpx;
  font-weight: 700;
  color: var(--text-primary);
  display: block;
  margin-bottom: 28rpx;
}

.login-card-btn {
  display: inline-block;
  background: var(--text-primary);
  color: #fff;
  padding: 18rpx 64rpx;
  border-radius: var(--radius-sm);
}

.login-card-btn-text {
  font-size: 28rpx;
  font-weight: 600;
}

/* Tab 切换 */
.tabs {
  display: flex;
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4rpx;
  margin: 24rpx 0;
}

.tab {
  flex: 1;
  text-align: center;
  padding: 16rpx 0;
  font-size: 26rpx;
  color: var(--text-hint);
  border-radius: var(--radius-sm);
  transition: all 0.15s;
}

.tab-active {
  background: var(--bg-card);
  color: var(--text-primary);
  font-weight: 600;
}

/* 日历 */
.calendar {
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-md);
  padding: 24rpx;
  margin-bottom: 24rpx;
}

.calendar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20rpx;
}

.calendar-nav {
  width: 60rpx;
  height: 60rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28rpx;
  color: var(--text-primary);
}

.calendar-title {
  font-size: 28rpx;
  font-weight: 600;
  color: var(--text-primary);
}

.calendar-weekdays {
  display: flex;
  margin-bottom: 8rpx;
}

.calendar-weekday {
  flex: 1;
  text-align: center;
  font-size: 20rpx;
  color: var(--text-hint);
  padding: 8rpx 0;
}

.calendar-grid {
  display: flex;
  flex-wrap: wrap;
}

.calendar-day {
  width: calc(100% / 7);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12rpx 0;
  position: relative;
}

.calendar-day-num {
  font-size: 26rpx;
  color: var(--text-primary);
}

.calendar-day-other .calendar-day-num {
  color: var(--text-hint);
  opacity: 0.4;
}

.calendar-day-selected {
  background: var(--text-primary);
  border-radius: 50%;
}

.calendar-day-selected .calendar-day-num {
  color: #fff;
}

.calendar-dot {
  width: 8rpx;
  height: 8rpx;
  border-radius: 50%;
  background: var(--text-primary);
  margin-top: 4rpx;
}

.calendar-day-selected .calendar-dot {
  background: #fff;
}

/* 记录列表 */
.records {
  margin-top: 8rpx;
}

.day-card {
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-md);
  padding: 24rpx;
  margin-bottom: 16rpx;
}

.day-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16rpx;
  padding-bottom: 16rpx;
  border-bottom: 1rpx solid var(--border);
}

.day-title {
  font-size: 28rpx;
  font-weight: 700;
  color: var(--text-primary);
}

.day-summary {
  font-size: 22rpx;
  color: var(--text-hint);
}

.meal-group {
  margin-bottom: 12rpx;
}

.meal-label {
  font-size: 22rpx;
  color: var(--text-hint);
  font-weight: 600;
  display: block;
  margin-bottom: 8rpx;
}

.record-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8rpx 0;
}

.record-name {
  font-size: 26rpx;
  color: var(--text-primary);
}

.record-value {
  font-size: 24rpx;
  color: var(--text-hint);
}

/* 加载更多 */
.load-more {
  text-align: center;
  padding: 24rpx;
}

.load-more-text {
  font-size: 24rpx;
  color: var(--text-hint);
}

/* 日期范围（图表 Tab） */
.range-tabs {
  display: flex;
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4rpx;
  margin: 24rpx 0;
}

.range-tab {
  flex: 1;
  text-align: center;
  padding: 14rpx 0;
  font-size: 24rpx;
  color: var(--text-hint);
  border-radius: var(--radius-sm);
  transition: all 0.15s;
  letter-spacing: 1rpx;
}

.range-on {
  background: var(--bg-card);
  color: var(--text-primary);
  font-weight: 600;
}

/* 图表 */
.chart-label {
  font-size: 20rpx;
  color: var(--text-hint);
  letter-spacing: 3rpx;
  font-weight: 600;
  display: block;
  margin-bottom: 16rpx;
}

.chart-canvas {
  width: 100%;
  height: 340rpx;
}

/* 卡片 */
.card {
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-md);
  padding: 24rpx;
  margin-bottom: 16rpx;
}

/* 空状态 */
.empty {
  text-align: center;
  padding: 80rpx;
}

.empty-text {
  font-size: 26rpx;
  color: var(--text-hint);
}
```

- [ ] **Step 5: 在微信开发者工具中验证**

打开 stats 页面，确认：
1. Tab 切换正常，"历史记录"和"统计图表"可切换
2. 日历显示当月，可左右切换月份
3. 有记录的日期下方有圆点
4. 点击日期可高亮
5. 记录列表按日期倒序显示，每天有饮食和运动分组
6. 触底可加载更多
7. 统计图表 Tab 显示净热量趋势折线图

- [ ] **Step 6: 提交**

```bash
git add miniprogram/pages/stats/
git commit -m "feat: 改造 stats 页面为历史记录 + 统计图表双 Tab"
```

---

### Task 3: 日历点击日期滚动到对应记录

**Files:**
- Modify: `miniprogram/pages/stats/stats.js` (selectDate 方法)
- Modify: `miniprogram/pages/stats/stats.wxml` (scroll-view 或 pageScrollTo)

- [ ] **Step 1: 在 selectDate 中实现滚动定位**

在 stats.js 的 `selectDate` 方法中，点击日历日期后滚动到对应的记录卡片。由于使用了 `id="date-{{item.date}}"` 作为锚点，需要用 `wx.pageScrollTo`。

注意：`wx.pageScrollTo` 的 `selector` 在小程序中需要用 `#` 前缀。修改 `selectDate` 方法：

```js
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
```

- [ ] **Step 2: 在微信开发者工具中验证**

点击日历上有圆点的日期，确认页面滚动到对应的记录卡片位置。

- [ ] **Step 3: 提交**

```bash
git add miniprogram/pages/stats/stats.js
git commit -m "feat: 日历点击日期滚动到对应记录"
```

---

### Task 4: 本地缓存增量更新（用户新增记录后）

**Files:**
- Modify: `miniprogram/pages/stats/stats.js` (onShow 生命周期)

- [ ] **Step 1: 添加 onShow 生命周期，页面显示时刷新缓存**

当用户从记录页面返回时，需要刷新当天的数据。添加 `onShow` 方法：

```js
onShow() {
  // 从其他页面返回时，如果有缓存则后台同步
  if (this.data.loggedIn) {
    this._syncFromServer()
  }
},
```

在 `_syncFromServer` 方法中已经处理了缓存对比和更新逻辑，无需额外修改。

- [ ] **Step 2: 在微信开发者工具中验证**

1. 打开历史记录页面（有缓存）
2. 跳转到记录页面，新增一条食物记录
3. 返回历史记录页面，确认当天数据已更新

- [ ] **Step 3: 提交**

```bash
git add miniprogram/pages/stats/stats.js
git commit -m "feat: 历史记录页面 onShow 时后台同步缓存"
```

---

### Task 5: 最终验证与清理

**Files:**
- 无新增文件

- [ ] **Step 1: 完整流程验证**

在微信开发者工具中完成以下验证：

1. **首页**：按钮显示"历史记录"，点击跳转正常
2. **历史记录 Tab**：
   - 日历显示当月，左右切换月份正常
   - 有记录日期显示圆点
   - 点击日期高亮 + 滚动定位
   - 记录列表按日期倒序，饮食按餐次分组，运动单独分组
   - 汇总行显示摄入/消耗热量
   - 触底加载更多正常
   - 首次打开有 loading，之后秒开（缓存命中）
3. **统计图表 Tab**：
   - 单线净热量趋势图
   - 零线虚线
   - 盈余区域红色填充，缺口区域绿色填充
   - 不显示数值标签
   - 近 7 天/近 30 天切换正常
4. **缓存**：清除小程序缓存后重新打开，首次 loading；之后打开秒开

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "feat: 历史记录页面完整实现"
```
