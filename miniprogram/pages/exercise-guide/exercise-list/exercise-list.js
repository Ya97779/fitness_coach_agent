const { exerciseData } = require('../../../data/exercises')

const DIFF_COLOR = {
  beginner: '#a8b5a0',
  intermediate: '#c4a882',
  advanced: '#c47a6c'
}

Page({
  data: {
    group: {},
    subRegions: [],
    activeSub: '',
    filteredExercises: []
  },

  onLoad(options) {
    const group = exerciseData[options.group]
    if (!group) return

    wx.setNavigationBarTitle({ title: group.name })

    // 子区域列表
    let subRegions = group.subRegions || []
    if (subRegions.length === 0) {
      // 没有子区域的肌群（如核心、有氧），用「全部」
      subRegions = [{ id: 'all', name: '全部' }]
    }

    // 给每个动作加上颜色
    const exercises = group.exercises.map(ex => ({
      ...ex,
      diffColor: DIFF_COLOR[ex.difficulty] || '#999'
    }))

    this.setData({
      group: { ...group, exercises },
      subRegions,
      activeSub: subRegions[0].id
    })
    this.filterExercises(subRegions[0].id)
  },

  switchSub(e) {
    const id = e.currentTarget.dataset.id
    this.setData({ activeSub: id })
    this.filterExercises(id)
  },

  filterExercises(subId) {
    const { group } = this.data
    let filtered
    if (subId === 'all') {
      filtered = group.exercises
    } else {
      filtered = group.exercises.filter(ex => ex.subRegion === subId)
    }
    this.setData({ filteredExercises: filtered })
  },

  goDetail(e) {
    const id = e.currentTarget.dataset.id
    wx.navigateTo({
      url: `/pages/exercise-guide/exercise-detail/exercise-detail?id=${id}&group=${this.data.group.id}`
    })
  }
})
