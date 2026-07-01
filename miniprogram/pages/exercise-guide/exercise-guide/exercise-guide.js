const { groupList, exerciseData } = require('../../../data/exercises')

Page({
  data: {
    groups: [],
    searchText: '',
    searchResults: []
  },

  onLoad() {
    this.setData({ groups: groupList })
  },

  onSearch(e) {
    const text = e.detail.value.trim().toLowerCase()
    this.setData({ searchText: text })

    if (!text) {
      this.setData({ searchResults: [] })
      return
    }

    const results = []
    for (const group of groupList) {
      for (const ex of group.exercises) {
        if (ex.name.toLowerCase().includes(text) || ex.summary.toLowerCase().includes(text)) {
          results.push({ ...ex, groupId: group.id, groupName: group.name })
        }
      }
    }
    this.setData({ searchResults: results })
  },

  goList(e) {
    wx.navigateTo({
      url: `/pages/exercise-guide/exercise-list/exercise-list?group=${e.currentTarget.dataset.id}`
    })
  },

  goDetail(e) {
    const { id, group } = e.currentTarget.dataset
    wx.navigateTo({
      url: `/pages/exercise-guide/exercise-detail/exercise-detail?id=${id}&group=${group}`
    })
  },

  onShareAppMessage() {
    return {
      title: '健身助手Agent - 健身动作指导',
      path: '/pages/exercise-guide/exercise-guide/exercise-guide'
    }
  },

  onShareTimeline() {
    return {
      title: '健身助手Agent - 健身动作指导'
    }
  }
})
