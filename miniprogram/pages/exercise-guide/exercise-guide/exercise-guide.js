const { groupList, exerciseData } = require('../../../data/exercises')

const EMOJI_MAP = {
  chest: '💪', back: '🔙', shoulder: '🏋️',
  arms: '💪', legs: '🦵', core: '🎯', cardio: '🏃'
}

Page({
  data: {
    groups: [],
    searchText: '',
    searchResults: []
  },

  onLoad() {
    const groups = groupList.map(g => ({
      ...g,
      emoji: EMOJI_MAP[g.id] || '🏃'
    }))
    this.setData({ groups })
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
  }
})
