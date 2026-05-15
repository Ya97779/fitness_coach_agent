const MEAL_MAP = {
  breakfast: '早餐', lunch: '午餐', dinner: '晚餐', snack: '加餐'
}

Component({
  properties: {
    item: { type: Object, value: {} }
  },
  computed: {},
  data: {
    mealTypeText: ''
  },
  observers: {
    'item.meal_type': function(val) {
      this.setData({ mealTypeText: MEAL_MAP[val] || '' })
    }
  },
  methods: {
    onTap() {
      this.triggerEvent('tap', { item: this.data.item })
    }
  }
})
