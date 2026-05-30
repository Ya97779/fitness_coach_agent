const TEMPLATES = [
  {
    id: 'chest', name: '胸部训练',
    exercises: [
      { name: '平板卧推', sets: 4, rest: 240, weight: 0 },
      { name: '上斜卧推', sets: 4, rest: 240, weight: 0 },
      { name: '龙门架夹胸', sets: 4, rest: 180, weight: 0 },
      { name: '双杠臂屈伸', sets: 4, rest: 180, weight: 0 }
    ]
  },
  {
    id: 'back', name: '背部训练',
    exercises: [
      { name: '引体向上', sets: 4, rest: 240, weight: 0 },
      { name: '杠铃划船', sets: 4, rest: 180, weight: 0 },
      { name: '高位下拉', sets: 4, rest: 180, weight: 0 },
      { name: '坐姿划船', sets: 4, rest: 180, weight: 0 }
    ]
  },
  {
    id: 'shoulder', name: '肩部训练',
    exercises: [
      { name: '站姿推举', sets: 4, rest: 180, weight: 0 },
      { name: '哑铃侧平举', sets: 4, rest: 180, weight: 0 },
      { name: '俯身哑铃飞鸟', sets: 4, rest: 180, weight: 0 },
      { name: '绳索面拉', sets: 4, rest: 180, weight: 0 }
    ]
  },
  {
    id: 'arms', name: '手臂训练',
    exercises: [
      { name: '杠铃弯举', sets: 4, rest: 120, weight: 0 },
      { name: '锤式弯举', sets: 4, rest: 120, weight: 0 },
      { name: '哑铃碎颅者', sets: 4, rest: 120, weight: 0 },
      { name: '绳索下压', sets: 4, rest:120, weight: 0 }
    ]
  },
  {
    id: 'legs', name: '腿部训练',
    exercises: [
      { name: '杠铃深蹲', sets: 4, rest: 240, weight: 0 },
      { name: '倒蹬', sets: 4, rest: 240, weight: 0 },
      { name: '器械腿屈伸', sets: 4, rest: 180, weight: 0 },
      { name: '卧姿腿弯举', sets: 4, rest: 180, weight: 0 }
    ]
  },
  {
    id: 'core', name: '核心训练',
    exercises: [
      { name: '卷腹', sets: 4, rest: 45, weight: 0 },
      { name: '俄罗斯转体', sets: 4, rest: 45, weight: 0 },
      { name: '悬垂举腿', sets: 4, rest: 60, weight: 0 }
    ]
  },
  {
    id: 'cardio', name: '有氧减脂',
    exercises: [
      { name: '波比跳', sets: 4, rest: 60, weight: 0 },
      { name: '开合跳', sets: 4, rest: 45, weight: 0 },
      { name: '爬坡', sets: 1, rest: 1800, weight: 0 },
    ]
  }
]

module.exports = { TEMPLATES }
