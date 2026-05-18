const TEMPLATES = [
  {
    id: 'chest', icon: '💪', name: '胸部训练',
    exercises: [
      { name: '杠铃卧推', sets: 4, rest: 90 },
      { name: '上斜哑铃卧推', sets: 3, rest: 75 },
      { name: '龙门架夹胸', sets: 3, rest: 60 },
      { name: '双杠臂屈伸', sets: 3, rest: 90 }
    ]
  },
  {
    id: 'back', icon: '🔙', name: '背部训练',
    exercises: [
      { name: '引体向上', sets: 4, rest: 90 },
      { name: '杠铃划船', sets: 4, rest: 90 },
      { name: '高位下拉', sets: 3, rest: 60 },
      { name: '坐姿划船', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'shoulder', icon: '🏋️', name: '肩部训练',
    exercises: [
      { name: '站姿推举', sets: 4, rest: 90 },
      { name: '侧平举', sets: 3, rest: 60 },
      { name: '俯身飞鸟', sets: 3, rest: 60 },
      { name: '面拉', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'arms', icon: '💪', name: '手臂训练',
    exercises: [
      { name: '杠铃弯举', sets: 3, rest: 60 },
      { name: '锤式弯举', sets: 3, rest: 60 },
      { name: '窄距卧推', sets: 3, rest: 75 },
      { name: '绳索下压', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'legs', icon: '🦵', name: '腿部训练',
    exercises: [
      { name: '深蹲', sets: 5, rest: 120 },
      { name: '腿举', sets: 4, rest: 90 },
      { name: '腿屈伸', sets: 3, rest: 60 },
      { name: '腿弯举', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'core', icon: '🎯', name: '核心训练',
    exercises: [
      { name: '卷腹', sets: 3, rest: 45 },
      { name: '平板支撑', sets: 3, rest: 45 },
      { name: '俄罗斯转体', sets: 3, rest: 45 },
      { name: '悬垂举腿', sets: 3, rest: 60 }
    ]
  },
  {
    id: 'cardio', icon: '🏃', name: '有氧减脂',
    exercises: [
      { name: '波比跳', sets: 4, rest: 60 },
      { name: '开合跳', sets: 4, rest: 45 },
      { name: '高抬腿', sets: 4, rest: 45 },
      { name: '登山跑', sets: 4, rest: 45 }
    ]
  }
]

module.exports = { TEMPLATES }
