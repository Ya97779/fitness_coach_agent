module.exports = {
  id: 'arms',
  name: '手臂训练',
  functions: '肘屈（肱二头肌）、肘伸（肱三头肌）、前臂旋前/旋后',
  icon: '/images/guide/arms.png',
  subRegions: [
    { id: 'biceps', name: '肱二头肌' },
    { id: 'triceps', name: '肱三头肌' },
    { id: 'forearm', name: '小臂' }
  ],
  exercises: [
    // === 肱二头肌 ===
    {
      id: 'barbell-curl',
      name: '杠铃弯举',
      subRegion: 'biceps',
      difficulty: 'beginner',
      summary: '二头肌基础动作，可使用大重量',
      gif: '/images/guide/barbell-curl.gif',
      equipment: '杠铃',
      targetMuscles: ['肱二头肌', '肱肌'],
      video: '',
      steps: ['双手反握杠铃，与肩同宽', '上臂夹紧身体两侧', '发力弯举至顶峰收缩', '有控制地下放'],
      tips: ['上臂不要前后晃动', '顶峰停留 1 秒', '下放时完全伸直'],
      mistakes: [{ wrong: '身体晃动借力', fix: '核心收紧，上臂固定' }, { wrong: '下放不完全', fix: '手臂完全伸直再弯举' }],
      variations: [{ id: 'ez-bar-curl', name: 'EZ 杠弯举', desc: '对手腕更友好' }]
    },
    {
      id: 'hammer-curl',
      name: '锤式弯举',
      subRegion: 'biceps',
      difficulty: 'beginner',
      summary: '中立握法，同时刺激肱肌和前臂',
      gif: '/images/guide/hammer-curl.gif',
      equipment: '哑铃',
      targetMuscles: ['肱肌', '肱桡肌', '肱二头肌'],
      video: '',
      steps: ['双手持哑铃，掌心相对', '发力弯举至肩部', '有控制地下放'],
      tips: ['全程保持掌心相对', '可以交替进行'],
      mistakes: [{ wrong: '手腕旋转', fix: '保持中立握法' }],
      variations: []
    },
    {
      id: 'preacher-curl',
      name: '牧师凳弯举',
      subRegion: 'biceps',
      difficulty: 'beginner',
      summary: '孤立二头肌，防止借力',
      gif: '/images/guide/preacher-curl.gif',
      equipment: '牧师凳 + 哑铃/杠铃',
      targetMuscles: ['肱二头肌'],
      video: '',
      steps: ['上臂放在牧师凳靠垫上', '发力弯举至顶峰', '有控制地下放至手臂伸直'],
      tips: ['下放要完全，拉伸二头肌', '不要使用过大重量'],
      mistakes: [{ wrong: '臀部离开座位', fix: '坐稳，不要借力' }],
      variations: []
    },

    // === 肱三头肌 ===
    {
      id: 'close-grip-bench',
      name: '窄距卧推',
      subRegion: 'triceps',
      difficulty: 'intermediate',
      summary: '三头肌复合动作，可上大重量',
      gif: '/images/guide/close-grip-bench.gif',
      equipment: '杠铃、卧推凳',
      targetMuscles: ['肱三头肌', '胸大肌内侧'],
      video: '',
      steps: ['仰卧在卧推凳上，双手握距与肩同宽或略窄', '下放杠铃至胸部', '发力推起至手臂伸直'],
      tips: ['握距不要过窄，保护手腕', '手肘贴近身体'],
      mistakes: [{ wrong: '握距太窄伤手腕', fix: '保持与肩同宽或略窄' }],
      variations: [{ id: 'tricep-pushdown', name: '绳索下压', desc: '孤立三头肌' }]
    },
    {
      id: 'tricep-pushdown',
      name: '绳索下压',
      subRegion: 'triceps',
      difficulty: 'beginner',
      summary: '三头肌孤立动作，感受度好',
      gif: '/images/guide/tricep-pushdown.gif',
      equipment: '龙门架 + 绳索',
      targetMuscles: ['肱三头肌'],
      video: '',
      steps: ['面对龙门架站立，双手握住绳索', '上臂夹紧身体', '发力下压至手臂完全伸直', '有控制地回到起始位置'],
      tips: ['只动前臂，上臂固定', '顶峰用力挤压三头肌'],
      mistakes: [{ wrong: '身体前倾借力', fix: '躯干保持直立' }],
      variations: []
    },
    {
      id: 'skull-crusher',
      name: '颈后臂屈伸',
      subRegion: 'triceps',
      difficulty: 'intermediate',
      summary: '三头肌长头重点刺激',
      gif: '/images/guide/skull-crusher.gif',
      equipment: '哑铃/EZ 杠、卧推凳',
      targetMuscles: ['肱三头肌（长头）'],
      video: '',
      steps: ['仰卧在卧推凳上，手臂伸直举在面部上方', '只弯曲肘部，将重量下放至额头附近', '发力推回起始位置'],
      tips: ['上臂保持固定，只动前臂', '下放时吸气，推起时呼气'],
      mistakes: [{ wrong: '上臂晃动', fix: '上臂垂直地面，保持不动' }],
      variations: []
    },

    // === 小臂 ===
    {
      id: 'wrist-curl',
      name: '腕弯举',
      subRegion: 'forearm',
      difficulty: 'beginner',
      summary: '前臂屈肌群孤立训练',
      gif: '/images/guide/wrist-curl.gif',
      equipment: '哑铃',
      targetMuscles: ['前臂屈肌群'],
      video: '',
      steps: ['坐在凳子上，前臂放在大腿上，手腕悬空', '掌心向上握住哑铃', '手腕向上弯举', '有控制地下放'],
      tips: ['只动手腕，前臂不动', '可以放在膝盖上做'],
      mistakes: [{ wrong: '前臂抬起借力', fix: '前臂固定不动' }],
      variations: []
    },
    {
      id: 'reverse-wrist-curl',
      name: '反握腕弯举',
      subRegion: 'forearm',
      difficulty: 'beginner',
      summary: '前臂伸肌群训练，提升握力',
      gif: '/images/guide/reverse-wrist-curl.gif',
      equipment: '哑铃',
      targetMuscles: ['前臂伸肌群'],
      video: '',
      steps: ['坐在凳子上，前臂放在大腿上，手腕悬空', '掌心向下握住哑铃', '手腕向上弯举', '有控制地下放'],
      tips: ['重量比正握轻一些', '控制速度'],
      mistakes: [{ wrong: '甩动借力', fix: '控制动作节奏' }],
      variations: []
    },
    {
      id: 'farmer-walk',
      name: '农夫行走',
      subRegion: 'forearm',
      difficulty: 'beginner',
      summary: '功能性握力训练，同时锻炼核心',
      gif: '/images/guide/farmer-walk.gif',
      equipment: '哑铃/壶铃',
      targetMuscles: ['前臂屈肌群', '核心', '斜方肌'],
      video: '',
      steps: ['双手各持重物', '挺胸收腹，正常行走', '保持握紧，走一定距离或时间'],
      tips: ['握紧不要松手', '保持正常步态'],
      mistakes: [{ wrong: '身体前倾', fix: '挺胸收腹，保持直立' }],
      variations: []
    }
  ]
}
