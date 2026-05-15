module.exports = {
  id: 'legs',
  name: '腿部训练',
  functions: '膝伸（股四头肌）、膝屈（腘绳肌）、髋伸（臀大肌）、踝跖屈（小腿）',
  icon: '/images/guide/legs.png',
  subRegions: [
    { id: 'quad', name: '股四头肌' },
    { id: 'hamstring', name: '腘绳肌' },
    { id: 'glute', name: '臀部' }
  ],
  exercises: [
    // === 股四头肌 ===
    {
      id: 'squat',
      name: '深蹲',
      subRegion: 'quad',
      difficulty: 'intermediate',
      summary: '腿部王牌动作，全身复合力量',
      gif: '/images/guide/squat.gif',
      equipment: '杠铃',
      targetMuscles: ['股四头肌', '臀大肌', '腘绳肌', '核心'],
      video: '',
      steps: ['杠铃放在斜方肌上，双脚与肩同宽', '屈髋屈膝下蹲至大腿与地面平行', '发力站起至起始位置'],
      tips: ['膝盖方向与脚尖一致', '挺胸收腹，背部挺直', '下蹲时吸气，站起时呼气'],
      mistakes: [{ wrong: '膝盖内扣', fix: '膝盖始终对准脚尖方向' }, { wrong: '弓背', fix: '挺胸收腹，目视前方' }],
      variations: [{ id: 'front-squat', name: '前蹲', desc: '杠铃在前方，更多刺激股四头肌' }]
    },
    {
      id: 'leg-press',
      name: '腿举',
      subRegion: 'quad',
      difficulty: 'beginner',
      summary: '器械替代深蹲，对脊柱压力小',
      gif: '/images/guide/leg-press.gif',
      equipment: '腿举机',
      targetMuscles: ['股四头肌', '臀大肌'],
      video: '',
      steps: ['坐在腿举机上，双脚踩在踏板上', '解锁安全锁', '屈膝下放至约 90 度', '发力推起'],
      tips: ['膝盖不要内扣', '下放时不要让臀部离开座位'],
      mistakes: [{ wrong: '膝盖内扣', fix: '保持膝盖与脚尖同向' }, { wrong: '下放过深', fix: '臀部不要离开座位' }],
      variations: []
    },
    {
      id: 'leg-extension',
      name: '腿屈伸',
      subRegion: 'quad',
      difficulty: 'beginner',
      summary: '股四头肌孤立动作',
      gif: '/images/guide/leg-extension.gif',
      equipment: '腿屈伸机',
      targetMuscles: ['股四头肌'],
      video: '',
      steps: ['坐在机器上，小腿抵住垫子', '发力伸直膝盖', '有控制地下放'],
      tips: ['顶峰收缩 1 秒', '不要使用惯性甩动'],
      mistakes: [{ wrong: '甩动借力', fix: '控制速度，慢起慢放' }],
      variations: []
    },

    // === 腘绳肌 ===
    {
      id: 'leg-curl',
      name: '腿弯举',
      subRegion: 'hamstring',
      difficulty: 'beginner',
      summary: '腘绳肌孤立动作',
      gif: '/images/guide/leg-curl.gif',
      equipment: '腿弯举机',
      targetMuscles: ['腘绳肌'],
      video: '',
      steps: ['俯卧在机器上，脚踝抵住垫子', '发力弯举小腿', '有控制地下放'],
      tips: ['臀部不要翘起', '顶峰挤压腘绳肌'],
      mistakes: [{ wrong: '臀部抬起借力', fix: '髋部贴紧垫子' }],
      variations: []
    },
    {
      id: 'romanian-deadlift',
      name: '罗马尼亚硬拉',
      subRegion: 'hamstring',
      difficulty: 'intermediate',
      summary: '腘绳肌和臀部为主，强化后链',
      gif: '/images/guide/romanian-deadlift.gif',
      equipment: '杠铃',
      targetMuscles: ['腘绳肌', '臀大肌', '竖脊肌'],
      video: '',
      steps: ['双手持杠铃站立', '保持膝盖微弯，屈髋将杠铃沿大腿下放', '感受腘绳肌拉伸后回到起始位置'],
      tips: ['杠铃紧贴大腿', '背部始终保持挺直', '下放到腘绳肌有拉伸感即可'],
      mistakes: [{ wrong: '弓背下放', fix: '挺胸收腹，背部挺直' }, { wrong: '膝盖弯曲太多', fix: '膝盖微弯即可，主要是髋关节运动' }],
      variations: [{ id: 'deadlift', name: '传统硬拉', desc: '更多刺激下背' }]
    },

    // === 臀部 ===
    {
      id: 'hip-thrust',
      name: '臀推',
      subRegion: 'glute',
      difficulty: 'intermediate',
      summary: '臀大肌最佳孤立动作',
      gif: '/images/guide/hip-thrust.gif',
      equipment: '杠铃、卧推凳',
      targetMuscles: ['臀大肌'],
      video: '',
      steps: ['上背靠在卧推凳上，杠铃放在髋部', '双脚踩实地面', '发力将髋部推起至身体呈一条直线', '有控制地下放'],
      tips: ['顶峰用力挤压臀部', '下巴微收，目视前方', '不要过度弓腰'],
      mistakes: [{ wrong: '过度弓腰', fix: '收紧核心，骨盆后倾' }, { wrong: '脚的位置不对', fix: '小腿在顶峰时应垂直地面' }],
      variations: []
    },
    {
      id: 'bulgarian-split-squat',
      name: '保加利亚分腿蹲',
      subRegion: 'glute',
      difficulty: 'intermediate',
      summary: '单腿训练，改善左右不平衡',
      gif: '/images/guide/bulgarian-split-squat.gif',
      equipment: '哑铃、卧推凳',
      targetMuscles: ['臀大肌', '股四头肌'],
      video: '',
      steps: ['后脚脚背搭在凳子上', '前脚向前一步', '屈膝下蹲至前大腿与地面平行', '发力站起'],
      tips: ['躯干微微前倾更多刺激臀部', '前膝不要超过脚尖太多'],
      mistakes: [{ wrong: '身体晃动', fix: '核心收紧，保持平衡' }],
      variations: []
    }
  ]
}
