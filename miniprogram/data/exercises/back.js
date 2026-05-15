module.exports = {
  id: 'back',
  name: '背部训练',
  functions: '肩关节水平外展、肩伸、肩外旋、脊柱伸展',
  icon: '/images/guide/back.png',
  subRegions: [
    { id: 'lats', name: '背阔肌' },
    { id: 'upper', name: '上背' },
    { id: 'lower', name: '下背' }
  ],
  exercises: [
    // === 背阔肌 ===
    {
      id: 'pull-up',
      name: '引体向上',
      subRegion: 'lats',
      difficulty: 'intermediate',
      summary: '自重背部王牌动作，侧重背阔肌宽度',
      gif: '/images/guide/pull-up.gif',
      equipment: '单杠',
      targetMuscles: ['背阔肌', '大圆肌', '肱二头肌'],
      video: '',
      steps: ['双手正握单杠，握距略宽于肩', '收紧核心，发力拉起身体至下巴过杠', '缓慢下放至手臂完全伸直'],
      tips: ['拉起时想象用肘部往下拉', '不要借力甩动身体', '下放时吸气，拉起时呼气'],
      mistakes: [{ wrong: '身体晃动借力', fix: '收紧核心，双腿微微前倾保持稳定' }, { wrong: '幅度不够', fix: '手臂完全伸直再拉起' }],
      variations: [{ id: 'lat-pulldown', name: '高位下拉', desc: '器械替代，可调节重量' }]
    },
    {
      id: 'lat-pulldown',
      name: '高位下拉',
      subRegion: 'lats',
      difficulty: 'beginner',
      summary: '引体向上的器械替代，可调节重量',
      gif: '/images/guide/lat-pulldown.gif',
      equipment: '高位下拉机',
      targetMuscles: ['背阔肌', '大圆肌', '肱二头肌'],
      video: '',
      steps: ['坐在高位下拉机上，大腿固定', '双手宽握横杆', '发力将横杆拉至锁骨位置', '缓慢回到起始位置'],
      tips: ['身体微微后倾约 15 度', '拉起时挤压背部', '不要用手臂硬拉'],
      mistakes: [{ wrong: '身体过度后仰', fix: '保持微倾，不要借力' }],
      variations: [{ id: 'pull-up', name: '引体向上', desc: '自重版本' }]
    },
    {
      id: 'straight-arm-pulldown',
      name: '绳索直臂下压',
      subRegion: 'lats',
      difficulty: 'beginner',
      summary: '孤立背阔肌的动作，感受度好',
      gif: '/images/guide/straight-arm-pulldown.gif',
      equipment: '龙门架',
      targetMuscles: ['背阔肌', '大圆肌'],
      video: '',
      steps: ['站在龙门架前，双手握住绳索', '手臂伸直，从高位向下压至大腿前侧', '缓慢回到起始位置'],
      tips: ['全程手臂保持伸直', '用力挤压背阔肌'],
      mistakes: [{ wrong: '弯曲手臂变成三头肌动作', fix: '手臂始终保持伸直' }],
      variations: []
    },

    // === 上背 ===
    {
      id: 'barbell-row',
      name: '杠铃划船',
      subRegion: 'upper',
      difficulty: 'intermediate',
      summary: '增加背部厚度的核心复合动作',
      gif: '/images/guide/barbell-row.gif',
      equipment: '杠铃',
      targetMuscles: ['背阔肌', '菱形肌', '斜方肌中束', '三角肌后束'],
      video: '',
      steps: ['双脚与肩同宽，屈髋俯身约 45 度', '双手握距与肩同宽，握住杠铃', '发力将杠铃拉至腹部', '缓慢下放至手臂伸直'],
      tips: ['拉起时挤压肩胛骨', '保持背部挺直，不要弓背', '核心收紧，身体不要晃动'],
      mistakes: [{ wrong: '弓背拉起', fix: '挺胸收腹，保持脊柱中立' }, { wrong: '身体晃动借力', fix: '降低重量，控制动作' }],
      variations: [{ id: 'seated-row', name: '坐姿划船', desc: '对脊柱压力更小' }]
    },
    {
      id: 'seated-row',
      name: '坐姿划船',
      subRegion: 'upper',
      difficulty: 'beginner',
      summary: '中背部厚度训练，对脊柱压力小',
      gif: '/images/guide/seated-row.gif',
      equipment: '坐姿划船机',
      targetMuscles: ['背阔肌', '菱形肌', '斜方肌中束'],
      video: '',
      steps: ['坐在划船机上，双脚踩实', '双手握住手柄', '发力将手柄拉至腹部', '缓慢回到起始位置'],
      tips: ['拉起时挺胸，挤压肩胛骨', '身体不要过度前后晃动'],
      mistakes: [{ wrong: '身体大幅前后晃动', fix: '固定躯干，只用手臂和背部发力' }],
      variations: [{ id: 'barbell-row', name: '杠铃划船', desc: '自由重量版本' }]
    },
    {
      id: 'reverse-fly',
      name: '反向飞鸟',
      subRegion: 'upper',
      difficulty: 'beginner',
      summary: '孤立三角肌后束和上背，改善圆肩',
      gif: '/images/guide/reverse-fly.gif',
      equipment: '哑铃或反向飞鸟机',
      targetMuscles: ['三角肌后束', '菱形肌', '斜方肌中束'],
      video: '',
      steps: ['俯身或坐在反向飞鸟机上', '双手向两侧打开', '挤压肩胛骨，缓慢回到起始位置'],
      tips: ['想象用肩胛骨夹笔', '控制速度，不要甩动'],
      mistakes: [{ wrong: '用手臂力量甩', fix: '用上背和后束发力' }],
      variations: []
    },

    // === 下背 ===
    {
      id: 'deadlift',
      name: '硬拉',
      subRegion: 'lower',
      difficulty: 'advanced',
      summary: '全身复合动作，强化后链和下背',
      gif: '/images/guide/deadlift.gif',
      equipment: '杠铃',
      targetMuscles: ['竖脊肌', '臀大肌', '腘绳肌', '背阔肌'],
      video: '',
      steps: ['双脚与肩同宽站立，杠铃在脚中上方', '屈髋屈膝握住杠铃', '挺胸收腹，发力站起', '有控制地下放杠铃'],
      tips: ['全程保持背部挺直', '杠铃紧贴身体', '用臀部和腿发力，不是背部'],
      mistakes: [{ wrong: '弓背拉起', fix: '挺胸收腹，如果弓背说明重量太大' }, { wrong: '杠铃远离身体', fix: '杠铃紧贴小腿和大腿' }],
      variations: [{ id: 'romanian-deadlift', name: '罗马尼亚硬拉', desc: '更侧重腘绳肌和臀部' }]
    },
    {
      id: 'hyperextension',
      name: '山羊挺身',
      subRegion: 'lower',
      difficulty: 'beginner',
      summary: '下背孤立动作，强化竖脊肌',
      gif: '/images/guide/hyperextension.gif',
      equipment: '山羊挺身凳',
      targetMuscles: ['竖脊肌', '臀大肌'],
      video: '',
      steps: ['俯卧在山羊挺身凳上，脚踝固定', '身体从髋部向下弯曲', '发力抬起身体至与腿呈一条直线'],
      tips: ['不要过度后仰', '可以抱在胸前增加难度'],
      mistakes: [{ wrong: '过度后仰', fix: '抬起到身体平直即可' }],
      variations: []
    }
  ]
}
