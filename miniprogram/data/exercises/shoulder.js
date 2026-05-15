module.exports = {
  id: 'shoulder',
  name: '肩部训练',
  functions: '肩屈（前束）、肩伸（后束）、肩外展（中束）、肩内旋/外旋',
  icon: '/images/guide/shoulder.png',
  subRegions: [
    { id: 'front', name: '前束' },
    { id: 'side', name: '中束' },
    { id: 'rear', name: '后束' }
  ],
  exercises: [
    // === 前束 ===
    {
      id: 'overhead-press',
      name: '站姿推举',
      subRegion: 'front',
      difficulty: 'intermediate',
      summary: '肩部核心复合动作，强化前束和中束',
      gif: '/images/guide/overhead-press.gif',
      equipment: '杠铃',
      targetMuscles: ['三角肌前束', '三角肌中束', '肱三头肌'],
      video: '',
      steps: ['双脚与肩同宽站立，杠铃置于锁骨位置', '发力将杠铃推举至头顶', '有控制地下放至起始位置'],
      tips: ['推举时头部稍微后仰让路', '锁定时手臂完全伸直', '核心收紧，不要过度弓腰'],
      mistakes: [{ wrong: '过度弓腰', fix: '收紧核心和臀部' }, { wrong: '杠铃轨迹偏离', fix: '杠铃贴近面部走直线' }],
      variations: [{ id: 'arnold-press', name: '阿诺德推举', desc: '旋转推举，全面刺激三角肌' }]
    },
    {
      id: 'arnold-press',
      name: '阿诺德推举',
      subRegion: 'front',
      difficulty: 'intermediate',
      summary: '旋转推举，全面刺激三角肌三个束',
      gif: '/images/guide/arnold-press.gif',
      equipment: '哑铃',
      targetMuscles: ['三角肌前束', '三角肌中束', '三角肌后束'],
      video: '',
      steps: ['双手持哑铃举在胸前，掌心朝向自己', '推举同时旋转手腕，至顶部掌心朝前', '下放时反向旋转回到起始位置'],
      tips: ['动作流畅，推举和旋转同步', '控制速度，不要甩动'],
      mistakes: [{ wrong: '旋转和推举脱节', fix: '保持动作流畅同步' }],
      variations: [{ id: 'overhead-press', name: '站姿推举', desc: '基础推举动作' }]
    },
    {
      id: 'front-raise',
      name: '前平举',
      subRegion: 'front',
      difficulty: 'beginner',
      summary: '孤立三角肌前束',
      gif: '/images/guide/front-raise.gif',
      equipment: '哑铃',
      targetMuscles: ['三角肌前束'],
      video: '',
      steps: ['双手持哑铃，手臂伸直在身体前方', '发力将哑铃举至与肩齐平', '缓慢下放至起始位置'],
      tips: ['手臂微微弯曲', '交替或同时进行均可'],
      mistakes: [{ wrong: '身体晃动借力', fix: '核心收紧，控制重量' }],
      variations: []
    },

    // === 中束 ===
    {
      id: 'lateral-raise',
      name: '侧平举',
      subRegion: 'side',
      difficulty: 'beginner',
      summary: '孤立三角肌中束，打造肩部宽度',
      gif: '/images/guide/lateral-raise.gif',
      equipment: '哑铃',
      targetMuscles: ['三角肌中束'],
      video: '',
      steps: ['双手持哑铃在身体两侧', '手臂微弯，向两侧举起至与肩齐平', '有控制地下放'],
      tips: ['想象倒水的动作，小拇指微微上翻', '不要耸肩', '使用能控制的重量'],
      mistakes: [{ wrong: '重量太大甩动', fix: '降低重量，感受中束发力' }, { wrong: '耸肩', fix: '放松斜方肌，专注中束' }],
      variations: []
    },
    {
      id: 'cable-lateral-raise',
      name: '绳索侧平举',
      subRegion: 'side',
      difficulty: 'beginner',
      summary: '全程恒定张力的中束训练',
      gif: '/images/guide/cable-lateral-raise.gif',
      equipment: '龙门架',
      targetMuscles: ['三角肌中束'],
      video: '',
      steps: ['站在龙门架侧面，单手握住低位绳索', '手臂微弯，向侧上方举起', '有控制地下放'],
      tips: ['全程保持张力', '比哑铃侧平举更稳定'],
      mistakes: [{ wrong: '身体侧倾借力', fix: '保持躯干稳定' }],
      variations: []
    },

    // === 后束 ===
    {
      id: 'reverse-fly-shoulder',
      name: '俯身飞鸟',
      subRegion: 'rear',
      difficulty: 'beginner',
      summary: '孤立三角肌后束，改善圆肩',
      gif: '/images/guide/reverse-fly.gif',
      equipment: '哑铃',
      targetMuscles: ['三角肌后束', '菱形肌'],
      video: '',
      steps: ['俯身约 90 度，双手持哑铃', '手臂微弯，向两侧打开', '挤压肩胛骨，缓慢下放'],
      tips: ['想象用肩胛骨夹笔', '控制速度，不要甩动'],
      mistakes: [{ wrong: '身体抬起借力', fix: '保持俯身角度不变' }],
      variations: []
    },
    {
      id: 'face-pull',
      name: '面拉',
      subRegion: 'rear',
      difficulty: 'beginner',
      summary: '强化后束和外旋肌群，改善肩部健康',
      gif: '/images/guide/face-pull.gif',
      equipment: '龙门架 + 绳索',
      targetMuscles: ['三角肌后束', '肩外旋肌群', '菱形肌'],
      video: '',
      steps: ['将龙门架滑轮调至面部高度', '双手握住绳索，拉向面部两侧', '外旋手臂，挤压后束'],
      tips: ['拉至大臂与地面平行', '外旋时双手在耳朵两侧'],
      mistakes: [{ wrong: '用手臂硬拉', fix: '用后束和肩胛骨发力' }],
      variations: []
    }
  ]
}
