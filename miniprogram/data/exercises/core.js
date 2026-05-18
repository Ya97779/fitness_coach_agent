module.exports = {
  id: 'core',
  name: '核心训练',
  functions: '脊柱屈曲（腹直肌）、脊柱旋转（腹斜肌）、脊柱抗伸展、骨盆前/后倾',
  icon: '/images/guide/core.png',
  exercises: [
    {
      id: 'crunch',
      name: '卷腹',
      difficulty: 'beginner',
      summary: '腹直肌基础孤立动作',
      gif: '/images/guide/crunch.gif',
      equipment: '瑜伽垫',
      targetMuscles: ['腹直肌', '腹横肌'],
      video: 'https://cdn.example.com/guide/crunch.mp4',
      steps: [
        '仰卧在瑜伽垫上，双膝弯曲，双脚平放地面',
        '双手轻放在耳侧或交叉放在胸前',
        '呼气时收紧腹部，将肩胛骨抬离地面',
        '在最高点停留 1 秒，感受腹肌收缩',
        '吸气缓慢下放至起始位置'
      ],
      tips: [
        '不需要完全坐起来，肩胛骨离地即可',
        '发力时呼气，下放时吸气',
        '下巴与胸口保持一拳距离，避免颈部代偿'
      ],
      mistakes: [
        { wrong: '双手抱头用力拉脖子', fix: '手轻放在耳侧，用腹肌发力而非手臂' },
        { wrong: '借助惯性快速弹起', fix: '放慢速度，全程控制，感受腹肌持续发力' },
        { wrong: '下背部离开地面', fix: '保持腰部贴紧地面，只抬起上背部' }
      ],
      variations: [
        { id: 'bicycle-crunch', name: '自行车卷腹', desc: '加入转体动作，同时刺激腹斜肌' },
        { id: 'reverse-crunch', name: '反向卷腹', desc: '抬起骨盆，侧重下腹' }
      ]
    },
    {
      id: 'plank',
      name: '平板支撑',
      difficulty: 'beginner',
      summary: '核心抗伸展等长收缩训练',
      gif: '/images/guide/plank.gif',
      equipment: '瑜伽垫',
      targetMuscles: ['腹直肌', '腹横肌', '竖脊肌', '臀大肌'],
      video: 'https://cdn.example.com/guide/plank.mp4',
      steps: [
        '俯卧，双肘撑地，肘关节在肩膀正下方',
        '双脚与肩同宽，脚尖着地',
        '收紧腹部和臀部，身体从头到脚呈一条直线',
        '保持自然呼吸，维持 30-60 秒',
        '结束后缓慢放下身体，休息'
      ],
      tips: [
        '想象肚脐向脊柱方向收紧',
        '臀部不要塌下去也不要翘太高',
        '眼睛看地面，保持颈椎中立'
      ],
      mistakes: [
        { wrong: '腰部塌陷', fix: '收紧核心和臀部，保持身体一条直线' },
        { wrong: '臀部抬太高', fix: '降低臀部，让身体呈一条直线' },
        { wrong: '憋气', fix: '保持均匀呼吸，不要憋气' }
      ],
      variations: [
        { id: 'side-plank', name: '侧平板支撑', desc: '侧重腹斜肌训练' },
        { id: 'plank-shoulder-tap', name: '平板支撑触肩', desc: '加入抗旋转挑战' }
      ]
    },
    {
      id: 'russian-twist',
      name: '俄罗斯转体',
      difficulty: 'beginner',
      summary: '腹斜肌旋转训练',
      gif: '/images/guide/russian-twist.gif',
      equipment: '瑜伽垫、药球（可选）',
      targetMuscles: ['腹外斜肌', '腹内斜肌', '腹直肌'],
      video: 'https://cdn.example.com/guide/russian-twist.mp4',
      steps: [
        '坐在瑜伽垫上，双膝弯曲，双脚微微离地',
        '身体后倾约 45 度，保持背部挺直',
        '双手合十或握住药球放在胸前',
        '呼气时转动躯干，将手移向身体一侧',
        '吸气回到中间，再呼气转向另一侧'
      ],
      tips: [
        '旋转来自躯干而非手臂，感受腹斜肌发力',
        '双脚离地可以增加难度，初学者可以脚跟着地',
        '保持核心收紧，不要弓背'
      ],
      mistakes: [
        { wrong: '只转动手臂，躯干不动', fix: '想象整个上半身在旋转，不只是手臂摆动' },
        { wrong: '背部弯曲', fix: '挺直背部，胸部打开，与地面保持 45 度角' },
        { wrong: '转动速度过快', fix: '放慢节奏，每侧停留半秒再转' }
      ],
      variations: [
        { id: 'weighted-russian-twist', name: '负重俄罗斯转体', desc: '手持哑铃或药球增加阻力' },
        { id: 'feet-elevated-russian-twist', name: '抬脚俄罗斯转体', desc: '双脚完全离地，核心持续紧张' }
      ]
    },
    {
      id: 'hanging-leg-raise',
      name: '悬垂举腿',
      difficulty: 'intermediate',
      summary: '下腹和髋屈肌复合训练',
      gif: '/images/guide/hanging-leg-raise.gif',
      equipment: '单杠',
      targetMuscles: ['腹直肌（尤其下腹）', '髂腰肌', '股直肌'],
      video: 'https://cdn.example.com/guide/hanging-leg-raise.mp4',
      steps: [
        '双手正握单杠，握距与肩同宽，身体自然悬垂',
        '收紧核心，保持身体稳定不要晃动',
        '呼气时双腿并拢缓慢抬起，直到与地面平行或更高',
        '在最高点停留 1 秒，感受腹部收缩',
        '吸气缓慢放下双腿，回到起始位置'
      ],
      tips: [
        '动作全程控制，不要靠惯性甩腿',
        '骨盆微微后倾可以让腹肌更好地参与',
        '如果直腿太难，可以先做屈膝版本'
      ],
      mistakes: [
        { wrong: '身体大幅晃动', fix: '先稳定核心再抬腿，动作放慢' },
        { wrong: '只弯曲髋关节，腹肌没有参与', fix: '骨盆后卷，想象用腹肌把骨盆卷起来' },
        { wrong: '放下时完全放松', fix: '下放也要控制速度，保持核心紧张' }
      ],
      variations: [
        { id: 'hanging-knee-raise', name: '悬垂屈膝举腿', desc: '屈膝降低难度，适合初学者' },
        { id: 'hanging-windshield-wiper', name: '悬垂雨刷', desc: '举腿后左右摆动，终极腹肌挑战' }
      ]
    },
    {
      id: 'dead-bug',
      name: '死虫式',
      difficulty: 'beginner',
      summary: '核心抗伸展训练，对腰椎友好',
      gif: '/images/guide/dead-bug.gif',
      equipment: '瑜伽垫',
      targetMuscles: ['腹横肌', '腹直肌', '竖脊肌'],
      video: 'https://cdn.example.com/guide/dead-bug.mp4',
      steps: [
        '仰卧，双臂伸直指向天花板，双腿屈膝抬起呈 90 度',
        '腰部贴紧地面，保持骨盆中立',
        '呼气时缓慢放下右手和左腿，直到接近地面',
        '吸气时回到起始位置',
        '换另一侧（左手和右腿）重复'
      ],
      tips: [
        '腰部始终贴紧地面，如果腰部弓起说明幅度太大',
        '动作越慢越好，每侧 3-5 秒完成',
        '对腰椎间盘突出人群非常友好'
      ],
      mistakes: [
        { wrong: '腰部离开地面', fix: '减小动作幅度，确保腰部始终贴地' },
        { wrong: '放下速度过快', fix: '放慢到 3-5 秒，感受核心对抗重力' },
        { wrong: '憋气', fix: '放下时呼气，收回时吸气' }
      ],
      variations: [
        { id: 'dead-bug-with-band', name: '弹力带死虫式', desc: '双手握住弹力带增加抗伸展阻力' },
        { id: 'single-limb-dead-bug', name: '单侧死虫式', desc: '只放一侧肢体，降低协调难度' }
      ]
    },
    {
      id: 'side-plank',
      name: '侧平板支撑',
      difficulty: 'beginner',
      summary: '腹斜肌等长收缩训练',
      gif: '/images/guide/side-plank.gif',
      equipment: '瑜伽垫',
      targetMuscles: ['腹外斜肌', '腹内斜肌', '腹横肌', '臀中肌'],
      video: 'https://cdn.example.com/guide/side-plank.mp4',
      steps: [
        '侧卧，下方手肘撑地，肘关节在肩膀正下方',
        '双脚叠放或前后错开（前后错开更稳定）',
        '收紧核心和臀部，将身体抬起',
        '身体从头到脚呈一条直线',
        '保持 20-30 秒，换另一侧'
      ],
      tips: [
        '初学者可以膝盖着地降低难度',
        '上方手叉腰或伸向天花板保持平衡',
        '保持颈部中立，眼睛看前方'
      ],
      mistakes: [
        { wrong: '臀部下沉', fix: '收紧臀部和核心，保持身体一条直线' },
        { wrong: '身体前倾或后仰', fix: '肩膀、髋部、脚踝在同一个平面' },
        { wrong: '肩膀耸起靠近耳朵', fix: '肩膀下沉远离耳朵，保护肩关节' }
      ],
      variations: [
        { id: 'side-plank-hip-dip', name: '侧平板支撑髋部下沉', desc: '加入髋部上下运动，动态刺激腹斜肌' },
        { id: 'side-plank-leg-lift', name: '侧平板支撑抬腿', desc: '上方腿抬起，增加臀中肌和核心挑战' }
      ]
    }
  ]
}
