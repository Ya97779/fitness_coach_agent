module.exports = {
  id: 'cardio',
  name: '有氧减脂',
  functions: '全身复合运动，提升心率，以燃脂为主要目标',
  icon: '/images/guide/cardio.png',
  exercises: [
    {
      id: 'running',
      name: '跑步',
      difficulty: 'beginner',
      summary: '最经典的有氧运动，燃脂效率高',
      gif: '/images/guide/running.gif',
      equipment: '跑鞋',
      targetMuscles: ['股四头肌', '腘绳肌', '臀大肌', '小腿三头肌', '核心肌群'],
      video: 'https://cdn.example.com/guide/running.mp4',
      steps: [
        '热身 5 分钟：慢走 + 动态拉伸',
        '保持身体微微前倾，目视前方',
        '手臂自然摆动，肘部弯曲约 90 度',
        '前脚掌或全脚掌着地，避免脚后跟重击',
        '保持均匀呼吸：两步一吸、两步一呼',
        '结束后慢走 5 分钟冷却，静态拉伸'
      ],
      tips: [
        '初学者用「能边跑边说话」的配速，不要追求速度',
        '每周跑量增加不超过 10%，避免受伤',
        '选择减震好的跑鞋，保护膝盖'
      ],
      mistakes: [
        { wrong: '步幅过大', fix: '缩短步幅、提高步频，减少关节冲击' },
        { wrong: '身体后仰或过度前倾', fix: '保持身体微微前倾，从脚踝处前倾而非腰部' },
        { wrong: '手臂横向摆动', fix: '手臂前后摆动，不要越过身体中线' }
      ],
      variations: [
        { id: 'interval-running', name: '间歇跑', desc: '快跑 1 分钟 + 慢跑 2 分钟交替，燃脂效率更高' },
        { id: 'incline-running', name: '上坡跑', desc: '增加坡度提升心率和臀部发力' }
      ]
    },
    {
      id: 'jump-rope',
      name: '跳绳',
      difficulty: 'beginner',
      summary: '高效燃氧，提升协调性',
      gif: '/images/guide/jump-rope.gif',
      equipment: '跳绳',
      targetMuscles: ['小腿三头肌', '前臂肌群', '核心肌群', '三角肌'],
      video: 'https://cdn.example.com/guide/jump-rope.mp4',
      steps: [
        '选择合适长度的跳绳：踩住绳子，手柄到腋下',
        '双脚并拢站立，手肘贴近身体两侧',
        '手腕发力转动跳绳，不要大臂甩绳',
        '前脚掌着地，膝盖微屈缓冲',
        '跳跃高度控制在 2-3 厘米即可',
        '保持节奏，初学者每组 1-2 分钟'
      ],
      tips: [
        '手腕发力而非手臂，省力且容易保持节奏',
        '跳绳前做好脚踝热身',
        '室内跳绳建议用减震垫，保护关节'
      ],
      mistakes: [
        { wrong: '跳得太高', fix: '只需跳过绳子即可，约 2-3 厘米' },
        { wrong: '大臂甩绳', fix: '手肘夹紧身体，只用手腕转动' },
        { wrong: '脚后跟着地', fix: '用前脚掌着地，膝盖微屈缓冲' }
      ],
      variations: [
        { id: 'double-under', name: '双摇跳', desc: '一次跳跃绳子转两圈，进阶燃脂' },
        { id: 'alternate-foot-jump', name: '交替脚跳', desc: '像原地跑步一样交替抬脚，更轻松' }
      ]
    },
    {
      id: 'burpee',
      name: '波比跳',
      difficulty: 'intermediate',
      summary: '全身 HIIT 动作，燃脂王者',
      gif: '/images/guide/burpee.gif',
      equipment: '无',
      targetMuscles: ['全身肌群', '心肺系统'],
      video: 'https://cdn.example.com/guide/burpee.mp4',
      steps: [
        '双脚与肩同宽站立',
        '下蹲，双手撑地在脚前方',
        '双脚向后跳，进入俯卧撑位置',
        '做一个俯卧撑（可选）',
        '双脚跳回双手旁边',
        '起身跳跃，双手举过头顶'
      ],
      tips: [
        '初学者可以省略俯卧撑和跳跃，先掌握基本流程',
        '动作连贯流畅，每个步骤衔接紧凑',
        '落地时膝盖微屈缓冲，保护关节'
      ],
      mistakes: [
        { wrong: '俯卧撑位置腰部塌陷', fix: '全程收紧核心，保持身体一条直线' },
        { wrong: '落地时膝盖锁死', fix: '膝盖微屈着地，用肌肉吸收冲击' },
        { wrong: '每个步骤之间停顿太久', fix: '提高动作连贯性，保持心率' }
      ],
      variations: [
        { id: 'half-burpee', name: '半波比', desc: '省略俯卧撑和跳跃，适合初学者' },
        { id: 'burpee-box-jump', name: '波比跳箱', desc: '最后跳跃改为跳上箱子，增加爆发力' }
      ]
    },
    {
      id: 'jumping-jack',
      name: '开合跳',
      difficulty: 'beginner',
      summary: '简单高效的热身和燃脂动作',
      gif: '/images/guide/jumping-jack.gif',
      equipment: '无',
      targetMuscles: ['三角肌', '小腿三头肌', '臀中肌', '核心肌群'],
      video: 'https://cdn.example.com/guide/jumping-jack.mp4',
      steps: [
        '双脚并拢站立，双臂放在身体两侧',
        '跳跃时双脚向两侧打开，同时双臂向上举过头顶',
        '再次跳跃回到起始位置',
        '保持节奏均匀，每组 30-60 秒'
      ],
      tips: [
        '前脚掌着地，膝盖微屈',
        '手臂完全伸展，举到最高点',
        '适合作为热身或 HIIT 间歇动作'
      ],
      mistakes: [
        { wrong: '脚后跟重重着地', fix: '用前脚掌轻盈着地，减少冲击' },
        { wrong: '手臂没有完全伸展', fix: '双臂伸直举过头顶，充分激活肩部' },
        { wrong: '动作节奏不均匀', fix: '保持稳定的节奏，不要忽快忽慢' }
      ],
      variations: [
        { id: 'seal-jack', name: '海豹开合', desc: '双臂向前伸展而非上举，刺激胸肌' },
        { id: 'star-jump', name: '星形跳', desc: '跳起时四肢充分展开呈星形' }
      ]
    },
    {
      id: 'high-knees',
      name: '高抬腿',
      difficulty: 'beginner',
      summary: '提升心率，锻炼核心和腿部',
      gif: '/images/guide/high-knees.gif',
      equipment: '无',
      targetMuscles: ['髂腰肌', '股四头肌', '核心肌群', '小腿三头肌'],
      video: 'https://cdn.example.com/guide/high-knees.mp4',
      steps: [
        '双脚与肩同宽站立',
        '交替抬起膝盖至腰部高度或更高',
        '手臂自然配合摆动，对侧手臂与腿配合',
        '前脚掌着地，保持轻盈弹跳',
        '保持躯干直立，不要过度前倾'
      ],
      tips: [
        '膝盖尽量抬高到腰部水平',
        '前脚掌着地，像在原地跑步',
        '保持上半身稳定，不要过度晃动'
      ],
      mistakes: [
        { wrong: '身体后仰', fix: '收紧核心，保持躯干直立微前倾' },
        { wrong: '膝盖抬得太低', fix: '至少抬到腰部高度，否则效果大打折扣' },
        { wrong: '脚后跟着地', fix: '前脚掌着地，保持弹性和节奏' }
      ],
      variations: [
        { id: 'slow-high-knees', name: '慢速高抬腿', desc: '放慢速度，每次抬高保持 1 秒，强化髋屈肌' },
        { id: 'high-knees-to-butt-kick', name: '高抬腿接后踢', desc: '抬膝后接后踢腿跑，全面激活腿部' }
      ]
    },
    {
      id: 'mountain-climber',
      name: '登山跑',
      difficulty: 'beginner',
      summary: '核心+有氧复合动作',
      gif: '/images/guide/mountain-climber.gif',
      equipment: '瑜伽垫',
      targetMuscles: ['腹直肌', '腹斜肌', '髂腰肌', '三角肌', '股四头肌'],
      video: 'https://cdn.example.com/guide/mountain-climber.mp4',
      steps: [
        '双手撑地，进入俯卧撑起始位置',
        '身体从头到脚呈一条直线',
        '交替将膝盖拉向胸部',
        '像在原地快速跑步一样',
        '保持核心收紧，臀部不要翘太高'
      ],
      tips: [
        '保持俯卧撑姿势，核心全程收紧',
        '速度可以快，但要保证膝盖拉到胸部',
        '呼吸配合：收腿时呼气，伸腿时吸气'
      ],
      mistakes: [
        { wrong: '臀部抬太高', fix: '压低臀部，保持身体接近一条直线' },
        { wrong: '膝盖没有拉到胸部', fix: '每一步都把膝盖拉到胸口位置' },
        { wrong: '手腕疼痛', fix: '手指张开分散压力，或用拳头撑地' }
      ],
      variations: [
        { id: 'slow-mountain-climber', name: '慢速登山跑', desc: '每个动作保持 2 秒，强化核心控制' },
        { id: 'cross-body-mountain-climber', name: '交叉登山跑', desc: '膝盖拉向对侧手肘，增加腹斜肌刺激' }
      ]
    }
  ]
}
