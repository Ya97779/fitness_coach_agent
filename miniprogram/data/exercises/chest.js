module.exports = {
  id: 'chest',
  name: '胸部训练',
  functions: '肩关节水平内收、肩屈、肩内旋',
  icon: '/images/guide/chest.png',
  subRegions: [
    { id: 'upper', name: '上胸' },
    { id: 'overall', name: '整体' },
    { id: 'lower', name: '下胸' }
  ],
  exercises: [
    // === 上胸 ===
    {
      id: 'incline-bench-press',
      name: '上斜杠铃卧推',
      subRegion: 'upper',
      difficulty: 'intermediate',
      summary: '侧重上胸和三角肌前束的复合推举动作',
      gif: '/images/guide/incline-bench-press.gif',
      equipment: '杠铃、可调卧推凳',
      targetMuscles: ['胸大肌上束', '三角肌前束', '肱三头肌'],
      video: 'https://cdn.example.com/guide/incline-bench-press.mp4',
      steps: [
        '将卧推凳调至 30-45 度角',
        '双手握距略宽于肩，全握杠铃',
        '将杠铃从架子上取下，手臂伸直',
        '缓慢下放杠铃至上胸部（锁骨附近）',
        '发力推起至起始位置'
      ],
      tips: [
        '凳子角度 30 度侧重上胸，45 度更多刺激肩部',
        '杠铃落点在锁骨到乳头之间',
        '下放时吸气，推起时呼气'
      ],
      mistakes: [
        { wrong: '凳子角度太陡（超过 45 度）', fix: '调至 30-45 度，主要刺激上胸' },
        { wrong: '杠铃落点太低', fix: '对准锁骨到乳头之间的位置' }
      ],
      variations: [
        { id: 'incline-dumbbell-press', name: '上斜哑铃卧推', desc: '更大的运动幅度' },
        { id: 'low-incline-press', name: '低角度上斜推举', desc: '15-20 度角，更侧重中胸偏上' }
      ]
    },
    {
      id: 'incline-dumbbell-press',
      name: '上斜哑铃卧推',
      subRegion: 'upper',
      difficulty: 'intermediate',
      summary: '针对上胸部的哑铃推举动作',
      gif: '/images/guide/incline-dumbbell-press.gif',
      equipment: '哑铃、可调卧推凳',
      targetMuscles: ['胸大肌上束', '三角肌前束', '肱三头肌'],
      video: 'https://cdn.example.com/guide/incline-dumbbell-press.mp4',
      steps: [
        '将卧推凳调至 30-45 度角',
        '双手各持一只哑铃，借助大腿力量举至肩部',
        '发力推起哑铃至手臂伸直，哑铃在胸部上方靠拢',
        '缓慢下放至起始位置，感受胸部拉伸'
      ],
      tips: [
        '下放时哑铃与胸部齐平，不要过度下沉',
        '推起时两只哑铃轻轻碰一下，增强收缩',
        '全程控制速度，不要借力甩动'
      ],
      mistakes: [
        { wrong: '凳子角度太陡', fix: '调至 30-45 度' },
        { wrong: '推起时腰部过度弓起', fix: '收紧核心，臀部贴紧凳面' }
      ],
      variations: [
        { id: 'incline-bench-press', name: '上斜杠铃卧推', desc: '可以推更大的重量' }
      ]
    },
    {
      id: 'low-cable-crossover',
      name: '低位龙门架夹胸',
      subRegion: 'upper',
      difficulty: 'intermediate',
      summary: '从低位向上的夹胸动作，侧重上胸内侧',
      gif: '/images/guide/low-cable-crossover.gif',
      equipment: '龙门架',
      targetMuscles: ['胸大肌上束', '三角肌前束'],
      video: '',
      steps: [
        '将龙门架两侧滑轮调至最低位',
        '双手各握一个手柄，站在龙门架中间',
        '手臂从低位向胸部上方合拢',
        '缓慢回到起始位置'
      ],
      tips: [
        '合拢时手臂轨迹从下往上',
        '用力挤压上胸，停留 1 秒'
      ],
      mistakes: [
        { wrong: '用手臂力量拉', fix: '想象用上胸发力' }
      ],
      variations: []
    },

    // === 整体 ===
    {
      id: 'barbell-bench-press',
      name: '杠铃卧推',
      subRegion: 'overall',
      difficulty: 'intermediate',
      summary: '经典胸部复合动作，侧重整体胸肌发展',
      gif: '/images/guide/barbell-bench-press.gif',
      equipment: '杠铃、卧推凳',
      targetMuscles: ['胸大肌', '三角肌前束', '肱三头肌'],
      video: 'https://cdn.example.com/guide/barbell-bench-press.mp4',
      steps: [
        '仰卧在卧推凳上，双脚踩实地面',
        '双手握距略宽于肩，全握杠铃',
        '将杠铃从架子上取下，手臂伸直支撑在胸部正上方',
        '缓慢下放杠铃至胸部中段，肘关节约 90 度',
        '发力推起至起始位置'
      ],
      tips: [
        '全程保持肩胛骨后缩下沉，收紧上背',
        '腰部保持自然弓起，不要过度拱腰',
        '下放时吸气，推起时呼气',
        '控制节奏：下放 2-3 秒，推起 1-2 秒'
      ],
      mistakes: [
        { wrong: '杠铃触胸位置过高', fix: '对准胸部中段，乳头连线位置' },
        { wrong: '手腕过度后弯', fix: '保持手腕中立，杠铃落在掌根' },
        { wrong: '臀部离开凳面', fix: '收紧核心，臀部始终贴紧凳面' }
      ],
      variations: [
        { id: 'incline-bench-press', name: '上斜卧推', desc: '侧重上胸' },
        { id: 'dumbbell-bench-press', name: '哑铃卧推', desc: '更大的运动幅度' }
      ]
    },
    {
      id: 'dumbbell-bench-press',
      name: '哑铃卧推',
      subRegion: 'overall',
      difficulty: 'intermediate',
      summary: '哑铃胸部推举，运动幅度更大，肌肉激活更充分',
      gif: '/images/guide/dumbbell-bench-press.gif',
      equipment: '哑铃、卧推凳',
      targetMuscles: ['胸大肌', '三角肌前束', '肱三头肌'],
      video: '',
      steps: [
        '仰卧在卧推凳上，双手各持一只哑铃',
        '哑铃位于胸部两侧，掌心朝前',
        '发力推起哑铃至手臂伸直，哑铃在胸部上方靠拢',
        '缓慢下放至起始位置，感受胸部拉伸'
      ],
      tips: [
        '下放时尽量拉伸胸肌，幅度比杠铃更大',
        '推起时两只哑铃可以轻轻碰一下',
        '手腕保持中立，不要过度后弯'
      ],
      mistakes: [
        { wrong: '手腕过度后弯', fix: '保持手腕中立，哑铃落在掌根' },
        { wrong: '下放时肘部过度下沉', fix: '保持肘部与肩部在同一水平面' }
      ],
      variations: [
        { id: 'barbell-bench-press', name: '杠铃卧推', desc: '可以上更大的重量' },
        { id: 'dumbbell-fly', name: '哑铃飞鸟', desc: '孤立拉伸动作' }
      ]
    },
    {
      id: 'push-up',
      name: '俯卧撑',
      subRegion: 'overall',
      difficulty: 'beginner',
      summary: '最经典的自重胸部训练动作',
      gif: '/images/guide/push-up.gif',
      equipment: '无（徒手）',
      targetMuscles: ['胸大肌', '三角肌前束', '肱三头肌', '核心'],
      video: 'https://cdn.example.com/guide/push-up.mp4',
      steps: [
        '双手撑地，略宽于肩，身体呈一条直线',
        '收紧核心，不要塌腰或撅臀',
        '弯曲肘部，身体下放至胸部接近地面',
        '发力推起至起始位置'
      ],
      tips: [
        '全程保持身体一条直线',
        '下放时吸气，推起时呼气',
        '手肘不要过度外展，保持约 45 度'
      ],
      mistakes: [
        { wrong: '塌腰或撅臀', fix: '收紧核心和臀部，保持身体一条线' },
        { wrong: '幅度不够', fix: '胸部接近地面再推起' }
      ],
      variations: [
        { id: 'incline-push-up', name: '上斜俯卧撑', desc: '降低难度' },
        { id: 'diamond-push-up', name: '钻石俯卧撑', desc: '侧重三头肌和内胸' }
      ]
    },

    // === 下胸 ===
    {
      id: 'dips',
      name: '双杠臂屈伸',
      subRegion: 'lower',
      difficulty: 'intermediate',
      summary: '自重复合动作，刺激下胸和三头肌',
      gif: '/images/guide/dips.gif',
      equipment: '双杠',
      targetMuscles: ['胸大肌下束', '肱三头肌', '三角肌前束'],
      video: 'https://cdn.example.com/guide/dips.mp4',
      steps: [
        '双手撑在双杠上，手臂伸直，身体悬空',
        '身体微微前倾（约 30 度），更多刺激胸部',
        '缓慢弯曲肘部，下放身体至上臂与地面平行',
        '发力推起至起始位置'
      ],
      tips: [
        '前倾角度越大，胸部参与越多',
        '下放时吸气，推起时呼气',
        '体重不够可以挂杠铃片增加负重'
      ],
      mistakes: [
        { wrong: '身体完全直立', fix: '前倾 30 度才能有效刺激胸部' },
        { wrong: '下放过深导致肩部疼痛', fix: '上臂与地面平行即可' }
      ],
      variations: [
        { id: 'bench-dips', name: '凳上臂屈伸', desc: '降低难度，适合初学者' }
      ]
    },
    {
      id: 'cable-crossover',
      name: '龙门架夹胸',
      subRegion: 'lower',
      difficulty: 'intermediate',
      summary: '从高位向下的夹胸动作，侧重下胸内侧',
      gif: '/images/guide/cable-crossover.gif',
      equipment: '龙门架',
      targetMuscles: ['胸大肌下束', '三角肌前束'],
      video: 'https://cdn.example.com/guide/cable-crossover.mp4',
      steps: [
        '将龙门架两侧滑轮调至高位',
        '双手各握一个手柄，站在龙门架中间',
        '身体微微前倾，手臂从上向腹部方向合拢',
        '缓慢回到起始位置，感受胸部拉伸'
      ],
      tips: [
        '合拢时用力挤压下胸',
        '身体前倾约 30 度，增加胸肌参与',
        '可以调整滑轮高度，刺激不同部位'
      ],
      mistakes: [
        { wrong: '用手臂力量拉', fix: '想象用胸肌发力，手臂只是传导' },
        { wrong: '身体晃动借力', fix: '双脚站稳，核心收紧' }
      ],
      variations: [
        { id: 'low-cable-crossover', name: '低位龙门架夹胸', desc: '侧重上胸' }
      ]
    },
    {
      id: 'decline-bench-press',
      name: '下斜卧推',
      subRegion: 'lower',
      difficulty: 'intermediate',
      summary: '下斜角度推举，集中刺激下胸部',
      gif: '/images/guide/decline-bench-press.gif',
      equipment: '杠铃、下斜卧推凳',
      targetMuscles: ['胸大肌下束', '肱三头肌'],
      video: '',
      steps: [
        '仰卧在下斜卧推凳上，双脚固定',
        '双手握距略宽于肩，握住杠铃',
        '缓慢下放杠铃至下胸部',
        '发力推起至起始位置'
      ],
      tips: [
        '下斜角度约 15-30 度',
        '注意安全，建议有人保护'
      ],
      mistakes: [
        { wrong: '下放位置过高', fix: '对准下胸部（乳头下方）' }
      ],
      variations: [
        { id: 'dips', name: '双杠臂屈伸', desc: '自重替代方案' }
      ]
    }
  ]
}
