module.exports = {
  id: 'cardio',
  name: '有氧减脂',
  functions: '全身复合运动，提升心率，以燃脂为主要目标',

  exercises: [
    {
      id: 'running',
      name: '跑步',
      difficulty: 'beginner',
      summary: '最经典的有氧运动，燃脂效率高',
      cover: 'https://gzyapi.gzyhm.xyz/guide/跑步.jpg',
      equipment: '跑鞋',
      targetMuscles: ['股四头肌', '腘绳肌', '臀大肌', '小腿三头肌', '核心肌群'],
      video: '',
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
        { id: 'incline-running', name: '爬坡', desc: '增加坡度提升心率和臀部发力' }
      ]
    },
    {
      "id": "incline-running",
      "name": "爬坡",
      "difficulty": "beginner",
      "summary": "进阶有氧跑步训练，强化臀部与下肢肌群，燃脂效率高于平路跑，减少膝盖压力",
      "cover": "https://gzyapi.gzyhm.xyz/guide/爬坡.jpg",
      "equipment": "减震跑鞋、跑步机/户外坡道",
      "targetMuscles": ["臀大肌", "腘绳肌", "股四头肌", "小腿三头肌", "深层核心"],
      "video": "",
      "steps": [
        "提前热身5分钟，重点激活臀部、小腿及踝关节",
        "跑步机调节5°-15°坡度，户外选择平缓坡道，不设置过快配速",
        "身体小幅前倾，重心贴合坡道，不要弯腰低头",
        "手臂前后小幅摆动，肘部保持90°，稳定上半身",
        "小步高频落地，脚掌全脚掌平稳着地，避免大步踩踏",
        "保持匀速呼吸，结束后下调坡度慢走放松，拉伸臀腿后侧肌群"
      ],
      "tips": [
        "新手坡度从3°起步，循序渐进增加坡度，切勿一开始高坡冲刺",
        "爬坡无需追求快速度，以心率平稳、说话不费力为标准",
        "全程收紧核心，稳住躯干，避免腰部代偿发力"
      ],
      "mistakes": [
        {
          "wrong": "上半身过度前倾弯腰塌腰",
          "fix": "挺直腰背，依靠整体躯干小幅前倾，不要折叠腰部"
        },
        {
          "wrong": "大步幅跨步，脚跟重重落地",
          "fix": "缩小步幅、加快步频，轻柔落地，降低下肢关节冲击"
        },
        {
          "wrong": "手扶跑步机扶手借力",
          "fix": "双手自然摆动，脱离扶手，保证核心和下肢正常发力"
        }
      ],
      "variations": []
    },
    
    {
      id: 'burpee',
      name: '波比跳',
      difficulty: 'intermediate',
      summary: '全身 HIIT 动作，燃脂王者',
      cover: 'https://gzyapi.gzyhm.xyz/guide/波比跳.gif',
      equipment: '无',
      targetMuscles: ['全身肌群', '心肺系统'],
      video: '',
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
      cover: 'https://gzyapi.gzyhm.xyz/guide/开合跳.gif',
      equipment: '无',
      targetMuscles: ['三角肌', '小腿三头肌', '臀中肌', '核心肌群'],
      video: '',
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
    
  ]
}
