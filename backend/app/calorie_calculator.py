"""统一运动热量计算模块 — 所有路径共用此模块"""

# ==================== 有氧运动 MET 表 ====================
# MET (Metabolic Equivalent of Task) 值
# 公式：calories = MET × 体重kg × (时长min / 60)

MET_VALUES = {
    "跑步":   {"light": 7,  "medium": 10, "intense": 14},
    "慢跑":   {"light": 6,  "medium": 9,  "intense": 12},
    "游泳":   {"light": 6,  "medium": 10, "intense": 14},
    "快走":   {"light": 4,  "medium": 5,  "intense": 7},
    "骑行":   {"light": 5,  "medium": 8,  "intense": 12},
    "跳绳":   {"light": 8,  "medium": 12, "intense": 15},
    "瑜伽":   {"light": 2,  "medium": 3,  "intense": 5},
    "HIIT":   {"light": 8,  "medium": 12, "intense": 17},
    "篮球":   {"light": 5,  "medium": 8,  "intense": 11},
    "足球":   {"light": 5,  "medium": 8,  "intense": 11},
    "羽毛球": {"light": 4,  "medium": 7,  "intense": 10},
    "乒乓球": {"light": 3,  "medium": 5,  "intense": 7},
    "网球":   {"light": 4,  "medium": 7,  "intense": 10},
    "爬山":   {"light": 5,  "medium": 7,  "intense": 10},
    "椭圆机": {"light": 4,  "medium": 6,  "intense": 8},
    "划船机": {"light": 4,  "medium": 7,  "intense": 9},
    "有氧操": {"light": 4,  "medium": 6,  "intense": 9},
    "拳击":   {"light": 6,  "medium": 10, "intense": 13},
    "力量训练": {"light": 4,  "medium": 6,  "intense": 8},
}

# 有氧运动名称集合（用于分类判断）
AEROBIC_EXERCISES = set(MET_VALUES.keys())

# ==================== 力量训练每组热量表 ====================
# kcal/set，基于 70kg 体重标准化
# 公式：calories = calories_per_set × 组数 × (体重kg / 70)

STRENGTH_CALORIES_PER_SET = {
    # 胸
    "平板卧推": 8,    "上斜卧推": 8,    "哑铃卧推": 7.5,
    "上斜哑铃卧推": 8, "龙门架夹胸": 6,  "蝴蝶机夹胸": 5.5,
    "俯卧撑": 5,     "双杠臂屈伸": 8,
    # 背
    "引体向上": 10,   "杠铃划船": 8,    "哑铃划船": 7,
    "坐姿划船": 6.5,  "高位下拉": 6,    "硬拉": 10,
    "罗马尼亚硬拉": 9,
    # 肩
    "杠铃推举": 8,    "哑铃推举": 7,    "侧平举": 5,
    "前平举": 4.5,    "俯身飞鸟": 5,    "面拉": 5,
    # 腿
    "深蹲": 10,      "前蹲": 9.5,     "腿举": 7,
    "箭步蹲": 8,     "腿弯举": 5,     "腿屈伸": 5,
    "小腿提踵": 4,
    # 手臂
    "杠铃弯举": 5,    "哑铃弯举": 4.5,  "锤式弯举": 4.5,
    "三头下压": 4.5,  "窄距卧推": 7,    "仰卧臂屈伸": 5,
    # 核心
    "平板支撑": 3,    "卷腹": 3,       "悬垂举腿": 5,
    "俄罗斯转体": 3.5,
    # 兼容旧数据中的名称
    "仰卧起坐": 3,
}

# 力量训练动作别名映射 → 标准名称
STRENGTH_ALIASES = {
    "卧推": "平板卧推",   "杠铃卧推": "平板卧推",   "平板杠铃卧推": "平板卧推",
    "上斜杠铃卧推": "上斜卧推",
    "平板哑铃卧推": "哑铃卧推",
    "夹胸": "龙门架夹胸", "绳索夹胸": "龙门架夹胸", "飞鸟": "龙门架夹胸", "蝴蝶机": "蝴蝶机夹胸",
    "双杠": "双杠臂屈伸", "臂屈伸": "双杠臂屈伸",
    "引体": "引体向上",   "正手引体": "引体向上",
    "俯身划船": "杠铃划船", "俯身杠铃划船": "杠铃划船",
    "单臂哑铃划船": "哑铃划船",
    "绳索坐姿划船": "坐姿划船", "坐姿绳索划船": "坐姿划船", "器械坐姿划船": "坐姿划船",
    "下拉": "高位下拉",   "引体下拉": "高位下拉",
    "传统硬拉": "硬拉",   "杠铃硬拉": "硬拉",   "罗拉": "罗马尼亚硬拉",
    "推举": "杠铃推举",   "站姿推举": "杠铃推举", "肩推": "杠铃推举",
    "坐姿哑铃推举": "哑铃推举",
    "哑铃侧平举": "侧平举",
    "哑铃前平举": "前平举",
    "俯身侧平举": "俯身飞鸟", "反向飞鸟": "俯身飞鸟",
    "绳索面拉": "面拉",
    "杠铃深蹲": "深蹲",   "颈后深蹲": "深蹲",
    "杠铃前蹲": "前蹲",
    "腿举机": "腿举",
    "弓步蹲": "箭步蹲",   "保加利亚深蹲": "箭步蹲",
    "俯卧腿弯举": "腿弯举",
    "坐姿腿屈伸": "腿屈伸",
    "提踵": "小腿提踵",   "站姿提踵": "小腿提踵",
    "弯举": "杠铃弯举",   "二头弯举": "杠铃弯举",
    "交替弯举": "哑铃弯举",
    "锤式": "锤式弯举",
    "绳索下压": "三头下压", "三头绳索下压": "三头下压",
    "窄握卧推": "窄距卧推",
    "碎颅者": "仰卧臂屈伸",
    "plank": "平板支撑",
}


def classify_exercise(name: str) -> str:
    """判断运动类型：'aerobic' 或 'strength'"""
    if name in AEROBIC_EXERCISES:
        return "aerobic"
    # 先查别名
    resolved = STRENGTH_ALIASES.get(name)
    if resolved and resolved in STRENGTH_CALORIES_PER_SET:
        return "strength"
    if name in STRENGTH_CALORIES_PER_SET:
        return "strength"
    return "unknown"


def _resolve_strength_name(name: str) -> str:
    """解析力量训练别名到标准名称"""
    if name in STRENGTH_CALORIES_PER_SET:
        return name
    return STRENGTH_ALIASES.get(name, name)


def estimate_calories(
    exercise_name: str,
    user_weight: float = 70,
    duration: int = None,
    sets: int = None,
    intensity: str = "medium",
) -> float:
    """统一热量估算入口

    Args:
        exercise_name: 运动名称
        user_weight: 用户体重 (kg)
        duration: 时长 (分钟)，有氧运动必填
        sets: 组数，力量训练必填
        intensity: 强度 (light/medium/intense)，仅对有氧有效

    Returns:
        估算热量 (kcal)
    """
    category = classify_exercise(exercise_name)

    if category == "aerobic":
        met = MET_VALUES.get(exercise_name, {}).get(intensity, 5)
        dur = duration or 30
        return round(met * user_weight * (dur / 60), 1)

    if category == "strength":
        # 有 sets 时用每组公式；无 sets 但有时长时用 MET 公式（如"力量训练30分钟"）
        if sets:
            resolved = _resolve_strength_name(exercise_name)
            cal_per_set = STRENGTH_CALORIES_PER_SET.get(resolved, 5)
            return round(cal_per_set * sets * (user_weight / 70), 1)
        dur = duration or 30
        return round(6 * user_weight * (dur / 60), 1)

    # 未知运动：优先用 sets 公式（如果有），否则用 MET 兜底
    if sets:
        return round(5 * sets * (user_weight / 70), 1)
    dur = duration or 30
    return round(5 * user_weight * (dur / 60), 1)


def get_calories_per_set(exercise_name: str) -> float:
    """获取力量训练的每组热量值，用于 DB 缓存等场景"""
    resolved = _resolve_strength_name(exercise_name)
    return STRENGTH_CALORIES_PER_SET.get(resolved, 5)
