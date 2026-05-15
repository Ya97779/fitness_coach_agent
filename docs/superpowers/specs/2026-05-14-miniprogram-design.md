# 微信小程序前端设计文档

日期：2026-05-14

## 概述

为 FitCoach AI 健身营养顾问系统开发微信小程序前端，替代现有 Streamlit 前端作为主要用户入口。采用原生小程序开发，底部五 Tab 导航（首页、聊天、计时器、记录、我的），清新健康视觉风格。

## 技术选型

- **框架**：微信原生小程序（wxml + wxss + js）
- **后端**：复用现有 FastAPI 后端，新增 2 个记录接口
- **鉴权**：微信登录 + JWT（已有 `POST /api/v1/auth/wx-login`）
- **通信**：普通接口用 `wx.request`，聊天用 SSE 流式

## 目录结构

```
miniprogram/
├── app.js                    # 小程序入口，登录逻辑、全局数据
├── app.json                  # 全局配置（页面路由、TabBar、窗口样式）
├── app.wxss                  # 全局样式（CSS 变量、通用类）
├── project.config.json       # 项目配置
├── sitemap.json              # 微信搜索配置
├── utils/
│   ├── request.js            # 封装 wx.request，自动带 JWT token
│   ├── auth.js               # 登录、token 管理
│   └── config.js             # API base URL 等常量
├── pages/
│   ├── home/                 # 首页 - 今日概览
│   │   ├── home.js
│   │   ├── home.json
│   │   ├── home.wxml
│   │   └── home.wxss
│   ├── chat/                 # AI 聊天
│   │   ├── chat.js
│   │   ├── chat.json
│   │   ├── chat.wxml
│   │   └── chat.wxss
│   ├── log/                  # 记录（食物/运动快捷录入）
│   │   ├── log.js
│   │   ├── log.json
│   │   ├── log.wxml
│   │   └── log.wxss
│   ├── profile/              # 个人档案
│   │   ├── profile.js
│   │   ├── profile.json
│   │   ├── profile.wxml
│   │   └── profile.wxss
│   └── stats/                # 历史统计（从首页进入）
│       ├── stats.js
│       ├── stats.json
│       ├── stats.wxml
│       └── stats.wxss
└── components/               # 可复用组件
    ├── calorie-ring/         # 卡路里环形进度条（canvas，显示摄入/消耗/剩余）
    ├── chat-bubble/          # 聊天气泡（区分用户/AI，支持流式文本渲染）
    ├── food-item/            # 食物记录卡片（名称、卡路里、餐次标签）
    └── exercise-item/        # 运动记录卡片（类型、时长、消耗卡路里）
```

## 页面设计

### Tab 1：首页（home）

**功能**：今日健康数据概览

**布局**：
- 顶部：问候语（根据时间段）+ 今日日期
- 中间：卡路里环形进度条（摄入 / 消耗 / 净摄入），使用 canvas 绘制
- 下方：今日食物记录列表 + 运动记录列表
- 底部：「查看历史趋势」按钮 → 跳转 stats 页

**数据来源**：
- `GET /api/v1/user/me/today` → intake_calories, burn_calories, food_items, exercise_items
- `GET /api/v1/user/me` → tdee（用于计算进度百分比）

**交互**：
- 页面 onShow 时刷新数据
- 记录新食物/运动后自动刷新
- 点击记录项查看详情

### Tab 2：聊天（chat）

**功能**：与 AI 健身营养顾问对话

**布局**：
- 消息列表：用户消息右对齐（绿色气泡），AI 回复左对齐（白色气泡）
- 底部：输入框 + 发送按钮
- 输入框上方：快捷入口横向滚动栏（记录早餐、记录运动、热量查询、训练建议）

**数据来源**：
- `POST /api/v1/chat/stream`（SSE 流式）

**交互**：
- 发送消息后显示加载动画，AI 回复逐字流式显示
- 快捷入口点击后将预设文本填入输入框（可编辑后发送）
- SSE 连接使用 `wx.request` 的 `enableChunked` 选项
- 消息列表自动滚动到底部

### Tab 4：记录（log）

**功能**：快捷记录饮食和运动

**布局**：
- 顶部：Tab 切换「饮食」/「运动」

**饮食 Tab**：
- 食物名称输入框
- 卡路里输入框（可选，留空则后端自动查询）
- 餐次选择：早餐 / 午餐 / 晚餐 / 加餐（四个按钮组）
- 提交按钮
- 下方：今日已记录食物列表（滑动删除）

**运动 Tab**：
- 运动类型选择（常见运动快选 + 自定义输入）
- 时长输入框（分钟）
- 提交按钮
- 下方：今日已记录运动列表（滑动删除）

**数据来源**：
- 提交：`POST /api/v1/food-log` / `POST /api/v1/exercise-log`
- 列表：`GET /api/v1/user/me/today` 的 food_items / exercise_items

### Tab 5：我的（profile）

**功能**：个人档案管理

**布局**：
- 头像 + 昵称（微信信息）
- 身体数据卡片：身高、体重、年龄、性别、BMI、BMR、TDEE
- 目标信息：目标体重、过敏源
- 编辑按钮 → 编辑表单页（或弹窗）
- 退出登录按钮

**数据来源**：
- `GET /api/v1/user/me`
- 更新：`POST /api/v1/user/`

### 非 Tab 页：历史统计（stats）

**入口**：首页「查看历史趋势」按钮

**布局**：
- 日期范围选择（近 7 天 / 近 30 天）
- 摄入 vs 消耗折线图（使用 echarts-for-weixin）
- 每日净热量柱状图
- 体重变化趋势（如有 weight_log 数据）

**数据来源**：
- `GET /api/v1/user/me/logs`

## 后端新增接口

### POST /api/v1/food-log

快捷记录饮食。JWT 鉴权。

**Request Body**：
```json
{
  "name": "鸡蛋",
  "calories": 150,
  "meal_type": "breakfast"
}
```

- `name`：食物名称（必填）
- `calories`：卡路里（可选，不填则调用 `search_food_nutrient` 查询）
- `meal_type`：餐次，枚举值 `breakfast` / `lunch` / `dinner` / `snack`（必填）

**Response**：
```json
{
  "id": 1,
  "name": "鸡蛋",
  "calories": 150,
  "meal_type": "breakfast",
  "log_id": 1
}
```

**实现逻辑**：
1. 获取或创建今日 DailyLog
2. 如果 calories 为空，调用 `search_food_nutrient(name)` 查询
3. 创建 FoodItem 记录
4. 更新 DailyLog.intake_calories
5. 返回创建的记录

### POST /api/v1/exercise-log

快捷记录运动。JWT 鉴权。

**Request Body**：
```json
{
  "type": "跑步",
  "duration": 30
}
```

- `type`：运动类型（必填）
- `duration`：时长，单位分钟（必填）

**Response**：
```json
{
  "id": 1,
  "type": "跑步",
  "duration": 30,
  "calories": 280,
  "log_id": 1
}
```

**实现逻辑**：
1. 获取或创建今日 DailyLog
2. 调用 `estimate_exercise_calories(type, duration, user.weight)` 计算卡路里
3. 创建 ExerciseItem 记录
4. 更新 DailyLog.burn_calories
5. 返回创建的记录

### Pydantic Schema

```python
class FoodLogCreate(BaseModel):
    name: str
    calories: Optional[float] = None
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"]

class ExerciseLogCreate(BaseModel):
    type: str
    duration: int  # 分钟
```

## 数据流

### 登录流

```
app.js onLaunch
  → wx.login() 获取 code
  → POST /api/v1/auth/wx-login {code}
  → 后端返回 {token, user}
  → 存 token 到 wx.setStorageSync('token')
  → 后续请求自动带 Authorization: Bearer <token>
```

### 首页加载

```
home.js onShow
  → GET /api/v1/user/me/today
  → 渲染卡路里环形进度（intake / burn / tdee）
  → 渲染食物记录列表
  → 渲染运动记录列表
```

### 记录饮食

```
log 页面表单提交
  → POST /api/v1/food-log {name, calories?, meal_type}
  → 后端创建记录，返回结果
  → 前端刷新列表
  → 切换到首页时自动刷新
```

### AI 聊天

```
chat 页面发送消息
  → POST /api/v1/chat/stream (SSE)
  → wx.request({ enableChunked: true })
  → 逐 chunk 接收，拼接显示
  → 收到 [DONE] 结束流式
```

## 视觉规范

- **主色调**：#4CAF50（绿色，健康活力）
- **辅助色**：#2196F3（蓝色，数据/运动相关）
- **背景色**：#F5F7FA（浅灰白）
- **卡片**：白色背景，圆角 16rpx，轻微阴影
- **字体**：系统默认字体，标题 32rpx，正文 28rpx，辅助文字 24rpx
- **间距**：页面边距 32rpx，卡片间距 24rpx

## 实施范围

本次实施包括：
1. 小程序全部前端页面和组件
2. 后端新增 food-log 和 exercise-log 两个接口
3. 后端新增对应的 Pydantic schema
4. 工具函数（request 封装、auth 管理）

不包括：
- 微信支付、分享等高级功能
- 后端其他改动
- 训练计时器、动作指导页面（独立功能，见下方设计）

## 新增页面设计（2026-05-15）

### 页面 A：训练计时器（timer）

**入口**：底部 TabBar 第三个 Tab（计时器图标）

**功能**：组间休息计时器，参考训计APP。用户自定义训练计划（动作列表 + 每个动作的组数 + 每个动作的休息时长），训练过程中自动计时、显示剩余组数、自动跳转下一个动作。

#### TabBar 更新

原四 Tab → 五 Tab：
```
首页 | 聊天 | 计时器 | 记录 | 我的
```

`app.json` tabBar 配置新增计时器 Tab，图标使用秒表/计时器风格。

#### 子页面 A1：训练计划配置页（timer-setup）

**布局**：
- 页面标题「训练计划」
- 动作列表（可拖拽排序）：
  - 每行：动作名称（可编辑输入） | 组数（数字选择器，默认 3） | 每组休息时长（数字输入，单位秒，默认 60） | 删除按钮
  - 底部「+ 添加动作」按钮
  - 每个动作可以单独设置不同的休息时长（力量动作可以设长一点，孤立动作设短一点）
- 顶部全局默认休息时长：数字输入框（秒），新增动作时自动填入此默认值
- 预设模板入口（横向滚动卡片）：
  - 「胸部训练」「背部训练」「肩部训练」「手臂训练」「腿部训练」「核心训练」「有氧减脂」
  - 点击模板自动填充动作列表（预设模板中每个动作已配好推荐组数和休息时长）
- 底部固定按钮：「开始训练」

**数据存储**：
- 训练计划保存到 `wx.setStorageSync('training_plan', {...})`，下次打开可复用
- 预设模板硬编码在 JS 中

#### 子页面 A2：训练进行页（timer-training）

**布局**：
- 顶部信息栏：当前动作名称 / 总进度（如 2/5 动作）
- 中心区域：大号倒计时数字（休息时显示倒计时，训练中显示「进行中」）
- 动作信息卡片：
  - 当前动作名
  - 当前第几组 / 共几组（如「第 2 组 / 共 4 组」）
  - 剩余组数进度条
- 控制按钮区：
  - 「完成本组」主按钮（绿色，大号）
  - 「跳过休息」次级按钮
  - 「上一个」「下一个」导航按钮
  - 「暂停/继续」按钮
  - 「结束训练」按钮（红色文字）
- 底部：下一个动作预览（名称 + 组数）

**交互流程**：
```
用户点击「完成本组」
  → 当前组数 +1
  → 如果当前动作还有剩余组：
      → 显示休息倒计时（全屏覆盖层，带进度环）
      → 倒计时结束或用户点「跳过」
      → 回到训练界面，等待下一组
  → 如果当前动作所有组完成：
      → 显示「动作完成」动画（短暂提示）
      → 自动跳转下一个动作，组数重置为 1
  → 如果所有动作完成：
      → 显示训练总结页面
```

#### 子页面 A3：训练完成页（timer-summary）

**布局**：
- 训练完成图标 + 祝贺文案
- 训练统计卡片：
  - 总训练时长
  - 完成动作数 / 总组数
  - 预估消耗卡路里（可选，基于时长粗略估算）
- 「保存记录」按钮（调用运动记录接口）
- 「再来一次」按钮
- 「返回首页」按钮

**文件结构**：
```
pages/
├── timer/
│   ├── timer-setup/          # 训练计划配置
│   │   ├── timer-setup.js
│   │   ├── timer-setup.json
│   │   ├── timer-setup.wxml
│   │   └── timer-setup.wxss
│   ├── timer-training/       # 训练进行中
│   │   ├── timer-training.js
│   │   ├── timer-training.json
│   │   ├── timer-training.wxml
│   │   └── timer-training.wxss
│   └── timer-summary/        # 训练完成总结
│       ├── timer-summary.js
│       ├── timer-summary.json
│       ├── timer-summary.wxml
│       └── timer-summary.wxss
```

### 页面 B：动作指导库（exercise-guide）

**入口**：首页「动作指导」卡片 / 聊天页快捷入口

**功能**：按肌群分类的固定动作指导库，每个动作包含演示视频（URL）、动作细节讲解、常见错误、目标肌群等信息。纯前端页面，数据硬编码在 JS 中。视频和 GIF 动图全部使用网络 URL，不放本地（避免超出小程序包体积限制）。

**素材策略**：
- 视频：使用网络 URL（CDN 托管），详情页用 `<video>` 组件播放
- 封面图/肌群示意图：本地 `/images/guide/` 目录（几 KB 一张，总量可控）
- 数据维护：动作数据拆成独立 JSON 文件，方便增删改，不修改页面逻辑

**数据文件结构**：
```
miniprogram/
├── data/
│   ├── exercises.js          # 统一导出入口，聚合所有肌群数据
│   └── exercises/
│       ├── chest.js          # 胸部动作数据
│       ├── back.js           # 背部动作数据
│       ├── shoulder.js       # 肩部动作数据
│       ├── arms.js           # 手臂动作数据
│       ├── legs.js           # 腿部动作数据
│       ├── core.js           # 核心动作数据
│       └── cardio.js         # 有氧减脂动作数据
```

每个肌群文件导出一个对象，`exercises.js` 聚合后统一导出。页面通过 `const { exerciseData } = require('../../data/exercises')` 引用。增删动作只需修改对应肌群的 JS 文件，不影响页面代码。

#### 子页面 B1：动作分类列表页（exercise-guide）

**布局**：
- 页面标题「动作指导库」
- 顶部搜索框（跨所有肌群搜索动作名称）
- 肌群分类卡片列表（纵向排列，每个卡片带封面图）：
  - 胸部训练（封面图 + 运动功能简介 + 动作数量，如「6 个动作」）
  - 背部训练
  - 肩部训练
  - 手臂训练
  - 腿部训练
  - 核心训练
  - 有氧减脂
- 每个卡片点击进入对应肌群的动作列表
- 卡片上显示该肌群的主要运动功能（一行简介）

**七个肌群及运动功能**：

| 肌群 | 主要运动功能 |
|------|-------------|
| 胸部 | 肩关节水平内收、肩屈、肩内旋 |
| 背部 | 肩关节水平外展、肩伸、肩外旋、脊柱伸展 |
| 肩部 | 肩屈（前束）、肩伸（后束）、肩外展（中束）、肩内旋/外旋 |
| 手臂 | 肘屈（肱二头肌）、肘伸（肱三头肌）、前臂旋前/旋后 |
| 腿部 | 膝伸（股四头肌）、膝屈（腘绳肌）、髋伸（臀大肌）、踝跖屈（小腿） |
| 核心 | 脊柱屈曲（腹直肌）、脊柱旋转（腹斜肌）、脊柱抗伸展、骨盆前/后倾 |
| 有氧减脂 | 全身复合运动，提升心率，以燃脂为主要目标 |

#### 子页面 B2：肌群动作列表页（exercise-guide/list）

**路径参数**：`?group=chest`（肌群标识）

**布局**：
- 顶部：肌群名称 + 运动功能说明（如「主要功能：肩关节水平内收、肩屈、肩内旋」）+ 肌群示意图（标注目标肌肉位置）
- 动作卡片列表（每行一个卡片）：
  - 左侧：动作缩略图
  - 右侧：动作名称 + 难度标签（初级/中级/高级）+ 简短描述（一行）
  - 点击进入动作详情页

**每个肌群约 6 个动作（示例，数量不固定，可按需增删）**：

| 肌群 | 动作列表 |
|------|---------|
| 胸部 | 杠铃卧推、上斜卧推、哑铃飞鸟、龙门架夹胸、双杠臂屈伸、俯卧撑 |
| 背部 | 引体向上、杠铃划船、高位下拉、坐姿划船、硬拉、山羊挺身 |
| 肩部 | 站姿推举、阿诺德推举、侧平举、俯身飞鸟、面拉、耸肩 |
| 手臂 | 杠铃弯举、锤式弯举、牧师凳弯举、窄距卧推、绳索下压、颈后臂屈伸 |
| 腿部 | 深蹲、腿举、腿屈伸、腿弯举、罗马尼亚硬拉、保加利亚分腿蹲 |
| 核心 | 卷腹、平板支撑、俄罗斯转体、悬垂举腿、死虫式、侧平板支撑 |
| 有氧减脂 | 跑步、跳绳、波比跳、开合跳、高抬腿、登山跑 |

**数据结构**：

单个肌群文件（如 `data/exercises/chest.js`）：
```javascript
module.exports = {
  id: 'chest',
  name: '胸部训练',
  functions: '肩关节水平内收、肩屈、肩内旋',
  icon: '/images/guide/chest.png',
  exercises: [
    {
      id: 'barbell-bench-press',
      name: '杠铃卧推',
      difficulty: 'intermediate',  // beginner / intermediate / advanced
      summary: '经典胸部复合动作，侧重整体胸肌发展',
      cover: '/images/guide/barbell-bench-press.jpg'
    },
    // ... 按需增删
  ]
}
```

聚合入口 `data/exercises.js`：
```javascript
const chest = require('./exercises/chest')
const back = require('./exercises/back')
const shoulder = require('./exercises/shoulder')
const arms = require('./exercises/arms')
const legs = require('./exercises/legs')
const core = require('./exercises/core')
const cardio = require('./exercises/cardio')

const exerciseData = { chest, back, shoulder, arms, legs, core, cardio }
module.exports = { exerciseData }
```

#### 子页面 B3：动作详情页（exercise-guide/detail）

**路径参数**：`?id=barbell-bench-press`

**布局**：
- 顶部视频区域：
  - 演示视频播放器（网络 URL，使用 `<video>` 组件）
  - 视频控制条
- 动作信息卡片：
  - 动作名称（大号）
  - 难度标签 + 目标肌群标签（如「胸大肌」「三角肌前束」「肱三头肌」）
  - 所需器械标签（如「杠铃」「卧推凳」）
- 分段内容（Tab 或纵向排列）：
  - **动作步骤**：编号步骤列表，每步配简要文字说明
  - **动作细节**：关键要点（呼吸、握距、角度等）
  - **常见错误**：错误描述 + 纠正方法（带 ✗/✓ 图标）
  - **变体动作**：相关变体的简要介绍和跳转链接
- 底部固定按钮：
  - 「加入训练计划」→ 跳转 timer-setup 并预填该动作
  - 「咨询 AI」→ 跳转聊天页，预填「请详细讲解 XXX 的动作要领」

**数据结构**：
```javascript
const exerciseDetail = {
  id: 'barbell-bench-press',
  name: '杠铃卧推',
  difficulty: 'intermediate',
  equipment: '杠铃、卧推凳',
  targetMuscles: ['胸大肌', '三角肌前束', '肱三头肌'],
  video: 'https://cdn.example.com/guide/bench-press.mp4',  // 网络 URL，不放本地
  steps: [
    '仰卧在卧推凳上，双脚踩实地面',
    '双手握距略宽于肩，握住杠铃',
    '将杠铃从架子上取下，手臂伸直',
    '缓慢下放杠铃至胸部中段，肘关节约 90°',
    '发力推起至起始位置'
  ],
  tips: [
    '全程保持肩胛骨后缩下沉',
    '腰部保持自然弓起，不要过度拱腰',
    '下放时吸气，推起时呼气',
    '控制节奏：下放 2-3 秒，推起 1-2 秒'
  ],
  mistakes: [
    { wrong: '杠铃触胸位置过高（颈部方向）', fix: '对准胸部中段（乳头连线位置）' },
    { wrong: '手腕过度后弯', fix: '保持手腕中立，杠铃落在掌根' },
    { wrong: '臀部离开凳面', fix: '收紧核心，臀部始终贴紧凳面' }
  ],
  variations: [
    { id: 'incline-bench-press', name: '上斜卧推', desc: '侧重上胸和三角肌前束' },
    { id: 'dumbbell-bench-press', name: '哑铃卧推', desc: '更大的运动幅度，更好的肌肉激活' }
  ]
}
```

**文件结构**：
```
pages/
├── exercise-guide/
│   ├── exercise-guide/           # 肌群分类列表
│   │   ├── exercise-guide.js
│   │   ├── exercise-guide.json
│   │   ├── exercise-guide.wxml
│   │   └── exercise-guide.wxss
│   ├── exercise-list/            # 某肌群的动作列表
│   │   ├── exercise-list.js
│   │   ├── exercise-list.json
│   │   ├── exercise-list.wxml
│   │   └── exercise-list.wxss
│   └── exercise-detail/          # 动作详情
│       ├── exercise-detail.js
│       ├── exercise-detail.json
│       ├── exercise-detail.wxml
│       └── exercise-detail.wxss
```

### app.json 路由更新

页面路由（五 Tab + 非 Tab 页）：
```json
{
  "pages": [
    "pages/home/home",
    "pages/chat/chat",
    "pages/timer/timer-setup/timer-setup",
    "pages/log/log",
    "pages/profile/profile",
    "pages/stats/stats",
    "pages/timer/timer-training/timer-training",
    "pages/timer/timer-summary/timer-summary",
    "pages/exercise-guide/exercise-guide/exercise-guide",
    "pages/exercise-guide/exercise-list/exercise-list",
    "pages/exercise-guide/exercise-detail/exercise-detail"
  ],
  "tabBar": {
    "list": [
      { "pagePath": "pages/home/home", "text": "首页", "iconPath": "...", "selectedIconPath": "..." },
      { "pagePath": "pages/chat/chat", "text": "聊天", "iconPath": "...", "selectedIconPath": "..." },
      { "pagePath": "pages/timer/timer-setup/timer-setup", "text": "计时器", "iconPath": "...", "selectedIconPath": "..." },
      { "pagePath": "pages/log/log", "text": "记录", "iconPath": "...", "selectedIconPath": "..." },
      { "pagePath": "pages/profile/profile", "text": "我的", "iconPath": "...", "selectedIconPath": "..." }
    ]
  }
}
```

### 视觉补充

- 训练计时器页面：深色背景（#1A1A2E）突出计时数字，绿色强调按钮，白色文字
- 动作指导页面：保持全局白底风格，视频区域占满屏宽，卡片带左侧彩色条标记难度

## 注意事项

### 数据库小改动

现有 `FoodItem` 模型没有 `meal_type` 字段。需要给 `food_items` 表新增 `meal_type` 列（String，可空），用于记录餐次。这是唯一需要的数据库改动，使用 SQLAlchemy 的 `create_all` 即可自动迁移（开发阶段，无需 Alembic）。
