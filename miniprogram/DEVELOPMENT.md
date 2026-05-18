# FitCoach AI 小程序二次开发指南

本文档面向后续维护者，覆盖常见的修改场景。所有路径相对于 `miniprogram/` 目录。

---

## 1. 修改页面标题

每个页面的导航栏标题在各自的 `.json` 文件中定义，修改 `navigationBarTitleText` 字段即可。

| 页面 | 文件 | 当前标题 |
|------|------|----------|
| 首页 | `pages/home/home.json` | FitCoach AI |
| 聊天 | `pages/chat/chat.json` | AI 顾问 |
| 训练计划 | `pages/timer/timer-setup/timer-setup.json` | 训练计划 |
| 训练中 | `pages/timer/timer-training/timer-training.json` | 训练中 |
| 训练完成 | `pages/timer/timer-summary/timer-summary.json` | 训练完成 |
| 周训练计划 | `pages/timer/training-plan/training-plan.json` | 周训练计划 |
| 记录 | `pages/log/log.json` | 记录 |
| 我的 | `pages/profile/profile.json` | 我的 |
| 历史统计 | `pages/stats/stats.json` | 历史统计 |
| 动作指导库 | `pages/exercise-guide/exercise-guide/exercise-guide.json` | 动作指导库 |
| 动作列表 | `pages/exercise-guide/exercise-list/exercise-list.json` | 动作列表 |
| 动作详情 | `pages/exercise-guide/exercise-detail/exercise-detail.json` | 动作详情 |

全局默认标题在 `app.json` 的 `window.navigationBarTitleText` 中设置。

---

## 2. 修改页面中的中文文案

页面上的中文文字直接写在 `.wxml` 模板文件中，直接搜索修改即可。以下列出各页面的主要文案位置：

**首页 `pages/home/home.wxml`**
- 问候语、日期、卡路里相关文字

**聊天页 `pages/chat/chat.wxml`**
- 快捷入口文字（`shortcuts` 数组中的 `text` 字段，定义在 `chat.js` 中）
- 输入框 placeholder

**训练计划 `pages/timer/timer-setup/timer-setup.wxml`**
- 「快速选择」「默认休息」「动作清单」「添加动作」等分区标题
- 「开始训练」「保存至周计划」按钮文字

**周计划 `pages/timer/training-plan/training-plan.wxml`**
- 「应用模板」「标记休息」「复制自...」等操作按钮
- 「今日开始训练」「今日休息日」等提示文字
- 星期名称定义在 `training-plan.js` 的 `DAYS` 数组中

**记录页 `pages/log/log.wxml`**
- 「饮食」「运动」Tab 名称、表单标签、按钮文字

**我的页 `pages/profile/profile.wxml`**
- 「身体数据」「编辑资料」「退出登录」等菜单项

**动作指导 `pages/exercise-guide/exercise-guide/exercise-guide.wxml`**
- 搜索框 placeholder

**快捷修改：** 用编辑器全局搜索中文关键词，定位到对应 `.wxml` 文件直接改。

---

## 3. 修改图片资源

### 3.1 TabBar 图标

位置：`images/tab-*.png`（共 10 个文件，5 个未选中态 + 5 个选中态）

| 图标 | 未选中 | 选中 |
|------|--------|------|
| 首页 | `tab-home.png` | `tab-home-active.png` |
| 聊天 | `tab-chat.png` | `tab-chat-active.png` |
| 计时器 | `tab-timer.png` | `tab-timer-active.png` |
| 动作指导 | `tab-guide.png` | `tab-guide-active.png` |
| 我的 | `tab-profile.png` | `tab-profile-active.png` |

替换规则：
- 尺寸建议 81x81px，PNG 格式
- 选中态颜色与 `app.json` 中 `tabBar.selectedColor` 一致（当前 `#1a1a1a`）
- 未选中态颜色与 `tabBar.color` 一致（当前 `#999`）
- 替换后文件名保持不变，无需改代码

### 3.2 动作指导 — 肌群封面图

位置：`images/guide/` 目录，共 7 张 PNG：

| 文件 | 对应肌群 |
|------|----------|
| `chest.png` | 胸部训练 |
| `back.png` | 背部训练 |
| `shoulder.png` | 肩部训练 |
| `arms.png` | 手臂训练 |
| `legs.png` | 腿部训练 |
| `core.png` | 核心训练 |
| `cardio.png` | 有氧减脂 |

路径定义在 `data/exercises/*.js` 的 `icon` 字段中。替换同名文件即可，无需改代码。

### 3.3 动作指导 — 动作 GIF 演示图

位置：`images/guide/*.gif`，共 50 个文件。

文件名与 `data/exercises/*.js` 中每个动作的 `gif` 字段对应。例如：
```js
// data/exercises/chest.js
{
  id: 'barbell-bench-press',
  gif: '/images/guide/barbell-bench-press.gif',  // ← 对应这个文件
  ...
}
```

替换规则：
- 准备同名 GIF 文件放入 `images/guide/` 目录
- 如果要改文件名，同步修改对应数据文件中的 `gif` 字段路径
- GIF 建议 300x300px 以内，文件控制在 1MB 以内

### 3.4 动作指导 — 视频链接

每个动作的 `video` 字段当前是占位 URL（`https://cdn.example.com/...`）。替换为真实视频地址即可：

```js
// data/exercises/chest.js
{
  id: 'barbell-bench-press',
  video: 'https://你的CDN地址/barbell-bench-press.mp4',
  ...
}
```

视频在 `pages/exercise-guide/exercise-detail/exercise-detail.wxml` 中通过 `<video>` 标签播放。

### 3.5 用户默认头像

位置：`images/default-avatar.png`

在 `pages/profile/profile.wxml` 中引用，当用户未设置头像时显示。

---

## 4. 修改颜色/主题

所有颜色通过 CSS 变量统一管理，修改 `app.wxss` 中的变量即可全局生效：

```css
/* app.wxss — page 选择器 */
page {
  /* 背景色 */
  --bg-base: #f5f5f5;        /* 页面背景 */
  --bg-card: #ffffff;         /* 卡片背景 */
  --bg-input: #f0f0f0;       /* 输入框背景 */
  --border: #e8e8e8;          /* 边框 */
  --border-light: #f0f0f0;    /* 浅色分割线 */

  /* 文字色 */
  --text-primary: #1a1a1a;    /* 主文字（标题、正文） */
  --text-secondary: #666666;  /* 次要文字 */
  --text-hint: #999999;       /* 提示/占位文字 */
  --text-muted: #cccccc;      /* 极淡文字 */

  /* 强调色 */
  --accent: #1a1a1a;          /* 主强调色（按钮、选中态） */
  --accent-dim: rgba(26,26,26,0.06);  /* 淡强调背景 */
  --danger: #c47a6c;          /* 警告/危险色 */

  /* 圆角 */
  --radius-sm: 8rpx;
  --radius-md: 12rpx;
}
```

切换主题示例（深色模式）：
```css
page {
  --bg-base: #111113;
  --bg-card: #1a1a1c;
  --bg-input: #2a2a2c;
  --border: #333;
  --text-primary: #f0ece4;
  --text-secondary: #999;
  --text-hint: #666;
  --accent: #a8b5a0;
}
```

Canvas 绘图颜色（环形图、折线图等）需要在对应的 `.js` 文件中单独修改：
- `pages/home/home.js` → `drawRing()` 方法中的颜色值
- `pages/stats/stats.js` → canvas 绘图中的颜色值
- `pages/timer/timer-training/timer-training.js` → 倒计时环颜色

---

## 5. 修改动作数据

### 5.1 数据文件结构

```
data/
  exercises.js          ← 聚合导出（新增肌群时在这里注册）
  exercises/
    chest.js            ← 胸部动作数据
    back.js             ← 背部
    shoulder.js         ← 肩部
    arms.js             ← 手臂
    legs.js             ← 腿部
    core.js             ← 核心
    cardio.js           ← 有氧减脂
  templates.js          ← 训练模板（计时器预设）
```

### 5.2 单个动作的数据结构

```js
{
  id: 'barbell-bench-press',       // 唯一标识（kebab-case）
  name: '杠铃卧推',                 // 显示名称
  subRegion: 'overall',             // 所属子区域（可选，core/cardio 无此字段）
  difficulty: 'intermediate',       // 难度：beginner / intermediate / advanced
  summary: '胸肌基础复合推举动作',    // 一句话描述
  gif: '/images/guide/barbell-bench-press.gif',  // 演示 GIF
  equipment: '杠铃、卧推凳',         // 所需装备
  targetMuscles: ['胸大肌', '三角肌前束', '肱三头肌'],  // 目标肌群
  video: 'https://cdn.example.com/...',  // 视频链接
  steps: [                          // 训练步骤
    '仰卧在卧推凳上，双脚踩实地面',
    '双手握距略宽于肩，全握杠铃',
    ...
  ],
  tips: [                           // 训练技巧
    '肩胛骨后缩下沉，挺起胸腔',
    ...
  ],
  mistakes: [                       // 常见错误
    { wrong: '杠铃落点太高', fix: '对准乳头位置下放' },
    ...
  ],
  variations: [                     // 变体动作
    { id: 'dumbbell-bench-press', name: '哑铃卧推', desc: '更大运动幅度' },
    ...
  ]
}
```

### 5.3 添加新动作

在对应肌群文件的 `exercises` 数组中追加即可：

```js
// data/exercises/chest.js
exercises: [
  // ... 现有动作
  {
    id: '新动作id',
    name: '新动作名称',
    difficulty: 'beginner',
    summary: '一句话描述',
    gif: '/images/guide/新动作id.gif',
    equipment: '所需装备',
    targetMuscles: ['目标肌群'],
    video: '',
    steps: ['步骤1', '步骤2'],
    tips: ['技巧1'],
    mistakes: [{ wrong: '错误做法', fix: '正确做法' }],
    variations: []
  }
]
```

### 5.4 添加新肌群

1. 新建 `data/exercises/新肌群.js`，按照上述结构编写
2. 在 `data/exercises.js` 中注册：

```js
const newGroup = require('./exercises/新肌群')
const exerciseData = { chest, back, ..., newGroup }
const groupList = [chest, back, ..., newGroup]
```

3. 在 `data/templates.js` 中添加对应的训练模板（可选）

### 5.5 修改训练模板

编辑 `data/templates.js`，每个模板对应计时器页面的一个预设：

```js
{
  id: '模板id',
  icon: '💪',           // 显示的 emoji
  name: '模板名称',
  exercises: [
    { name: '动作名称', sets: 4, rest: 90 },  // 组数、休息秒数
    ...
  ]
}
```

模板会同时出现在计时器页面和周计划页面的模板选择器中。

---

## 6. 修改 TabBar

编辑 `app.json` 的 `tabBar` 部分：

```json
{
  "tabBar": {
    "color": "#999999",              // 未选中文字色
    "selectedColor": "#1a1a1a",      // 选中文字色
    "backgroundColor": "#ffffff",    // 背景色
    "borderStyle": "black",          // 顶部分割线：black / white
    "list": [
      {
        "pagePath": "pages/xxx/xxx",      // 页面路径（必须在 pages 数组中）
        "text": "标签名",                   // 显示文字
        "iconPath": "images/xxx.png",      // 未选中图标
        "selectedIconPath": "images/xxx-active.png"  // 选中图标
      }
    ]
  }
}
```

限制：最多 5 个 Tab。

---

## 7. 本地存储 Key 一览

| Key | 类型 | 用途 | 操作位置 |
|-----|------|------|----------|
| `token` | string | JWT 登录凭证 | `app.js`、`utils/auth.js`、`utils/request.js` |
| `training_plan` | object | 当前训练计划 `{ exercises, defaultRest }` | `timer-setup.js`、`exercise-detail.js` |
| `weekly_plan` | object | 周训练计划 `{ days: { mon, tue, ... } }` | `timer-setup.js`、`training-plan.js` |

---

## 8. 后端 API 配置

API 基础地址在 `utils/config.js` 中：

```js
module.exports = {
  API_BASE_URL: 'http://49.233.181.116:8000'
}
```

修改为你的服务器地址即可。所有请求通过 `utils/request.js` 封装，自动携带 JWT token。

---

## 9. 页面路由注册

新增页面需要在 `app.json` 的 `pages` 数组中注册：

```json
{
  "pages": [
    "pages/home/home",
    "pages/chat/chat",
    "pages/timer/timer-setup/timer-setup",
    ...
    "pages/你的新页面/你的新页面"
  ]
}
```

---

## 10. 文件速查表

```
miniprogram/
  app.json              ← 路由、TabBar、全局窗口配置
  app.wxss              ← 全局 CSS 变量、通用类
  app.js                ← 全局生命周期、登录逻辑
  data/
    exercises.js        ← 动作数据聚合导出
    exercises/*.js      ← 各肌群动作数据（改动作内容在这里）
    templates.js        ← 训练模板（改预设计划在这里）
  utils/
    config.js           ← API 地址
    request.js          ← 请求封装
    auth.js             ← 登录/鉴权
  components/
    food-item/          ← 食物记录卡片组件
    exercise-item/      ← 运动记录卡片组件
  images/
    tab-*.png           ← TabBar 图标（10 个）
    guide/              ← 肌群封面 + 动作 GIF（待补充）
    default-avatar.png  ← 默认头像
  pages/
    home/               ← 首页
    chat/               ← AI 聊天
    log/                ← 记录
    profile/            ← 我的
    stats/              ← 统计
    timer/
      timer-setup/      ← 训练配置
      timer-training/   ← 训练中
      timer-summary/    ← 训练完成
      training-plan/    ← 周计划
    exercise-guide/
      exercise-guide/   ← 肌群列表
      exercise-list/    ← 动作列表
      exercise-detail/  ← 动作详情
```
