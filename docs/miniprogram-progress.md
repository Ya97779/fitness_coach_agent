# 微信小程序前端开发进度

更新日期：2026-05-15

## 当前状态：代码框架已搭建，待联调测试

## 已完成

### 1. 项目功能梳理

后端现有功能：

| 模块 | API | 说明 |
|------|-----|------|
| 登录 | `POST /api/v1/auth/wx-login` | 微信 code → JWT |
| 用户档案 | `GET /api/v1/user/me` | 获取用户信息 |
| 用户档案 | `POST /api/v1/user/` | 创建/更新档案 |
| 今日数据 | `GET /api/v1/user/me/today` | 今日摄入/消耗/记录列表 |
| 历史数据 | `GET /api/v1/user/me/logs` | 全部日志（趋势图用） |
| 聊天 | `POST /api/v1/chat/stream` | SSE 流式 AI 对话 |

数据模型：User、DailyLog、FoodItem、ExerciseItem、ConversationLog

### 2. 设计方案确认

- **框架**：微信原生小程序（wxml + wxss + js）
- **导航**：底部五 Tab — 首页、聊天、计时器、记录、我的
- **视觉**：清新健康风（#4CAF50 绿色主色调，白底圆角卡片）
- **记录方式**：聊天触发 + 快捷表单两种都要

### 3. 设计文档

已提交到 `docs/superpowers/specs/2026-05-14-miniprogram-design.md`

内容包括：
- 目录结构（pages / utils / components）
- 四个 Tab 页面详细设计
- 后端新增接口设计（POST /food-log、POST /exercise-log）
- 数据流（登录、首页加载、记录、聊天）
- 视觉规范（色值、字号、间距）
- 注意事项（FoodItem 需加 meal_type 字段）

### 4. 小程序前端代码框架（2026-05-15）

- 11 个页面全部创建（5 Tab + 6 非 Tab）
- 2 个通用组件（food-item、exercise-item）
- 3 个工具函数（request、auth、config）
- 7 个肌群动作数据文件（胸部含完整示例，其余为骨架）
- 后端新增 2 个记录接口（POST /food-log、POST /exercise-log）
- 后端 FoodItem 模型新增 meal_type 字段

## 未完成

1. **TabBar 图标** — 需要准备 10 个 PNG 图标文件（81x81 像素）
2. **动作指导视频素材** — 需要准备或链接到动作演示视频 URL
3. **动作数据填充** — 背部/肩部/手臂/腿部/核心/有氧的动作详情（steps/tips/mistakes）待补充
4. **后端 Today 接口扩展** — 当前 /today 不返回 food_items 和 exercise_items 列表，首页记录列表需要后端配合
5. **联调测试** — 微信开发者工具中编译运行、接口联调
6. **echarts 集成** — 统计页折线图/柱状图需要引入 echarts-for-weixin

## 下次继续的步骤

1. 准备 TabBar 图标文件（10 个 PNG）放入 `miniprogram/images/`
2. 配置 `project.config.json` 中的真实 appid
3. 修改 `utils/config.js` 中的 API_BASE_URL 为实际后端地址
4. 在微信开发者工具中编译运行，检查页面渲染
5. 补充动作数据详情（其余 6 个肌群的 steps/tips/mistakes）
6. 后端扩展 /today 接口返回 items 列表
7. 统计页引入 echarts-for-weixin 组件

## 目录结构预览

```
miniprogram/
├── app.js / app.json / app.wxss
├── utils/
│   ├── request.js        # wx.request 封装，自动带 JWT
│   ├── auth.js           # 登录、token 管理
│   └── config.js         # API base URL
├── pages/
│   ├── home/             # 首页：今日概览、卡路里环
│   ├── chat/             # AI 聊天：SSE 流式
│   ├── log/              # 记录：饮食/运动快捷录入
│   ├── profile/          # 我的：个人档案
│   ├── stats/            # 统计：历史趋势图
│   ├── timer/            # 训练计时器（新增 2026-05-15）
│   │   ├── timer-setup/      # 训练计划配置
│   │   ├── timer-training/   # 训练进行中（组间休息计时）
│   │   └── timer-summary/    # 训练完成总结
│   └── exercise-guide/   # 动作指导库（新增 2026-05-15）
│       ├── exercise-guide/   # 肌群分类列表
│       ├── exercise-list/    # 某肌群的动作列表
│       └── exercise-detail/  # 动作详情（视频+讲解）
├── data/
│   ├── exercises.js          # 动作数据聚合入口
│   └── exercises/            # 按肌群拆分的数据文件
│       ├── chest.js / back.js / shoulder.js
│       ├── arms.js / legs.js / core.js / cardio.js
└── components/
    ├── calorie-ring/     # 卡路里环形进度条
    ├── chat-bubble/      # 聊天气泡
    ├── food-item/        # 食物记录卡片
    └── exercise-item/    # 运动记录卡片
```
