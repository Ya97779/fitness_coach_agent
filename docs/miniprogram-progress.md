# 微信小程序前端开发进度

更新日期：2026-05-14

## 当前状态：设计完成，待实施

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
- **导航**：底部四 Tab — 首页、聊天、记录、我的
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

## 未完成

1. **审阅设计文档** — 确认是否需要修改
2. **编写实施计划** — 按模块拆分开发任务
3. **代码开发** — 小程序前端 + 后端新接口

## 下次继续的步骤

1. 读 `docs/superpowers/specs/2026-05-14-miniprogram-design.md`
2. 确认/修改设计方案
3. 生成实施计划
4. 按计划开发：先 utils → 再 pages → 最后 components
5. 后端新增 /food-log 和 /exercise-log 接口
6. 联调测试

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
│   └── stats/            # 统计：历史趋势图
└── components/
    ├── calorie-ring/     # 卡路里环形进度条
    ├── chat-bubble/      # 聊天气泡
    ├── food-item/        # 食物记录卡片
    └── exercise-item/    # 运动记录卡片
```
