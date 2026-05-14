# 微信小程序前端设计文档

日期：2026-05-14

## 概述

为 FitCoach AI 健身营养顾问系统开发微信小程序前端，替代现有 Streamlit 前端作为主要用户入口。采用原生小程序开发，底部四 Tab 导航，清新健康视觉风格。

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

### Tab 3：记录（log）

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

### Tab 4：我的（profile）

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

## 注意事项

### 数据库小改动

现有 `FoodItem` 模型没有 `meal_type` 字段。需要给 `food_items` 表新增 `meal_type` 列（String，可空），用于记录餐次。这是唯一需要的数据库改动，使用 SQLAlchemy 的 `create_all` 即可自动迁移（开发阶段，无需 Alembic）。
