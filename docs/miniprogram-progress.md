# 微信小程序前端开发进度

更新日期：2026-05-20

## 当前状态：功能开发完成，上线前安全审查中

---

## 已完成

### 1. 核心页面（12 个）

| 页面 | 路径 | 说明 |
|------|------|------|
| 首页 | `pages/home/home` | 今日卡路里环形进度、食物/运动记录列表 |
| 聊天 | `pages/chat/chat` | SSE 流式 AI 对话、快捷入口 |
| 训练计划 | `pages/timer/timer-setup/timer-setup` | 预设模板、动作列表配置、同步周计划 |
| 训练中 | `pages/timer/timer-training/timer-training` | 倒计时、组间休息、完成本组 |
| 训练完成 | `pages/timer/timer-summary/timer-summary` | 训练统计、逐动作保存记录 |
| 周训练计划 | `pages/timer/training-plan/training-plan` | 周一至周日计划、模板应用、休息日 |
| 记录 | `pages/log/log` | 饮食/运动快捷录入 |
| 我的 | `pages/profile/profile` | 身体数据、编辑资料 |
| 历史统计 | `pages/stats/stats` | 摄入/消耗折线图、净卡路里柱状图 |
| 动作指导库 | `pages/exercise-guide/exercise-guide/` | 肌群分类列表、搜索 |
| 动作列表 | `pages/exercise-guide/exercise-list/` | 某肌群动作列表、子区域筛选 |
| 动作详情 | `pages/exercise-guide/exercise-detail/` | 视频播放、步骤/技巧/常见错误/变体 |

### 2. 动作数据（7 个肌群，52 个动作）

全部动作包含完整数据：steps / tips / mistakes / variations / equipment / targetMuscles / gif

| 肌群 | 动作数 | 文件 |
|------|--------|------|
| 胸部 | 9 | `data/exercises/chest.js` |
| 背部 | 8 | `data/exercises/back.js` |
| 肩部 | 7 | `data/exercises/shoulder.js` |
| 手臂 | 9 | `data/exercises/arms.js` |
| 腿部 | 7 | `data/exercises/legs.js` |
| 核心 | 6 | `data/exercises/core.js` |
| 有氧减脂 | 6 | `data/exercises/cardio.js` |

### 3. 训练模板（`data/templates.js`）

7 个预设模板，每个含 4 个动作，带 weight 字段。

### 4. 后端接口

| 接口 | 说明 |
|------|------|
| `POST /api/v1/auth/wx-login` | 微信登录 → JWT |
| `GET /api/v1/user/me` | 用户信息 |
| `POST /api/v1/user/` | 创建/更新档案 |
| `GET /api/v1/user/me/today` | 今日数据（含 food_items、exercise_items） |
| `GET /api/v1/user/me/logs` | 历史日志 |
| `POST /api/v1/food-log` | 记录食物 |
| `POST /api/v1/exercise-log` | 记录运动（含 name/sets/weight） |
| `POST /api/v1/chat/stream` | SSE 流式 AI 对话 |

### 5. 通用组件

- `components/food-item/` — 食物记录卡片
- `components/exercise-item/` — 运动记录卡片

### 6. 二次开发文档

- `miniprogram/DEVELOPMENT.md` — 覆盖 10 个常见修改场景

---

## 上线前安全审查（2026-05-20）

### 已修复

| 修复项 | 状态 | 说明 |
|--------|------|------|
| JWT 密钥硬编码默认值 | **已修复** | 移除 `"default-secret-change-me"` 默认值，未配置时启动直接报错 |
| JWT 过期时间 72h | **已修复** | 调整为 24h |
| API 地址 HTTP 裸 IP | **已修复** | 改为 HTTPS 域名占位，需替换为真实域名 |
| `project.private.config.json` 入库 | **已修复** | 已加入 `.gitignore` |

### 待修复 — 上线前必做

**致命（不修必炸）**

| # | 问题 | 操作 |
|---|------|------|
| 1 | `.env` 在 git 历史中泄露过 API Key | **立即轮换**智谱 API Key 和天行 API Key，在微信公众平台和智谱控制台重新生成 |
| 2 | `images/guide/` 目录为空（57 个文件缺失） | 补齐 50+ 个动作 GIF + 7 个肌群封面 PNG |
| 3 | `images/default-avatar.png` 缺失 | 准备默认头像图片 |
| 4 | 视频 URL 全是占位符 `cdn.example.com` | 替换为真实 CDN 地址或暂时清空 video 字段 |
| 5 | `config.js` 中的 `api.yourdomain.com` | 替换为实际 HTTPS 域名，并在微信公众平台配置白名单 |

**高危（上线后大概率出事）**

| # | 问题 | 操作 |
|---|------|------|
| 6 | 无频率限制，LLM 接口可被刷到欠费 | 后端加 `slowapi` 限流 |
| 7 | 401 处理死循环 | `request.js` 改为弹登录弹窗而非跳首页 |
| 8 | `request.js` 无超时 | 加 10s 超时 + 用户提示 |
| 9 | 聊天 `sending` 标志永久卡住 | catch/finally 中重置 |
| 10 | SSE 流断开无重连 | 加断连提示和重试按钮 |
| 11 | 聊天无上下文，每条消息独立 | 传历史消息实现多轮对话 |
| 12 | 多个页面静默吞错误 | 加 toast 提示 |
| 13 | `requirements.txt` 未锁版本 | `pip freeze` 锁版本 |
| 14 | `session_key` 持久化在数据库 | 改用 Redis 短期缓存 |
| 15 | Pydantic 无输入校验 | 加 `Field` 约束 |

**中等（影响体验）**

| # | 问题 |
|---|------|
| 16 | SQLite 做生产库，多人并发会锁死 |
| 17 | 热量估算硬编码 `duration * 6` |
| 18 | 统计页无数据时白屏 |
| 19 | 保存训练记录 `Promise.all` 无部分失败处理 |
| 20 | `create_all` 每次启动建表 |

---

## 目录结构

```
miniprogram/
├── app.js / app.json / app.wxss
├── DEVELOPMENT.md              # 二次开发指南
├── utils/
│   ├── request.js              # wx.request 封装，自动带 JWT
│   ├── auth.js                 # 登录、token 管理
│   └── config.js               # API base URL（HTTPS）
├── pages/
│   ├── home/                   # 首页
│   ├── chat/                   # AI 聊天
│   ├── log/                    # 记录
│   ├── profile/                # 我的
│   ├── stats/                  # 统计
│   ├── timer/
│   │   ├── timer-setup/        # 训练计划配置
│   │   ├── timer-training/     # 训练中
│   │   ├── timer-summary/      # 训练完成
│   │   └── training-plan/      # 周训练计划
│   └── exercise-guide/
│       ├── exercise-guide/     # 肌群列表
│       ├── exercise-list/      # 动作列表
│       └── exercise-detail/    # 动作详情
├── data/
│   ├── exercises.js            # 动作数据聚合
│   ├── exercises/              # 7 个肌群数据文件
│   └── templates.js            # 训练模板
├── components/
│   ├── food-item/
│   └── exercise-item/
└── images/
    ├── tab-*.png               # TabBar 图标（10 个）
    ├── guide/                  # 肌群封面 + 动作 GIF（待补充）
    └── default-avatar.png      # 默认头像（待补充）
```
