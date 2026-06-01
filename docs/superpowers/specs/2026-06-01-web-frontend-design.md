# FitCoach AI 桌面端网页版设计文档

## 概述

为 FitCoach AI 健身营养顾问系统开发桌面端网页版前端，1:1 复刻微信小程序的全部功能。采用 Vue 3 + Vite + Vant 技术栈，通过 Nginx 反向代理部署，与现有 FastAPI 后端集成。

## 目标与约束

**目标：**
- 完全替代小程序，功能 1:1 对齐
- 桌面端优先的 UI 设计（侧边栏导航）
- 完整 PWA 支持（离线缓存、后台计时、推送通知）
- 微信扫码登录，与小程序共享账号体系
- 两端数据实时同步（定时轮询）

**约束：**
- 服务器：4 核 4 线程，需同时运行 FastAPI + PostgreSQL + Nginx
- 技术栈：Vue 3 + Vite + Vant 4
- 后端：复用现有 FastAPI 接口，无需新增（微信登录除外）

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                         Nginx (80/443)                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  静态文件服务          │  │  反向代理 /api/*                 │ │
│  │  /var/www/fitcoach    │──│  → http://127.0.0.1:8000        │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────┼───────────────────────────────┐
│                               ▼                               │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   FastAPI        │  │  PostgreSQL  │  │   ChromaDB      │  │
│  │   (8000)         │  │  (5432)      │  │                 │  │
│  └─────────────────┘  └──────────────┘  └─────────────────┘  │
│                    4C4T 服务器                                  │
└───────────────────────────────────────────────────────────────┘
```

## 技术栈

| 类别 | 技术选型 | 说明 |
|------|----------|------|
| 框架 | Vue 3 | Composition API，响应式系统 |
| 构建工具 | Vite | 快速开发，优化构建 |
| UI 组件库 | Vant 4 | 移动端组件库，适配桌面端 |
| 状态管理 | Pinia | 轻量，TypeScript 友好 |
| 路由 | Vue Router | SPA 路由管理 |
| HTTP 客户端 | Axios | API 请求，拦截器 |
| 图表 | ECharts | 数据统计可视化 |
| PWA | Workbox | Service Worker 缓存策略 |

## 页面结构

```
/ (侧边栏布局)
├── /home              首页 — 今日概览（摄入/消耗/TDEE）+ 记录列表
├── /chat              AI 对话 — 流式 SSE，快捷指令
├── /log               记录 — 饮食/运动双 Tab 表单
├── /profile           个人档案 — 身体指标、目标设置
├── /stats             数据统计 — 日历视图 + 图表（7天/30天）
├── /timer             训练计时器
│   ├── /timer/setup       计时器设置
│   ├── /timer/training    训练中（后台 Web Worker）
│   └── /timer/summary     训练总结
├── /guide             动作指导
│   ├── /guide/list        按部位分类的动作列表
│   └── /guide/:id         动作详情
├── /feedback          用户反馈
└── /login             登录页（微信扫码）
```

**侧边栏导航结构：**
- 首页
- AI 对话
- 记录
- 数据统计
- 训练计时
- 动作指导
- ─── 分割线 ───
- 个人档案
- 反馈

## 核心功能模块

### AI 对话

- SSE 流式响应，复用 `/api/v1/chat/stream` 接口
- 快捷指令栏：记录早餐、记录运动、查询热量、训练建议、饮食计划
- 本地消息缓存，支持历史消息查看

### 饮食/运动记录

- Vant 表单组件
- 食物搜索（调用天行 API）
- 份量选择（克/份/碗）
- 运动选择（按部位分类：手臂/背部/胸部/核心/腿部/肩部/有氧）
- 支持编辑和删除

### 训练计时器

- Web Worker 后台计时（切标签页/最小化继续运行）
- 页面可见性 API 检测
- Notification API 完成提醒
- 训练模板管理

### 数据统计

- ECharts 图表（热量趋势、摄入/消耗对比）
- 日历视图标记有记录的日期
- 7天/30天切换

### 动作指导

- 按部位分类（手臂/背部/胸部/核心/腿部/肩部/有氧）
- 动作详情页（要领、常见错误、呼吸方法）

### 微信登录

前端调用微信 OAuth 获取 code → 后端 `/api/v1/wechat/web-login` 换 JWT → 存储 Token，Axios 拦截器自动附加。

### PWA

- Service Worker 缓存静态资源
- 离线时显示缓存页面
- Web App Manifest 支持"添加到主屏幕"

## 数据流与状态管理

### Pinia Store 结构

```
stores/
├── auth.js        用户认证状态（token、用户信息）
├── user.js        用户档案（身体数据、目标）
├── daily.js       今日数据（摄入/消耗/记录列表）
├── chat.js        对话消息列表、快捷指令
├── timer.js       计时器状态（训练模板、当前组数/时间）
└── cache.js       食物/运动本地缓存（减少 API 调用）
```

### 数据同步流程

```
小程序记录饮食 → 写入 FastAPI → PostgreSQL
                                      ↑
网页端每 10 秒轮询 → GET /daily-stats → 返回最新数据 → 更新 Pinia → 视图刷新
```

### API 复用

- 直接复用现有 FastAPI 后端的所有接口（`/api/v1/*`）
- 无需新增后端接口，前端适配即可
- 微信登录接口需新增（`/api/v1/wechat/web-login`，用 OAuth code 换 token）

## 项目目录结构

```
fitness_coach_agent/
├── backend/                  # 现有后端
├── frontend/
│   └── web/                  # Vue 网页端（替代原 Streamlit 前端）
│       ├── public/
│       │   ├── manifest.json     # PWA 配置
│       │   └── icons/            # 应用图标
│       ├── src/
│       │   ├── main.js           # 入口
│       │   ├── App.vue           # 根组件
│       │   ├── router/           # Vue Router 路由
│       │   ├── stores/           # Pinia 状态管理
│       │   ├── views/            # 页面组件（9 个主页面 + 子页面）
│       │   ├── components/       # 通用组件（记录卡片、图表等）
│       │   ├── api/              # Axios 封装、接口定义
│       │   ├── utils/            # 工具函数
│       │   └── assets/           # 静态资源
│       ├── vite.config.js
│       └── package.json
├── miniprogram/              # 现有小程序
└── knowledge_base/           # RAG 知识库
```

**废弃：** 删除 `frontend/app.py`（Streamlit 前端）

## 关键技术细节

### 微信扫码登录流程

```
网页端 → 跳转微信 OAuth 页面 → 用户扫码授权 → 回调带 code
  → POST /api/v1/wechat/web-login {code}
  → 后端用 code 换 openid（微信开放平台）
  → 查找/创建用户 → 返回 JWT Token
```

### 训练计时器（Web Worker）

```javascript
// timer.worker.js — 后台计时，不受页面切换影响
let interval = null;
onmessage = (e) => {
  if (e.data === 'start') {
    interval = setInterval(() => postMessage('tick'), 1000);
  } else if (e.data === 'stop') {
    clearInterval(interval);
  }
};
```

### PWA 缓存策略

- 静态资源（JS/CSS/图标）：Cache First，离线可用
- API 请求：Network First，离线时返回缓存
- SSE 流式：不缓存，仅在线可用

### Nginx 配置

```nginx
server {
    listen 80;
    server_name fitcoach.example.com;

    # 静态文件
    location / {
        root /var/www/fitcoach;
        try_files $uri $uri/ /index.html;  # SPA 路由
    }

    # API 反向代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_buffering off;  # SSE 流式需要关闭缓冲
    }
}
```

## 部署流程

1. 前端构建：`cd frontend/web && npm run build`
2. 产出目录：`frontend/web/dist/`
3. 部署到服务器：复制 `dist/` 到 `/var/www/fitcoach/`
4. 配置 Nginx（见上方配置）
5. （可选）配置 SSL 证书（Let's Encrypt）

## 实现阶段

详见实现计划（待生成）。
