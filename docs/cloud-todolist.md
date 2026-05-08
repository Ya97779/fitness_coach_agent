# FitCoach AI 云部署 TODO 清单

## 目标
将现有后端部署到云服务器，同时供**微信小程序**和 **Web 网页**使用。

## 架构概览

```
┌──────────────────┐    ┌──────────────────┐
│   微信小程序      │    │   Web 浏览器      │
│  (miniprogram/)  │    │  (Vue/React/     │
│                  │    │   Streamlit)     │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         │   HTTPS + SSE         │   HTTPS + SSE
         ▼                       ▼
┌──────────────────────────────────────────┐
│            Nginx (443/SSL)               │
│  /api/v1/  → 127.0.0.1:8000 (直通)      │
│  /         → 127.0.0.1:8501 (Web 前端)  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Gunicorn + Uvicorn (FastAPI :8000)      │
│  ├── agents/   (Chat/Nutrition/Fitness)  │
│  ├── rag/      (知识库检索)              │
│  ├── memory/   (记忆模块)               │
│  └── wechat.py (微信登录) ← 新增        │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   ┌───────────┐      ┌───────────┐
   │  SQLite   │      │ ChromaDB  │
   │ (数据持久化)│      │ (向量库)  │
   └───────────┘      └───────────┘
```

## 分支策略

```
main   ← 本地开发（Streamlit + HTTP）
  │
  └── deploy  ← 云部署（Nginx + HTTPS + 微信登录 + WebSocket）
```

两个分支共享后端核心代码（agents/rag/memory），deploy 分支仅新增部署配置和少量适配代码。

---

## 阶段一：后端代码适配

### 1.1 Git 分支准备
- [ ] 确保 main 分支代码已提交并推送
- [ ] `git checkout -b deploy` 创建部署分支
- [ ] 创建 `.env.example` 模板文件（列出所有环境变量，不含真实值）
- [ ] 更新 `.gitignore`，确保以下条目存在：
  ```
  fitness_coach.db
  chroma_db/
  checkpoints.db
  data/
  ```

### 1.2 后端新增微信登录模块
- [ ] 创建 `backend/app/wechat.py`
  - 实现 `wx_code_to_session(code: str)` 函数 — 小程序登录，调用 `jscode2session` 换取 openid + session_key
  - 实现 `wx_web_code_to_token(code: str)` 函数 — Web 微信扫码登录，调用公众号 OAuth2 接口（`sns/oauth2/access_token`）
  - 实现 JWT token 生成与验证工具函数
- [ ] 修改 `backend/app/models.py`
  - User 表新增 `openid` 字段（String, unique, index）
  - User 表新增 `unionid` 字段（String, nullable, index — 用于跨端用户关联）
  - User 表新增 `session_key` 字段（String, nullable — 微信敏感数据解密需要）
  - User 表新增 `nickname` 字段
  - User 表新增 `avatar_url` 字段
- [ ] 在 `backend/app/main.py` 新增登录接口
  - `POST /api/v1/auth/wx-login` — 小程序登录（code → openid → JWT）
  - `POST /api/v1/auth/wx-scan-login` — Web 微信扫码登录（公众号 OAuth2 code → JWT）
  - 创建 JWT 鉴权依赖函数 `get_current_user()`，从 Authorization header 解析 token
- [ ] 鉴权后改造现有端点：`user_id` 从 token 中解析，不再由前端传入（防伪造身份）
- [ ] 更新 `requirements.txt` 新增依赖
  - `PyJWT` — JWT token 生成与验证
  - `httpx` — 异步 HTTP 客户端（调用微信 API）

### 1.3 流式对话方案
> **优先使用 SSE（当前已实现）**，微信小程序从基础库 2.20.2 起支持 `wx.request` 的 `enableChunked`，可接收 SSE 流式响应。如果目标用户的小程序基础库版本较旧，再考虑 WebSocket 方案。

**方案 A：SSE（推荐，当前已有）**
- [ ] 现有 `/chat/stream` 端点（SSE）已满足需求，仅需接入鉴权
- [ ] 小程序端使用 `wx.request({ enableChunked: true })` 接收

**方案 B：WebSocket（备选，仅在 SSE 不可用时启用）**
- [ ] 在 `backend/app/main.py` 新增 WebSocket 路由
  - `WS /api/v1/ws/chat` — 支持流式对话，user_id 从 token 解析
  - ⚠️ `stream_user_message()` 是同步生成器，必须在 `loop.run_in_executor()` 中运行，不能直接在 async handler 中调用（会阻塞事件循环）
  - 逐 chunk 通过 WebSocket 推送
- [ ] 支持 WebSocket 鉴权（token 通过 query 参数或首条消息传递）
- [ ] 添加心跳机制（30s ping/pong 防断连）
- [ ] 更新 `requirements.txt` 新增 `websockets`（uvicorn 已包含）

### 1.4 后端 CORS 配置
- [ ] 在 `backend/app/main.py` 添加 CORS 中间件
  ```python
  ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
  app.add_middleware(
      CORSMiddleware,
      allow_origins=ALLOWED_ORIGINS,
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
- [ ] 环境变量 `CORS_ORIGINS` 配置允许的域名

### 1.5 API 路由规范化
- [ ] 给所有路由加统一前缀 `/api/v1/`（避免与 Nginx 静态资源冲突）
- [ ] 保持旧路由兼容（可选：添加重定向）
- [ ] Nginx 配置同步修改为 `/api/v1/` → `127.0.0.1:8000/api/v1/`（直通，不做路径重写）

### 1.6 数据库适配
- [ ] 确认 SQLite 在生产环境的并发能力（当前单机足够）
- [ ] `backend/app/database.py` 的数据库路径改为环境变量
  ```python
  DB_PATH = os.getenv("DB_PATH", "./fitness_coach.db")
  SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
  ```
- [ ] 确保 `fitness_coach.db` 和 `chroma_db/` 路径在部署时可写

### 1.7 错误处理标准化
- [ ] 统一 API 错误响应格式：
  ```json
  {"code": 400, "message": "参数错误: xxx"}
  ```
- [ ] 全局异常处理器捕获未处理异常，返回标准格式（避免泄露堆栈信息）
- [ ] 鉴权失败统一返回 401 + 标准错误体

### 1.8 前端适配（仅保留 Streamlit 方案时需要）
- [ ] `BACKEND_URL` 改为从环境变量读取
- [ ] 所有 `requests` 调用加 `verify=SSL_VERIFY` 参数
- [ ] 新增 `SSL_VERIFY` 环境变量（自签名证书时设为 false）

### 1.9 单元测试验证
- [ ] 运行现有测试确保适配没有破坏原有功能
  ```bash
  cd backend && python -m unittest discover tests
  ```
- [ ] 为新增的微信登录模块编写测试
- [ ] 如果启用了 WebSocket，为 WebSocket 端点编写测试

---

## 阶段二：云服务器环境准备

### 2.1 服务器购买与基础配置
- [ ] 购买云服务器（推荐 4 核 4G 以上）
  - 阿里云 ECS / 腾讯云 CVM / 华为云 ECS
  - 操作系统：Ubuntu 22.04 LTS 或 24.04 LTS
- [ ] 购买域名（如 `fitcoach.yourdomain.com`）
- [ ] 域名备案（国内服务器必须，约 1-2 周）
- [ ] DNS 解析：将域名指向服务器公网 IP
- [ ] 安全组/防火墙开放端口：22（SSH）、80（HTTP）、443（HTTPS）

### 2.2 服务器基础软件安装
- [ ] 系统更新
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```
- [ ] 安装 Python 3.12+
  ```bash
  sudo apt install python3.12 python3.12-venv python3-pip -y
  ```
- [ ] 安装 Nginx
  ```bash
  sudo apt install nginx -y
  ```
- [ ] 安装 Git
  ```bash
  sudo apt install git -y
  ```
- [ ] 安装 tesseract-ocr（图片 OCR，RAG 文档处理需要）
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-chi-sim -y
  ```
- [ ] 安装 certbot（SSL 证书申请）
  ```bash
  sudo apt install certbot python3-certbot-nginx -y
  ```

### 2.3 项目代码部署
- [ ] 克隆项目并切换到 deploy 分支
  ```bash
  cd /var/www
  sudo git clone -b deploy https://github.com/Ya97779/fitness_coach_agent.git fitness_coach
  sudo chown -R $USER:$USER /var/www/fitness_coach
  ```
- [ ] 创建 Python 虚拟环境
  ```bash
  cd /var/www/fitness_coach
  python3.12 -m venv venv
  source venv/bin/activate
  ```
- [ ] 安装 Python 依赖
  ```bash
  pip install -r requirements.txt
  pip install gunicorn uvicorn[standard]
  ```
- [ ] 创建日志目录
  ```bash
  sudo mkdir -p /var/log/fitness_coach
  sudo chown $USER:$USER /var/log/fitness_coach
  ```

### 2.4 环境变量配置
- [ ] 在服务器创建 `/var/www/fitness_coach/.env`
  ```env
  # ===== LLM 配置 =====
  LLM_MODEL=glm-4.7
  OPENAI_API_KEY=your_zhipu_api_key
  OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4

  # ===== Embedding =====
  EMBEDDING_MODEL=embedding-2

  # ===== 食物营养 API =====
  TianxingFood_API_KEY=your_food_api_key

  # ===== 微信小程序 =====
  WECHAT_APPID=wx_your_appid
  WECHAT_SECRET=your_secret

  # ===== JWT =====
  JWT_SECRET_KEY=your_random_32char_secret
  JWT_EXPIRE_HOURS=72

  # ===== CORS（逗号分隔） =====
  CORS_ORIGINS=https://fitcoach.yourdomain.com,https://yourdomain.com

  # ===== 数据库路径 =====
  DB_PATH=/var/www/fitness_coach/data/fitness_coach.db

  # ===== 前端（如果同一服务器部署 Streamlit） =====
  BACKEND_URL=https://fitcoach.yourdomain.com
  SSL_VERIFY=true
  ```
- [ ] 确保 `.env` 文件权限 `chmod 600 .env`

---

## 阶段三：Nginx + SSL 配置

### 3.1 Nginx 反向代理配置
- [ ] 创建 `/etc/nginx/conf.d/fitcoach.conf`
  ```nginx
  server {
      listen 80;
      server_name fitcoach.yourdomain.com;
      return 301 https://$host$request_uri;
  }

  server {
      listen 443 ssl http2;
      server_name fitcoach.yourdomain.com;

      # SSL 证书（certbot 自动生成路径）
      ssl_certificate     /etc/letsencrypt/live/fitcoach.yourdomain.com/fullchain.pem;
      ssl_certificate_key /etc/letsencrypt/live/fitcoach.yourdomain.com/privkey.pem;

      # SSL 安全配置
      ssl_protocols TLSv1.2 TLSv1.3;
      ssl_ciphers HIGH:!aNULL:!MD5;
      ssl_prefer_server_ciphers on;

      # ===== FastAPI 后端 API（直通，不做路径重写） =====
      location /api/v1/ {
          proxy_pass http://127.0.0.1:8000/api/v1/;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
          proxy_read_timeout 120s;
      }

      # ===== WebSocket（仅在启用 WebSocket 方案时需要） =====
      location /api/v1/ws/ {
          proxy_pass http://127.0.0.1:8000/api/v1/ws/;
          proxy_http_version 1.1;
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_read_timeout 300s;
          proxy_send_timeout 300s;
      }

      # ===== Web 前端（Streamlit 或 Vue/React 打包） =====
      location / {
          proxy_pass http://127.0.0.1:8501/;
          proxy_http_version 1.1;
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";
          proxy_set_header Host $host;
          proxy_read_timeout 86400;
      }
  }
  ```
- [ ] 测试 Nginx 配置 `sudo nginx -t`
- [ ] 重启 Nginx `sudo systemctl restart nginx`

### 3.2 SSL 证书申请
- [ ] 申请 Let's Encrypt 免费证书
  ```bash
  sudo certbot --nginx -d fitcoach.yourdomain.com
  ```
- [ ] 验证自动续期 `sudo certbot renew --dry-run`
- [ ] 测试 HTTPS 访问 `curl https://fitcoach.yourdomain.com`

---

## 阶段四：后端服务启动

### 4.1 Gunicorn 配置
- [ ] 创建 `/var/www/fitness_coach/gunicorn.conf.py`
  ```python
  import multiprocessing

  bind = "127.0.0.1:8000"
  workers = min(multiprocessing.cpu_count() * 2 + 1, 4)
  worker_class = "uvicorn.workers.UvicornWorker"
  timeout = 120
  keepalive = 5
  max_requests = 1000
  max_requests_jitter = 50
  accesslog = "/var/log/fitness_coach/access.log"
  errorlog = "/var/log/fitness_coach/error.log"
  loglevel = "info"
  ```

### 4.2 Systemd 服务配置
- [ ] 创建 `/etc/systemd/system/fitcoach.service`
  ```ini
  [Unit]
  Description=FitCoach AI Backend
  After=network.target

  [Service]
  Type=simple
  User=www-data
  Group=www-data
  WorkingDirectory=/var/www/fitness_coach
  Environment="PATH=/var/www/fitness_coach/venv/bin"
  EnvironmentFile=/var/www/fitness_coach/.env
  ExecStart=/var/www/fitness_coach/venv/bin/gunicorn backend.app.main:app -c gunicorn.conf.py
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```
- [ ] 启用并启动服务
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable fitcoach
  sudo systemctl start fitcoach
  sudo systemctl status fitcoach
  ```
- [ ] 验证后端启动日志
  ```bash
  sudo journalctl -u fitcoach -f
  # 应看到：[RAG 启动] 增量索引结果: 新增 X 文件...
  ```

### 4.3 后端健康检查
- [ ] 测试 API 文档页面 `https://fitcoach.yourdomain.com/api/docs`
- [ ] 测试基础接口 `curl https://fitcoach.yourdomain.com/api/agents`
- [ ] 测试对话接口 `curl -X POST https://fitcoach.yourdomain.com/api/chat -H "Content-Type: application/json" -d '{"message":"你好"}'`

---

## 阶段五：微信小程序后端对接

### 5.1 微信公众平台配置
- [ ] 注册微信小程序账号（https://mp.weixin.qq.com）
- [ ] 获取 AppID 和 AppSecret
- [ ] 在「开发 → 开发管理 → 服务器域名」配置：
  - request 合法域名：`https://fitcoach.yourdomain.com`
  - wss 合法域名：`wss://fitcoach.yourdomain.com`
  - uploadFile 合法域名：`https://fitcoach.yourdomain.com`（如需上传图片）
  - downloadFile 合法域名：`https://fitcoach.yourdomain.com`

### 5.2 微信登录接口验证
- [ ] 小程序端 `wx.login()` 获取 code
- [ ] 调用 `POST /api/v1/auth/wx-login` 传入 code
- [ ] 验证返回 JWT token 和用户信息
- [ ] 验证 token 过期和刷新机制

### 5.3 流式对话验证
**SSE 方案（推荐）：**
- [ ] 小程序端使用 `wx.request({ url, enableChunked: true, success(res) { ... } })` 调用 `POST /api/v1/chat/stream`
- [ ] 验证流式消息逐 chunk 接收
- [ ] 验证 `[DONE]` 结束标记正确处理

**WebSocket 方案（备选）：**
- [ ] 小程序端 `wx.connectSocket()` 连接 `wss://fitcoach.yourdomain.com/api/v1/ws/chat?token=xxx`
- [ ] 发送消息并接收流式回复
- [ ] 验证断线重连机制
- [ ] 验证心跳保活（30s ping/pong）

### 5.4 工具调用功能验证
- [ ] 营养师 Agent — API 食物查询（search_food_nutrition）
- [ ] 营养师 Agent — RAG 营养知识检索（search_nutrition_knowledge）
- [ ] 健身教练 Agent — RAG 健身知识检索（search_fitness_knowledge）
- [ ] 记录食物/运动（log_food_intake / log_exercise）
- [ ] 每日统计查询（get_daily_nutrition_summary）

---

## 阶段六：Web 前端部署

### 6.1 Streamlit 方案（快速启动）
- [ ] 安装 Streamlit 到虚拟环境
- [ ] 配置 Streamlit 启动参数
  ```bash
  # /var/www/fitness_coach/.streamlit/config.toml
  [server]
  port = 8501
  address = "127.0.0.1"
  headless = true
  ```
- [ ] 创建 systemd 服务 `/etc/systemd/system/fitcoach-web.service`
  ```ini
  [Unit]
  Description=FitCoach Web Frontend
  After=network.target fitcoach.service

  [Service]
  Type=simple
  User=www-data
  WorkingDirectory=/var/www/fitness_coach
  Environment="PATH=/var/www/fitness_coach/venv/bin"
  EnvironmentFile=/var/www/fitness_coach/.env
  ExecStart=/var/www/fitness_coach/venv/bin/streamlit run frontend/app.py
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```
- [ ] 启用并启动 `sudo systemctl enable fitcoach-web && sudo systemctl start fitcoach-web`
- [ ] 验证访问 `https://fitcoach.yourdomain.com/`

### 6.2 Vue/React 方案（可选，后续升级）
- [ ] 新建 `web/` 目录，使用 Vue 3 + Vite 或 React + Vite
- [ ] 实现聊天页面（对接 HTTP + WebSocket）
- [ ] `npm run build` 打包到 `web/dist/`
- [ ] Nginx `location /` 指向 `web/dist/`
- [ ] 对接微信扫码登录

---

## 阶段七：安全加固

### 7.1 服务器安全
- [ ] 禁用 root SSH 登录，使用普通用户 + sudo
- [ ] 配置 SSH 密钥登录，禁用密码登录
- [ ] 安装 fail2ban 防暴力破解
  ```bash
  sudo apt install fail2ban -y
  sudo systemctl enable fail2ban
  ```
- [ ] 配置 UFW 防火墙
  ```bash
  sudo ufw allow 22/tcp
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  sudo ufw enable
  ```

### 7.2 应用安全
- [ ] `.env` 文件权限设为 600，仅 owner 可读
- [ ] JWT Secret Key 使用 32 位以上随机字符串
- [ ] API 接口添加频率限制（防刷）
- [ ] 微信 AppSecret 不要提交到 Git（已在 .gitignore 中？）
- [ ] CORS 白名单仅包含实际使用的域名

### 7.3 数据安全
- [ ] 配置 SQLite 数据库定时备份（cron job）
  ```bash
  # /etc/cron.d/fitcoach-backup
  0 3 * * * cp /var/www/fitness_coach/data/fitness_coach.db /var/backups/fitcoach/$(date +\%Y\%m\%d).db
  ```
- [ ] ChromaDB 向量库目录定期备份
- [ ] 备份保留策略（保留最近 30 天）

---

## 阶段八：监控与运维

### 8.1 日志管理
- [ ] 配置 logrotate 管理 Gunicorn 日志
  ```
  /var/log/fitness_coach/*.log {
      daily
      rotate 14
      compress
      delaycompress
      missingok
      notifempty
  }
  ```
- [ ] Nginx 日志定期清理
- [ ] 建议接入日志聚合（可选：ELK / Loki）

### 8.2 服务监控
- [ ] 配置 systemd 服务自动重启（Restart=always 已配置）
- [ ] 简单监控脚本（检测进程是否存活）
  ```bash
  #!/bin/bash
  # /var/www/fitness_coach/scripts/health_check.sh
  if ! systemctl is-active --quiet fitcoach; then
      systemctl restart fitcoach
      echo "$(date) - fitcoach restarted" >> /var/log/fitness_coach/watchdog.log
  fi
  ```
- [ ] cron 定时执行健康检查（每 5 分钟）
  ```
  */5 * * * * /var/www/fitness_coach/scripts/health_check.sh
  ```

### 8.3 更新部署流程
- [ ] 编写更新脚本 `scripts/deploy.sh`
  ```bash
  #!/bin/bash
  cd /var/www/fitness_coach
  git pull origin deploy
  source venv/bin/activate
  pip install -r requirements.txt --quiet
  sudo systemctl restart fitcoach
  sudo systemctl restart fitcoach-web
  echo "Deploy completed at $(date)"
  ```
- [ ] 后续可接入 CI/CD（GitHub Actions 自动部署）

---

## 阶段九：小程序前端开发

### 9.1 小程序项目初始化
- [ ] 使用微信开发者工具创建项目 `miniprogram/`
- [ ] 配置 `app.json`（页面路由、tabBar、窗口样式）
- [ ] 配置 `project.config.json`（appid、编译选项）

### 9.2 核心页面开发
- [ ] 首页 `pages/index/` — 功能入口
- [ ] 对话页 `pages/chat/` — 聊天界面（核心页面）
  - 聊天气泡组件
  - 底部输入栏
  - 流式消息渲染
- [ ] 个人中心 `pages/profile/` — 身高体重等信息
- [ ] 统计页 `pages/stats/` — 每日热量/运动统计
- [ ] 饮食记录 `pages/food-log/` — 记录今日饮食

### 9.3 工具与服务封装
- [ ] `utils/api.js` — 封装 HTTP 请求（统一 base URL、token 注入）
- [ ] `utils/auth.js` — 微信登录逻辑（wx.login + 后端换取 token）
- [ ] `utils/socket.js` — WebSocket 管理（连接、重连、心跳）
- [ ] `utils/storage.js` — 本地缓存管理

### 9.4 测试与审核
- [ ] 真机调试（Android + iOS）
- [ ] 提交微信审核
- [ ] 审核通过后发布

---

## 环境变量完整清单

| 变量名 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `LLM_MODEL` | 是 | LLM 模型名称 | `glm-4.7` |
| `OPENAI_API_KEY` | 是 | 智谱 AI API Key | `xxx` |
| `OPENAI_API_BASE` | 是 | 智谱 AI API Base URL | `https://open.bigmodel.cn/api/paas/v4` |
| `EMBEDDING_MODEL` | 是 | Embedding 模型 | `embedding-2` |
| `TianxingFood_API_KEY` | 否 | 天行食物营养 API | `xxx` |
| `WECHAT_APPID` | 小程序必填 | 微信小程序 AppID | `wx1234567890` |
| `WECHAT_SECRET` | 小程序必填 | 微信小程序 AppSecret | `xxx` |
| `JWT_SECRET_KEY` | 是 | JWT 签名密钥 | `random_32_chars` |
| `JWT_EXPIRE_HOURS` | 否 | Token 有效期（小时） | `72` |
| `CORS_ORIGINS` | 生产必填 | 允许的跨域域名（逗号分隔） | `https://xxx.com` |
| `DB_PATH` | 否 | SQLite 数据库路径 | `/var/www/.../fitness_coach.db` |
| `BACKEND_URL` | Web 前端 | 后端 API 地址 | `https://fitcoach.xxx.com` |
| `SSL_VERIFY` | 否 | 是否验证 SSL 证书 | `true` / `false` |

---

## 服务器推荐配置

| 项目 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| CPU | 2 核 | 4 核 | Gunicorn workers |
| 内存 | 2 GB | 4 GB | RAG + ChromaDB + LLM 请求 |
| 硬盘 | 40 GB SSD | 50 GB SSD | 向量库 + 数据库 + 日志 |
| 带宽 | 3 Mbps | 5 Mbps | 流式响应需持续连接 |
| 系统 | Ubuntu 22.04 | Ubuntu 24.04 | LTS |

---

## 常用运维命令速查

```bash
# 服务管理
sudo systemctl status fitcoach          # 查看后端状态
sudo systemctl restart fitcoach         # 重启后端
sudo systemctl status fitcoach-web      # 查看前端状态
sudo systemctl restart fitcoach-web     # 重启前端

# 日志查看
sudo journalctl -u fitcoach -f          # 实时查看后端日志
sudo tail -f /var/log/fitness_coach/error.log  # Gunicorn 错误日志
sudo tail -f /var/log/nginx/access.log  # Nginx 访问日志

# SSL 证书
sudo certbot certificates               # 查看证书状态
sudo certbot renew --dry-run           # 测试证书续期

# 代码更新
cd /var/www/fitness_coach && bash scripts/deploy.sh

# 数据库备份
cp /var/www/fitness_coach/data/fitness_coach.db /var/backups/fitcoach/manual_$(date +%Y%m%d).db

# 进程检查
ps aux | grep gunicorn                  # 查看 Gunicorn 进程
ss -tlnp | grep 8000                    # 查看 8000 端口占用
```

---

## 预计工期

| 阶段 | 工作量 | 说明 |
|---|---|---|
| 一、后端代码适配 | 1.5-2 天 | 微信登录 + SSE 鉴权 + CORS + 错误处理 |
| 二、服务器环境准备 | 0.5 天 | 购买服务器 + 基础软件 |
| 三、Nginx + SSL | 0.5 天 | 配置 + 证书申请 |
| 四、后端服务启动 | 0.5 天 | Gunicorn + systemd |
| 五、小程序后端对接 | 1 天 | 登录 + SSE 流式验证 |
| 六、Web 前端部署 | 0.5 天 | Streamlit 部署 |
| 七、安全加固 | 0.5 天 | 防火墙 + 备份 |
| 八、监控运维 | 0.5 天 | 日志 + 监控脚本 |
| 九、小程序前端开发 | 3-5 天 | 页面 + 联调 + 审核 |
| **合计** | **约 8-11 天** | |

> 如需启用 WebSocket 方案，额外增加 0.5-1 天。
