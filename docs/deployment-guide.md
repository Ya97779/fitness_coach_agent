# FitCoach AI 部署指南

更新日期：2026-05-20

## 域名规划

| 域名 | 用途 | 指向 |
|------|------|------|
| `gzyhm.xyz` | 网页前端（Streamlit） | Nginx → Streamlit (8501) |
| `www.gzyhm.xyz` | 同上（301 跳转） | Nginx → `gzyhm.xyz` |
| `gzyapi.gzyhm.xyz` | 后端 API + 图片资源 | Nginx → FastAPI (8000) |

---

## 一、DNS 解析配置

登录你的域名注册商控制台（阿里云/腾讯云/Cloudflare），添加以下 DNS 记录：

| 主机记录 | 类型 | 记录值 | 说明 |
|---------|------|--------|------|
| `@` | A | `<服务器公网IP>` | 主域名 |
| `www` | CNAME | `gzyhm.xyz` | www 跳转主域名 |
| `gzyapi` | A | `<服务器公网IP>` | 后端 API |

添加后等待 DNS 生效（通常 5-30 分钟），可用以下命令验证：

```bash
nslookup gzyapi.gzyhm.xyz
ping gzyapi.gzyhm.xyz
```

---

## 二、云服务器准备

### 2.1 购买与连接

推荐配置：2 核 4G 内存，Ubuntu 22.04 LTS，开放安全组端口 80、443、22。

```bash
# SSH 连接服务器
ssh root@<服务器公网IP>
```

### 2.2 系统初始化

```bash
# 更新系统
apt update && apt upgrade -y

# 安装基础工具
apt install -y git curl wget nginx certbot python3-certbot-nginx python3-pip python3-venv
```

### 2.3 创建项目用户（可选，推荐）

```bash
# 创建非 root 用户
adduser fitcoach
usermod -aG sudo fitcoach

# 切换到该用户
su - fitcoach
```

---

## 三、后端部署

### 3.1 拉取代码

```bash
# 创建项目目录
sudo mkdir -p /var/www/fitcoach
sudo chown $USER:$USER /var/www/fitcoach

# 克隆仓库
cd /var/www/fitcoach
git clone https://github.com/<你的用户名>/fitness_coach.git .
# 或者从本地 scp 上传：
# scp -r D:\fitness_coach root@<服务器IP>:/var/www/fitcoach/
```

### 3.2 创建虚拟环境

```bash
cd /var/www/fitcoach

# 创建 Python 虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3.3 配置环境变量

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# LLM 配置
LLM_MODEL=glm-4.7
OPENAI_API_KEY=<你的智谱API Key>
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-2

# 微信小程序
WECHAT_APPID=<你的小程序AppID>
WECHAT_SECRET=<你的小程序AppSecret>

# JWT（至少 32 位随机字符串）
JWT_SECRET_KEY=<用 openssl rand -hex 32 生成>
JWT_EXPIRE_HOURS=24

# 天行数据食物API
TianxingFood_API_KEY=<你的天行API Key>

# CORS（逗号分隔）
CORS_ORIGINS=https://gzyhm.xyz,https://www.gzyhm.xyz

# 数据库
DB_PATH=/var/www/fitcoach/fitness_coach.db

# 其他
SSL_VERIFY=true
EOF

# 设置权限（防止泄露）
chmod 600 .env
```

生成 JWT 密钥的命令：

```bash
openssl rand -hex 32
```

### 3.4 放置图片资源

```bash
# 创建静态资源目录
mkdir -p backend/static/guide

# 上传图片文件（从本地 scp 或其他方式）
# 图片文件名需与 miniprogram/data/exercises/*.js 中 cover 字段一致
scp -r D:\fitness_coach\backend\static\guide\* root@<服务器IP>:/var/www/fitcoach/backend/static/guide/
```

验证图片可访问：

```bash
ls -la /var/www/fitcoach/backend/static/guide/
```

### 3.5 测试后端启动

```bash
cd /var/www/fitcoach
source venv/bin/activate

# 直接启动测试（前台运行，Ctrl+C 退出）
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 看到类似输出说明成功：
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Started reloader process

# 按 Ctrl+C 停止
```

### 3.6 配置 Systemd 服务（后台常驻运行）

```bash
# 创建 systemd 服务文件
sudo cat > /etc/systemd/system/fitcoach.service << 'EOF'
[Unit]
Description=FitCoach AI Backend
After=network.target

[Service]
Type=simple
User=fitcoach
Group=fitcoach
WorkingDirectory=/var/www/fitcoach
Environment="PATH=/var/www/fitcoach/venv/bin"
ExecStart=/var/www/fitcoach/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable fitcoach    # 开机自启
sudo systemctl start fitcoach

# 查看状态
sudo systemctl status fitcoach

# 查看日志
sudo journalctl -u fitcoach -f
```

### 3.7 验证后端运行

```bash
# 测试本地访问
curl http://127.0.0.1:8000/docs

# 测试图片资源
curl -I http://127.0.0.1:8000/guide/杠铃卧推.gif
```

---

## 四、Nginx 配置

### 4.1 创建配置文件

```bash
sudo cat > /etc/nginx/conf.d/gzyhm.conf << 'EOF'
# === 后端 API ===
server {
    listen 443 ssl http2;
    server_name gzyapi.gzyhm.xyz;

    ssl_certificate     /etc/nginx/ssl/gzyapi.gzyhm.xyz.pem;
    ssl_certificate_key /etc/nginx/ssl/gzyapi.gzyhm.xyz.key;

    # SSL 优化
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # 请求大小限制（上传文件用）
    client_max_body_size 10m;

    # API 代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式响应（聊天功能必须）
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}

# === 网页前端 ===
server {
    listen 443 ssl http2;
    server_name gzyhm.xyz www.gzyhm.xyz;

    ssl_certificate     /etc/nginx/ssl/gzyhm.xyz.pem;
    ssl_certificate_key /etc/nginx/ssl/gzyhm.xyz.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Streamlit 代理
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# === HTTP 强制跳转 HTTPS ===
server {
    listen 80;
    server_name gzyhm.xyz www.gzyhm.xyz gzyapi.gzyhm.xyz;
    return 301 https://$host$request_uri;
}
EOF
```

### 4.2 申请 SSL 证书

**方式 A：Let's Encrypt（免费，自动续期，推荐）**

```bash
# 申请证书（会自动修改 Nginx 配置）
sudo certbot --nginx -d gzyhm.xyz -d www.gzyhm.xyz -d gzyapi.gzyhm.xyz

# 验证自动续期
sudo certbot renew --dry-run
```

**方式 B：云服务商免费证书**

1. 登录阿里云/腾讯云控制台
2. SSL 证书 → 免费证书 → 申请
3. 填写域名 `gzyhm.xyz` 和 `gzyapi.gzyhm.xyz`（各申请一个）
4. 验证域名所有权（DNS 验证或文件验证）
5. 下载 Nginx 格式证书（.pem + .key）
6. 上传到服务器：

```bash
sudo mkdir -p /etc/nginx/ssl
scp gzyhm.xyz.pem root@<服务器IP>:/etc/nginx/ssl/
scp gzyhm.xyz.key root@<服务器IP>:/etc/nginx/ssl/
scp gzyapi.gzyhm.xyz.pem root@<服务器IP>:/etc/nginx/ssl/
scp gzyapi.gzyhm.xyz.key root@<服务器IP>:/etc/nginx/ssl/
```

### 4.3 启动 Nginx

```bash
# 测试配置语法
sudo nginx -t

# 启动/重载
sudo systemctl enable nginx
sudo systemctl start nginx
sudo systemctl reload nginx
```

### 4.4 验证 HTTPS 访问

```bash
# 测试 API
curl https://gzyapi.gzyhm.xyz/docs

# 测试图片
curl -I https://gzyapi.gzyhm.xyz/guide/杠铃卧推.gif

# 测试前端（Streamlit 未启动时会 502，这是正常的）
curl -I https://gzyhm.xyz
```

---

## 五、Streamlit 前端部署（可选）

如果需要网页版前端：

```bash
# 创建 systemd 服务
sudo cat > /etc/systemd/system/fitcoach-web.service << 'EOF'
[Unit]
Description=FitCoach Streamlit Frontend
After=network.target fitcoach.service

[Service]
Type=simple
User=fitcoach
Group=fitcoach
WorkingDirectory=/var/www/fitcoach
Environment="PATH=/var/www/fitcoach/venv/bin"
ExecStart=/var/www/fitcoach/venv/bin/streamlit run frontend/app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fitcoach-web
sudo systemctl start fitcoach-web
```

---

## 六、微信公众平台配置

1. 登录 [微信公众平台](https://mp.weixin.qq.com)
2. 开发管理 → 开发设置 → 服务器域名
3. 配置以下域名：

| 类型 | 域名 |
|------|------|
| request 合法域名 | `https://gzyapi.gzyhm.xyz` |
| uploadFile 合法域名 | `https://gzyapi.gzyhm.xyz` |
| downloadFile 合法域名 | `https://gzyapi.gzyhm.xyz` |

4. 点击保存并验证

---

## 七、小程序配置更新

确认 `miniprogram/utils/config.js` 中的域名正确：

```js
const API_BASE_URL = 'https://gzyapi.gzyhm.xyz'
const IMG_BASE_URL = 'https://gzyapi.gzyhm.xyz/guide'
```

然后在微信开发者工具中重新上传小程序。

---

## 八、日常运维

### 常用命令

```bash
# 查看后端状态
sudo systemctl status fitcoach

# 重启后端
sudo systemctl restart fitcoach

# 查看后端日志
sudo journalctl -u fitcoach -f --lines=100

# 查看 Nginx 日志
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# 重启 Nginx
sudo systemctl reload nginx
```

### 更新代码

```bash
cd /var/www/fitcoach
git pull origin main
source venv/bin/activate
pip install -r requirements.txt   # 如有新依赖
sudo systemctl restart fitcoach
```

### SSL 证书续期

Let's Encrypt 证书 90 天过期，certbot 会自动续期。手动续期：

```bash
sudo certbot renew
sudo systemctl reload nginx
```

---

## 九、常见问题

| 问题 | 排查 |
|------|------|
| 小程序请求失败 | 检查微信公众平台域名白名单是否配置 |
| SSE 流式响应卡住 | 检查 Nginx `proxy_buffering off` 是否生效 |
| 图片 404 | 检查 `backend/static/guide/` 目录是否存在且文件名匹配 |
| 后端启动失败 | `journalctl -u fitcoach` 查看错误日志 |
| 502 Bad Gateway | 后端服务未启动，`systemctl start fitcoach` |
| SSL 证书错误 | `certbot certificates` 查看证书状态 |
