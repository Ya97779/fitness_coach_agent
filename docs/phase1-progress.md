# 阶段一：后端代码适配 — 进度文档

**完成日期：** 2026-05-07
**分支：** deploy

## 已完成项

### 1.1 Git 分支准备
- [x] deploy 分支已创建
- [x] `.env.example` 已存在
- [x] `.gitignore` 新增 `fitness_coach.db`、`chroma_db/`、`checkpoints.db`

### 1.2 微信登录模块
- [x] 新建 `backend/app/auth.py`
  - `wx_code_to_session(code)` — 调用微信 jscode2session 接口
  - `create_access_token(user_id)` — 生成 JWT（HS256，payload 含 sub/exp）
  - `decode_access_token(token)` — 解析 JWT，返回 user_id
  - `get_current_user` — FastAPI Depends，从 Bearer token 解析用户
- [x] `backend/app/models.py` User 表新增字段：`openid`(unique, index)、`unionid`(index)、`session_key`、`nickname`、`avatar_url`；身体数据字段改为 nullable+default，兼容微信登录先于资料填写的场景
- [x] `POST /api/v1/auth/wx-login` — 小程序登录端点，code 换 JWT + 自动创建用户
- [x] `requirements.txt` 新增 `PyJWT`、`httpx`

### 1.3 流式对话
- [x] 现有 SSE 端点 `/api/v1/chat/stream` 保持不变，已接入鉴权（user_id 从 token 解析）
- [ ] WebSocket 方案未实现（文档中标注为备选，SSE 优先）

### 1.4 CORS 配置
- [x] `main.py` 添加 CORS 中间件，从 `CORS_ORIGINS` 环境变量读取允许的域名
- [x] 配置 `allow_credentials=True`、`allow_methods=["*"]`、`allow_headers=["*"]`

### 1.5 API 路由规范化
- [x] 所有业务路由迁移到 `APIRouter(prefix="/api/v1")`
- [x] 根路由保留 `/agents` 作为健康检查（无需鉴权）
- [x] 用户相关端点改为 `/user/me`、`/user/me/logs`、`/user/me/today`（从 token 获取用户，不再暴露 user_id 路径参数）

### 1.6 数据库适配
- [x] `backend/app/database.py` 从 `DB_PATH` 环境变量读取数据库路径
- [x] `get_db()` 从 `main.py` 移至 `database.py`（解决 auth 模块的循环依赖）

### 1.7 错误处理标准化
- [x] 全局 HTTPException 处理器：返回 `{"code": int, "message": str}`
- [x] 全局 Exception 处理器：捕获未处理异常，返回 `{"code": 500, "message": "服务器内部错误"}`

### 1.8 前端适配
- [x] `BACKEND_URL` 从环境变量读取
- [x] `SSL_VERIFY` 从环境变量读取
- [x] API 路径更新为 `/api/v1/...`
- [x] 所有请求附带 `Authorization: Bearer {token}` 头
- [x] 新增微信登录流程（输入 code 获取 token）

### 1.9 单元测试
- [x] `backend/tests/test_auth.py` — 12 个测试用例
  - JWT 生成/解析（正常、过期、无效、错误密钥）
  - get_current_user（正常、无效 token、用户不存在）
  - wx_code_to_session（成功、微信返回错误）
- [x] 全部 162 个测试通过（12 新增 + 150 已有）

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `.gitignore` | 修改 | 新增 DB 文件忽略规则 |
| `requirements.txt` | 修改 | 新增 PyJWT、httpx |
| `backend/app/database.py` | 修改 | DB_PATH 环境变量 + get_db() |
| `backend/app/models.py` | 修改 | User 表新增微信字段 |
| `backend/app/auth.py` | 新建 | 微信登录 + JWT + 鉴权依赖 |
| `backend/app/main.py` | 重写 | CORS + APIRouter + 鉴权端点 + 错误处理 |
| `frontend/app.py` | 修改 | 环境变量 + API 路径 + token 注入 + 登录流程 |
| `backend/tests/test_auth.py` | 新建 | auth 模块单元测试 |

## 未完成项

| 项目 | 原因 | 后续计划 |
|------|------|----------|
| WebSocket 端点 | SSE 方案已满足需求 | 如小程序基础库不支持 enableChunked 再启用 |
| Web 微信扫码登录 | 需要公众号 OAuth2，当前只做小程序 | 阶段六 Web 前端开发时实现 |

## 已知限制

1. **User 表 height/weight/age/gender 有默认值**：微信登录创建的用户身体数据为 0/"未知"，需要用户后续填写资料更新
2. **前端 wx.login() 需要真机环境**：Streamlit 前端的 code 输入需要从微信开发者工具获取，本地调试流程不同于生产
3. **数据库迁移**：新增字段全部 nullable，兼容已有 SQLite 数据库，无需手动迁移

## 验证结果

```
启动后端: cd backend && python -m uvicorn app.main:app --reload
Swagger 文档: http://127.0.0.1:8000/docs (所有 /api/v1/ 路由可见)
未鉴权访问 /api/v1/user/me: 返回 401 {"code":401,"message":"Not authenticated"}
测试: cd backend && python -m unittest discover tests → 162 tests OK
```
