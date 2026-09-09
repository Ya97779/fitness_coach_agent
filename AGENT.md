# Fitness Coach 项目维护指南

本文档面向后续维护者和编码 Agent。内容以当前仓库代码为依据；当本文档、`README.md`、`frontend/README.md`、`CLAUDE.md` 与代码不一致时，应先核对代码和测试，不要直接沿用旧文档中的结论。

## 1. 项目定位

这是一个 AI 健身与饮食助手，主要包含：

- FastAPI 后端：用户认证、饮食/运动记录、对话、RAG、反馈等。
- 微信小程序：当前主要用户端，支持聊天、记录、训练计时、动作指南和个人中心。
- Streamlit 前端：次要或调试用途的 Web 客户端。
- LangGraph 多 Agent 流程：路由、闲聊、营养、健身和专家审核。
- Chroma + BM25 混合检索知识库。

仓库根目录是 Python 包导入和多数相对路径的基准目录。除非有特殊原因，后端命令都应从仓库根目录执行。

## 2. 代码结构

```text
backend/
  app/
    config.py                统一选择并加载 dotenv 配置
    main.py                 FastAPI 应用、路由、启动逻辑、SSE 接口
    auth.py                 微信登录、JWT 生成与认证
    database.py             SQLAlchemy 引擎和 Session
    models.py               数据库模型
    schemas.py              API 请求/响应模型
    calorie_calculator.py   运动热量算法与别名，热量计算的唯一事实来源
    llm_manager.py          LLM 实例缓存、并发限制、排队和调用代理
    memory.py               用户画像、日志统计、对话历史和 Prompt 注入
    food_api.py             食物信息外部接口及回退逻辑
    agents/                 路由、聊天、营养、健身、专家 Agent 和 LangGraph
    rag/                    文档加载、索引、混合检索、重排与高级 RAG
    seed_data.py            初始运动热量数据
  tests/                    unittest 测试
  static/                   头像、反馈和部署时提供的动作指南资源
frontend/
  app.py                    Streamlit 客户端
miniprogram/
  app.js / app.json         小程序入口、全局状态和页面注册
  pages/                    页面代码
  utils/                    API、认证、SSE、Markdown 等公共逻辑
  data/exercises/           动作和训练数据
knowledgebase/              当前仓库实际跟踪的 Markdown 知识文档
chroma_db/                  本地向量索引，不应作为业务源码提交
config.yaml                 非敏感运行配置、profile 和功能开关
docs/                       设计、模型评估和项目说明
requirements.txt            Python 依赖
update.sh                   服务器从 deploy 分支更新并重启服务的脚本
```

RAG 默认读取 `knowledgebase/`；如果通过 `KNOWLEDGE_BASE_DIR` 指向其他目录，必须确认目标目录确实存在，避免无意中创建第二份空知识库。

## 3. 核心运行流程

### 3.1 API 与认证

API 统一使用 `/api/v1` 前缀，主要接口包括：

- `POST /api/v1/auth/wx-login`
- `GET /api/v1/user/me`
- `POST /api/v1/user/profile`
- `POST /api/v1/user/avatar`
- `GET /api/v1/user/me/logs`
- `GET /api/v1/user/me/today`
- `POST|PATCH|DELETE /api/v1/food-log`
- `POST|PATCH|DELETE /api/v1/exercise-log`
- `DELETE /api/v1/user/me/data`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`（请求体支持 `session_id`、`request_id`）
- `GET /api/v1/chat/history`
- `POST /api/v1/feedback`
- `GET /api/v1/agents`
- `POST /api/v1/estimate-calories`

除登录等公开接口外，请继续通过 `get_current_user` 获取用户，不要信任客户端提交的用户 ID。任何查询、修改或删除都必须限制在当前用户范围内。

本地开发的 localhost 无 Token 兼容逻辑只在 `dev` profile 的
`allow_local_auth: true` 时启用；生产 profile 明确关闭。不要通过反向代理
地址判断生产请求是否来自本机。

### 3.2 聊天与 Agent

非流式 `/chat` 使用完整 LangGraph：

```text
router -> chat
       -> nutrition -> expert_review -> 必要时重试
       -> fitness   -> expert_review -> 必要时重试
```

流式 `/chat/stream` 为降低延迟走独立路径，不经过完整专家审核。它先发送
状态事件，再在后台线程中运行同步生成器，通过队列发送 SSE，并在成功完成后
保存完整回复。明确的饮食/运动记录会复用现有工具走确定性快速路径；复杂咨询
继续走原有工具循环。`stream=True` 下，聊天、营养和健身 Agent 的首轮工具决策、
工具后的最终回答都使用 `.stream()`；工具参数必须完整后才会执行，确定性记录则
在业务写入完成后返回一条结果。非流式 `/chat` 仍保留同步图流程。

因此，修改 Agent 时必须同时检查两条执行路径：

- `process_user_message`：完整图流程。
- `stream_user_message`：线上小程序主要使用的流式流程。

路由采用高精度规则、会话承接和结构化 LLM 判断的混合策略。流式与非流式
都传入相同的会话上下文；修改意图识别规则时，要测试普通聊天、饮食记录、
运动记录、跨领域输入和“第二个/换成哑铃”等短追问。

营养和健身工具会直接写数据库。工具描述、Prompt、工具参数模型和回退解析逻辑必须一起维护。当前代码包含针对模型未正确返回结构化 tool call 时的文本解析和自动记录回退，不要只验证理想的 tool-calling 路径。

### 3.3 SSE 协议

小程序聊天页依赖当前 SSE 格式，包括：

- `event: status`
- `event: queue`
- `event: intent`
- 普通 `data:` 文本块
- `data: [DONE]`
- 心跳注释

修改流式响应、事件名称、JSON 字段或结束标记时，必须同步修改 `miniprogram/pages/chat/chat.js` 和相关请求解码逻辑。还要验证 UTF-8 分块、半个 JSON 跨 chunk、断线、超时以及页面切换后恢复显示的情况。

### 3.4 用户记忆

`MemoryManager` 继续负责用户画像、今日/周统计和历史日志，但三层记忆的职责
现在明确为：`ConversationSession` 保存结构化工作记忆，现有
`ConversationLog` 保存逐轮情景记忆，`UserMemory` 保存有来源/置信度/有效期的
语义事实。最近对话通过有字符预算的 `HumanMessage`/`AIMessage` 列表进入上下文，
不再把原始用户历史拼入 System Prompt。可通过 `/api/v1/memory` 查看、PUT
修正、DELETE 忘记用户语义记忆。

新增记忆字段时要保持两条路径行为一致，并检查：

- 新字段是否来自当前用户。
- 空值是否安全降级。
- 是否会把历史对话重复注入。
- 保存和摘要失败是否会影响主回复。

### 3.5 热量与日志

运动热量计算必须复用 `backend/app/calorie_calculator.py`。这里集中维护 MET、力量训练按组计算和动作别名；不要在 API、Agent 或小程序中再复制一套服务端热量公式。

新增或修改饮食/运动日志时，应保证：

- `DailyLog`、明细记录和汇总值在同一事务语义内更新。
- 更新和删除后重新计算汇总，避免累计误差。
- 失败时回滚 Session。
- 后台线程自行创建和关闭数据库 Session，不复用请求线程 Session。

`FoodItem.calories` 表示用户本次摄入的总热量；`FoodCalorieCache` 只保存可复用
的热量基准。固体重量统一换算成 `per_100g + g`，液体换算成
`per_100ml + ml`，产品规则允许液体查询按 `1g = 1ml` 交叉匹配；个/份/碗等单位
使用 `per_unit + 单位`。查询必须同时匹配规范化名称、基准类型和单位。所有缓存读写
复用 `backend/app/food_cache.py`，缓存失败不能回滚用户的饮食记录。版本控制中的
`backend/data/common_food_calories.json` 是基础参考数据；生产迁移会幂等导入，
人工数据优先级高于参考数据，参考数据优先级高于 API、LLM 和本地降级值。
`FoodItem.calorie_status` 必须在 `pending`、`ready`、`failed` 中取值。后台估算无论
成功或失败都要写入终态；读取今日记录时会把超时的 `pending` 转为 `failed`。

### 3.6 RAG

RAG 使用 Chroma 向量检索与 BM25，通过 RRF 融合；还包含查询扩展、HyDE、CoT、Self-RAG、Agentic RAG 和可选 Jina 重排。

索引状态保存在 `chroma_db/indexed_files.json`，用文件哈希判断新增或变更。文档变更可能触发全量重建，所以执行重建前要确认 `CHROMA_DIR` 指向项目内预期目录，避免删除错误路径。

支持的主要文档格式包括 PDF、DOC/DOCX、TXT/MD、HTML 和常见图片。新增格式时，应同时补充加载、文本清洗、元数据和索引测试。

不要提交本机生成的 Chroma 数据。知识文档与向量索引是两类资产：文档可评审、可版本化；索引应由目标环境重新生成。

## 4. 配置

真实 `.env` 含密钥，不要读取后复制到日志、提交、测试快照或回复中。维护 `.env.example` 时只放占位符。

配置由 `config.yaml` 和 `backend/app/config.py` 统一加载：

- `config.yaml` 的 `active_profile` 和 `profiles` 选择 `.env.dev` 或 `.env`。
- `config.yaml` 管理模型、路径、CORS、JWT 有效期和 RAG 功能开关等非敏感配置。
- `.env` / `.env.dev` 只放 API Key、JWT Secret、微信密钥、数据库连接串等敏感值。
- `FITNESS_PROFILE=dev|prod` 可以临时覆盖当前 profile；`FITNESS_CONFIG_FILE` 可以选择另一份 YAML。
- `ENV_FILE` 仍保留为 CI/故障排查时的直接覆盖方式，但不作为日常配置入口。

当前代码的重要环境变量如下：

### 必需或接近必需

- `DATABASE_URL`：SQLAlchemy 数据库连接串；当前后端导入时即要求存在。
- `JWT_SECRET_KEY`：至少 32 个字符，生产必须使用随机密钥。
- `OPENAI_API_KEY`：LLM、Embedding 和 RAG 初始化所需；没有时部分 RAG 启动逻辑会跳过。

### LLM 与 Embedding

- `OPENAI_API_BASE`、`LLM_MODEL`、`LLM_REASONING_EFFORT`、`LLM_THINKING_TYPE`、
  `LLM_CLEAR_THINKING`、`EMBEDDING_MODEL`、`API_BASE_URL`：由 `config.yaml` 管理。
- `glm-5.3-flash` 只支持开启思考；当前固定 `thinking.type=enabled`、
  `reasoning_effort=low`。不要为该模型配置 `disabled`。

### 微信与 Web

- `WECHAT_APPID`
- `WECHAT_SECRET`
- `JWT_EXPIRE_HOURS`、`CORS_ORIGINS`、`BACKEND_URL`、`SSL_VERIFY`：由 `config.yaml` 管理。

### 外部数据与 RAG

- `TianxingFood_API_KEY`
- `JINA_API_KEY`
- `CHROMA_DIR`、`KNOWLEDGE_BASE_DIR` 及 `ENABLE_*` 功能开关：由 `config.yaml` 管理。

`.env.example` 是不含机密的配置模板；涉及配置的改动应同步修正文档和示例，不能假设线上服务器的 `.env` 与本地配置完全一致。

## 5. 本地启动

使用当前项目支持的 Python 3.12 环境，从仓库根目录执行：

推荐使用一键开发脚本。脚本默认选择 `config.yaml` 的 `dev` profile，使用 `.env.dev`、
创建本地测试用户、关闭启动时 RAG 索引，并只监听 `127.0.0.1`；脚本本身不再硬编码
数据库、JWT 或模型配置：

```powershell
.\scripts\dev-backend.ps1
```

如果要模拟服务器的“无热重载、固定进程数”启动方式，在 Windows 本地使用：

```powershell
.\scripts\start-backend.ps1 -SeedDevUser
```

该脚本使用 Uvicorn worker（Windows 本地不依赖 Gunicorn），服务器仍由
`systemd -> gunicorn -> UvicornWorker` 启动；可以通过 `-Workers 2` 调整本地进程数。

如果要验证 `prod` profile 的配置，显式选择它；确认 `.env` 指向的是目标环境后再执行，
因为该 profile 不会创建本地测试用户：

```powershell
.\scripts\dev-backend.ps1 -Profile prod
```

如果只是临时使用某个本地 dotenv 文件，可以保留旧的直接覆盖方式：

```powershell
.\scripts\dev-backend.ps1 -Profile dev -EnvFile .env
```

需要验证知识库增量索引时显式开启；该操作可能调用 Embedding 服务：

```powershell
.\scripts\dev-backend.ps1 -WithRagIndex
```

也可以手动启动：

```powershell
python -m pip install -r requirements.txt
$env:FITNESS_PROFILE = "dev"
python -m uvicorn backend.app.main:app --reload --port 8000
```

启动 Streamlit 客户端：

```powershell
python -m streamlit run frontend/app.py
```

微信小程序使用微信开发者工具导入 `miniprogram/`。开发版可在开发者工具控制台设置本地 API 覆盖，然后重新编译：

```javascript
wx.setStorageSync('DEV_API_BASE_URL', 'http://127.0.0.1:8000')
```

移除覆盖并恢复生产 API：

```javascript
wx.removeStorageSync('DEV_API_BASE_URL')
```

该覆盖只对微信 `develop` 环境生效，体验版和正式版始终使用生产地址。本地设置中还需勾选“不校验合法域名、TLS 版本及 HTTPS 证书”。切换本地后清除旧登录 Token，避免把生产 JWT 发给本地后端。

后端启动仍会执行 `Base.metadata.create_all()`，它只负责兼容创建缺失表，不能替代
生产迁移。阶段 0–2 的会话/语义记忆表及每日记录、食物缓存结构由
`scripts/migrate_phase02.py` 单独执行。脚本发现重复每日记录时会停止；发现旧版
食物缓存表时会将其完整复制到带时间戳的备份表，重建 v2 缓存表并导入常见食物
参考数据。后续若增加非兼容字段，仍应引入正式迁移版本和回滚方案。

## 6. 测试与检查

项目测试以 `unittest` 为主：

```powershell
python -m unittest discover -s backend/tests -v
```

可按模块运行，例如：

```powershell
python -m unittest backend.tests.test_rag -v
python -m unittest backend.tests.test_calorie_calculator -v
python -m unittest backend.tests.test_memory -v
```

提交前至少执行：

```powershell
python -m compileall -q backend/app frontend/app.py
git diff --check
```

当前本地基线：`python -m unittest discover -s backend/tests -p 'test_*.py' -q`
共 178 项通过。RAG 测试可能打印外部模型/提示词降级日志，但不影响该基线的
退出状态；涉及真实模型的评估仍需单独配置测试 Key。

RAG 路由器示例中的 JSON 花括号已按 LangChain 模板规则转义；如果真实模型不可用，
测试仍会按既有逻辑记录降级日志并使用默认策略。

这些是维护开始时的已知基线，不应把它们误报为新改动引入的问题；但修改相关区域时，应顺手修复或至少准确记录剩余失败。`pytest` 当前未列在 `requirements.txt`，不要把 `pytest` 命令当作所有环境都可用的默认验证方式。

### 按改动范围选择验证

- API/Schema：接口成功、鉴权失败、资源归属、无数据、非法参数。
- 数据模型：新库建表、已有库迁移、事务回滚、级联关系。
- Agent/Prompt：普通回复、结构化工具调用、文本工具调用回退、写库副作用。
- 流式聊天：SSE 事件顺序、中文分块、`[DONE]`、中断和保存历史。
- Memory：画像、今日统计、周统计、最近历史、空用户和批量加载一致性。
- RAG：空知识库、新增/更新/删除文档、混合检索、缓存和关闭可选能力。
- 小程序：真机或开发者工具验证授权、Token 过期、缓存恢复和页面跳转。

## 7. 编码约束

- 保持路由薄：复杂计算和可复用逻辑放到独立模块，不继续扩大 `main.py`。
- 所有数据库写操作都应明确提交、回滚和关闭 Session。
- 不在日志中输出 Token、openid、完整用户 Prompt、数据库连接串或外部 API 密钥。
- 对外错误返回稳定、可理解的信息；内部异常保留可定位日志，但不要泄露密钥和数据库细节。
- LLM 调用优先通过 `llm_manager.py`，保留并发限制、缓存和排队能力。
- 修改模型名称或默认参数时，同时检查聊天、路由、摘要、RAG 和测试中的默认值，避免不同模块悄悄使用不同模型。
- Prompt 与代码里的结构化输出格式必须同步修改，并为解析失败提供明确回退。
- 不直接修改 `miniprogram/components/mp-html`、`showdown.js` 等第三方或供应商代码，除非任务就是升级依赖，并能说明来源和验证范围。
- 小程序页面变更通常需要同步检查 `.js`、`.wxml`、`.wxss`、`.json` 四类文件。
- 新增静态资源时确认部署脚本和服务器目录是否包含它们。`backend/static/guide` 的图片并非完整跟踪在仓库中。
- 不提交 `.env`、数据库文件、日志、用户上传、反馈附件或生成的向量索引。

## 8. 分支与部署

- `main` 用于日常维护和新功能开发；建议从最新 `main` 创建 `codex/<topic>` 或团队约定的功能分支。
- `deploy` 是服务器部署快照。`update.sh` 会在服务器上将工作区强制重置到 `origin/deploy` 并重启 `fitcoach` 服务。
- 不要在 `deploy` 上长期开发，也不要默认认为分支名较特殊就一定更新；比较提交拓扑和实际 diff 后再判断。
- 合并到 `deploy` 前，确认数据库迁移、环境变量、静态资源、依赖安装和服务重启方式均已准备。
- 未经明确要求，不执行强制推送、重置用户本地修改或覆盖部署环境数据。

## 9. 已知技术债与优先处理项

维护或开发新功能前，优先关注这些问题：

1. 统一 `knowledgebase/` 与 `knowledge_base/`，并修正默认配置、测试和部署目录。
2. 保持 `config.yaml` profile、`.env.example` 和实际读取的配置项一致。
3. 修复现有 unittest 基线错误，使测试真正具备回归价值。
4. 为数据库引入明确的迁移机制，避免依赖 `create_all()` 修改生产结构。
5. 减少 `backend/app/main.py` 的职责，把业务逻辑逐步下沉到 service/repository 层。
6. 让流式和非流式 Agent 流程共享更多行为与测试，避免结果长期漂移。
7. 统一运行时模型默认值、文档和评估脚本中的模型配置。

## 10. 完成定义

一次维护或功能开发只有在以下条件基本满足时才算完成：

- 已定位实际调用链，而不是只修改表面入口。
- 已考虑用户数据隔离、事务、配置和部署兼容性。
- 前后端协议改动已同步。
- 新逻辑有针对性测试，相关旧测试已运行。
- 完整测试中的新增失败与既有失败已区分说明。
- `git diff --check` 通过，未包含密钥、生成文件或无关改动。
- 数据库、环境变量或索引格式变化附带可执行的升级说明。
- 文档与当前实现一致；若仍有已知差异，已明确记录而不是隐藏。
