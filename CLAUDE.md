# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FitCoach AI — 基于多 Agent 的健身营养顾问系统。后端 FastAPI + LangGraph 多 Agent 编排，前端 Streamlit + 微信小程序，RAG 检索增强生成，SQLite 持久化。

## 常用命令

```bash
# 启动后端
uvicorn backend.app.main:app --reload --port 8000

# 启动前端（Streamlit Web 版）
streamlit run frontend/app.py

# 运行全部测试（项目根目录执行）
python -m pytest backend/tests/ -v --tb=short

# 运行单个测试文件
python -m pytest backend/tests/test_agents.py -v
python -m pytest backend/tests/test_rag.py -v

# RAG 效果评估（会调用 API，消耗 token）
python -m tests.test_rag_evaluation --quick             # 5 条快速评估
python -m tests.test_rag_evaluation --output report.json # 导出报告

# 安装依赖
pip install -r requirements.txt
```

## 架构

### 请求流转

```
Streamlit 前端 / 微信小程序
  → POST /api/v1/chat/stream (SSE)
    → FastAPI (JWT 鉴权，HTTPBearer)
      → MemoryManager (加载用户画像、对话历史、统计数据)
      → LangGraph StateGraph
        → router.py: 关键词预筛 + LLM 兜底分类
          → chat_agent / nutrition_agent / fitness_agent
            → nutrition: food_api (天行API) + RAG 检索
            → fitness: RAG 检索
          → expert_agent 评审 (评分 < 3 重试，最多 3 次)
            → 简单事实查询跳过评审 (skip_review / quick_patterns)
        → MemoryManager.save_conversation() 持久化
```

### 目录结构

```
fitness_coach_agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口，路由注册，启动时增量索引
│   │   ├── auth.py              # JWT 鉴权 + 微信 jscode2session 登录
│   │   ├── database.py          # SQLAlchemy engine/session（SQLite）
│   │   ├── models.py            # 数据模型：User, DailyLog, FoodItem, ExerciseItem, ConversationLog
│   │   ├── llm_manager.py       # ChatOpenAI 按 temperature 分桶缓存（单例）
│   │   ├── food_api.py          # 天行数据食物营养 API + 本地兜底数据
│   │   ├── agents/
│   │   │   ├── __init__.py      # 模块导出
│   │   │   ├── base.py          # AgentState TypedDict, AGENT_SYSTEM_PROMPTS（4 个 Agent 的 prompt）
│   │   │   ├── router.py        # 混合路由：关键词预筛 + LLM 兜底
│   │   │   ├── chat_agent.py    # 闲聊 Agent（无工具，直接 LLM 对话）
│   │   │   ├── nutrition_agent.py # 营养师 Agent（5 个 @tool）
│   │   │   ├── fitness_agent.py # 健身教练 Agent（4 个 @tool）
│   │   │   ├── expert_agent.py  # 专家评审 Agent（1-5 分评分）
│   │   │   └── graph.py         # LangGraph StateGraph 编排，流式/非流式入口
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── memory_manager.py    # MemoryManager 整合入口
│   │   │   ├── user_profile.py      # 用户画像加载（从 DB）
│   │   │   ├── conversation_summary.py # 对话历史摘要（LLM 压缩）
│   │   │   └── stats_summary.py     # 每日/每周统计汇总
│   │   └── rag/
│   │       ├── __init__.py      # ModernRAG 主类 + get_rag_instance() 全局单例
│   │       └── modules/
│   │           ├── __init__.py  # 模块导出
│   │           ├── loader.py    # 文档加载器（PDF/DOCX/TXT/MD）
│   │           ├── splitter.py  # 智能文本分割器
│   │           ├── preprocessor.py # 文本预处理（去重）
│   │           ├── doc_processor.py # 高级文档处理（表格检测、代码块、语义分块）
│   │           ├── bm25.py      # BM25 关键词检索（jieba 分词）
│   │           ├── hybrid_search.py # 混合检索（向量 + BM25，RRF 融合）
│   │           ├── query_expansion.py # 多查询扩展
│   │           ├── hyde.py      # 假设性文档嵌入
│   │           ├── cot.py       # 思维链推理
│   │           ├── self_rag.py  # 自我反思纠正
│   │           └── agentic_rag.py # Agentic RAG（LLM 自主决策检索策略）
│   └── tests/
│       ├── test_agents.py       # Agent 模块单元测试
│       ├── test_auth.py         # 鉴权测试
│       ├── test_memory.py       # 记忆系统测试
│       ├── test_rag.py          # RAG 测试
│       └── test_rag_evaluation.py # RAG 效果评估（需 API）
├── frontend/
│   └── app.py                   # Streamlit 单文件前端，聊天/档案/统计三模式
├── miniprogram/                 # 微信小程序
│   ├── app.js / app.json / app.wxss
│   ├── utils/
│   │   ├── config.js            # API_BASE_URL 配置
│   │   ├── request.js           # wx.request 封装 + SSE 流式请求
│   │   └── auth.js              # 微信登录逻辑
│   ├── components/              # 自定义组件
│   │   ├── exercise-item/       # 运动记录组件
│   │   └── food-item/           # 食物记录组件
│   ├── data/
│   │   ├── exercises.js         # 运动数据汇总
│   │   ├── exercises/           # 按部位分类：arms/back/cardio/chest/core/legs/shoulder
│   │   └── templates.js         # 训练模板
│   └── pages/
│       ├── home/                # 首页
│       ├── chat/                # AI 对话（SSE 流式）
│       ├── log/                 # 快捷记录
│       ├── profile/             # 个人档案
│       ├── stats/               # 数据统计
│       ├── timer/               # 训练计时器
│       │   ├── timer-setup/     # 计时器设置
│       │   ├── timer-training/  # 训练中
│       │   ├── timer-summary/   # 训练总结
│       │   └── training-plan/   # 周训练计划
│       └── exercise-guide/      # 动作指导
│           ├── exercise-guide/  # 指南首页（按部位分类）
│           ├── exercise-list/   # 动作列表
│           └── exercise-detail/ # 动作详情
├── knowledge_base/              # RAG 知识库（PDF 等文档）
├── chroma_db/                   # ChromaDB 向量库（gitignore）
├── fitness_coach.db             # SQLite 数据库（gitignore）
├── checkpoints.db               # LangGraph checkpoint（gitignore）
└── requirements.txt
```

### 核心模块详解

| 模块 | 路径 | 职责 |
|------|------|------|
| API 入口 | `backend/app/main.py` | FastAPI 路由（`/api/v1`），启动时增量索引，SSE 流式，CORS，全局异常处理 |
| 鉴权 | `backend/app/auth.py` | 微信 `jscode2session` 换 openid，JWT 生成/解析，`get_current_user` 依赖注入 |
| Agent 编排 | `backend/app/agents/graph.py` | LangGraph StateGraph，5 个节点：router→chat/nutrition/fitness→expert_review |
| 路由 | `backend/app/agents/router.py` | 混合路由：`_keyword_match()` 关键词预筛（40+ 中文关键词）→ `_llm_route()` LLM 兜底 |
| 闲聊 Agent | `backend/app/agents/chat_agent.py` | 直接 LLM 对话，无工具调用，无评审 |
| 营养师 Agent | `backend/app/agents/nutrition_agent.py` | 5 个工具：查用户信息、记录食物、查日统计、搜食物 API、搜营养知识 RAG |
| 健身教练 Agent | `backend/app/agents/fitness_agent.py` | 4 个工具：查用户信息、记录运动、估算热量(MET)、搜健身知识 RAG |
| 专家评审 | `backend/app/agents/expert_agent.py` | 1-5 分评分（专业性/个性化/实用性/安全性），评分 < 3 重试，最多 3 次 |
| RAG | `backend/app/rag/__init__.py` | ModernRAG 主类，支持 hybrid/vector/bm25/query_expansion/hyde 五种检索模式 |
| RAG 模块 | `backend/app/rag/modules/` | BM25(jieba)、混合检索(RRF)、HyDE、Self-RAG、Agentic RAG、语义分块 |
| 记忆系统 | `backend/app/memory/` | MemoryManager 整合 3 个子模块：UserProfileLoader、ConversationSummarizer、StatsSummarizer |
| LLM 管理 | `backend/app/llm_manager.py` | ChatOpenAI 实例按 temperature 分桶缓存（单例），避免重复创建 |
| 数据模型 | `backend/app/models.py` | SQLAlchemy 5 张表：User、DailyLog、FoodItem、ExerciseItem、ConversationLog |
| 食物 API | `backend/app/food_api.py` | 天行数据 API 查询食物营养，API 不可用时回退到本地 8 条兜底数据 |
| Streamlit 前端 | `frontend/app.py` | 单文件，三模式：chat(SSE 流式)、profile(登录/档案)、stats(图表) |
| 微信小程序 | `miniprogram/` | 6 个 tabBar 页面 + 训练计时器 + 动作指导，SSE 流式对话 |

### 关键设计模式

- **LLM 调用**：统一通过 `LLMManager.get_llm(temperature)` 获取实例，不要自行创建 `ChatOpenAI`
- **Agent 工具**：nutrition/fitness agent 用 `@tool` 装饰器定义工具，LLM 自主决定调用；工具只做检索/记录，不生成回答
- **流式响应**：`stream_user_message()` 在 graph.py 中独立实现，直接调用 agent 函数的 `stream=True` 模式，跳过 expert review 以保证实时性
- **非流式响应**：`process_user_message()` 走完整 LangGraph 工作流，包含 expert review 和重试
- **RAG 单例**：通过 `get_rag_instance()` 获取全局 ModernRAG 实例，内部带 LRU 查询缓存（128 条，5 分钟 TTL）
- **增量索引**：启动时 `check_and_update_index()` 扫描 `knowledge_base/`，MD5 对比增量更新 ChromaDB；文件更新时全量重建
- **记忆注入**：MemoryManager 在 graph 入口处加载，通过 `enhance_system_prompt()` 将用户画像/统计/对话历史注入各 Agent 的 system prompt
- **快速通道**：`should_skip_review()` 匹配简单事实查询模式（热量查询等）或短回复（< 150 字符），跳过 expert review
- **鉴权**：HTTPBearer 方案，微信 code → openid → JWT token，`get_current_user` 依赖注入

## 技术栈

- **LLM**：智谱 GLM-4.7，通过 OpenAI 兼容接口调用 (`open.bigmodel.cn`)
- **Embedding**：智谱 embedding-2
- **向量库**：ChromaDB（本地持久化 `./chroma_db`）
- **数据库**：SQLite + SQLAlchemy（`./fitness_coach.db`）
- **Agent 框架**：LangChain + LangGraph（StateGraph 编排）
- **鉴权**：JWT (PyJWT) + 微信小程序登录 (httpx)
- **前端 Web**：Streamlit + Plotly（图表）
- **前端小程序**：微信原生小程序
- **外部 API**：天行数据食物营养 API（有本地兜底数据）
- **NLP**：jieba（BM25 分词）

## 环境变量

创建 `.env` 文件（已 gitignore），关键变量：
- `LLM_MODEL` — LLM 模型名（默认 `glm-4.7`）
- `OPENAI_API_KEY` — 智谱 API Key
- `OPENAI_API_BASE` — 智谱 API 地址（`https://open.bigmodel.cn/api/paas/v4`）
- `EMBEDDING_MODEL` — 向量模型名（默认 `embedding-2`）
- `WECHAT_APPID` / `WECHAT_SECRET` — 微信小程序登录
- `JWT_SECRET_KEY` — JWT 签名密钥
- `JWT_EXPIRE_HOURS` — JWT 过期时间（默认 72 小时）
- `TianxingFood_API_KEY` — 天行数据食物营养 API Key
- `CORS_ORIGINS` — CORS 允许的来源（逗号分隔）
- `DB_PATH` — SQLite 数据库路径（默认 `./fitness_coach.db`）
- `SSL_VERIFY` — Streamlit 前端是否验证 SSL（默认 `true`）
- `BACKEND_URL` — Streamlit 前端连接的后端地址

## 代码规范

- 注释、docstring、prompt、UI 文本用中文；变量名、函数名用英文
- 测试用 `unittest` + `unittest.mock`，mock LLM 用 `@patch('app.llm_manager.LLMManager.get_llm')`
- 测试文件顶部需要 `sys.path.insert(0, ...)` 确保导入正确
- 改动后跑验证：`python -m pytest backend/tests/ -v`

## 分支策略

- `main` — 日常开发分支
- `deploy` — 部署分支，服务器从 deploy 拉取
- 开发完成后合并 main → deploy
