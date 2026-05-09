# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FitCoach AI — 基于多 Agent 的健身营养顾问系统。后端 FastAPI + LangGraph 多 Agent 编排，前端 Streamlit，RAG 检索增强生成，SQLite 持久化。

## 常用命令

```bash
# 启动后端
uvicorn backend.app.main:app --reload --port 8000

# 启动前端
streamlit run frontend/app.py

# 运行全部测试（项目根目录执行）
python -m pytest backend/tests/ -v --tb=short

# 运行单个测试文件
python -m pytest backend/tests/test_agents.py -v
python -m unittest tests.test_rag -v

# RAG 效果评估（会调用 API，消耗 token）
python -m tests.test_rag_evaluation --quick             # 5 条快速评估
python -m tests.test_rag_evaluation --output report.json # 导出报告

# 安装依赖
pip install -r requirements.txt
```

## 架构

### 请求流转

```
Streamlit 前端
  → POST /api/v1/chat/stream (SSE)
    → FastAPI (JWT 鉴权)
      → MemoryManager (加载用户画像、对话历史、统计数据)
      → LangGraph StateGraph
        → router.py: 关键词预筛 + LLM 兜底分类
          → chat_agent / nutrition_agent / fitness_agent
            → nutrition: food_api (天行API) + RAG
            → fitness: RAG
          → expert_agent 评审 (评分 < 3 重试，最多 3 次)
            → 简单事实查询跳过评审 (skip_review)
        → MemoryManager.save_conversation() 持久化
```

### 核心模块

| 模块 | 路径 | 职责 |
|------|------|------|
| API 入口 | `backend/app/main.py` | FastAPI 路由、启动时增量索引、SSE 流式 |
| Agent 编排 | `backend/app/agents/graph.py` | LangGraph StateGraph，节点：router→chat/nutrition/fitness→expert_review |
| 路由 | `backend/app/agents/router.py` | 关键词匹配（40+ 中文关键词）+ LLM 分类 |
| RAG | `backend/app/rag/__init__.py` | ModernRAG 主类，全局单例 `get_rag_instance()` |
| RAG 模块 | `backend/app/rag/modules/` | BM25(jieba)、混合检索(RRF)、HyDE、Self-RAG、Agentic RAG |
| 记忆系统 | `backend/app/memory/memory_manager.py` | 用户画像、对话摘要、统计摘要，注入 Agent prompt |
| LLM 管理 | `backend/app/llm_manager.py` | ChatOpenAI 实例按 temperature 缓存（单例） |
| 数据模型 | `backend/app/models.py` | SQLAlchemy: User, DailyLog, FoodItem, ExerciseItem, ConversationLog |
| 前端 | `frontend/app.py` | Streamlit 单文件，聊天/档案/统计三模式 |

### 关键设计模式

- **LLM 调用**：统一通过 `LLMManager.get_llm(temperature)` 获取实例，不要自行创建 `ChatOpenAI`
- **Agent 工具**：nutrition/fitness agent 用 `@tool` 装饰器定义工具，LLM 自主决定调用
- **流式响应**：`stream_user_message()` 做 SSE 流式，会跳过 expert review 以保证实时性
- **RAG 单例**：通过 `get_rag_instance()` 获取全局 ModernRAG 实例
- **增量索引**：启动时 `check_and_update_index()` 扫描 knowledge_base，MD5 对比增量更新 ChromaDB

## 技术栈

- **LLM**：智谱 GLM-4.7，通过 OpenAI 兼容接口调用 (`open.bigmodel.cn`)
- **Embedding**：智谱 embedding-2
- **向量库**：ChromaDB（本地持久化 `./chroma_db`）
- **数据库**：SQLite + SQLAlchemy（`./fitness_coach.db`）
- **鉴权**：JWT + 微信小程序登录
- **外部 API**：天行数据食物营养 API（有本地兜底数据）

## 环境变量

参考 `.env.example`，关键变量：
- `LLM_MODEL` / `OPENAI_API_KEY` / `OPENAI_API_BASE` — LLM 配置
- `EMBEDDING_MODEL` — 向量模型
- `WECHAT_APPID` / `WECHAT_SECRET` — 微信登录
- `JWT_SECRET_KEY` — JWT 签名
- `TianxingFood_API_KEY` — 食物营养 API

## 代码规范

- 注释、docstring、prompt、UI 文本用中文；变量名、函数名用英文
- 测试用 `unittest` + `unittest.mock`，mock LLM 用 `@patch('app.llm_manager.LLMManager.get_llm')`
- 测试文件顶部需要 `sys.path.insert(0, ...)` 确保导入正确
- 改动后跑验证：`python -m pytest backend/tests/ -v`

## 分支策略

- `main` — 日常开发分支
- `deploy` — 部署分支，服务器从 deploy 拉取
- 开发完成后合并 main → deploy
