# FitCoach AI - 会话上下文完整摘要

## 📅 文档创建日期：2026-04-27

---

## 一、项目概述

**FitCoach AI 智能私教系统** 是一个基于多 Agent 架构的大模型应用，用于健身和营养指导。

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + SQLite + SQLAlchemy |
| Agent 编排 | LangGraph（微软开源的多 Agent 框架） |
| RAG 系统 | ChromaDB 向量数据库 + 混合检索 |
| 前端 | Streamlit（ChatGPT 风格 UI） |
| LLM | 智谱 GLM-4（通过 `openai` 接口调用） |
| Embedding | 智谱 embedding-2 模型 |

### 项目定位
- 面向 fitness 领域的 AI 助手
- 支持饮食计划、热量计算、健身指导
- 可接入外部知识库（RAG）

---

## 二、项目演进历程

### Phase 1: 基础架构
- FastAPI 后端 + SQLite 数据库
- Streamlit 前端（仿 ChatGPT 风格）
- 食物热量 API 接入（天行数据）

### Phase 2: 多 Agent 系统
- 闲聊 Agent（Chat Agent）
- 营养师 Agent（Nutrition Agent）
- 健身教练 Agent（Fitness Agent）
- 专家评审 Agent（Expert Agent）
- 动态路由器（Router Agent）
- LangGraph 工作流编排

### Phase 3: RAG 系统
- ChromaDB 向量数据库
- 混合检索（向量 + BM25 + RRF）
- 高级特性：Query Expansion、HyDE、CoT、Self-RAG
- Agentic RAG（LLM 自主决策检索策略）

### Phase 4: 记忆模块
- 用户画像加载（UserProfileLoader）
- 对话历史摘要（ConversationSummarizer）
- 统计数据汇总（StatsSummarizer）
- 跨会话对话持久化（ConversationLog）

---

## 三、完整项目结构

```
d:\fitness_coach/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI 主入口
│       ├── models.py                # 数据库模型（含 ConversationLog）
│       ├── database.py              # 数据库连接
│       ├── food_api.py              # 食物营养 API
│       ├── agents/                  # 多 Agent 系统
│       │   ├── __init__.py
│       │   ├── base.py              # System Prompt 配置
│       │   ├── router.py            # 意图识别路由
│       │   ├── chat_agent.py        # 闲聊 Agent
│       │   ├── nutrition_agent.py   # 营养师 Agent
│       │   ├── fitness_agent.py     # 健身教练 Agent
│       │   ├── expert_agent.py      # 专家评审 Agent
│       │   └── graph.py             # LangGraph 工作流
│       ├── rag/                     # RAG 系统
│       │   ├── __init__.py          # ModernRAG 主入口
│       │   └── modules/             # RAG 核心模块
│       │       ├── loader.py        # 文档加载
│       │       ├── splitter.py      # 文本分割
│       │       ├── preprocessor.py  # 预处理
│       │       ├── bm25.py          # BM25 检索
│       │       ├── hybrid_search.py # 混合检索
│       │       ├── query_expansion.py
│       │       ├── hyde.py          # 假设性文档嵌入
│       │       ├── cot.py           # 思维链
│       │       ├── self_rag.py      # 自我反思
│       │       ├── agentic_rag.py   # Agentic RAG
│       │       └── doc_processor.py # 高级文档处理
│       └── memory/                   # 记忆模块
│           ├── __init__.py
│           ├── memory_manager.py    # 记忆管理器
│           ├── user_profile.py      # 用户画像
│           ├── conversation_summary.py  # 对话摘要
│           ├── stats_summary.py     # 统计汇总
│           ├── ANALYSIS.md          # 记忆模块分析报告
│           └── todolist.md
├── frontend/
│   └── app.py                      # Streamlit 前端
├── chroma_db/                      # ChromaDB 向量库
├── knowledge_base/                  # RAG 知识库
├── session1.md                      # 本会话上下文摘要
├── Development.md                   # 开发进度文档
├── README.md                        # 项目说明
├── requirements.txt
└── .env                            # 环境变量
```

---

## 四、多 Agent 系统详解

### 4.1 Agent 职责

| Agent | 职责 | 工具 |
|-------|------|------|
| **Router** | 分析用户意图，分发到对应 Agent | 无 |
| **Chat** | 闲聊、情感支持 | 无 |
| **Nutrition** | 饮食计划、热量计算、营养建议 | `search_food_nutrition`, `log_food_intake`, `get_daily_nutrition_summary` |
| **Fitness** | 训练计划、动作指导、RAG 检索 | `search_fitness_knowledge`, `estimate_exercise_calories`, `log_exercise` |
| **Expert** | 评审输出质量（1-5分），不达标则重试 | 无 |

### 4.2 Router 路由规则

```python
# LLM 分析用户输入，返回数字
1 = 闲聊助手（日常问候、情感交流、通用知识）
2 = 营养师（食物热量、饮食计划、食谱推荐）
3 = 健身教练（运动训练、动作指导、训练计划）
```

### 4.3 Expert 评审机制

| 评分 | 等级 | 动作 |
|------|------|------|
| 5 | 卓越 | 直接通过 |
| 4 | 优秀 | 直接通过 |
| 3 | 合格 | 直接通过 |
| 2 | 不足 | 重试（最多3次） |
| 1 | 很差 | 重试（最多3次） |

---

## 五、RAG 系统架构

### 5.1 检索流程

```
用户问题 → RouterAgent 分析问题类型
                │
        ┌───────┼───────┬────────┐
        ▼       ▼       ▼        ▼
    no_retrieval  basic  hyde   cot/self_rag
        │        │       │        │
        └────────┴───────┴────────┘
                        │
                    质量评估
                        │
                   最终回答
```

### 5.2 检索模式

| 模式 | 说明 |
|------|------|
| `hybrid` | 混合检索（向量+BM25） |
| `vector` | 仅向量检索 |
| `bm25` | 仅 BM25 关键词 |
| `query_expansion` | 多查询扩展 |
| `hyde` | 假设性文档嵌入 |

### 5.3 高级特性

- **Query Expansion**: LLM 生成 3-5 个查询变体
- **HyDE**: 生成"假设答案"再检索
- **CoT**: 思维链推理
- **Self-RAG**: 自我反思与纠正
- **Agentic RAG**: LLM 自主决策策略

---

## 六、记忆模块架构

### 6.1 记忆类型

| 类型 | 存储 | 说明 |
|------|------|------|
| **长期记忆** | SQLite | 用户画像、统计、对话历史持久化 |
| **短期记忆** | 内存 | 当前对话、10条后自动摘要 |
| **Agent上下文** | 动态注入 | 记忆格式化后注入 System Prompt |

### 6.2 核心组件

| 组件 | 类名 | 功能 |
|------|------|------|
| 用户画像 | `UserProfileLoader` | 从数据库加载用户信息 |
| 对话摘要 | `ConversationSummarizer` | 超过10条消息自动摘要 |
| 统计汇总 | `StatsSummarizer` | 当日/本周统计 |
| 持久化 | `ConversationLog` 模型 | 跨会话对话存储 |
| 记忆管理 | `MemoryManager` | 统一管理所有记忆 |

### 6.3 ConversationLog 表结构

```python
class ConversationLog(Base):
    user_id: int           # 用户 ID
    session_id: str        # 会话 ID
    agent_type: str        # Agent 类型
    user_message: str      # 用户消息
    agent_response: str    # Agent 回复
    created_at: datetime   # 创建时间
```

---

## 七、本会话完成的工作

### 7.1 记忆模块分析
- 分析长期记忆/短期记忆实现方式
- 识别待解决问题（见 7.3）
- 生成 `memory/ANALYSIS.md` 分析报告

### 7.2 跨会话对话持久化
**问题**：对话历史仅存于内存，关闭对话后丢失

**解决**：
1. 新增 `ConversationLog` 数据库模型
2. `MemoryManager` 新增 `save_conversation()` 和 `load_conversation_history()` 方法
3. `graph.py` 在处理消息后自动保存对话
4. 各 Agent 注入历史对话上下文

### 7.3 待解决问题

| 问题 | 优先级 |
|------|--------|
| 记忆上下文信息量不足 | P1 |
| `enhance_system_prompt()` 未被使用 | P1 |
| 对话要点提取未完成 | P2 |
| 用户偏好未持久化 | P2 |

---

## 八、数据库模型完整清单

| 模型 | 表名 | 说明 |
|------|------|------|
| User | users | 用户信息（身高、体重、目标等） |
| DailyLog | daily_logs | 每日摄入/消耗记录 |
| FoodItem | food_items | 食物记录 |
| ExerciseItem | exercise_items | 运动记录 |
| ConversationLog | conversation_logs | 对话历史持久化 🆕 |

---

## 九、快速启动

```bash
# 后端
cd backend
uvicorn app.main:app --reload

# 前端（新终端）
streamlit run frontend/app.py
```

---

## 十、API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/user/` | 创建用户 |
| GET | `/user/{user_id}` | 获取用户信息 |
| GET | `/user/{user_id}/logs` | 获取用户所有日志 |
| GET | `/user/{user_id}/today` | 获取当日日志 |
| POST | `/chat` | 非流式聊天 |
| POST | `/chat/stream` | 流式聊天 |
| GET | `/agents` | 获取所有 Agent |

---

## 十一、重要文件路径

| 文件 | 作用 |
|------|------|
| `backend/app/models.py` | 所有数据库模型 |
| `backend/app/memory/memory_manager.py` | 记忆管理器核心 |
| `backend/app/agents/graph.py` | LangGraph 工作流入口 |
| `backend/app/agents/router.py` | 意图识别 |
| `backend/app/rag/__init__.py` | ModernRAG 主入口 |
| `backend/app/rag/modules/agentic_rag.py` | Agentic RAG 实现 |
| `Development.md` | 开发进度文档 |
| `memory/ANALYSIS.md` | 记忆模块分析报告 |

---

*本文件用于为新对话提供完整上下文，确保 AI 能够理解项目的全部历史和当前状态。*
