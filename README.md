# FitCoach AI - 智能私人营养师与健身教练 (持续迭代中)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)

> 基于大语言模型（LLM）和多 Agent 架构的智能私人营养师与健身教练应用

## 🎯 项目简介

FitCoach AI 通过动态路由将用户请求分发到最合适的专业 Agent，提供**个性化的饮食规划**、**运动指导**和**健康管理**服务。系统采用 LangGraph 框架实现状态机驱动的多 Agent 协作，并集成现代化的 RAG 系统获取专业知识。

***

## 🆕 近期更新

### v2.0 - Agent 专业化升级

- ✨ **提示词全面优化**：为每个 Agent 设计专业化的 System Prompt，明确职责边界
- 🔄 **Router 增强**：扩展路由判断标准，支持更精准的意图识别
- 📝 **Expert Agent 完善**：引入评分维度和权重（专业性30%/个性化25%/实用性25%/安全性20%）

### v1.5 - ModernRAG 系统

- ✨ **混合检索**：向量 + BM25 + RRF 融合
- ✨ **查询增强**：Query Expansion 多查询扩展
- ✨ **HyDE**：假设性文档嵌入
- ✨ **CoT + Self-RAG**：思维链推理与自我反思

### v1.0 - 核心架构

- ✅ 多 Agent 系统（Chat/Nutrition/Fitness/Expert）
- ✅ LangGraph 工作流
- ✅ 动态路由机制

***

## ✨ 核心特性

### 🤖 多 Agent 协作系统

| Agent    | 职责             | 处理场景                  |
| -------- | -------------- | --------------------- |
| **闲聊助手** | 日常对话、情感支持      | 问候、闲聊、通用问题            |
| **营养师**  | 饮食计划、热量计算、营养建议 | 食物查询、饮食记录、餐单推荐        |
| **健身教练** | 训练计划、动作指导、运动建议 | 运动计划、动作要领、健身问题        |
| **专家评审** | 评估输出质量，确保专业性   | 评审其他 Agent 的输出（1-5 分） |

### 🔄 动态路由机制

```
用户输入 → 意图识别 → 智能分发
   │
   ├─ 包含"饮食/热量/营养" ──→ 营养师 Agent ──→ 专家评审（≥3分通过）
   │
   ├─ 包含"运动/训练/健身" ──→ 健身教练 Agent ──→ 专家评审（≥3分通过）
   │
   └─ 日常闲聊/问候 ──→ 闲聊 Agent ──→ 直接回复
```

### 📚 Agentic RAG 系统

| 特性       | 说明                                 |
| -------- | ---------------------------------- |
| **混合检索** | 向量检索 + BM25 关键词检索（RRF 融合）          |
| **查询增强** | Query Expansion 多查询扩展、HyDE 假设性文档嵌入 |
| **生成增强** | CoT 思维链推理、Self-RAG 自我反思纠正          |
| **自主决策** | Agentic RAG - LLM 自主选择检索/生成策略      |
| **高级处理** | 语义分割、表格提取、代码块保留                    |

### 👤 用户状态管理

- 生理数据：身高、体重、年龄、性别
- 自动计算：BMR（基础代谢率）、TDEE（每日总消耗）
- 热量追踪：摄入 - 消耗 = 热量缺口

***

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         FitCoach AI 架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│   │  用户     │ ──▶ │ FastAPI  │ ──▶ │  Router  │                │
│   │  输入     │     │  Backend │     │  Agent   │                │
│   └──────────┘     └──────────┘     └────┬─────┘                │
│                                           │                     │
│              ┌────────────────────────────┼────────────────────┐│
│              │                            │                    ││
│              ▼                            ▼                     │
│       ┌──────────┐              ┌──────────────┐        ┌──────────┐
│       │   Chat   │              │   Nutrition  │        │  Fitness │
│       │  Agent   │              │    Agent     │        │  Agent   │
│       └────┬─────┘              └──────┬───────┘        └────┬─────┘
│            │                            │                    │     │
│            │                            ▼                    │     │
│            │                     ┌──────────────┐            │     │
│            │                     │    Expert    │◄───────────┘     │
│            │                     │    Agent     │                  │
│            │                     └──────┬───────┘                  │
│            │                            │                          │
│            ▼                     评分 ≥ 3 ──▶ 用户                  │
│          END                      │                                │
│                               评分 ≤ 2                              │
│                                 │ (最多重试3次)                      │
│                                 ▼                                  │
│                              重新生成                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

***

## 📂 项目结构

```
fitness_coach/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI 主入口
│       ├── models.py                 # SQLAlchemy 数据模型
│       ├── database.py               # 数据库连接
│       ├── food_api.py              # 食物营养 API
│       ├── agents/                   # 多 Agent 系统
│       │   ├── __init__.py          # 包入口
│       │   ├── base.py              # Agent 基础配置
│       │   ├── router.py            # 动态路由器
│       │   ├── chat_agent.py        # 闲聊 Agent
│       │   ├── nutrition_agent.py   # 营养师 Agent
│       │   ├── fitness_agent.py      # 健身教练 Agent
│       │   ├── expert_agent.py      # 专家评审 Agent
│       │   └── graph.py             # LangGraph 工作流
│       └── rag/                      # 现代化 RAG 系统
│           ├── __init__.py          # ModernRAG 主入口
│           └── modules/             # 核心模块
│               ├── loader.py        # 文档加载器
│               ├── splitter.py       # 智能文本分割
│               ├── preprocessor.py  # 文本预处理
│               ├── bm25.py          # BM25 检索
│               ├── hybrid_search.py # 混合检索
│               ├── query_expansion.py  # 多查询扩展
│               ├── hyde.py          # 假设性文档嵌入
│               ├── cot.py           # 思维链推理
│               ├── self_rag.py      # 自我反思纠正
│               └── agentic_rag.py   # Agentic RAG
├── frontend/
│   └── app.py                      # Streamlit 前端
├── knowledge_base/                  # RAG 知识库
├── chroma_db/                       # ChromaDB 向量库
├── .env                            # 环境变量
├── requirements.txt                 # 依赖
└── README.md
```

***

## 🚀 快速开始

### 环境要求

- Python 3.12+
- 智谱 AI API Key
- 天行数据 API Key（食物营养查询）

### 1. 克隆并安装

```bash
git clone https://github.com/Ya97779/fitness_coach_agent.git
cd fitness_coach_agent
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env`：

```env
# LLM 配置
LLM_MODEL=glm-4.7
OPENAI_API_KEY=your_zhipu_api_key
#对应的API Base URL,
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
# 食物营养API（天行数据）（注册后每日免费100次查询）
TianxingFood_API_KEY=your_food_api_key_here
# Embedding模型
EMBEDDING_MODEL=embedding-2

```

### 3. 启动服务

```bash
# 终端 1 - 后端
uvicorn backend.app.main:app --reload --port 8000

# 终端 2 - 前端
streamlit run frontend/app.py
```

### 4. 访问

- 前端：<http://localhost:8501>
- API 文档：<http://localhost:8000/docs>

***

## 📖 使用示例

### 营养师 Agent

```
用户: 100g 鸡胸肉的热量是多少？
AI:  【API查询】→ 133 kcal, 蛋白质 31g, 脂肪 3.6g
     → 优化回答，提供营养建议
```

### 健身教练 Agent

```
用户: 上斜卧推怎么做？
AI:  【RAG检索】→ 动作详解、发力技巧、常见错误
     → 结合专业知识生成回答
```

### 专家评审

| 评分 | 含义 | 动作       |
| -- | -- | -------- |
| 5  | 卓越 | 直接通过     |
| 4  | 优秀 | 直接通过     |
| 3  | 合格 | 直接通过     |
| 2  | 不足 | 重试（最多3次） |
| 1  | 很差 | 重试（最多3次） |

***

## 🛠️ 技术栈

| 层级           | 技术                   | 说明           |
| ---------     |-------------------     | ----------    |
| **前端**       | Streamlit             | 交互式 Web UI   |
| **后端**       | FastAPI               | 异步 API 框架   |
| **AI 编排**    | LangGraph             | Agent 工作流    |
| **向量数据库**  | ChromaDB              | 本地知识存储     |
| **数据库**     | SQLite + SQLAlchemy    |  数据持久化     |

***

## 📈 项目进度

| 模块            | 状态 | 说明                            |
| ------------- | -- | ----------------------------- |
| 基础架构          | ✅  | FastAPI + Streamlit           |
| 多 Agent 系统    | ✅  | Chat/Nutrition/Fitness/Expert |
| LangGraph 工作流 | ✅  | 状态机驱动                         |
| 动态路由          | ✅  | LLM 意图识别                      |
| 专家评审          | ✅  | 1-5分评分 + 重试机制                 |
| 食物营养 API      | ✅  | 天行数据 API                      |
| RAG 基础        | ✅  | ChromaDB 向量检索                 |
| 现代 RAG        | ✅  | 混合检索/查询扩展/HyDE                |
| Agentic RAG   | ✅  | LLM 自主决策                      |
| 高级文档处理        | ✅  | 语义分割/表格提取                     |
| 优化提示词         | ✅  | 各 Agent 专业化                   |

***

## 🔮 未来规划

- [ ] 记忆模块：用户画像加载、对话历史摘要
- [ ] 上下文工程优化：分级上下文注入策略
- [ ] 前端重构：TypeScript + React
- [ ] 语音输入功能
- [ ] 用户认证系统

***

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent 工作流编排
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Streamlit](https://github.com/streamlit/streamlit) - 前端框架
- [智谱 AI](https://open.bigmodel.cn/) - 大模型 API

