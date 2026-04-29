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

### v3.0 - 智能化与性能优化

- ✨ **记忆模块**：用户画像加载、对话历史摘要、统计汇总，支持跨会话持久化
- ✨ **混合路由**：关键词预筛选 + LLM 二次确认，大幅提升路由可靠性
- ✨ **营养师 RAG**：Nutrition Agent 新增 RAG 知识库检索，支持营养专业问答
- ✨ **流式响应**：SSE 流式输出，基于 LLM stream 方法逐 chunk 推送
- ✨ **增量索引**：后端启动时自动检测新文档并增量添加到知识库
- ✨ **专家评审快速通道**：简单查询跳过评审，减少不必要 LLM 调用
- ⚡ **性能优化**：LLM 单例复用、数据库查询缓存、RAG 检索 LRU 缓存
- 🧪 **单元测试**：Memory / Agents / RAG 模块共 150+ 测试用例

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
| **营养师**  | 饮食计划、热量计算、营养知识 | 食物查询、饮食记录、营养原理、膳食策略 |
| **健身教练** | 训练计划、动作指导、健身知识 | 运动计划、动作要领、训练原理       |
| **专家评审** | 评估输出质量，确保专业性   | 评审其他 Agent 的输出（1-5 分） |

### 🔄 混合路由机制

```
用户输入 → 关键词预筛选（快速路径）
   │
   ├─ 包含"吃/食物/热量/饮食" ──→ 营养师 Agent ──→ 专家评审（≥3分通过）
   │
   ├─ 包含"运动/训练/健身" ──→ 健身教练 Agent ──→ 专家评审（≥3分通过）
   │
   └─ 模糊场景 ──→ LLM 二次确认 ──→ 分发到对应 Agent
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
│       ├── models.py                # SQLAlchemy 数据模型
│       ├── database.py              # 数据库连接
│       ├── llm_manager.py           # LLM 实例管理器（单例复用）
│       ├── food_api.py              # 食物营养 API（天行数据）
│       ├── agents/                  # 多 Agent 系统
│       │   ├── __init__.py          # 包入口
│       │   ├── base.py              # Agent 基础配置
│       │   ├── router.py            # 混合路由器（关键词+LLM）
│       │   ├── chat_agent.py        # 闲聊 Agent
│       │   ├── nutrition_agent.py   # 营养师 Agent（API+RAG）
│       │   ├── fitness_agent.py     # 健身教练 Agent（RAG）
│       │   ├── expert_agent.py      # 专家评审 Agent
│       │   └── graph.py             # LangGraph 工作流
│       ├── rag/                     # 现代化 RAG 系统
│       │   ├── __init__.py          # ModernRAG 主入口
│       │   └── modules/             # 核心模块
│       │       ├── loader.py        # 文档加载器
│       │       ├── splitter.py      # 智能文本分割
│       │       ├── preprocessor.py  # 文本预处理
│       │       ├── bm25.py          # BM25 检索
│       │       ├── hybrid_search.py # 混合检索
│       │       ├── query_expansion.py # 多查询扩展
│       │       ├── hyde.py          # 假设性文档嵌入
│       │       ├── cot.py           # 思维链推理
│       │       ├── self_rag.py      # 自我反思纠正
│       │       ├── agentic_rag.py   # Agentic RAG
│       │       └── doc_processor.py # 高级文档处理
│       └── memory/                  # 记忆模块
│           ├── __init__.py          # 模块导出
│           ├── memory_manager.py    # 记忆管理器（统一接口）
│           ├── user_profile.py      # 用户画像加载
│           ├── conversation_summary.py # 对话历史摘要
│           └── stats_summary.py     # 统计数据汇总
├── backend/tests/                   # 单元测试
│   ├── test_agents.py               # Agent 模块测试（61）
│   ├── test_memory.py               # Memory 模块测试（32）
│   └── test_rag.py                  # RAG 模块测试（57）
├── frontend/
│   └── app.py                       # Streamlit 前端
├── knowledge_base/                   # RAG 知识库（PDF/DOCX）
├── chroma_db/                       # ChromaDB 向量库
├── .env                            # 环境变量（需手动创建）
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
AI:  【API检索】→ 133 kcal, 蛋白质 31g, 脂肪 3.6g
     → 优化回答，提供营养建议

用户: 增肌期应该怎么安排饮食？
AI:  【RAG检索】→ 《健身营养全书》中的增肌饮食策略
     → 结合知识库和专业知识生成个性化建议
```

### 健身教练 Agent

```
用户: 上斜卧推怎么做？
AI:  【RAG检索】→ 动作详解、发力技巧、常见错误
     → 结合知识库和专业知识生成回答
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
| **LLM**       | 智谱 GLM-4           | 大语言模型      |
| **向量数据库**  | ChromaDB              | 本地知识存储     |
| **数据库**     | SQLite + SQLAlchemy    | 数据持久化      |
| **测试**       | unittest              | 150+ 单元测试   |

***

## 📈 项目进度

| 模块            | 状态 | 说明                            |
| ------------- | -- | ----------------------------- |
| 基础架构          | ✅  | FastAPI + SQLite + Streamlit  |
| 多 Agent 系统    | ✅  | Chat/Nutrition/Fitness/Expert |
| LangGraph 工作流 | ✅  | 状态机驱动                         |
| 混合路由          | ✅  | 关键词预筛选 + LLM 二次确认           |
| 专家评审          | ✅  | 1-5分评分 + 重试 + 快速通道           |
| 食物营养 API      | ✅  | 天行数据 API                      |
| RAG 基础        | ✅  | ChromaDB 向量检索                 |
| 现代 RAG        | ✅  | 混合检索/查询扩展/HyDE/Self-RAG     |
| Agentic RAG   | ✅  | LLM 自主决策                      |
| 高级文档处理        | ✅  | 语义分割/表格提取                     |
| 记忆模块          | ✅  | 用户画像/对话摘要/统计汇总/跨会话持久化  |
| 上下文工程         | ✅  | enhance_system_prompt 完整上下文注入 |
| 增量文档索引        | ✅  | 启动时自动检测并增量添加               |
| 流式响应          | ✅  | 真正的 SSE 流式输出                  |
| 性能优化          | ✅  | LLM 单例/查询缓存/RAG 缓存          |
| 营养师 RAG 检索   | ✅  | Nutrition Agent 支持知识库检索       |
| 单元测试          | ✅  | 150+ 测试用例（Memory/Agents/RAG） |

***

## 🔮 未来规划

- [ ] 对话要点提取与持久化存储
- [ ] 用户偏好持久化
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

