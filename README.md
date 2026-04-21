# FitCoach AI - 智能私人营养师与健身教练

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

基于大语言模型（LLM）和多 Agent 架构的智能私人营养师与健身教练应用。系统通过动态路由将用户请求分发到最合适的专业 Agent，提供个性化的饮食规划、运动指导和健康管理服务。

## ✨ 核心特性

### 🤖 多 Agent 协作系统

| Agent | 职责 | 处理场景 |
|-------|------|----------|
| **闲聊助手** | 日常对话、情感支持 | 问候、闲聊、通用问题 |
| **营养师** | 饮食计划、热量计算、营养建议 | 食物查询、饮食记录、餐单推荐 |
| **健身教练** | 训练计划、动作指导、运动建议 | 运动计划、动作要领、健身问题 |
| **专家评审** | 评估输出质量，确保专业性 | 评审其他 Agent 的输出（1-5 分） |

### 🔄 动态路由机制

用户输入 → 意图识别 → 智能分发：

- 包含"饮食/热量/营养" → 营养师 Agent
- 包含"运动/训练/健身" → 健身教练 Agent
- 日常闲聊/问候 → 闲聊 Agent
- Expert Agent 评分 ≤ 2 → 打回重试（最多 3 次）

### 📚 现代化 RAG 系统

- **混合检索**：向量检索 + BM25 关键词检索（RRF 融合）
- **高级处理**：语义分割、表格提取、代码块保留
- **查询增强**：Query Expansion、HyDE（假设性文档嵌入）
- **生成增强**：CoT 思维链、Self-RAG 自我反思
- **自主决策**：Agentic RAG - 大模型自主选择检索/生成策略

### 👤 用户状态管理

- 生理数据：身高、体重、年龄、性别
- 自动计算：BMR（基础代谢率）、TDEE（每日总消耗）
- 热量追踪：摄入 - 消耗 = 热量缺口

## 🏗️ 系统架构

```
                         ┌─────────────┐
                         │   用户输入   │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   FastAPI   │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   Graph     │
                         │  (Router)   │
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
       ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
       │    Chat     │   │  Nutrition  │   │   Fitness   │
       │   Agent     │   │   Agent     │   │   Agent     │
       └─────────────┘   └──────┬──────┘   └──────┬──────┘
                                │                 │
                                └────────┬────────┘
                                         │
                                  ┌──────▼──────┐
                                  │   Expert    │
                                  │   Agent     │
                                  └──────┬──────┘
                                         │
                                  评分 ≥ 3 → 用户
                                  评分 ≤ 2 → 重试（≤3次）
```

## 📂 项目结构

```
fitness_coach/
├── backend/                         # 后端服务
│   └── app/
│       ├── main.py                  # FastAPI 主入口
│       ├── models.py                 # SQLAlchemy 数据模型
│       ├── database.py               # 数据库连接
│       ├── food_api.py              # 食物营养 API
│       ├── rag_utils.py              # RAG 向量检索
│       ├── agents/                   # 多 Agent 系统
│       │   ├── base.py              # Agent 基础配置
│       │   ├── router.py            # 动态路由器
│       │   ├── chat_agent.py        # 闲聊 Agent
│       │   ├── nutrition_agent.py   # 营养师 Agent
│       │   ├── fitness_agent.py     # 健身教练 Agent
│       │   ├── expert_agent.py      # 专家评审 Agent
│       │   └── graph.py             # 主控工作流
│       └── rag/                      # 现代化 RAG 系统
│           ├── __init__.py          # ModernRAG 主入口
│           └── modules/             # 核心模块
│               ├── loader.py        # 文档加载器
│               ├── splitter.py       # 智能文本分割
│               ├── preprocessor.py  # 文本预处理
│               ├── bm25.py          # BM25 检索
│               ├── hybrid_search.py # 混合检索
│               ├── query_expansion.py # 多查询扩展
│               ├── hyde.py          # 假设性文档嵌入
│               ├── cot.py           # 思维链推理
│               ├── self_rag.py      # 自我反思纠正
│               ├── agentic_rag.py   # Agentic RAG
│               └── doc_processor.py # 高级文档处理
├── frontend/
│   └── app.py                      # Streamlit 前端
├── knowledge_base/                  # RAG 知识库文档
├── chroma_db/                       # ChromaDB 向量库
├── .env                            # 环境变量
├── requirements.txt                 # Python 依赖
└── README.md                        # 本文档
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- API 密钥（智谱 AI）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/fitness_coach.git
cd fitness_coach
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_zhipu_api_key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4
EMBEDDING_MODEL=embedding-2
TIANAPI_KEY=your_tianapi_key
```

### 4. 启动服务

**终端 1 - 启动后端：**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**终端 2 - 启动前端：**

```bash
streamlit run frontend/app.py
```

### 5. 访问应用

- 前端：http://localhost:8501
- 后端 API：http://localhost:8000/docs

## 📖 使用指南

### 创建用户

首次使用时，系统会根据您的生理数据自动计算 BMR 和 TDEE。

### 营养师功能

- 查询食物热量：`"100g 鸡胸肉的热量是多少？"`
- 记录饮食：`"我中午吃了 200g 米饭和一份番茄炒蛋"`
- 获取建议：`"我想减肥，今天还能吃什么？"`

### 健身教练功能

- 动作指导：`"上斜卧推怎么做？"`
- 训练计划：`"帮我制定一个增肌计划"`
- 运动记录：`"我今天跑了 30 分钟"`

### 专家评审

营养师和健身教练的回答都会经过专家 Agent 评审：
- 评分 1-2 分：打回重试
- 评分 3-5 分：通过，返回给用户

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Streamlit | 交互式 Web UI |
| 后端 | FastAPI | 异步 API 框架 |
| AI 编排 | LangGraph | Agent 工作流 |
| 向量数据库 | ChromaDB | 本地知识存储 |
| 嵌入模型 | 智谱 embedding-2 | 文本向量化 |
| 大模型 | 智谱 GLM-4 | 对话能力 |
| 数据库 | SQLite + SQLAlchemy | 数据持久化 |

## 📈 项目进度

| 阶段 | 状态 |
|------|------|
| 基础架构 | ✅ 完成 |
| 多 Agent 核心 | ✅ 完成 |
| Agent 实现 | ✅ 完成 |
| 动态路由 | ✅ 完成 |
| 流式输出 | ✅ 完成 |
| ChatGPT 风格 UI | ✅ 完成 |
| RAG 检索 | ✅ 完成 |
| 食物营养 API | ✅ 完成 |
| 专家评审机制 | ✅ 完成 |
| 现代 RAG 升级 | ✅ 完成 |
| Agentic RAG | ✅ 完成 |
| 高级文档处理 | ✅ 完成 |

## 🔮 未来优化方向

- [ ] 语音输入功能
- [ ] 导出月度健康报告
- [ ] 用户认证系统
- [ ] 健身计划自动生成
- [ ] 多语言支持
- [ ] 知识库可视化编辑

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent 工作流编排
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Streamlit](https://github.com/streamlit/streamlit) - 前端框架
- [智谱 AI](https://open.bigmodel.cn/) - 大模型 API
