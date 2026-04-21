# 项目开发状态：FitCoach AI 智能私教系统

## 📅 最后更新日期：2026-04-21

---

## 📁 项目目录结构 (Project Structure)

```
d:\fitness_coach/
├── backend/                         # 后端服务
│   └── app/
│       ├── main.py                  # FastAPI 主入口，API 路由定义
│       ├── models.py                 # SQLAlchemy 数据库模型
│       ├── database.py               # 数据库连接配置
│       ├── food_api.py              # 天行数据食物营养 API（带本地 fallback）
│       ├── rag_utils.py              # RAG 向量检索工具（ChromaDB）
│       └── agents/                   # 多 Agent 系统核心
│           ├── __init__.py          # 模块导出
│           ├── base.py              # Agent 基础配置（系统提示词）
│           ├── router.py            # 动态路由器（意图识别）
│           ├── chat_agent.py        # 闲聊 Agent
│           ├── nutrition_agent.py   # 营养师 Agent（RAG + API fallback）
│           ├── fitness_agent.py     # 健身教练 Agent（RAG + LLM fallback）
│           ├── expert_agent.py      # 专家评审 Agent
│           └── graph.py             # 主控工作流（整合所有 Agent）
├── frontend/
│   └── app.py                      # Streamlit 前端界面
├── chroma_db/                       # ChromaDB 向量数据库持久化
├── knowledge_base/                  # RAG 知识库文档（PDF/Word/图片）
├── ragTodolist.md                   # RAG 系统升级计划
├── .env                            # 环境变量（API密钥等）
├── requirements.txt                 # Python 依赖列表
├── Development.md                  # 本文档
├── design.md                       # 产品设计文档
└── study.md                        # 开发学习文档
```

---

## 🏗️ 系统架构 (System Architecture)

### 整体架构图

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
                         │   Graph     │ ◄── 主控工作流
                         │  (Router)   │
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
       ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
       │    Chat     │   │  Nutrition  │   │   Fitness   │
       │   Agent     │   │   Agent     │   │   Agent     │
       └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
              │                 │                 │
              │          ┌──────┴──────┐          │
              │          │  Food API   │          │
              │          │  (天行数据)  │          │
              │          └─────────────┘          │
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Expert    │
                         │   Agent     │
                         │  (评分 1-5) │
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
             ┌──────▼──────┐          ┌──────▼──────┐
             │ 评分 >= 3   │          │ 评分 <= 2   │
             │   通过 ✓    │          │  打回重试   │
             └──────┬──────┘          │  (最多3次)  │
                    │                 └──────┬──────┘
                    │                        │
                    │                 ┌──────▼──────┐
                    │                 │  重新生成   │
                    │                 └──────┬──────┘
                    │                        │
                    └────────────────────────┘
                                │
                         ┌──────▼──────┐
                         │   最终回复  │
                         └─────────────┘
```

### 流程说明

1. **路由决策**：Graph 根据用户输入路由到 Chat/Nutrition/Fitness Agent
2. **Agent 执行**：Chat 直接返回；Nutrition/Fitness 执行并调用工具
3. **Expert 评审**：Nutrition/Fitness 的输出经过 Expert Agent 评分（1-5分）
4. **评分判断**：
   - 评分 >= 3 → 通过，直接返回给用户
   - 评分 <= 2 → 打回重试（最多 MAX_RETRIES=3 次）
5. **防死循环**：超过最大重试次数后返回最后一次结果

### 核心模块说明

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **主入口** | `main.py` | FastAPI 应用定义，API 路由，BMR/TDEE 计算 |
| **数据模型** | `models.py` | User, DailyLog, FoodItem, ExerciseItem 四个表 |
| **数据库** | `database.py` | SQLAlchemy engine 和 SessionLocal 配置 |
| **食物API** | `food_api.py` | 天行数据食物营养查询，带本地 fallback |
| **RAG工具** | `rag_utils.py` | ChromaDB 向量检索，支持 PDF/DOCX/图片 |
| **Agent基座** | `base.py` | 系统提示词定义，MultiAgentState 状态类型 |
| **路由器** | `router.py` | 意图识别（闲聊/营养/健身），返回数字 1/2/3 |
| **主工作流** | `graph.py` | 整合路由和 Agent 调用，process_user_message 入口 |
| **闲聊Agent** | `chat_agent.py` | 日常对话，无工具调用 |
| **营养师Agent** | `nutrition_agent.py` | 饮食记录，热量查询（API + LLM fallback） |
| **健身教练Agent** | `fitness_agent.py` | 运动记录，动作指导（RAG + LLM fallback） |
| **专家Agent** | `expert_agent.py` | 评审其他 Agent 输出质量 |

---

## 🤖 多 Agent 协作机制

### Agent 职责与工具

| Agent | 职责 | 可用工具 |
|-------|------|----------|
| **Router** | 分析用户输入，路由到合适 Agent | 无 |
| **Chat** | 日常闲聊、情感支持 | 无 |
| **Nutrition** | 饮食计划、热量计算、营养查询 | `search_food_nutrition`, `log_food_intake`, `get_daily_nutrition_summary`, `get_user_nutrition_info` |
| **Fitness** | 训练计划、动作指导、运动消耗 | `search_fitness_knowledge`, `estimate_exercise_calories`, `log_exercise`, `get_user_fitness_info` |
| **Expert** | 评审 Nutrition/Fitness 输出质量 | `get_user_info` |

### 路由规则

```
用户输入包含以下关键词时：
- "吃"、"食物"、"热量"、"饮食"、"营养" → Nutrition Agent
- "运动"、"跑步"、"训练"、"健身"、"动作" → Fitness Agent
- 其他（问候、天气、日常闲聊） → Chat Agent
```

### RAG + LLM Fallback 机制

**健身教练 Agent (`fitness_agent.py`)**
```
1. 用户询问健身动作（如"上斜卧推怎么做"）
2. 调用 search_fitness_knowledge 工具
3. 工具从 RAG 检索：
   - 检索到 → 返回【RAG检索】+ 检索内容
   - 未检索到 → 返回"【RAG检索】未在知识库中找到相关信息"
4. LLM 基于检索结果生成优化回答：
   - 有检索结果 → 基于检索内容整理后回答
   - 无检索结果 → LLM 基于自身健身知识回答
5. 重要原则：工具只负责检索，回答由 LLM 生成
```

**营养师 Agent (`nutrition_agent.py`)**
```
1. 用户询问食物营养（如"100g鸡胸肉热量"）
2. 调用 search_food_nutrition 工具
3. 工具从天行数据 API 查询：
   - 查询到 → 返回【API检索】+ 营养数据
   - 未查询到 → 返回"【API检索】未找到..."
4. LLM 基于检索结果生成优化回答：
   - 有检索结果 → 基于数据整理后回答
   - 无检索结果 → LLM 基于自身营养知识回答
```

### Expert Agent 评审机制

**评分标准（1-5分）**
| 分数 | 等级 | 说明 |
|------|------|------|
| 1 | 严重不足 | 内容严重不足、错误或不相关 |
| 2 | 需要改进 | 内容有较大问题，需要显著改进 |
| 3 | 基本合格 | 内容基本合格，有小问题 |
| 4 | 良好 | 内容良好，专业且实用 |
| 5 | 优秀 | 内容优秀，完美符合要求 |

**评审流程**
```
1. Agent 生成回答
2. Expert Agent 评审并打分
3. 评分判断：
   - 评分 >= 3 → 通过，直接返回给用户
   - 评分 <= 2 → 打回重试，Agent 重新生成
4. 重试循环：最多 MAX_RETRIES = 3 次
5. 超过最大重试次数 → 返回最后一次结果（即使不通过）
```

**防死循环机制**
- `MAX_RETRIES = 3`：限制最多重试次数
- 每次重试的评审历史 (`review_history`) 都会被记录
- 超过最大次数后返回结果和完整的评审历史，供后续分析

**Expert Agent 返回结构**
```python
{
    "score": 4,                          # 评分 1-5
    "approved": True,                     # 是否通过（score >= 3）
    "feedback": "内容专业且实用...",      # 评审意见
    "retries": 1,                        # 重试次数
    "review_history": [                  # 评审历史
        {"attempt": 1, "score": 2, "feedback": "内容不够详细..."},
        {"attempt": 2, "score": 4, "feedback": "内容专业且实用..."}
    ]
}
```

---

## 🔌 API 接口 (API Endpoints)

| 方法 | 路径 | 功能 | 请求体/参数 |
|------|------|------|-------------|
| POST | `/user/` | 创建用户（自动计算 BMR/TDEE） | `UserCreate` |
| GET | `/user/{user_id}` | 获取用户信息 | - |
| GET | `/user/{user_id}/logs` | 获取用户所有日志 | - |
| GET | `/user/{user_id}/today` | 获取当日日志 | - |
| POST | `/chat` | 非流式聊天 | `ChatRequest` |
| POST | `/chat/stream` | 流式聊天 | `StreamChatRequest` |
| GET | `/agents` | 获取所有可用 Agent | - |

### 请求/响应模型

```python
# UserCreate
{
    "height": 175,      # 身高(cm)
    "weight": 70,      # 体重(kg)
    "age": 25,         # 年龄
    "gender": "男",    # 性别
    "target_weight": 65,  # 目标体重(可选)
    "allergies": ""    # 过敏史(可选)
}

# ChatRequest
{
    "user_id": 1,      # 用户ID(可选)
    "message": "我想减肥应该怎么吃"
}

# ChatResponse
{
    "response": "根据您的TDEE...",
    "agent": "nutrition",
    "nutrition_response": "...",
    "fitness_response": "",
    "expert_review": {}
}
```

---

## 💾 数据库模型 (Database Schema)

### Users 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | Integer | 主键 |
| `height` | Float | 身高(cm) |
| `weight` | Float | 体重(kg) |
| `age` | Integer | 年龄 |
| `gender` | String | 性别 |
| `target_weight` | Float | 目标体重 |
| `allergies` | String | 过敏史 |
| `bmr` | Float | 基础代谢率 |
| `tdee` | Float | 每日总消耗 |
| `created_at` | DateTime | 创建时间 |

### DailyLogs 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | Integer | 主键 |
| `user_id` | Integer | 外键 |
| `date` | Date | 日期 |
| `intake_calories` | Float | 摄入热量 |
| `burn_calories` | Float | 消耗热量 |
| `weight_log` | Float | 体重记录 |
| `notes` | String | 备注 |

### FoodItems 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | Integer | 主键 |
| `log_id` | Integer | 外键 |
| `name` | String | 食物名称 |
| `calories` | Float | 热量 |

### ExerciseItems 表
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | Integer | 主键 |
| `log_id` | Integer | 外键 |
| `type` | String | 运动类型 |
| `duration` | Integer | 时长(分钟) |
| `calories` | Float | 消耗热量 |

---

## 🛠️ 工具详解 (Tools)

### Nutrition Agent 工具

**search_food_nutrition(query: str)**
```python
"""
从天行数据API查询食物营养信息
- 查询成功：返回【API检索】+ 热量/蛋白质/脂肪/碳水
- 查询失败：返回"【API检索】未找到..."
- 不直接返回数据，而是标记前缀供LLM识别处理
"""
```

**log_food_intake(user_id, food_name, calories, protein, fat, carbs)**
```python
"""
记录用户食物摄入到数据库
返回："已记录: {food_name}, {calories} kcal"
"""
```

**get_daily_nutrition_summary(user_id)**
```python
"""
获取当日营养摄入总结
返回：{
    "intake_calories": 1500,
    "burn_calories": 300,
    "net_calories": 1200,
    "tdee": 2000
}
"""
```

### Fitness Agent 工具

**search_fitness_knowledge(query: str)**
```python
"""
从RAG知识库检索健身专业知识
- 检索成功：返回【RAG检索】+ 检索内容
- 检索失败：返回"【RAG检索】未在知识库中找到相关信息"
- 注意：只负责检索，回答由LLM基于检索结果生成
"""
```

**estimate_exercise_calories(exercise_type, duration, intensity, user_weight)**
```python
"""
使用MET值估算运动消耗
MET表：
- 跑步: light=7, medium=10, intense=14
- 游泳: light=6, medium=10, intense=14
- 力量训练: light=4, medium=6, intense=8
公式：热量 = MET * 体重(kg) * 时间(h)
"""
```

**log_exercise(user_id, exercise_type, duration, calories, sets, reps)**
```python
"""
记录用户运动数据到数据库
返回："已记录: {exercise_type}, {notes}, 消耗 {calories} kcal"
"""
```

---

## 📚 RAG 系统 (RAG System)

### 架构

```
knowledge_base/  ──(加载文档)──►  ChromaDB向量库  ──(混合检索)──►  LLM生成回答
     │                                        │
  PDF/DOCX/图片                           embedding-2模型
                                              │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                    Query Expansion        HyDE              Self-RAG
                    (多查询扩展)        (假设性文档)        (自我反思)
```

### 目录结构

```
backend/app/rag/
├── __init__.py              # ModernRAG 主入口
├── modules/                  # 核心模块
│   ├── __init__.py         # 模块导出
│   ├── loader.py           # 文档加载器（重试机制）
│   ├── splitter.py         # 智能文本分割器
│   ├── preprocessor.py     # 文本预处理器（去重/清洗）
│   ├── bm25.py             # BM25 关键词检索
│   ├── hybrid_search.py    # 混合检索（向量+BM25 RRF融合）
│   ├── query_expansion.py  # 多查询扩展
│   ├── hyde.py             # 假设性文档嵌入
│   ├── cot.py              # 思维链推理
│   ├── self_rag.py         # 自我反思纠正
│   ├── agentic_rag.py      # 自主决策 RAG Agent
│   └── doc_processor.py    # 高级文档处理器
└── rag_utils.py            # 兼容旧接口
```

### 高级文档处理器 (doc_processor.py)

| 组件 | 说明 |
|------|------|
| **SemanticChunker** | 基于嵌入相似度的语义分割，识别主题边界 |
| **TableDetector** | Markdown/CSV/HTML 表格检测与提取 |
| **CodeBlockDetector** | 代码块检测与保留，防止错误分割 |
| **DocumentStructureAnalyzer** | 标题层级识别，构建文档树 |
| **ContextAwareCleaner** | 上下文感知清洗，保留格式信息 |
| **AdvancedDocumentProcessor** | 端到端文档处理流程整合 |

### 支持文档格式

- `.pdf` - PDF 文档
- `.docx` / `.doc` - Word 文档
- `.jpg` / `.jpeg` / `.png` - 图片（OCR）

### 检索模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `hybrid` | 混合检索（向量+BM25） | 默认模式 |
| `vector` | 仅向量检索 | 快速检索 |
| `bm25` | 仅 BM25 关键词检索 | 精确关键词匹配 |
| `query_expansion` | 多查询扩展 | 复杂问题 |
| `hyde` | 假设性文档嵌入 | 查询与文档表述差异大 |

### 高级特性

| 特性 | 说明 |
|------|------|
| **Query Expansion** | LLM 生成 3-5 个查询变体，提高召回率 |
| **HyDE** | 先用 LLM 生成"假设答案"，再用假设答案检索 |
| **CoT** | 思维链推理，逐步分析后生成答案 |
| **Self-RAG** | 评估检索相关性和回答质量，必要时自动纠正 |
| **Agentic RAG** | 大模型自主决定使用什么检索/生成策略 |

### Agentic RAG 工作流程

```
用户问题 → RouterAgent 分析问题类型
                │
        ┌───────┼───────┬────────┐
        ▼       ▼       ▼        ▼
    no_retrieval  basic  hyde   cot/self_rag
        │        │       │        │
        └────────┴───────┴────────┘
                        │
                    质量评估（如需要）
                        │
                   最终回答
```

### 检索流程

1. 用户查询 → embedding-2 向量化
2. ChromaDB 相似度搜索 + BM25 关键词搜索
3. RRF 融合分数排序
4. 拼接检索结果作为上下文
5. LLM 基于上下文生成回答（可选用 CoT/Self-RAG）

---

## 🚀 运行方式 (Quick Start)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端服务
cd backend
uvicorn app.main:app --reload

# 3. 启动前端（新终端）
streamlit run frontend/app.py
```

---

## 📈 项目进度 (Project Progress)

| 阶段 | 状态 | 说明 |
|------|------|------|
| 第一阶段：基础架构 | ✅ 完成 | FastAPI + SQLite 搭建 |
| 第二阶段：多 Agent 核心 | ✅ 完成 | LangGraph 状态机 |
| 第三阶段：Agent 实现 | ✅ 完成 | Chat/Nutrition/Fitness/Expert |
| 第四阶段：动态路由 | ✅ 完成 | 意图识别 + Agent 分发 |
| 第五阶段：流式输出 | ✅ 完成 | StreamingResponse |
| 第六阶段：UI 重构 | ✅ 完成 | ChatGPT 风格界面 |
| 第七阶段：RAG 检索 | ✅ 完成 | ChromaDB + embedding-2 |
| 第八阶段：API 集成 | ✅ 完成 | 天行数据食物营养 API |
| 第九阶段：RAG + LLM Fallback | ✅ 完成 | 检索失败时 LLM 自回答 |
| 第十阶段：专家评审 | ✅ 完成 | Expert Agent 评分 + 重试机制 |
| 第十一阶段：现代 RAG | ✅ 完成 | Hybrid Search + HyDE + CoT + Self-RAG |
| 第十二阶段：Agentic RAG | ✅ 完成 | 大模型自主决定检索/生成策略 |
| 第十三阶段：高级文档处理 | ✅ 完成 | 语义分割/表格提取/代码块保留/结构分析 |

---

## 🔮 未来优化方向 (Upcoming Enhancements)

- [ ] 语音输入功能
- [ ] 导出月度健康报告
- [ ] 用户认证系统
- [ ] 健身计划自动生成（基于用户目标）
- [ ] 增强 UI 交互（动画、主题切换）
- [ ] 支持更多运动类型和动作库
- [ ] Nutrition 和 Fitness Agent 协同工作（同时评审两个 Agent）
