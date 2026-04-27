# FitCoach AI 智能私教系统 - 开发文档

## 📅 最后更新日期：2026-04-27

---

## 📁 项目目录结构

```
d:\fitness_coach/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI 主入口，API 路由定义
│       ├── models.py                 # SQLAlchemy 数据库模型
│       ├── database.py               # 数据库连接配置
│       ├── food_api.py              # 天行数据食物营养 API
│       ├── agents/                   # 多 Agent 系统核心
│       │   ├── __init__.py          # 模块导出
│       │   ├── base.py              # Agent 基础配置（System Prompt）
│       │   ├── router.py            # 动态路由器（LLM 意图识别）
│       │   ├── chat_agent.py        # 闲聊 Agent
│       │   ├── nutrition_agent.py   # 营养师 Agent
│       │   ├── fitness_agent.py      # 健身教练 Agent
│       │   ├── expert_agent.py      # 专家评审 Agent
│       │   └── graph.py             # LangGraph 工作流
│       ├── rag/                      # 现代化 RAG 系统
│       │   ├── __init__.py          # ModernRAG 主入口
│       │   └── modules/             # RAG 核心模块
│       │       ├── loader.py        # 文档加载器
│       │       ├── splitter.py       # 智能文本分割
│       │       ├── preprocessor.py  # 文本预处理
│       │       ├── bm25.py          # BM25 检索
│       │       ├── hybrid_search.py # 混合检索
│       │       ├── query_expansion.py  # 多查询扩展
│       │       ├── hyde.py          # 假设性文档嵌入
│       │       ├── cot.py           # 思维链推理
│       │       ├── self_rag.py      # 自我反思纠正
│       │       ├── agentic_rag.py   # Agentic RAG
│       │       └── doc_processor.py # 高级文档处理
│       └── memory/                   # 记忆模块 🆕
│           ├── __init__.py          # 模块导出
│           ├── memory_manager.py    # 记忆管理器
│           ├── user_profile.py      # 用户画像加载
│           ├── conversation_summary.py  # 对话历史摘要
│           ├── stats_summary.py     # 统计数据汇总
│           └── todolist.md          # 记忆模块待办
├── frontend/
│   └── app.py                      # Streamlit 前端界面
├── chroma_db/                       # ChromaDB 向量库
├── knowledge_base/                  # RAG 知识库文档
├── .env                            # 环境变量
├── requirements.txt                 # 依赖
└── README.md                        # 项目说明
```

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            FitCoach AI 架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                       │
│   │  用户    │ ──▶ │ FastAPI  │ ──▶ │  Router  │                       │
│   │  输入    │     │  Backend │     │  Agent   │                       │
│   └──────────┘     └──────────┘     └────┬─────┘                       │
│                                          │                               │
│              ┌───────────────────────────┼───────────────────────────┐  │
│              │                           │                            │  │
│              ▼                           ▼                            ▼  │
│       ┌──────────┐              ┌──────────────┐              ┌──────────┐
│       │   Chat   │              │   Nutrition  │              │  Fitness │
│       │  Agent   │              │    Agent     │              │  Agent   │
│       └────┬─────┘              └──────┬───────┘              └────┬─────┘
│            │                            │                            │     │
│            │                            ▼                            │     │
│            │                     ┌──────────────┐                    │     │
│            │                     │    Expert    │◄────────────────────┘     │
│            │                     │    Agent     │                          │
│            │                     └──────┬───────┘                          │
│            │                            │                                 │
│            ▼                     评分 ≥ 3 ──▶ 用户                        │
│          END                      │                                      │
│                               评分 ≤ 2                                    │
│                                 │ (最多重试3次)                            │
│                                 ▼                                         │
│                              重新生成                                      │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────────┐ │
│   │                        Memory 模块 🆕                                │ │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │ │
│   │  │  用户画像    │  │  对话摘要    │  │  统计汇总    │           │ │
│   │  │ UserProfile  │  │ Conversation │  │    Stats     │           │ │
│   │  │   Loader     │  │  Summarizer │  │  Summarizer │           │ │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │ │
│   │                           │                                      │ │
│   │              ┌────────────┴────────────┐                         │ │
│   │              │     Memory Manager      │                         │ │
│   │              │   (统一记忆上下文)       │                         │ │
│   │              └─────────────────────────┘                         │ │
│   └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心模块说明

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **主入口** | `main.py` | FastAPI 应用、API 路由、BMR/TDEE 计算 |
| **数据模型** | `models.py` | User, DailyLog, FoodItem, ExerciseItem, **ConversationLog** |
| **数据库** | `database.py` | SQLAlchemy engine 和 SessionLocal |
| **食物API** | `food_api.py` | 天行数据食物营养查询 |
| **Agent基座** | `base.py` | System Prompt、AgentState 类型 |
| **路由器** | `router.py` | 意图识别（LLM 分析） |
| **主工作流** | `graph.py` | LangGraph 工作流整合 |
| **闲聊Agent** | `chat_agent.py` | 日常对话 + 记忆上下文 |
| **营养师Agent** | `nutrition_agent.py` | 饮食计划 + 记忆上下文 |
| **健身教练Agent** | `fitness_agent.py` | 训练指导 + 记忆上下文 |
| **专家Agent** | `expert_agent.py` | 评审输出质量 |
| **记忆模块** | `memory/*` | 用户画像、对话历史持久化、统计汇总 ✅ |

---

## 🤖 多 Agent 协作机制

### Agent 职责与工具

| Agent | 职责 | 可用工具 |
|-------|------|----------|
| **Router** | 分析意图，路由分发 | 无 |
| **Chat** | 闲聊、情感支持 | 无 |
| **Nutrition** | 饮食计划、热量计算 | `search_food_nutrition`, `log_food_intake`, `get_daily_nutrition_summary`, `get_user_nutrition_info` |
| **Fitness** | 训练计划、动作指导 | `search_fitness_knowledge`, `estimate_exercise_calories`, `log_exercise`, `get_user_fitness_info` |
| **Expert** | 评审质量（1-5分） | 无 |

### Router 路由规则

```python
# LLM 分析用户输入，返回 1/2/3
类型 1 - 闲聊助手：
  - 日常问候、情感交流
  - 通用知识问答
  - 不涉及具体的饮食或运动计划

类型 2 - 营养师：
  - 食物热量查询、营养成分计算
  - 饮食计划、食谱推荐
  - 增肌/减脂/维持的饮食策略

类型 3 - 健身教练：
  - 运动训练、动作指导
  - 健身计划、训练强度
  - 具体运动姿势和发力技巧
```

### Expert Agent 评审机制

| 评分 | 等级 | 说明 | 动作 |
|------|------|------|------|
| 5 | 卓越 | 内容极其专业、数据精确、完全个性化 | 直接通过 |
| 4 | 优秀 | 内容专业、数据准确、有小的改进空间 | 直接通过 |
| 3 | 合格 | 内容基本正确、有小问题 | 直接通过 |
| 2 | 不足 | 内容有较大问题、需要显著改进 | 重试 |
| 1 | 很差 | 内容错误、存在安全隐患 | 重试 |

**防死循环**：`MAX_RETRIES = 3`，超过最大次数返回最后结果

---

## 🧠 记忆模块（Memory）

### 模块架构

```
memory/
├── __init__.py              # 模块导出
├── memory_manager.py        # 记忆管理器（统一接口）
├── user_profile.py         # 用户画像加载
├── conversation_summary.py  # 对话历史摘要（内存）
├── stats_summary.py        # 统计数据汇总
└── todolist.md             # 待办事项
```

### 核心组件

| 组件 | 类名 | 功能 |
|------|------|------|
| 用户画像 | `UserProfileLoader` | 从数据库加载用户信息，格式化为 Agent 上下文 |
| 对话摘要 | `ConversationSummarizer` | 超过10条消息时自动摘要，压缩早期对话 |
| 统计汇总 | `StatsSummarizer` | 获取当日/本周的营养运动统计 |
| 记忆管理 | `MemoryManager` | 统一管理所有记忆，提供上下文注入接口 |

### 记忆类型

| 类型 | 存储 | 说明 |
|------|------|------|
| **长期记忆** | SQLite | 用户画像、每日统计、历史汇总、**对话历史持久化** |
| **短期记忆** | 内存 | 当前对话、10条后自动摘要 |
| **Agent上下文** | 动态注入 | 记忆格式化后注入 System Prompt |

### 跨会话对话持久化 ✅ 新增

```python
# ConversationLog 表结构
class ConversationLog(Base):
    user_id: int           # 用户 ID
    session_id: str        # 会话 ID
    agent_type: str        # Agent 类型
    user_message: str      # 用户消息
    agent_response: str    # Agent 回复
    created_at: datetime   # 创建时间

# 保存对话
memory.save_conversation(
    user_message="今天吃什么好？",
    agent_response="推荐鸡胸肉...",
    agent_type="nutrition",
    session_id="session_20260427"
)

# 加载历史对话
history = memory.load_conversation_history(days=7, limit=50)
# 返回格式：
# [
#   {"role": "user", "content": "今天吃什么好？", "agent_type": "nutrition"},
#   {"role": "assistant", "content": "推荐鸡胸肉...", "agent_type": "nutrition"},
#   ...
# ]
```

### 使用方式

```python
from .memory import MemoryManager

# 初始化
memory = MemoryManager(user_id=1)

# 获取完整上下文
context = memory.get_full_context()
# {
#     "profile": {...},      # 用户画像
#     "goal": "减脂",        # 用户目标
#     "today_stats": {...},  # 当日统计
#     "week_stats": {...}    # 本周统计
# }

# 格式化注入 Agent
profile_text = memory.format_profile_for_agent()
# "用户目标: 减脂, 当前体重: 75kg, 目标体重: 65kg, TDEE: 2000kcal"

# 获取记忆上下文（用于传入 Agent）
memory_context = memory.get_memory_context(agent_type="nutrition")
# "用户目标: 减脂\n今日已摄入: 1500 kcal\n今日剩余: ~500 kcal"
```

### Agent 中的记忆注入

```python
# graph.py - 入口函数中预生成增强 prompt
memory_manager = MemoryManager(user_id=user_id)
memory_summary = memory_manager.get_memory_summary()
conversation_history = memory_manager.load_conversation_history(days=7, limit=20)
memory_summary["conversation_history"] = conversation_history

# 使用 enhance_system_prompt() 生成完整的增强 prompt
messages_for_prompt = [HumanMessage(content=user_message)]
enhanced_prompts = {
    "chat": memory_manager.enhance_system_prompt(
        AGENT_SYSTEM_PROMPTS["chat"], "chat", messages_for_prompt
    ),
    "nutrition": memory_manager.enhance_system_prompt(
        AGENT_SYSTEM_PROMPTS["nutrition"], "nutrition", messages_for_prompt
    ),
    "fitness": memory_manager.enhance_system_prompt(
        AGENT_SYSTEM_PROMPTS["fitness"], "fitness", messages_for_prompt
    )
}

# 传递给 LangGraph 工作流
initial_state = {
    ...
    "enhanced_prompts": enhanced_prompts
}

# graph.py - 节点函数中使用增强 prompt
def nutrition(state: AgentState) -> Dict[str, Any]:
    enhanced_prompt = state.get("enhanced_prompts", {}).get("nutrition")
    response = nutrition_with_user(messages, user_id, memory_summary, enhanced_prompt)
    ...

# nutrition_agent.py / fitness_agent.py / chat_agent.py
def nutrition_with_user(messages, user_id, memory_summary=None, enhanced_prompt=None):
    if enhanced_prompt:
        system_content = enhanced_prompt  # 使用完整的增强上下文
    else:
        # 回退到简化版 memory_summary
        system_content = AGENT_SYSTEM_PROMPTS["nutrition"]
        if memory_summary:
            system_content += format_nutrition_memory(memory_summary)
    ...
```

**增强 Prompt 包含的完整上下文**：
1. 基础 System Prompt（Agent 角色定义）
2. 用户画像（身高/体重/年龄/性别/BMR/TDEE/过敏史/目标）
3. 当日统计（摄入/消耗/净热量）
4. 本周统计（营养师和健身教练 Agent 专用）
5. 对话要点（讨论话题、用户目标）✅

---

## 📚 RAG 系统

### 架构

```
knowledge_base/  ──(加载文档)──▶  ChromaDB向量库  ──(混合检索)──▶  LLM生成回答
     │                                        │
  PDF/DOCX/图片                           embedding-2模型
                                            │
                         ┌──────────────────┼──────────────────┐
                         │                  │                  │
                    Query Expansion      HyDE            Self-RAG
                    (多查询扩展)      (假设性文档)      (自我反思)
```

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

---

## 🔌 API 接口

| 方法 | 路径 | 功能 | 请求体/参数 |
|------|------|------|-------------|
| POST | `/user/` | 创建用户 | `UserCreate` |
| GET | `/user/{user_id}` | 获取用户信息 | - |
| GET | `/user/{user_id}/logs` | 获取用户所有日志 | - |
| GET | `/user/{user_id}/today` | 获取当日日志 | - |
| POST | `/chat` | 非流式聊天 | `ChatRequest` |
| POST | `/chat/stream` | 流式聊天 | `StreamChatRequest` |
| GET | `/agents` | 获取所有可用 Agent | - |

---

## 💾 数据库模型

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

## 🚀 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端服务
cd backend
uvicorn app.main:app --reload

# 3. 启动前端（新终端）
streamlit run frontend/app.py

# 4. 运行单元测试
cd backend
python -m unittest discover -s tests -v
```

---

## 🧪 单元测试

### 测试文件结构

```
backend/tests/
├── __init__.py
├── test_memory.py     # Memory 模块测试 (32 tests)
├── test_agents.py     # Agents 模块测试 (61 tests)
└── test_rag.py        # RAG 模块测试 (57 tests)
```

### 测试覆盖

| 模块 | 测试类数 | 测试数 | 覆盖内容 |
|------|----------|--------|----------|
| **Memory** | 5 | 32 | UserProfileLoader, ConversationSummarizer, StatsSummarizer, MemoryManager, 集成测试 |
| **Agents** | 22 | 61 | AgentConfig, AgentResponse, Router, ChatAgent, NutritionTools, FitnessTools, ExpertAgent, Graph 节点 |
| **RAG** | 19 | 57 | DocumentLoader, IntelligentSplitter, TextPreprocessor, BM25, HybridSearch, QueryExpander, HyDE, CoT, SelfRAG, AgenticRAG, TableDetector, CodeBlockDetector, SemanticChunker, DocumentStructureAnalyzer, ContextAwareCleaner, AdvancedDocumentProcessor |
| **总计** | 46 | **150** | |

### Agents 模块测试详情

| 测试类 | 测试数 | 覆盖的方法/功能 |
|--------|--------|-----------------|
| TestAgentConfig | 1 | AgentConfig 创建 |
| TestAgentResponse | 2 | AgentResponse 默认值/评审信息 |
| TestAgentSystemPrompts | 4 | prompts 键完整性、角色定义 |
| TestRouter | 4 | route_with_context 路由到各 Agent |
| TestFormatMemoryContext | 5 | format_memory_context 格式化 |
| TestChatAgent | 2 | chat_with_user 返回值/增强 prompt |
| TestNutritionTools | 4 | get_user_nutrition_info, log_food_intake, get_daily_nutrition_summary, tools 列表 |
| TestFormatNutritionMemory | 4 | format_nutrition_memory 格式化 |
| TestFitnessTools | 5 | get_user_fitness_info, log_exercise, estimate_exercise_calories, tools 列表 |
| TestFormatFitnessMemory | 4 | format_fitness_memory 格式化 |
| TestExpertAgent | 4 | review_output 返回值/评分/异常处理 |
| TestGraphConstants | 2 | MAX_RETRIES, MIN_APPROVAL_SCORE |
| TestAgentState | 1 | AgentState 键完整性 |
| TestRouterNode | 3 | router 节点路由逻辑 |
| TestChatNode | 1 | chat 节点返回消息 |
| TestNutritionNode | 1 | nutrition 节点返回消息 |
| TestFitnessNode | 1 | fitness 节点返回消息 |
| TestExpertReviewNode | 3 | expert_review 重试逻辑 |
| TestShouldContinue | 4 | should_continue 条件判断 |
| TestRouteAfterRouter | 3 | route_after_router 路由分发 |
| TestBuildGraph | 1 | build_graph 返回 StateGraph |

### RAG 模块测试详情

| 测试类 | 测试数 | 覆盖的方法/功能 |
|--------|--------|-----------------|
| TestDocumentLoader | 4 | loader 初始化, LOADER_MAP, load_directory |
| TestRetryDecorator | 3 | retry_on_failure 装饰器重试逻辑 |
| TestIntelligentSplitter | 5 | splitter 初始化, split_text, split_documents |
| TestTextPreprocessor | 8 | preprocessor 初始化, clean_text, normalize_text, preprocess_document, preprocess_documents, deduplicate_by_similarity |
| TestBM25 | 4 | BM25 初始化, tokenize, fit, search |
| TestBM25Search | 2 | BM25Search 初始化, index |
| TestHybridSearch | 3 | HybridSearch 初始化, _rrf_fusion, index |
| TestQueryExpander | 3 | QueryExpander 初始化, expand |
| TestHyDEGenerator | 2 | HyDEGenerator 初始化, generate |
| TestCoTReasoner | 2 | CoTReasoner 初始化, reason |
| TestSelfRAG | 2 | SelfRAG 初始化, is_retrieval |
| TestSelfRAGScorer | 1 | SelfRAGScorer 初始化 |
| TestRouterAgent | 2 | RouterAgent 初始化, decide |
| TestAgenticRAG | 1 | AgenticRAG 初始化 |
| TestAutoRAG | 1 | AutoRAG 初始化 |
| TestQueryClassifier | 1 | QueryClassifier 初始化 |
| TestTableDetector | 2 | TableDetector 初始化, detect_markdown_tables |
| TestCodeBlockDetector | 2 | CodeBlockDetector 初始化, detect_fenced_code_blocks |
| TestSemanticChunker | 1 | SemanticChunker 初始化 |
| TestDocumentStructureAnalyzer | 2 | DocumentStructureAnalyzer 初始化, analyze |
| TestContextAwareCleaner | 2 | ContextAwareCleaner 初始化, clean |
| TestAdvancedDocumentProcessor | 2 | AdvancedDocumentProcessor 初始化, process_documents |
| TestTableData | 1 | TableData 数据类创建 |
| TestCodeBlock | 1 | CodeBlock 数据类创建 |
| TestSection | 1 | Section 数据类创建 |

---

## 📈 项目进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 基础架构 | ✅ 完成 | FastAPI + SQLite |
| 多 Agent 系统 | ✅ 完成 | Chat/Nutrition/Fitness/Expert |
| LangGraph 工作流 | ✅ 完成 | 状态机驱动 |
| 动态路由 | ✅ 完成 | LLM 意图识别 |
| 专家评审 | ✅ 完成 | 1-5分评分 + 重试机制 |
| 食物营养 API | ✅ 完成 | 天行数据 API |
| RAG 基础 | ✅ 完成 | ChromaDB 向量检索 |
| 现代 RAG | ✅ 完成 | 混合检索/查询扩展/HyDE |
| Agentic RAG | ✅ 完成 | LLM 自主决策 |
| 高级文档处理 | ✅ 完成 | 语义分割/表格提取 |
| 记忆模块 | ✅ 完成 | 用户画像/对话摘要/统计汇总 |
| 跨会话对话持久化 | ✅ 完成 | ConversationLog 持久化 🆕 |
| 上下文工程 | ✅ 完成 | enhance_system_prompt() 完整上下文注入 Agent |
| 增量文档索引 | ✅ 完成 | 启动时自动检测新文档并增量添加到知识库 |
| Memory 模块单元测试 | ✅ 完成 | 32 tests ✅ |
| Agents 模块单元测试 | ✅ 完成 | 61 tests ✅ |
| RAG 模块单元测试 | ✅ 完成 | 57 tests ✅ |

---

## 🔮 未来优化方向

- [x] 记忆上下文信息量不足 (P1) - ✅ 已通过 enhance_system_prompt() 解决
- [x] enhance_system_prompt() 未被使用 (P1) - ✅ 已集成到 Agent 工作流
- [x] Memory 模块单元测试 - ✅ 32 tests
- [x] Agents 模块单元测试 - ✅ 61 tests
- [x] RAG 模块单元测试 - ✅ 57 tests
- [ ] 对话要点提取与持久化存储
- [ ] 用户偏好持久化
- [ ] 前端重构：TypeScript + React
- [ ] 语音输入功能
- [ ] 用户认证系统
