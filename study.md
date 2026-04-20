# 🤖 FitCoach AI 大模型应用开发学习指南

欢迎来到 FitCoach AI 项目的学习文档！这是一个基于大语言模型（LLM）开发的智能健身与营养教练应用。通过学习这个项目，您将掌握：

- 如何搭建一个完整的 LLM Agent 应用
- 如何使用 LangGraph 构建状态机工作流
- 如何实现 RAG（检索增强生成）系统
- 如何构建前后端分离的 Web 应用

---

## 📁 项目结构概览

```
fitness_coach/
├── backend/                    # 后端服务
│   └── app/
│       ├── main.py             # FastAPI 主入口，API 定义
│       ├── agent.py            # LangGraph Agent 核心逻辑
│       ├── database.py         # SQLAlchemy 数据库配置
│       ├── models.py           # 数据库模型定义
│       ├── rag_utils.py        # RAG 检索工具
│       └── food_api.py         # 食物热量 API 调用模块
├── frontend/                   # 前端应用
│   └── app.py                  # Streamlit 前端界面
├── chroma_db/                  # ChromaDB 向量数据库存储
├── knowledge_base/             # RAG 知识库（PDF/Word/图片）
├── checkpoints.db              # LangGraph 对话状态持久化
├── fitness_coach.db            # 业务数据 SQLite 数据库
└── requirements.txt            # Python 依赖列表
```

---

## 🔧 技术栈说明

| 层级 | 技术 | 作用 |
|------|------|------|
| 前端 | Streamlit | 快速构建交互式 Web UI |
| 后端 | FastAPI | 高性能异步 API 框架 |
| 大模型 | LangChain + LangGraph | LLM 编排与状态管理 |
| 向量数据库 | ChromaDB | 本地知识存储与检索 |
| 数据库 | SQLite + SQLAlchemy | 业务数据持久化 |

---

## 🚀 第一部分：后端核心模块

### 1.1 main.py - FastAPI 主入口

**作用**：定义 API 接口，连接前端和后端逻辑

#### 关键函数解析

```python
def calculate_metrics(height, weight, age, gender):
    """计算 BMR 和 TDEE（基础代谢率和每日总能量消耗）"""
    # BMR 计算公式（Mifflin-St Jeor 公式）
    # 男性: 10 * 体重 + 6.25 * 身高 - 5 * 年龄 + 5
    # 女性: 10 * 体重 + 6.25 * 身高 - 5 * 年龄 - 161
```

**API 接口列表**：

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/user/` | 创建用户（自动计算 BMR/TDEE） |
| GET | `/user/{user_id}` | 获取用户信息 |
| GET | `/user/{user_id}/today` | 获取当日日志 |
| POST | `/chat` | 非流式聊天 |
| POST | `/chat/stream` | **流式聊天（重点）** |

**流式输出实现原理**：

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. 构建初始状态（用户信息 + 每日数据）
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "user_profile": {...},
        "daily_stats": {...}
    }
    
    # 2. 使用 astream_events 流式获取结果
    async for event in agent_app.astream_events(initial_state, config=config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content  # 逐块返回给前端
```

---

### 1.2 agent.py - LangGraph Agent 核心

**作用**：实现智能体的思考和决策逻辑（ReAct 模式）

#### 核心概念：State（状态）

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # 对话历史
    user_id: int                                                # 当前用户ID
    user_profile: Dict[str, Any]                                # 用户生理数据
    daily_stats: Dict[str, Any]                                 # 当日统计数据
```

**关键点**：`Annotated[List[BaseMessage], lambda x, y: x + y]` 表示消息会自动累积，无需手动合并。

#### 工具列表（Tools）

智能体可以调用以下工具来完成任务：

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `get_user_info` | 获取用户生理数据 | `user_id` |
| `log_food_intake` | 记录食物摄入 | `user_id`, `food_name`, `calories` |
| `log_exercise_burn` | 记录运动消耗 | `user_id`, `activity_type`, `duration`, `calories` |
| `get_daily_summary` | 获取当日卡路里总结 | `user_id` |
| `search_food_calories` | 查询食物热量 | `food_name` |
| `estimate_exercise_burn` | 估算运动消耗 | `exercise_type`, `duration` |
| `rag_medical_search_tool` | 专业知识检索 | `query` |

**工具定义示例**：

```python
@tool
def log_food_intake(user_id: int, food_name: str, calories: float):
    """记录用户摄入的食物及其卡路里。"""
    db = database.SessionLocal()
    try:
        today = date.today()
        # 查找或创建当日日志
        log = db.query(models.DailyLog).filter(...).first()
        if not log:
            log = models.DailyLog(user_id=user_id, date=today)
            db.add(log)
        
        # 创建食物条目
        food_item = models.FoodItem(log_id=log.id, name=food_name, calories=calories)
        log.intake_calories += calories
        db.add(food_item)
        db.commit()
        return f"已记录: {food_name}, {calories} kcal"
    finally:
        db.close()
```

#### 状态机工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 状态机                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐      决定调用工具       ┌─────────┐          │
│   │  Agent  │ ──────────────────────→ │  Tools  │          │
│   │ (LLM)   │ ←────────────────────── │  Node   │          │
│   └────┬────┘      返回工具结果       └─────────┘          │
│        │                                                    │
│        │ 直接回答用户                                        │
│        ↓                                                    │
│    ┌─────────┐                                             │
│    │  END    │                                             │
│    └─────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键代码**：

```python
# 1. 创建状态机
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("agent", call_model)    # LLM 思考节点
workflow.add_node("tools", tool_node)     # 工具执行节点

# 3. 设置入口点
workflow.set_entry_point("agent")

# 4. 添加条件边（决定是否调用工具）
workflow.add_conditional_edges("agent", should_continue)

# 5. 添加工具执行后的返回边
workflow.add_edge("tools", "agent")

# 6. 编译（在 main.py 中进行，因为需要管理异步检查点生命周期）
# agent_app = workflow.compile(checkpointer=checkpointer)
```

---

### 1.3 database.py - 数据库配置

**作用**：配置 SQLAlchemy 连接和会话

```python
SQLALCHEMY_DATABASE_URL = "sqlite:///./fitness_coach.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite 特殊配置
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # 所有模型的基类
```

---

### 1.4 models.py - 数据库模型

**作用**：定义数据库表结构

#### 表关系图

```
Users (用户)
    │
    └──→ DailyLogs (每日日志)
            │
            ├──→ FoodItems (食物条目)
            │
            └──→ ExerciseItems (运动条目)
```

#### 核心模型

```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    height = Column(Float)    # 身高(cm)
    weight = Column(Float)    # 体重(kg)
    age = Column(Integer)     # 年龄
    gender = Column(String)   # 性别
    bmr = Column(Float)       # 基础代谢率
    tdee = Column(Float)      # 每日总能量消耗
    allergies = Column(String) # 过敏史

class DailyLog(Base):
    __tablename__ = "daily_logs"
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    intake_calories = Column(Float, default=0.0)  # 摄入卡路里
    burn_calories = Column(Float, default=0.0)    # 消耗卡路里
```

---

### 1.5 food_api.py - 食物热量 API

**作用**：调用天行数据 API 查询食物营养信息

**API 配置**：
- 接口地址：`https://apis.tianapi.com/nutrient/index`
- 请求方式：POST
- API Key：从 `.env` 文件读取 (`TianxingFood_API_KEY`)

**核心函数**：

```python
def search_food_nutrient(food_name: str) -> dict:
    """调用天行数据API查询食物营养信息"""
    # 返回：{"calories": 热量, "protein": 蛋白质, "fat": 脂肪, "carbs": 碳水}
    
def search_food_calories(food_name: str) -> str:
    """搜索食物热量（用于Agent工具）"""
```

**数据来源优先级**：
1. 天行数据 API（优先）
2. 本地 fallback 数据（API 不可用时）
3. 大模型回答（两者都查不到时）

**支持的字段**：
- `rl` = 热量 (kcal)
- `dbz` = 蛋白质 (g)
- `zf` = 脂肪 (g)
- `shhf` = 碳水化合物 (g)

---

### 1.6 rag_utils.py - RAG 检索工具

**作用**：实现检索增强生成（RAG），让模型能够引用专业知识库

#### 核心流程

```
知识库文档 → 文本分割 → 向量化 → 存储到 ChromaDB
                                    │
用户查询 → 向量化 → 相似度搜索 → 获取相关知识 → 传入 LLM
```

#### 关键函数

```python
def init_rag(force_rebuild=False):
    """初始化向量数据库"""
    # 如果向量库已存在，直接加载
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        return vectorstore
    
    # 否则创建新的向量库
    # 1. 加载内置知识
    initial_texts = [
        "硬拉时腰疼通常是因为核心没有收紧...",
        "增肌需要热量盈余，通常建议在 TDEE 基础上增加 200-500 卡路里..."
    ]
    
    # 2. 加载外部文档（PDF/Word/图片）
    external_docs = load_knowledge_base()
    
    # 3. 文本分割（避免单块文本过长）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)
    
    # 4. 创建并持久化向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def rag_medical_search(query: str):
    """执行 RAG 检索"""
    # 搜索最相关的 3-4 个文档块
    results = vectorstore.similarity_search(query, k=4)
    # 拼接结果作为上下文返回
    context = "\n\n".join([doc.page_content for doc in results])
    return context
```

**支持的文档格式**：
- PDF (PyPDFLoader)
- Word (.docx, .doc)
- 图片 (.jpg, .jpeg, .png) - 通过 OCR 识别

---

## 🎨 第二部分：前端界面

### 2.1 app.py - Streamlit 前端

**作用**：提供用户友好的交互界面

#### 三大功能模块

| 模块 | 功能 |
|------|------|
| 🤖 智能教练 | 与 AI 对话，获取健身和营养建议 |
| 👤 个人档案 | 管理用户生理数据 |
| 📊 数据统计 | 可视化展示热量摄入/消耗趋势 |

#### 聊天界面核心逻辑

```python
# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("输入您的问题..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 调用后端流式接口
    response = requests.post(
        f"{BACKEND_URL}/chat/stream",
        json={"message": prompt, "user_id": user_id},
        stream=True
    )
    
    # 流式输出（打字机效果）
    message_placeholder = st.empty()
    full_response = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")  # 光标效果
```

---

## 🧠 第三部分：大模型应用核心概念

### 3.1 ReAct 模式

**什么是 ReAct？**  
ReAct = Reasoning + Acting（推理 + 行动）

智能体通过以下步骤处理用户请求：
1. **思考**：分析用户问题，决定是否需要调用工具
2. **行动**：调用合适的工具获取信息
3. **总结**：根据工具返回结果，给出最终回答

### 3.2 持久化记忆

**为什么需要持久化？**  
普通对话模型每次都是独立的，无法记住之前的对话历史。

**实现方式**：使用 `AsyncSqliteSaver` 存储对话状态

```python
# 在 main.py 中
checkpointer_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
checkpointer = await checkpointer_context.__aenter__()
agent_app = agent.workflow.compile(checkpointer=checkpointer)

# 使用 thread_id 区分不同用户
config = {"configurable": {"thread_id": str(user_id)}}
```

### 3.3 RAG（检索增强生成）

**解决什么问题？**  
- LLM 知识截止到训练时间，无法获取最新信息
- 对于专业领域知识（如医学、健身），需要确保准确性

**工作原理**：
1. 将专业文档向量化存储
2. 用户提问时，先搜索相关文档
3. 将搜索结果作为上下文传入 LLM
4. LLM 基于上下文生成回答

---

## 🚀 运行项目

### 环境要求

1. Python 3.10+
2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量（创建 `.env` 文件）：
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.zhipuai.com/v4/
LLM_MODEL=glm-4.7
```

### 启动服务

```bash
# 启动后端（终端1）
cd backend
uvicorn app.main:app --reload

# 启动前端（终端2）
cd frontend
streamlit run app.py
```

访问 `http://localhost:8501` 即可使用应用！

---

## 💡 扩展建议

### 添加新工具

1. 在 `agent.py` 中定义新工具：
```python
@tool
def your_new_tool(param1, param2):
    """工具功能描述（让 LLM 理解）"""
    # 实现逻辑
    return result
```

2. 将工具添加到工具列表：
```python
tools = [..., your_new_tool]
```

### 扩展知识库

只需将 PDF、Word 或图片文件放入 `knowledge_base/` 目录，下次启动时会自动加载。

### 接入真实 API

当前使用的是模拟数据（Mock），可以接入：
- 食物热量 API（如 USDA FoodData Central）
- 运动消耗计算 API
- 天气预报 API（影响运动建议）

---

## 📝 学习路线

| 阶段 | 学习目标 | 实践任务 |
|------|----------|----------|
| 1 | 理解项目结构 | 画出架构图 |
| 2 | 掌握 API 开发 | 修改或添加新接口 |
| 3 | 理解 Agent 工作流 | 添加一个新工具 |
| 4 | 掌握 RAG | 添加自定义知识库文档 |
| 5 | 前端开发 | 修改界面样式或添加新功能 |

---

## 🎯 总结

FitCoach AI 是一个完整的大模型应用示例，涵盖：

1. **后端架构**：FastAPI + SQLAlchemy + LangGraph
2. **智能体逻辑**：ReAct 模式 + 工具调用
3. **知识检索**：RAG + ChromaDB
4. **前端展示**：Streamlit 交互式界面
5. **持久化**：SQLite + SqliteSaver

通过学习这个项目，您可以掌握现代 LLM 应用开发的核心技术栈！

---

*文档版本：1.0*  
*最后更新：2026-04-20*
