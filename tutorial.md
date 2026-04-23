# FitCoach AI 项目教程 - 大模型应用开发实战

## 📖 目录

1. [项目概述](#1-项目概述)
2. [系统架构总览](#2-系统架构总览)
3. [核心技术详解](#3-核心技术详解)
4. [代码流程分析](#4-代码流程分析)
5. [关键设计思想](#5-关键设计思想)
6. [面试重点总结](#6-面试重点总结)

---

## 1. 项目概述

### 1.1 项目是什么

**FitCoach AI** 是一个基于大语言模型（LLM）和多 Agent 架构的智能私人营养师与健身教练应用。

简单来说：
- 用户可以像跟真人私教聊天一样，询问饮食建议、运动计划
- 系统会自动理解用户意图，调用合适的"专家"来回答
- 每个"专家"都有专用的工具和知识库

### 1.2 解决了什么问题

| 传统方式 | 本项目 |
|----------|--------|
| 请私教费用高 | AI 私教免费 |
| 查食物热量要手动搜索 | 语音/文字直接问 |
| 健身知识要自己学习 | AI 基于知识库回答 |
| 计划制定不科学 | AI 根据个人情况定制 |

### 1.3 核心技术栈

```
前端：Streamlit（快速构建 Web UI）
后端：FastAPI（高性能 API）
AI：LangChain + 智谱 GLM-4
数据库：SQLite + SQLAlchemy
向量库：ChromaDB（RAG 知识检索）
```

---

## 2. 系统架构总览

### 2.1 整体架构图

```
用户 ──► 前端(Streamlit) ──► 后端(FastAPI) ──► Agent 路由 ──► 专业 Agent
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                    闲聊 Agent           营养师 Agent         健身教练 Agent
                    (无需工具)          (食物 API)           (RAG 知识库)
                                                │                     │
                                                └──────────┬──────────┘
                                                           │
                                                    Expert Agent
                                                    (质量评审)
```

### 2.2 模块职责

| 模块 | 文件位置 | 职责 |
|------|----------|------|
| **前端** | `frontend/app.py` | 用户界面，ChatGPT 风格 |
| **后端入口** | `backend/app/main.py` | API 定义，用户管理 |
| **Agent 路由** | `backend/app/agents/router.py` | 意图识别，路由分发 |
| **闲聊 Agent** | `backend/app/agents/chat_agent.py` | 日常对话 |
| **营养师 Agent** | `backend/app/agents/nutrition_agent.py` | 饮食建议、热量查询 |
| **健身教练 Agent** | `backend/app/agents/fitness_agent.py` | 运动计划、动作指导 |
| **专家 Agent** | `backend/app/agents/expert_agent.py` | 评审输出质量 |
| **RAG 系统** | `backend/app/rag/` | 知识库检索 |
| **食物 API** | `backend/app/food_api.py` | 食物营养数据 |

### 2.3 数据模型

```
User（用户）
├── height, weight, age, gender（生理数据）
├── bmr, tdee（代谢数据）
└── DailyLog（每日记录）
    ├── intake_calories（摄入热量）
    ├── burn_calories（消耗热量）
    ├── food_items（饮食明细）
    └── exercise_items（运动明细）
```

---

## 3. 核心技术详解

### 3.1 多 Agent 系统

**什么是 Agent？**

Agent = 大模型 + 工具 + 行为逻辑

类比：
- 大模型 = 大脑（能思考）
- 工具 = 四肢（能执行动作）
- Agent = 完整的人

**本项目的四个 Agent：**

| Agent | 用的模型 | 能做什么 | 用什么工具 |
|-------|----------|----------|------------|
| 闲聊 | GLM-4 | 聊天 | 无 |
| 营养师 | GLM-4 | 饮食建议 | 食物 API、数据库 |
| 健身教练 | GLM-4 | 健身指导 | RAG 检索、数据库 |
| 专家 | GLM-4 | 评审质量 | 无 |

**Agent 的工作方式：**

```python
# Agent 本质是一个"带工具调用能力的大模型"
llm = ChatOpenAI(model="glm-4")

# 绑定工具
llm_with_tools = llm.bind_tools(tools)

# 调用
response = llm_with_tools.invoke(user_message)

# 如果需要调用工具，response.tool_calls 会有内容
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
```

### 3.2 动态路由机制

**路由解决的问题：**

用户说"我想减肥" → 应该去营养师还是健身教练？

**路由实现：**

```python
def route_with_context(user_message: str) -> str:
    # 方案：让大模型判断
    prompt = f"用户输入: {user_message}\n判断类型：1=闲聊 2=营养师 3=健身教练\n只返回一个数字。"

    llm = ChatOpenAI(temperature=0.1)  # 低温度保证稳定
    response = llm.invoke([HumanMessage(content=prompt)])

    if "2" in response.content:
        return "nutrition"
    elif "3" in response.content:
        return "fitness"
    return "chat"
```

**路由规则（简化版）：**

```
关键词匹配：
- "吃"、"食物"、"热量"、"饮食" → nutrition
- "运动"、"跑步"、"训练"、"健身" → fitness
- 其他 → chat
```

### 3.3 RAG 系统（检索增强生成）

**RAG 是什么？**

RAG = Retrieval Augmented Generation（检索增强生成）

**为什么需要 RAG？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 纯大模型 | 通用能力强 | 知识可能过时/不准确 |
| 纯检索 | 实时性好 | 缺乏理解能力 |
| RAG | 实时 + 理解 | 需要维护知识库 |

**本项目 RAG 架构：**

```
知识库文档（PDF/Word）
        │
        ▼
文档加载 ──► 文本分割 ──► 向量化 ──► 存入 ChromaDB
                                            │
用户查询 ──► 向量化 ──► 相似度检索 ──► 返回结果
```

**现代化 RAG 特性：**

1. **混合检索**：向量检索 + BM25 关键词检索
2. **Query Expansion**：一个查询变多个，提高召回率
3. **HyDE**：先生成"假设答案"再检索
4. **Self-RAG**：生成后自我反思，检查质量
5. **Agentic RAG**：让大模型自己决定用什么检索策略

### 3.4 混合检索原理

```python
# 向量检索：理解语义
vector_results = vectorstore.similarity_search(query)
# "上斜卧推" 可能匹配到 "哑铃卧推"（同义理解）

# BM25 检索：精确匹配
bm25_results = bm25.search(query)
# 必须包含"上斜卧推"这个关键词

# RRF 融合：结合两者优势
# RRF_score = Σ 1/(k + rank)
# rank=第1名给60分，rank=第2名给59分...
```

### 3.5 专家评审机制

**为什么需要评审？**

避免营养师/教练给出质量差或不安全的建议。

**评审流程：**

```
用户问题 ──► 营养师回答 ──► 专家评审
                                │
                        评分 1-5 分
                        │
            ┌─────────────┴─────────────┐
            │                           │
        评分 ≥ 3                    评分 ≤ 2
        通过 ✓                     打回重试
                                    │
                              最多重试 3 次
```

**评分标准：**

| 分数 | 含义 |
|------|------|
| 1 | 严重不足/错误 |
| 2 | 需要显著改进 |
| 3 | 基本合格 |
| 4 | 良好 |
| 5 | 优秀 |

---

## 4. 代码流程分析

### 4.1 用户发起对话的完整流程

```
用户在前端输入"上斜卧推怎么做"
        │
        ▼
前端发送 POST /chat/stream
        │
        ▼
后端 main.py 接收请求
        │
        ▼
调用 graph.process_user_message()
        │
        ▼
router.route_with_context() 决定走哪个 Agent
        │
        ├── "fitness" → fitness_agent 处理
        │
        ▼
fitness_agent 调用 search_fitness_knowledge 工具
        │
        ▼
rag_utils.rag_medical_search() 查询 ChromaDB
        │
        ▼
返回检索结果，LLM 生成优化回答
        │
        ▼
expert_agent.review_output() 评审质量
        │
        ├── 评分 ≥ 3 → 返回给用户
        └── 评分 ≤ 2 → 重新生成（最多 3 次）
        │
        ▼
前端流式显示响应
```

### 4.2 关键代码解读

#### 4.2.1 工具定义（以营养师为例）

```python
@tool
def search_food_nutrition(food_name: str):
    """搜索食物营养信息"""
    from ..food_api import search_food_nutrient
    result = search_food_nutrient(food_name)

    # 返回格式化的检索结果，而不是直接回答
    # 这样 LLM 可以基于检索结果生成更好的回答
    if result:
        return f"【API检索】{food_name}: 热量 {result['calories']} kcal..."
    return f"【API检索】未找到 {food_name} 的营养信息"

# 工具列表
nutrition_tools = [
    get_user_nutrition_info,  # 获取用户信息
    log_food_intake,          # 记录饮食
    get_daily_nutrition_summary,  # 获取当日总结
    search_food_nutrition     # 搜索食物
]
```

#### 4.2.2 Agent 对话函数

```python
def nutrition_with_user(messages: list, user_id: int) -> str:
    llm = ChatOpenAI(model="glm-4")

    # 关键：绑定工具
    llm_with_tools = llm.bind_tools(nutrition_tools)

    for _ in range(2):  # 最多调用工具 2 次
        response = llm_with_tools.invoke(messages)

        # 没有工具调用？直接返回回答
        if not response.tool_calls:
            return response.content

        # 有工具调用？执行工具
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # 执行工具
            result = execute_tool(tool_name, tool_args)

            # 把工具结果加回消息
            messages.append(response)
            messages.append(HumanMessage(content=result, type="tool"))

    return response.content
```

#### 4.2.3 食物 API 调用

```python
def search_food_nutrient(food_name: str) -> dict:
    # 1. 调用天行数据 API
    params = {'key': API_KEY, 'word': food_name}
    conn.request('POST', '/nutrient/index', params)
    response = conn.getresponse()

    # 2. 解析响应
    data = json.loads(response.read())
    if data['code'] == 200:
        nutrient = data['result']['list'][0]
        return {
            'calories': nutrient.get('rl'),    # 热量
            'protein': nutrient.get('dbz'),     # 蛋白质
            'fat': nutrient.get('zf'),         # 脂肪
            'carbs': nutrient.get('shhf')     # 碳水
        }

    # 3. API 失败则返回 None，让 LLM 自己回答
    return None
```

#### 4.2.4 RAG 检索

```python
def rag_medical_search(query: str) -> str:
    # 1. 加载向量库
    vectorstore = Chroma(persist_directory="./chroma_db")

    # 2. 相似度检索
    docs = vectorstore.similarity_search(query, k=3)

    # 3. 拼接结果
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    return "未找到相关信息"
```

---

## 5. 关键设计思想

### 5.1 工具调用模式（Tool Use Pattern）

**核心思想：** 大模型不直接回答，而是判断是否需要工具辅助。

```
传统：大模型直接回答 → 可能不准确
本项目：大模型决定是否查资料 → 查完再回答 → 更准确
```

**代码体现：**

```python
# 绑定工具后，LLM 会自动判断何时调用工具
llm_with_tools = llm.bind_tools(tools)

# LLM 决定调用工具，返回的 response.tool_calls 不为空
# LLM 决定不调用工具，直接返回文本回答
```

### 5.2 检索与生成分离

**设计原则：**
- 工具只负责"检索"，不负责"回答"
- 大模型负责"理解检索结果"并"生成回答"

**好处：**
- 检索结果可以被大模型优化
- 检索不到时，大模型可以用自己的知识回答

### 5.3 质量门禁机制

**设计原则：** 重要输出必须经过评审。

```
Agent 输出 ──► Expert Agent 评分 ──► 通过？──► 返回用户
                  │                     │
                  │ 否                  │ 是
                  ▼                     ▼
              打回重试              返回用户
```

**防死循环：** 最多重试 3 次。

### 5.4 上下文管理

```python
# 每个 Agent 都能获取用户上下文
def nutrition_with_user(messages: list, user_id: int):
    # 可以查询用户信息
    user_info = get_user_nutrition_info(user_id)

    # 可以查询当日统计
    daily_summary = get_daily_nutrition_summary(user_id)

    # 上下文帮助生成个性化回答
```

---

## 6. 面试重点总结

### 6.1 必须掌握的核心概念

| 概念 | 解释 | 项目中的应用 |
|------|------|--------------|
| **Agent** | 大模型 + 工具 + 行为逻辑 | 营养师/健身教练/专家 Agent |
| **Tool Use** | 让大模型调用外部工具 | 食物 API、RAG 检索 |
| **RAG** | 检索增强生成 | ChromaDB 知识库 |
| **Router** | 意图识别 + 路由分发 | router.py |
| **Hybrid Search** | 向量 + 关键词混合检索 | ChromaDB + BM25 |
| **Self-RAG** | 生成后自我反思纠正 | 评审机制 |

### 6.2 项目亮点（面试可说）

1. **多 Agent 协作**：四个专业 Agent，各司其职
2. **动态路由**：LLM 判断用户意图，自动分发
3. **质量门禁**：Expert Agent 评审，不合格打回重试
4. **现代化 RAG**：混合检索 + Query Expansion + HyDE + Self-RAG
5. **容错设计**：API 失败有 fallback，大模型作为最后保障

### 6.3 技术难点与解决

| 难点 | 解决思路 |
|------|----------|
| 如何让 Agent 调用合适的工具 | 使用 `bind_tools()` 让 LLM 决定 |
| RAG 检索不准怎么办 | 混合检索 + Query Expansion |
| 如何保证输出质量 | Expert Agent 评审 + 重试机制 |
| API 失败怎么处理 | 本地 fallback + LLM 自回答 |

### 6.4 项目改进方向（面试加分）

1. **多模态**：支持上传食物照片识别
2. **个性化**：根据用户反馈微调模型
3. **评估体系**：RAGAS 评估检索质量
4. **流式输出**：减少等待时间

---

## 📚 推荐学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [RAG 综述论文](https://arxiv.org/abs/2005.11401)
- [Self-RAG 论文](https://arxiv.org/abs/2312.05933)
- [ChromaDB 文档](https://docs.trychroma.com/)
