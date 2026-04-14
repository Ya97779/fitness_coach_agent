# 目规划文档：智能私人营养师与健身教练 Agent

## 1. 项目概述

本项目旨在开发一款基于大语言模型（LLM）驱动的“私人营养师与健身教练” Web 应用程序。核心系统采用 Agent 架构，不仅具备多轮对话能力，更能通过记忆用户的生理状态、调用外部工具（计算卡路里、查询食物热量等）以及检索专业健康文献，为用户提供长期、个性化的饮食与运动规划闭环服务。

## 2. 产品功能需求 (PRD)

### 2.1 用户状态管理

支持输入并持久化存储用户的生理基准数据（身高、体重、年龄、性别、目标体重、过敏史）。

计算并维护用户的每日基础代谢率（BMR）和每日总能量消耗（TDEE）。
- **BMR (Mifflin-St Jeor Equation)**:
  - 男: 10 * weight(kg) + 6.25 * height(cm) - 5 * age(y) + 5
  - 女: 10 * weight(kg) + 6.25 * height(cm) - 5 * age(y) - 161
- **TDEE**: BMR * Activity Factor (默认 1.2 - 1.55)

### 2.2 日常打卡与数据追踪

饮食录入： 用户通过自然语言描述日常饮食（如“今天中午吃了一碗兰州拉面和一瓶可乐”），系统自动解析并计算摄入热量。

运动录入： 用户描述运动情况（如“慢跑了 40 分钟”、“卧推80kg 4x10”），系统估算并记录消耗热量。

热量缺口计算： 实时展示当日“摄入 - 消耗”的热量缺口，并给出进度反馈。

身体数据图表： 在用户界面按日显示用户的身高、体重、BMR、TDEE 等数据变化趋势，以可视化方式呈现用户的生理状态，让用户可以看到自己身体的变化。

### 2.3 智能教练咨询 (核心 Agent 场景)

动态规划： 当用户询问“我今晚还能吃顿烧烤吗？”时，系统需结合当日剩余热量配额给出建议，若超标则提供补救运动方案。用户询问“我想增肌该如何制定计划”时，系统需根据用户的目标体重和当前体重，结合用户的运动习惯，给出一个增肌的计划。

专业解答： 针对特定的营养学或营养学或运动损伤问题（如“硬拉时腰疼怎么回事”、“肌酸怎么吃”），系统需基于权威知识库给出专业、安全的解答。

情绪提供： 根据用户的执行情况，给予鼓励或严格的监督反馈。

## 3. 技术栈选型

### 3.1 前端展现层 (Frontend)

框架： Streamlit

原因： 纯 Python 编写，极速构建交互式 Web UI。非常适合大模型应用的原型验证和展示，能够轻松处理 Markdown 渲染、会话状态保持（Session State）以及简单的数据图表展示。

### 3.2 后端服务层 (Backend)

框架： FastAPI

原因： 高性能异步框架，易于构建 RESTful API。用于前后端分离，处理业务逻辑并与大模型接口进行交互。

### 3.3 AI 与大模型中间件

编排框架： LangChain + LangGraph

核心逻辑： 摒弃传统的无状态对话或死板的 AgentExecutor，全面采用 LangGraph 构建具备状态机（State Machine）特性的 ReAct Agent。

知识检索库 (RAG)： 使用 ChromaDB 或 FAISS 作为本地向量数据库，存储营养学指南、运动解剖学等文档。

LLM 接口： 兼容 OpenAI API 格式（智谱 GLM）。

### 3.4 数据持久化层

关系型数据库： SQLite 

ORM： SQLAlchemy

## 4. 核心系统架构设计 (LangGraph 工作流)

向 Trae 强调：后端的 Agent 必须使用 LangGraph 来管理状态。

### 4.1 全局状态定义 (Graph State)

定义一个继承自 TypedDict 的全局状态字典，贯穿整个图的执行周期。包含以下字段：

- `messages`: 历史对话列表 (List[BaseMessage])。
- `user_id`: 当前用户 ID。
- `user_profile`: 包含身高、体重、BMR、TDEE 的字典。
- `daily_stats`: 当日已摄入卡路里、已消耗卡路里、步数等。
- `is_final`: 是否已生成最终回复。

### 4.2 核心节点设计 (Nodes)

- `LLM Node (大模型节点)`: 负责接收当前 State 并进行推理。决定是调用工具（输出 ToolCall）还是直接回复用户（输出文本）。
- `Tool Node (工具执行节点)`: 当 LLM 决定调用工具时，路由到此节点执行具体的 Python 函数，并将结果附加到 messages 中。
- `Update State Node`: 将工具执行结果同步到数据库，并更新 State 中的 `daily_stats`。

### 4.3 提供给大模型的工具列表 (Tools)

这些工具需使用 LangChain 的 @tool 装饰器进行封装，包含清晰的 Docstring 供模型理解。

- `get_user_info(user_id)`: 获取用户的生理数据。
- `log_food_intake(user_id, food_items)`: 记录摄入的食物及其估算卡路里。
- `log_exercise_burn(user_id, activity, duration)`: 记录运动及其估算消耗。
- `get_daily_summary(user_id)`: 获取当日卡路里收支总结。
- `rag_medical_search(query)`: 检索增强工具，从本地知识库查询专业建议。

### 4.4 数据库设计 (Schema)

- **Users 表**: `id`, `height`, `weight`, `age`, `gender`, `target_weight`, `allergies`, `bmr`, `tdee`, `created_at`
- **DailyLogs 表**: `id`, `user_id`, `date`, `intake_calories`, `burn_calories`, `weight_log`, `notes`
- **FoodItems 表**: `id`, `log_id`, `name`, `calories` (记录每一项食物的细节)
- **ExerciseItems 表**: `id`, `log_id`, `type`, `duration`, `calories` (记录每一项运动的细节)

## 5. 给 Trae 的分步开发指令建议

请按照以下三个阶段进行代码生成：

1.阶段一：脚手架搭建。 帮我初始化 FastAPI 的项目结构，以及一个基础的 Streamlit 前端页面。实现用户基础数据的表单录入并保存在后端的 SQLite 中。

2.阶段二：LangGraph Agent 核心逻辑。 帮我定义 Graph State，实现 search\_food\_calories 和 estimate\_exercise\_burn 这两个基础工具（目前只需返回 mock 数据即可）。构建包含 LLM Node 和 Tool Node 的循环执行图，并对外暴露 API 供 Streamlit 调用。

3.阶段三：RAG 接入与前后端联调。 添加向量数据库查询工具 rag\_medical\_search 到大模型的可用工具列表中。完善 Streamlit 界面，使其呈现类似微信的聊天气泡框，并在侧边栏添加菜单，点击身体数据选项后弹出子页面展示卡路里、身体数据图表等。
