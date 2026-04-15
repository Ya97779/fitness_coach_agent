# 项目目规划文档：智能私人营养师与健身教练 Agent

## 1. 项目概述

本项目旨在开发一款基于大语言模型（LLM）驱动的“私人营养师与健身教练” Web 应用程序。核心系统采用 Agent 架构，不仅具备多轮对话能力，更能通过记忆用户的生理状态、调用外部工具（计算卡路里、查询食物热量等）以及检索专业健康文献，为用户提供长期、个性化的饮食与运动规划闭环服务。

## 2. 产品功能需求 (PRD)

### 2.1 用户状态管理

支持输入并持久化存储用户的生理基准数据（身高、体重、年龄、性别、目标体重、运动习惯）。

计算并维护用户的每日基础代谢率（BMR）和每日总能量消耗（TDEE）。
- **BMR (Mifflin-St Jeor Equation)**:
  - 男: 10 * weight(kg) + 6.25 * height(cm) - 5 * age(y) + 5
  - 女: 10 * weight(kg) + 6.25 * height(cm) - 5 * age(y) - 161
- **TDEE**: BMR * Activity Factor (默认 1.2 - 1.55)

### 2.2 日常打卡与数据追踪

饮食录入： 用户通过自然语言描述日常饮食（如“今天中午吃了一碗兰州拉面和一瓶可乐”），系统自动解析并计算摄入热量。

运动录入： 用户描述运动情况（如“慢跑了 40 分钟”、“卧推80kg 4x10”），系统估算并记录消耗热量。

热量缺口计算： 实时展示当日“摄入 - 消耗”的热量缺口，并给出进度反馈。

身体数据图表： 在用户界面按日显示用户的身高、体重、BMI、每日基础代谢率、每日总能量消耗、每日热量缺口 等数据变化趋势，以可视化方式呈现用户的生理状态，让用户可以看到自己身体的变化。

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

持久化记忆： 使用 `langgraph-checkpoint-sqlite` 提供的 `SqliteSaver` 实现基于 `thread_id` 的持久化对话记忆。

知识检索库 (RAG)： 使用 ChromaDB 作为本地向量数据库，通过智谱 AI 的 `embedding-2` 模型进行向量化。支持从 `./knowledge_base` 目录自动加载 PDF、Word、图片 (OCR) 等多格式文档。采用 `langchain-chroma` 库实现持久化存储与检索。

### 3.4 数据持久化层

关系型数据库： SQLite (用于存储用户数据、每日日志、食物和运动记录)

ORM： SQLAlchemy

对话持久化： SQLite (单独的 `checkpoints.db` 用于存储 LangGraph 的对话状态)

# 4. 核心系统架构设计 (LangGraph 工作流)

向 Trae 强调：后端的 Agent 必须使用 LangGraph 来管理状态。

### 4.1 全局状态定义 (Graph State)

定义一个继承自 TypedDict 的全局状态字典，贯穿整个图的执行周期。包含以下字段：

- `messages`: 历史对话列表 (List[BaseMessage])。
- `user_id`: 当前用户 ID。
- `user_profile`: 包含身高、体重、BMR、TDEE 的字典。
- `daily_stats`: 当日已摄入热量、已消耗热量、步数等。

### 4.2 核心节点设计 (Nodes)

- `LLM Node (大模型节点)`: 负责接收当前 State 并进行推理。决定是调用工具（输出 ToolCall）还是直接回复用户（输出文本）。
- `Tool Node (工具执行节点)`: 当 LLM 决定调用工具时，路由到此节点执行具体的 Python 函数，并将结果附加到 messages 中。

### 4.3 提供给大模型的工具列表 (Tools)

这些工具已使用 LangChain 的 @tool 装饰器进行封装，包含清晰的 Docstring 供模型理解。

- `get_user_info(user_id)`: 获取用户的生理数据。
- `log_food_intake(user_id, food_name, calories)`: 记录摄入的食物及其估算卡路里。
- `log_exercise_burn(user_id, activity_type, duration, calories)`: 记录运动及其估算消耗。
- `get_daily_summary(user_id)`: 获取当日卡路里收支总结。
- `search_food_calories(food_name)`: 模拟接口，根据食物名称返回估算的卡路里。
- `estimate_exercise_burn(exercise_type, duration)`: 模拟接口，根据运动类型和时长返回消耗卡路里。
- `rag_medical_search_tool(query)`: 检索增强工具，从本地知识库查询专业建议。

### 4.4 数据库设计 (Schema)

- **Users 表**: `id`, `height`, `weight`, `age`, `gender`, `target_weight`, `allergies`, `bmr`, `tdee`, `created_at`
- **DailyLogs 表**: `id`, `user_id`, `date`, `intake_calories`, `burn_calories`, `weight_log`, `notes`
- **FoodItems 表**: `id`, `log_id`, `name`, `calories` (记录每一项食物的细节)
- **ExerciseItems 表**: `id`, `log_id`, `type`, `duration`, `calories` (记录每一项运动的细节)

## 5. 当前进度与未来优化 (Update: 2026-04-15)

### 已完成：
- ✅ **阶段一**：初始化 FastAPI 结构，实现 SQLite 存储用户信息。
- ✅ **阶段二**：构建 LangGraph Agent 核心逻辑，实现基础工具（MOCK）与 LLM 节点的循环图。
- ✅ **阶段三**：接入 ChromaDB 实现 RAG 专业解答，完善 Streamlit 聊天界面与数据可视化。
- ✅ **持久化升级**：将对话记忆从内存 `MemorySaver` 升级为持久化的 `SqliteSaver` (`checkpoints.db`)。
- ✅ **RAG 增强**：支持从 `./knowledge_base` 目录自动加载 PDF、Word、图片等多种格式文档并入库。

### 未来优化：
- 🛠️ 接入真实的食物卡路里查询 API。
- 🛠️ 完善 RAG 知识库，支持更多专业文档的自动入库，并实现增量更新。
- 🛠️ 增加用户运动计划的自动生成与跟踪功能。
- 🛠️ 增强 UI 体验，支持语音输入饮食/运动记录。
