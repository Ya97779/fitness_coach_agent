# 项目开发状态：智能私教营养师 & 健身教练

## 📅 最后更新日期：2026-04-17

---

## � 项目目录结构 (Project Structure)

```
d:\fitness_coach/
├── backend/                    # 后端服务
│   └── app/
│       ├── agent.py            # LangGraph 智能体核心逻辑
│       ├── database.py         # SQLAlchemy 数据库连接配置
│       ├── main.py             # FastAPI 主入口
│       ├── models.py           # 数据库模型定义
│       └── rag_utils.py        # RAG 检索工具
├── frontend/                   # 前端应用
│   └── app.py                  # Streamlit 前端界面
├── chroma_db/                  # ChromaDB 向量数据库存储
├── knowledge_base/             # RAG 知识库（PDF/Word/图片）
├── .gitignore                  # Git 忽略配置
├── Development.md              # 开发状态文档
├── design.md                   # 设计文档
├── README.md                   # 项目说明
├── checkpoints.db              # LangGraph 对话状态持久化
├── fitness_coach.db            # 业务数据 SQLite 数据库
└── requirements.txt            # Python 依赖列表
```

---

## �🚀 已完成功能 (Completed Features)

### 1. 后端架构 (Backend Architecture)
- **框架**: FastAPI (高性能，异步)。
- **数据库**: SQLite 搭配 SQLAlchemy ORM (涵盖用户、每日日志、食物条目、运动条目)。
- **智能体工作流**: 使用 **LangGraph** 构建的状态机 (ReAct Agent)。
- **持久化**: 使用 **AsyncSqliteSaver** 实现基于线程的历史记录持久化对话记忆。
- **大模型集成**: 通过 OpenAI 兼容 API 支持 **智谱 AI GLM-4.7**。

### 2. RAG 系统 (RAG System)
- **向量数据库**: 使用 ChromaDB 进行本地知识存储。
- **嵌入模型 (Embedding)**: 使用 **智谱 AI embedding-2** 模型进行向量化。
- **文档加载**: 支持 `./knowledge_base` 路径下的 PDF、Word (.docx) 以及图片（通过 pytesseract 实现 OCR）。
- **检索功能**: 集成了 `rag_medical_search_tool` 用于获取专业的健身与营养建议。

### 3. 前端 UI (Frontend UI)
- **框架**: Streamlit。
- **设计风格**: 仿照 OpenAI ChatGPT 的现代化深色主题界面。
- **侧边栏**: 包含"新对话"按钮、导航菜单（智能教练、个人档案、数据统计）和快捷操作按钮。
- **聊天界面**: 
  - 支持 **流式输出 (Streaming)**，提供打字机效果
  - 消息气泡采用圆角设计，用户消息右对齐，助手消息左对齐
  - 初始欢迎消息展示功能介绍和使用示例
- **个人资料**: 卡片式布局展示用户生理数据（身高、体重、年龄、BMR、TDEE、BMI）。
- **数据可视化**: 通过 Plotly 展示摄入/消耗热量趋势图和每日热量缺口柱状图。
- **快捷操作**: 提供"记录早餐"、"记录运动"、"热量查询"、"训练建议"等一键操作。

### 4. API 接口 (API Endpoints)

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/user/` | 创建用户（计算 BMR/TDEE） |
| GET | `/user/{user_id}` | 获取用户信息 |
| GET | `/user/{user_id}/logs` | 获取用户所有日志 |
| GET | `/user/{user_id}/today` | 获取当日日志 |
| POST | `/chat` | 非流式聊天接口 |
| POST | `/chat/stream` | 流式聊天接口 |

### 5. Agent 工具列表 (Tools)

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `get_user_info` | 获取用户生理数据 | `user_id` |
| `log_food_intake` | 记录食物摄入 | `user_id`, `food_name`, `calories` |
| `log_exercise_burn` | 记录运动消耗 | `user_id`, `activity_type`, `duration`, `calories` |
| `get_daily_summary` | 获取当日卡路里总结 | `user_id` |
| `search_food_calories` | 查询食物热量（天行数据API） | `food_name` |
| `estimate_exercise_burn` | 估算运动消耗（模拟） | `exercise_type`, `duration` |
| `rag_medical_search_tool` | 专业健康知识检索 | `query` |

### 6. 食物热量 API (food_api.py)

使用**天行数据**提供的食物营养信息 API：

**API 配置**：
- 接口地址：`https://apis.tianapi.com/nutrient/index`
- 请求方式：POST
- API Key：存储在 `.env` 文件中 (`TianxingFood_API_KEY`)

**功能特性**：
- 通过真实 API 查询食物热量和营养信息
- 返回热量、蛋白质、脂肪、碳水化合物
- 内置 fallback 机制，API 不可用时使用本地数据

**使用示例**：
```python
from .food_api import search_food_nutrient, get_food_details

# 查询食物营养信息
result = search_food_nutrient("苹果")
# 返回: {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14, "source": "天行数据API"}

# 获取详细营养信息（格式化输出）
details = get_food_details("油条")
# 输出：🍽️ 油条
# 数据来源: 天行数据API
# 🔥 热量: 385 kcal/100g
# 💪 蛋白质: 6 g
# 🥑 脂肪: 17 g
# 🍞 碳水: 51 g
```

### 7. 数据库模型 (Database Schema)

**Users 表**: 用户基础信息
- `id`, `height`, `weight`, `age`, `gender`, `target_weight`, `allergies`, `bmr`, `tdee`, `created_at`

**DailyLogs 表**: 每日日志
- `id`, `user_id`, `date`, `intake_calories`, `burn_calories`, `weight_log`, `notes`

**FoodItems 表**: 食物条目
- `id`, `log_id`, `name`, `calories`

**ExerciseItems 表**: 运动条目
- `id`, `log_id`, `type`, `duration`, `calories`

---

## 📈 项目进度 (Project Progress)

- [√] 第一阶段：基础 FastAPI + SQLite 搭建。
- [√] 第二阶段：基于模拟工具 (Mock tools) 的 LangGraph Agent。
- [√] 第三阶段：使用 SqliteSaver 实现异步持久化记忆。
- [√] 第四阶段：多格式 RAG 系统 (PDF/Word/图片)。
- [√] 第五阶段：聊天流式输出。
- [√] **第六阶段**: 类 ChatGPT 风格的 UI 重新设计。

---

## 🛠️ 运行方式 (Quick Start)

```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端服务
uvicorn backend.app.main:app --reload

# 启动前端（新终端）
streamlit run frontend/app.py
```

---

## 🛠️ 未来优化 (Upcoming Enhancements)
- [✓] 集成真实的营养/食物数据库 API（已完成：天行数据API）。
- [ ] 根据用户目标自动生成健身计划。
- [ ] 语音输入功能，用于记录食物和运动。
- [ ] 导出月度健康报告。
- [ ] 增强 UI 交互，添加更好的动画和主题支持。