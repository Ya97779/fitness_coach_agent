# 记忆模块分析报告

## 📅 分析日期：2026-04-24

---

## 1. 长期记忆实现

### 1.1 用户画像（User Profile）

**存储位置**：SQLite `users` 表

**加载方式**：`UserProfileLoader.load_user_profile(user_id)`

```python
# 加载内容
{
    "user_id": int,
    "basic_info": {
        "height": float,      # cm
        "weight": float,      # kg
        "age": int,
        "gender": str
    },
    "body_metrics": {
        "bmr": float,        # 基础代谢率
        "tdee": float         # 每日总消耗
    },
    "goal": {
        "target_weight": float,
        "current_weight": float,
        "weight_diff": float
    },
    "constraints": {
        "allergies": str
    }
}
```

**注入方式**：
```python
# memory_manager.py
profile_text = memory.format_profile_for_agent()
# 输出格式：
# 【用户基本信息】
# - 身高: 175 cm
# - 体重: 70 kg
# ...
```

### 1.2 每日统计（Daily Stats）

**存储位置**：SQLite `daily_logs` 表 + `food_items` + `exercise_items`

**加载方式**：`StatsSummarizer.get_today_stats(user_id)`

```python
# 加载内容
{
    "date": "2026-04-24",
    "intake_calories": float,
    "burn_calories": float,
    "net_calories": float,
    "tdee": float,
    "calorie_balance": float,
    "food_count": int,
    "exercise_count": int,
    "food_items": [{"name": str, "calories": float}],
    "exercise_items": [{"type": str, "duration": int, "calories": float}]
}
```

### 1.3 每周统计（Weekly Stats）

**加载方式**：`StatsSummarizer.get_week_stats(user_id)`

```python
# 加载内容
{
    "week_start": str,
    "week_end": str,
    "days_logged": int,
    "total_intake": float,
    "total_burn": float,
    "avg_intake": float,
    "avg_burn": float,
    "days_below_tdee": int,
    "days_above_tdee": int,
    "daily_logs": [...]
}
```

---

## 2. 短期记忆实现

### 2.1 当前对话历史

**存储方式**：LangGraph `AgentState.messages`

**管理机制**：
- 消息列表在 LangGraph 工作流中传递
- 每个 Agent 节点可以访问完整消息历史
- 对话结束时消息生命周期结束（不持久化）

### 2.2 对话摘要机制

**实现方式**：`ConversationSummarizer`

```python
# 触发条件
MAX_MESSAGES_BEFORE_SUMMARY = 10  # 超过10条消息触发摘要

# 摘要策略
- 保留最近 5 条消息不变
- 对早期消息调用 LLM 生成摘要
- 摘要格式：
  1. 用户主要讨论的话题/问题：
  2. 达成的共识或建议：
  3. 用户的特殊需求或限制（如有）：
```

### 2.3 记忆摘要（Agent 上下文用）

**实现方式**：`MemoryManager.get_memory_summary()`

```python
# 返回内容
{
    "user_id": int,
    "goal": str,                    # "减脂" / "增肌" / "维持"
    "today_intake": float,           # 今日摄入 kcal
    "today_burn": float,             # 今日消耗 kcal
    "week_avg_intake": float         # 本周日均摄入 kcal
}
```

**格式化注入**：
```python
# 各 Agent 的 format_*_memory() 函数
format_nutrition_memory(memory_summary):
    "用户目标: 减脂\n今日已摄入: 1500 kcal\n今日剩余: ~500 kcal"

format_fitness_memory(memory_summary):
    "用户目标: 减脂\n今日已消耗: 300 kcal"
```

---

## 3. Agent 集成现状

### 3.1 集成流程

```
process_user_message()
    │
    ▼
MemoryManager(user_id) ──▶ get_memory_summary()
    │
    ▼
initial_state["memory_summary"] = {
    "goal": "减脂",
    "today_intake": 1500,
    "today_burn": 300,
    "week_avg_intake": 1800
}
    │
    ▼
agent_graph.invoke(initial_state)
    │
    ├──▶ router() ──▶ route_after_router()
    │                      │
    │                      ▼
    │                 chat() / nutrition() / fitness()
    │                      │
    │                      ▼
    │                 从 state 获取 memory_summary
    │                      │
    │                      ▼
    │                 format_*_memory(memory_summary)
    │                      │
    │                      ▼
    │                 注入到 System Prompt
    │                      │
    ▼                      ▼
   END               返回结果
```

### 3.2 各 Agent 注入情况

| Agent | memory_summary 注入 | 注入位置 |
|-------|-------------------|----------|
| Chat | ✅ 已实现 | System Prompt 末尾 |
| Nutrition | ✅ 已实现 | System Prompt 末尾 |
| Fitness | ✅ 已实现 | System Prompt 末尾 |

### 3.3 注入示例

```python
# nutrition_agent.py - nutrition_with_user()

# 原始 System Prompt（来自 AGENT_SYSTEM_PROMPTS["nutrition"]）
# + 记忆上下文

系统: 你是一位专业的营养师 AI 助手...

【用户基本信息】
- 身高: 175 cm
- 体重: 70 kg
- 年龄: 25 岁
- 性别: 男

【身体指标】
- 基础代谢率(BMR): 1750 kcal
- 每日总消耗(TDEE): 2200 kcal

用户目标: 减脂
今日已摄入: 1500 kcal
今日剩余: ~700 kcal
本周日均摄入: 1650 kcal
```

---

## 4. 记忆模块架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        记忆模块架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    长期记忆（持久化）                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │  用户画像    │  │  每日统计    │  │  每周统计    │     │  │
│  │  │   (users)   │  │ (daily_logs) │  │ (汇总计算)   │     │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │  │
│  │         │                  │                  │             │  │
│  │         ▼                  ▼                  ▼             │  │
│  │  ┌─────────────────────────────────────────────┐          │  │
│  │  │           UserProfileLoader                 │          │  │
│  │  │           StatsSummarizer                    │          │  │
│  │  └─────────────────────┬───────────────────────┘          │  │
│  └────────────────────────┼────────────────────────────────────┘  │
│                           │                                        │
│                           ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  MemoryManager                              │  │
│  │  • load_profile()        • get_full_context()             │  │
│  │  • format_profile()      • get_memory_summary()           │  │
│  │  • format_today_stats()  • enhance_system_prompt()        │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                            │                                       │
│                            ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    短期记忆（内存）                          │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │              ConversationSummarizer                  │ │  │
│  │  │  • should_summarize() - 超过10条触发                 │ │  │
│  │  │  • summarize_messages() - LLM 摘要                    │ │  │
│  │  │  • extract_key_info() - 提取对话要点                  │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                            │                                       │
│                            ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Agent System Prompt                      │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  原始 System Prompt                                 │  │  │
│  │  │  + 用户基本信息                                     │  │  │
│  │  │  + 今日统计（摄入/消耗/剩余）                       │  │  │
│  │  │  + 本周趋势                                         │  │  │
│  │  │  + 对话摘要（可选）                                 │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 待解决问题

### 5.1 跨会话对话记忆缺失 🔴

**问题描述**：
- 当前对话历史仅存在于内存中
- 用户关闭对话/重启服务后，历史对话完全丢失
- Agent 无法记住之前的讨论内容和达成的共识

**影响**：
- 用户每次开启新对话都需要重新说明背景
- 无法建立长期的用户偏好和需求记忆

**解决方案**：
```sql
-- 需要新增表
CREATE TABLE conversation_logs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    agent_type VARCHAR(50),
    user_message TEXT,
    agent_response TEXT,
    created_at DATETIME
);
```

### 5.2 记忆上下文信息量不足 🟡

**问题描述**：
- 当前 `memory_summary` 仅包含 4 个字段
- 缺少用户偏好、当前计划、特殊需求等关键信息

**当前返回**：
```python
{
    "user_id": 1,
    "goal": "减脂",
    "today_intake": 1500,
    "today_burn": 300,
    "week_avg_intake": 1800
}
```

**应该增加**：
```python
{
    "user_id": 1,
    "goal": "减脂",
    "target_weight": 65,
    "current_weight": 75,
    "tdee": 2200,
    "allergies": "海鲜",
    "today_intake": 1500,
    "today_burn": 300,
    "week_avg_intake": 1800,
    "preferences": ["喜欢吃鸡胸肉", "不喜欢的食物: 牛肉"],
    "current_plan": "每周4次训练",
    "exercise_history": ["深蹲", "硬拉"]  # 最近训练的肌肉群
}
```

### 5.3 `enhance_system_prompt()` 未被使用 🟡

**问题描述**：
- `MemoryManager.enhance_system_prompt()` 方法存在但未被调用
- 当前使用更简单的 `get_memory_summary()` + `format_*_memory()` 方式
- `enhance_system_prompt()` 提供了更完整的上下文注入（包括对话摘要）

**当前代码路径**：
```python
# graph.py
memory_manager = MemoryManager(user_id=user_id)
memory_summary = memory_manager.get_memory_summary()  # <-- 使用这个

# 而非
enhanced_prompt = memory_manager.enhance_system_prompt(base_prompt, agent_type, messages)
```

### 5.4 对话要点提取未完成 🟡

**问题描述**：
- `ConversationSummarizer.extract_key_info()` 方法存在
- 但返回的 `topics` 和 `goals` 未被正确传递到 Agent

**当前实现**：
```python
# memory_manager.py - enhance_system_prompt() 中
key_info = self.summarizer.extract_key_info(messages)
if key_info["topics"] or key_info["goals"]:
    enhanced_parts.append("\n【对话要点】")
    # ...
```

**问题**：这个逻辑在 `enhance_system_prompt()` 中，但该方法未被使用

### 5.5 用户偏好未持久化 🟡

**问题描述**：
- 用户的饮食偏好、运动偏好、过敏信息只在 `users` 表的 `allergies` 字段
- 真正的偏好（如喜欢吃什么、不喜欢什么）没有被记录
- 这些信息只能从对话中临时提取，无法跨会话记忆

---

## 6. 改进优先级

| 优先级 | 问题 | 预期收益 | 复杂度 |
|-------|------|---------|-------|
| P0 | 跨会话对话持久化 | 支持多轮对话记忆 | 高 |
| P1 | 丰富 memory_summary | Agent 更个性化 | 低 |
| P1 | 使用 enhance_system_prompt() | 注入更完整上下文 | 中 |
| P2 | 完成对话要点提取 | 记忆关键讨论 | 中 |
| P2 | 用户偏好持久化 | 长期偏好记忆 | 高 |

---

## 7. 总结

### 已实现
- ✅ 长期记忆：用户画像、每日统计、每周统计
- ✅ 短期记忆：当前对话、对话摘要机制
- ✅ Agent 集成：memory_summary 注入到各 Agent
- ✅ 上下文格式化：format_*_memory() 函数

### 待实现
- ❌ 跨会话对话记忆（最大缺失）
- ⚠️ 记忆上下文信息量不足
- ⚠️ enhance_system_prompt() 未被使用
- ⚠️ 对话要点提取未完成
- ⚠️ 用户偏好持久化

### 集成状态
记忆模块已经与 Agent 系统集成，但使用的是简化版的 `get_memory_summary()` 方式，`enhance_system_prompt()` 提供的更完整功能尚未启用。
