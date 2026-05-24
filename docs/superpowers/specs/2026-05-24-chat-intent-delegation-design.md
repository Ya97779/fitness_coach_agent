# Chat Agent 意图检测 + 前端确认 + 委托记录 设计

## 背景

路由不准确导致食物/运动相关消息被分到 chat agent，chat agent 没有记录能力，用户意图丢失。

## 方案

Chat agent 检测到记录意图后，通过 SSE 传递意图信息给前端，前端显示记录按钮，用户确认后直接调 API 记录。

## 流程

```
用户: "我晚餐吃了一份鸡腿饭"
  → 路由到 chat（误判）
  → chat agent 检测到食物记录意图
  → 返回: 文本回复 + 意图标记
  → 前端聊天底部弹出按钮: "记录鸡腿饭到饮食日志 (650 kcal)"
  → 用户点击 → 前端调用 /api/v1/food-log
  → 提示"已记录"
```

## 改动清单

### 1. 后端 chat_agent.py

system prompt 添加意图检测指令：

```
当你检测到用户有记录饮食或运动的意图时，在回复末尾单独一行输出意图标记：
[INTENT:food]食物名|餐次|估算热量
[INTENT:exercise]运动名|时长|估算热量

示例：
用户: "我晚餐吃了一份鸡腿饭"
回复: 鸡腿饭是经典快餐，味道不错！
[INTENT:food]鸡腿饭|dinner|650

用户: "今天跑步30分钟"
回复: 跑步是很好的有氧运动！
[INTENT:exercise]跑步|30|300

规则：
- 只在用户明确提到吃了/喝了/做了运动时才输出标记
- 闲聊、提问、咨询等不输出标记
- 热量用常识估算，不需要精确
- 餐次根据上下文推断：早餐/午餐/晚餐/加餐
```

### 2. 后端 graph.py

chat 节点处理后，检查 response 中是否有 `[INTENT:xxx]` 标记：

- 提取标记信息
- 从 response 中移除标记（用户看不到）
- 将意图信息附加到返回结果中

### 3. 后端 main.py

`/chat/stream` SSE 响应格式扩展：

```
data: {"type": "text", "content": "鸡腿饭是经典快餐，味道不错！"}
data: {"type": "intent", "intent": "food", "data": {"food_name": "鸡腿饭", "meal_type": "dinner", "calories": 650}}
```

`/chat` 非流式响应格式扩展：

```json
{
  "response": "鸡腿饭是经典快餐，味道不错！",
  "agent": "chat",
  "intent": {
    "type": "food",
    "data": {"food_name": "鸡腿饭", "meal_type": "dinner", "calories": 650}
  }
}
```

### 4. 前端 chat.js

- 解析 SSE 数据中的 `type: "intent"` 消息
- 存储意图信息到 data
- 用户点击记录按钮时：
  - food 类型 → 调用 `POST /api/v1/food-log`
  - exercise 类型 → 调用 `POST /api/v1/exercise-log`
- 记录成功后清除意图、显示 toast

### 5. 前端 chat.wxml

聊天底部条件显示记录按钮：

```xml
<view class="intent-bar" wx:if="{{pendingIntent}}">
  <view class="intent-btn" bindtap="recordFromIntent">
    <text class="intent-text">{{intentButtonText}}</text>
  </view>
</view>
```

### 6. 前端 chat.wxss

记录按钮样式：固定在聊天区域底部，醒目标色。

## 不改动的部分

- 路由逻辑（router.py）：保持不变
- nutrition agent / fitness agent：保持不变
- 数据库模型：无新增表或字段

## 边界情况

1. **用户不点击按钮**：意图信息在下次发消息时清除
2. **多次检测到意图**：后一次覆盖前一次
3. **chat agent 误检测**：用户不点击即可，无副作用
4. **热量不精确**：用户可以在记录前修改（编辑弹窗已有）
