# 饮食/运动记录编辑与删除

## Context

用户在首页查看今日记录后，需要修改或删除已保存的饮食和运动记录。当前系统只支持创建和查看，无编辑/删除功能。

## 设计

### 后端 API（main.py）

新增 4 个端点，均校验 `item.log.user_id == current_user.id`：

**PATCH `/api/v1/food-log/{item_id}`**
- 请求体：`{ name?, calories?, meal_type? }`（全部可选）
- 更新 FoodItem 字段，同步调整 DailyLog.intake_calories（旧值扣减 + 新值增加）
- 返回更新后的 FoodLogResponse

**DELETE `/api/v1/food-log/{item_id}`**
- 硬删除 FoodItem，同步扣减 DailyLog.intake_calories
- 返回 204

**PATCH `/api/v1/exercise-log/{item_id}`**
- 请求体：`{ type?, name?, sets?, weight?, duration?, calories? }`（全部可选）
- 更新 ExerciseItem 字段，同步调整 DailyLog.burn_calories
- 返回更新后的 ExerciseLogResponse

**DELETE `/api/v1/exercise-log/{item_id}`**
- 硬删除 ExerciseItem，同步扣减 DailyLog.burn_calories
- 返回 204

### 前端（miniprogram/pages/home/）

**交互方式**：
- 点击记录 → 底部弹出编辑弹窗
- 左滑记录 → 右侧露出红色删除按钮，点击弹确认框后删除

**编辑弹窗**：
- 食物：名称输入、热量输入、餐次选择（早餐/午餐/晚餐/加餐）
- 运动：类型输入、名称输入、组数步进、重量输入、时长输入、热量输入
- 底部：保存按钮 + 删除按钮
- 点击蒙层关闭

**数据流**：
1. 编辑 → PATCH API → 成功后重新加载今日数据
2. 删除 → 确认弹窗 → DELETE API → 成功后重新加载今日数据

### 文件变更

| 文件 | 变更 |
|------|------|
| `backend/app/main.py` | 新增 4 个端点 + 2 个 Pydantic 模型 |
| `miniprogram/pages/home/home.js` | 新增编辑/删除逻辑、弹窗状态、滑动处理 |
| `miniprogram/pages/home/home.wxml` | 新增编辑弹窗、左滑删除模板 |
| `miniprogram/pages/home/home.wxss` | 新增弹窗、滑动、删除按钮样式 |
