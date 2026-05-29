# Agent 工具 DB 缓存层设计

> 日期: 2026-05-29

## 背景

nutrition agent 的 `search_food_nutrition` 工具直接调 `food_api`，从不读写 `FoodCalorieCache` DB 表。导致聊天中查询过的食物热量不持久化，每次都要重新调外部 API。此外 `update_food_log` 的 LLM 估算路径不回写缓存。

## 改动范围

| 文件 | 改动 |
|------|------|
| `backend/app/agents/nutrition_agent.py` | `search_food_nutrition` 工具加 DB 缓存查找 + 回写；`_get_food_nutrition` 同步改造 |
| `backend/app/main.py` | `update_food_log` LLM 估算后回写 FoodCalorieCache |

## 不动的部分

- `ExerciseCalorie` 缓存链已完整（硬编码 → DB → LLM → 回写）
- `food_api.py` 内存缓存保留
- `FoodCalorieCache` 表结构不变

## nutrition_agent.py 改动

### search_food_nutrition 工具

当前流程：
```
search_food_nutrient(food_name) → food_api → 返回
```

改为：
```
1. 查 FoodCalorieCache（name 匹配，无份量限制）
2. 命中 → 直接返回（来源标记 DB 缓存）
3. 未命中 → search_food_nutrient(food_name) → food_api
4. API 返回结果 → 回写 FoodCalorieCache（source="api"）
5. API 未找到 → 返回空，让 LLM 自行估算
```

### _get_food_nutrition 函数（fallback 记录用）

当前流程：
```
search_food_nutrient(food_name) → food_api → 未找到则用本地估算
```

改为：
```
1. 查 FoodCalorieCache
2. 命中 → 直接返回
3. 未命中 → search_food_nutrient → food_api
4. API 返回 → 回写 FoodCalorieCache
5. API 未找到 → 本地估算（不写 DB，因为不准确）
```

### DB 查找逻辑

复用 main.py 中 `_find_cached` 的模糊匹配思路，但简化为：
- 精确匹配：`name == food_name AND portion_qty IS NULL`
- 不做份量换算（agent 场景不涉及具体份量）

## main.py 改动

### update_food_log LLM 估算回写

当前 `_bg_estimate_update`（line ~640）只更新 FoodItem，不写 FoodCalorieCache。

改为：LLM 估算成功后，额外写入 FoodCalorieCache（与 create 路径一致）。

## 数据流对比

### 改造前
```
用户问"鸡蛋热量" → search_food_nutrition → food_api(144kcal) → 返回
用户再问"鸡蛋热量" → search_food_nutrition → food_api(144kcal) → 返回（重复 API 调用）
```

### 改造后
```
用户问"鸡蛋热量" → search_food_nutrition → DB miss → food_api(144kcal) → 回写 DB → 返回
用户再问"鸡蛋热量" → search_food_nutrition → DB hit(144kcal) → 直接返回（无 API 调用）
```
