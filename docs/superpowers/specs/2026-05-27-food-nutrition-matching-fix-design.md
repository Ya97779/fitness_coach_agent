# 食物营养检索匹配修复设计

## 问题

用户说"我晚上吃了一根香蕉"，系统出现三个不同的热量值：

| 来源 | 热量 | 原因 |
|------|------|------|
| API 检索结果 | 49 kcal | `food_list[0]` 取到"红香蕉苹果"而非"香蕉[甘蕉]"(91kcal) |
| 兜底记录 | 105 kcal | `_FOOD_CALORIE_ESTIMATES["香蕉"]` 硬编码值 |
| API 正确值 | 91 kcal | "香蕉[甘蕉]"的 rl 字段 |

根因：
1. `food_api.py` 的 `search_food_nutrient()` 用 `food_list[0]` 取第一条结果，不做匹配
2. `nutrition_agent.py` 的兜底记录机制完全无视 API 已查到的数据，直接用硬编码估算

## 修复方案

### 修改 1：food_api.py — 精确匹配优先

在 `search_food_nutrient()` 中增加 `_select_best_match()` 函数，替换 `food_list[0]`：

```python
def _select_best_match(food_list: list, query: str) -> dict:
    """从 API 返回的食物列表中选择最匹配的结果"""
    # 1. 精确匹配（name 去掉方括号后缀后等于 query）
    for item in food_list:
        clean_name = item['name'].split('[')[0].strip()
        if clean_name == query or item['name'] == query:
            return item
    # 2. 包含匹配（query 在 name 中，选最短的 = 最精确的）
    candidates = [i for i in food_list if query in i['name']]
    if candidates:
        return min(candidates, key=lambda x: len(x['name']))
    # 3. 兜底取第一条
    return food_list[0]
```

调用处改为：
```python
nutrient = _select_best_match(food_list, food_name)
```

### 修改 2：nutrition_agent.py — 兜底复用 API 数据

当前兜底逻辑（`_extract_food_names` → `_estimate_calories`）完全不查 API。修改为：

```python
# 兜底记录时，先查 API，查不到再用硬编码
for food_name, meal_type in food_items:
    from ..food_api import search_food_nutrient
    api_result = search_food_nutrient(food_name)
    if api_result:
        calories = float(api_result['calories'])
        protein = api_result.get('protein', 0)
        fat = api_result.get('fat', 0)
        carbs = api_result.get('carbs', 0)
    else:
        calories = float(_estimate_calories(food_name))
        protein, fat, carbs = 0, 0, 0
    # 记录时传入完整营养数据
    log_food_intake.invoke({
        "user_id": user_id,
        "food_name": food_name,  # 用原始输入名，不用 API 返回名
        "calories": calories,
        "meal_type": meal_type,
        "protein": protein,
        "fat": fat,
        "carbs": carbs
    })
```

食物名用用户原始输入（如"香蕉"），不用 API 返回的"香蕉[甘蕉]"。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `backend/app/food_api.py` | 新增 `_select_best_match()`，替换 `food_list[0]` |
| `backend/app/agents/nutrition_agent.py` | 兜底记录先查 API，传入完整营养数据 |

## 验证

1. 搜"香蕉" → 应匹配"香蕉[甘蕉]"(91kcal)，不再匹配"红香蕉苹果"(49kcal)
2. 兜底记录"香蕉" → 应使用 API 返回的 91kcal，不再用硬编码 105kcal
3. API 查不到的食物 → 兜底仍用硬编码估算（向后兼容）
