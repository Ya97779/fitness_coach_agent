# 食物热量缓存 v2

## 数据定位

`food_calorie_cache` 保存用于快速估算的热量参考值，不是用户饮食记录，也不是
医疗或食品标签数据库。实际食材品种、品牌、烹饪方式和可食部比例都会造成差异，
复杂菜品仍应让用户补充重量、份量或包装营养标签。

基础数据位于 `backend/data/common_food_calories.json`，首批覆盖主食、肉蛋奶、
豆制品、水果、蔬菜、坚果和常见饮料。数值采用统一口径：

- 固体食物优先保存为每 100g；
- 饮料可按每 ml 换算；
- 鸡蛋、苹果、香蕉、米饭等同时保存常用单位参考值；
- 每份参考值只在用户单位明确匹配时使用。

数据表达遵循国家卫生健康委员会
[WS/T 464—2015《食物成分数据表达规范》](https://www.nhc.gov.cn/wjw/yingyang/201505/3cbe4ecd6e48465899557a25a5ae1be9.shtml)，
基础热量参考主要来自美国农业部
[FoodData Central](https://fdc.nal.usda.gov/)。种子数据经过中文名称和常用份量映射，
以 `curated_reference_v1` 标记，并进行整数化，便于消费级饮食记录使用。

## 导入和更新

生产迁移会自动导入：

```bash
/var/www/fitcoach/venv/bin/python scripts/migrate_phase02.py
```

如果数据库已经是 v2，仅重新导入基础数据，可以执行：

```bash
/var/www/fitcoach/venv/bin/python scripts/seed_food_calorie_cache.py
```

导入采用 upsert，可以重复运行，不会产生重复行。优先级为：

```text
人工维护 > 基础参考数据 > 外部 API > LLM 估算 > 本地降级值
```

因此重新部署可以更新基础参考值，但不会覆盖管理员人工校正过的数据。

## 验证

```sql
SELECT source, COUNT(*)
FROM food_calorie_cache
GROUP BY source
ORDER BY source;

SELECT name, basis_type, portion_qty, portion_unit, calories, source
FROM food_calorie_cache
WHERE normalized_name IN ('苹果', '鸡蛋', '米饭')
ORDER BY normalized_name, basis_type, portion_unit;
```
