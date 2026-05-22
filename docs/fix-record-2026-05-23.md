# 修复记录 - 2026-05-23

## 问题一：保存训练记录时数据库报错 `column users.goal does not exist`

### 原因

User 模型新增了 `goal` 字段，但生产 PostgreSQL 数据库未执行迁移，缺少该列。

### 解决

在生产数据库执行 ALTER TABLE：

```bash
PGPASSWORD=<密码> psql -U fitcoach -d fitcoach -h 127.0.0.1 -c "ALTER TABLE users ADD COLUMN IF NOT EXISTS goal VARCHAR;"
```

然后重启服务：

```bash
sudo systemctl restart fitcoach
```

---

## 问题二：保存训练记录时 `TypeError: 'StructuredTool' object is not callable`

### 原因

`estimate_exercise_calories` 在 `fitness_agent.py` 中被 `@tool` 装饰器包装为 LangChain 的 `StructuredTool` 对象。在 `main.py` 中直接用 `estimate_exercise_calories(...)` 调用会报错，因为 StructuredTool 不是普通函数。

### 解决

修改 `backend/app/main.py` 第 454 行，将直接调用改为 `.invoke()`：

```python
# 修改前
calories = data.calories if data.calories else estimate_exercise_calories(data.type, data.duration, "medium", body_weight)

# 修改后
calories = data.calories if data.calories else estimate_exercise_calories.invoke({"exercise_type": data.type, "duration": data.duration, "intensity": "medium", "user_weight": body_weight}).get("calories", 0)
```

### 经验总结

- `@tool` 装饰的函数是 `StructuredTool` 对象，调用需用 `.invoke({"参数名": 值})` 格式
- 数据库模型变更后需同步执行生产数据库迁移（ALTER TABLE）
