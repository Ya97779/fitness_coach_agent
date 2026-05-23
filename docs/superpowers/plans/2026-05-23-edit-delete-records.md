# 饮食/运动记录编辑与删除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户可以编辑和删除今日记录的饮食和运动条目

**Architecture:** 后端新增 PATCH/DELETE 端点，前端首页增加点击编辑弹窗和左滑删除交互

**Tech Stack:** FastAPI, SQLAlchemy, 微信小程序原生

---

### Task 1: 后端 — 食物记录 PATCH/DELETE

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: 新增请求模型和端点**

在 `main.py` 的 `ExerciseLogCreate` 类之后添加：

```python
class FoodLogUpdate(BaseModel):
    name: Optional[str] = None
    calories: Optional[float] = None
    meal_type: Optional[str] = None

class ExerciseLogUpdate(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None
    sets: Optional[int] = None
    weight: Optional[float] = None
    duration: Optional[int] = None
    calories: Optional[float] = None
```

- [ ] **Step 2: 添加食物 PATCH 端点**

在 `create_exercise_log` 函数之前添加：

```python
@router.patch("/food-log/{item_id}", response_model=FoodLogResponse)
def update_food_log(
    item_id: int,
    data: FoodLogUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.FoodItem).join(models.DailyLog).filter(
        models.FoodItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    old_calories = item.calories or 0
    if data.name is not None:
        item.name = data.name
    if data.calories is not None:
        item.calories = data.calories
    if data.meal_type is not None:
        item.meal_type = data.meal_type

    log = db.query(models.DailyLog).get(item.log_id)
    if log and data.calories is not None:
        log.intake_calories = (log.intake_calories or 0) - old_calories + data.calories

    db.commit()
    db.refresh(item)
    return FoodLogResponse(
        id=item.id, name=item.name, calories=item.calories,
        meal_type=item.meal_type, log_id=item.log_id,
    )
```

- [ ] **Step 3: 添加食物 DELETE 端点**

```python
@router.delete("/food-log/{item_id}", status_code=204)
def delete_food_log(
    item_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.FoodItem).join(models.DailyLog).filter(
        models.FoodItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    log = db.query(models.DailyLog).get(item.log_id)
    if log:
        log.intake_calories = (log.intake_calories or 0) - (item.calories or 0)

    db.delete(item)
    db.commit()
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: 食物记录 PATCH/DELETE 端点"
```

---

### Task 2: 后端 — 运动记录 PATCH/DELETE

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: 添加运动 PATCH 端点**

在 `delete_food_log` 函数之后添加：

```python
@router.patch("/exercise-log/{item_id}", response_model=ExerciseLogResponse)
def update_exercise_log(
    item_id: int,
    data: ExerciseLogUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.ExerciseItem).join(models.DailyLog).filter(
        models.ExerciseItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    old_calories = item.calories or 0
    if data.type is not None:
        item.type = data.type
    if data.name is not None:
        item.name = data.name
    if data.sets is not None:
        item.sets = data.sets
    if data.weight is not None:
        item.weight = data.weight
    if data.duration is not None:
        item.duration = data.duration
    if data.calories is not None:
        item.calories = data.calories

    log = db.query(models.DailyLog).get(item.log_id)
    if log and data.calories is not None:
        log.burn_calories = (log.burn_calories or 0) - old_calories + data.calories

    db.commit()
    db.refresh(item)
    return ExerciseLogResponse(
        id=item.id, type=item.type, name=item.name, sets=item.sets,
        weight=item.weight, duration=item.duration,
        calories=item.calories, log_id=item.log_id,
    )
```

- [ ] **Step 2: 添加运动 DELETE 端点**

```python
@router.delete("/exercise-log/{item_id}", status_code=204)
def delete_exercise_log(
    item_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.ExerciseItem).join(models.DailyLog).filter(
        models.ExerciseItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    log = db.query(models.DailyLog).get(item.log_id)
    if log:
        log.burn_calories = (log.burn_calories or 0) - (item.calories or 0)

    db.delete(item)
    db.commit()
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: 运动记录 PATCH/DELETE 端点"
```

---

### Task 3: 前端 — 首页编辑弹窗 + 左滑删除

**Files:**
- Modify: `miniprogram/pages/home/home.js`
- Modify: `miniprogram/pages/home/home.wxml`
- Modify: `miniprogram/pages/home/home.wxss`

- [ ] **Step 1: 更新 home.js — 添加编辑/删除逻辑**

在 `data` 中添加：

```javascript
// 编辑弹窗
editModalVisible: false,
editType: '', // 'food' or 'exercise'
editItem: null,
editForm: {},
// 左滑
swipeIndex: -1,
```

在 `Page({})` 的 methods 中添加（`goFeedback` 之后）：

```javascript
// === 编辑 ===
onItemTap(e) {
  const { type, item } = e.currentTarget.dataset
  const editForm = type === 'food'
    ? { name: item.name, calories: item.calories, meal_type: item.meal_type }
    : { type: item.type, name: item.name || '', sets: item.sets || 1, weight: item.weight || '', duration: item.duration, calories: item.calories }
  this.setData({ editModalVisible: true, editType: type, editItem: item, editForm })
},

closeEditModal() {
  this.setData({ editModalVisible: false, editItem: null })
},

onEditInput(e) {
  const field = e.currentTarget.dataset.field
  let value = e.detail.value
  if (['calories', 'sets', 'weight', 'duration'].includes(field)) {
    value = parseFloat(value) || 0
  }
  this.setData({ [`editForm.${field}`]: value })
},

selectEditMeal(e) {
  this.setData({ 'editForm.meal_type': e.currentTarget.dataset.meal })
},

adjustEditSets(e) {
  const delta = parseInt(e.currentTarget.dataset.delta)
  let sets = (this.data.editForm.sets || 1) + delta
  if (sets < 1) sets = 1
  this.setData({ 'editForm.sets': sets })
},

saveEdit() {
  const { editType, editItem, editForm } = this.data
  const url = editType === 'food'
    ? `/api/v1/food-log/${editItem.id}`
    : `/api/v1/exercise-log/${editItem.id}`

  wx.showLoading({ title: '保存中...' })
  request({ url, method: 'PATCH', data: editForm }).then(() => {
    wx.hideLoading()
    wx.showToast({ title: '已更新', icon: 'success' })
    this.closeEditModal()
    this.loadData()
  }).catch(err => {
    wx.hideLoading()
    wx.showToast({ title: err.message || '更新失败', icon: 'none' })
  })
},

deleteFromEdit() {
  const { editType, editItem } = this.data
  wx.showModal({
    title: '确认删除',
    content: '删除后不可恢复',
    confirmColor: '#c47a6c',
    success: (res) => {
      if (res.confirm) {
        const url = editType === 'food'
          ? `/api/v1/food-log/${editItem.id}`
          : `/api/v1/exercise-log/${editItem.id}`
        request({ url, method: 'DELETE' }).then(() => {
          wx.showToast({ title: '已删除', icon: 'success' })
          this.closeEditModal()
          this.loadData()
        })
      }
    }
  })
},

// === 左滑删除 ===
onItemTouchStart(e) {
  this.setData({ touchStartX: e.touches[0].clientX })
},

onItemTouchEnd(e) {
  const startX = this.data.touchStartX || 0
  const endX = e.changedTouches[0].clientX
  const { type, index } = e.currentTarget.dataset
  if (startX - endX > 60) {
    this.setData({ swipeIndex: `${type}-${index}` })
  } else {
    this.setData({ swipeIndex: -1 })
  }
},

resetSwipe() {
  this.setData({ swipeIndex: -1 })
},

deleteItem(e) {
  const { type, item } = e.currentTarget.dataset
  wx.showModal({
    title: '确认删除',
    content: '删除后不可恢复',
    confirmColor: '#c47a6c',
    success: (res) => {
      if (res.confirm) {
        const url = type === 'food'
          ? `/api/v1/food-log/${item.id}`
          : `/api/v1/exercise-log/${item.id}`
        request({ url, method: 'DELETE' }).then(() => {
          wx.showToast({ title: '已删除', icon: 'success' })
          this.loadData()
        })
      }
    }
  })
},
```

- [ ] **Step 2: 更新 home.wxml — 编辑弹窗 + 左滑模板**

替换整个"今日记录"section（从 `<view class="section-head"` 到 `</view>` 闭合的运动 card）：

```xml
<!-- 今日记录 -->
<view class="section-head" wx:if="{{loggedIn}}">
  <text class="section-title">今日记录</text>
</view>

<!-- 饮食记录 -->
<view class="card" wx:if="{{loggedIn && foodItems.length > 0}}">
  <text class="record-group-label">饮食</text>
  <view class="record-item-wrap" wx:for="{{foodItems}}" wx:key="id"
    data-type="food" data-index="{{index}}"
    bindtouchstart="onItemTouchStart" bindtouchend="onItemTouchEnd">
    <view class="record-item record-item-swipe {{swipeIndex === 'food-' + index ? 'record-item-swiped' : ''}}"
      data-type="food" data-item="{{item}}" bindtap="onItemTap" catchtap="resetSwipe">
      <view class="record-dot dot-food"></view>
      <view class="record-info">
        <text class="record-name">{{item.name}}</text>
        <text class="record-meta">{{item.meal_type_text}}</text>
      </view>
      <text class="record-val" wx:if="{{item.calories > 0}}">{{item.calories}} kcal</text>
      <text class="record-val record-estimating" wx:else>计算中...</text>
    </view>
    <view class="record-delete-btn" wx:if="{{swipeIndex === 'food-' + index}}"
      data-type="food" data-item="{{item}}" catchtap="deleteItem">
      <text class="record-delete-text">删除</text>
    </view>
  </view>
</view>

<!-- 运动记录 -->
<view class="card" wx:if="{{loggedIn && exerciseItems.length > 0}}">
  <text class="record-group-label">运动</text>
  <view class="record-item-wrap" wx:for="{{exerciseItems}}" wx:key="id"
    data-type="exercise" data-index="{{index}}"
    bindtouchstart="onItemTouchStart" bindtouchend="onItemTouchEnd">
    <view class="record-item record-item-swipe {{swipeIndex === 'exercise-' + index ? 'record-item-swiped' : ''}}"
      data-type="exercise" data-item="{{item}}" bindtap="onItemTap" catchtap="resetSwipe">
      <view class="record-dot dot-exercise"></view>
      <view class="record-info">
        <text class="record-name">{{item.type}}</text>
        <text class="record-meta" wx:if="{{item.sets}}">{{item.sets}} 组{{item.weight ? ' · ' + item.weight + 'kg' : ''}}</text>
        <text class="record-meta" wx:else>{{item.duration}} 分钟</text>
      </view>
      <text class="record-val">-{{item.calories}} kcal</text>
    </view>
    <view class="record-delete-btn" wx:if="{{swipeIndex === 'exercise-' + index}}"
      data-type="exercise" data-item="{{item}}" catchtap="deleteItem">
      <text class="record-delete-text">删除</text>
    </view>
  </view>
</view>
```

在 `</view>` (page closing tag) 之前添加编辑弹窗：

```xml
<!-- 编辑弹窗 -->
<view class="edit-mask {{editModalVisible ? 'edit-mask-show' : ''}}" bindtap="closeEditModal">
  <view class="edit-sheet {{editModalVisible ? 'edit-sheet-show' : ''}}" catchtap="">
    <!-- 食物编辑 -->
    <block wx:if="{{editType === 'food'}}">
      <view class="edit-field">
        <text class="edit-label">食物名称</text>
        <input class="edit-input" value="{{editForm.name}}" data-field="name" bindinput="onEditInput" />
      </view>
      <view class="edit-field">
        <text class="edit-label">热量 (kcal)</text>
        <input class="edit-input" type="digit" value="{{editForm.calories}}" data-field="calories" bindinput="onEditInput" />
      </view>
      <view class="edit-field">
        <text class="edit-label">餐次</text>
        <view class="meal-types">
          <view class="meal-btn {{editForm.meal_type === 'breakfast' ? 'meal-active' : ''}}" bindtap="selectEditMeal" data-meal="breakfast">早餐</view>
          <view class="meal-btn {{editForm.meal_type === 'lunch' ? 'meal-active' : ''}}" bindtap="selectEditMeal" data-meal="lunch">午餐</view>
          <view class="meal-btn {{editForm.meal_type === 'dinner' ? 'meal-active' : ''}}" bindtap="selectEditMeal" data-meal="dinner">晚餐</view>
          <view class="meal-btn {{editForm.meal_type === 'snack' ? 'meal-active' : ''}}" bindtap="selectEditMeal" data-meal="snack">加餐</view>
        </view>
      </view>
    </block>

    <!-- 运动编辑 -->
    <block wx:if="{{editType === 'exercise'}}">
      <view class="edit-field">
        <text class="edit-label">运动类型</text>
        <input class="edit-input" value="{{editForm.type}}" data-field="type" bindinput="onEditInput" />
      </view>
      <view class="edit-field">
        <text class="edit-label">动作名称</text>
        <input class="edit-input" value="{{editForm.name}}" data-field="name" bindinput="onEditInput" />
      </view>
      <view class="edit-field">
        <text class="edit-label">组数</text>
        <view class="qty-stepper">
          <view class="stepper-btn" bindtap="adjustEditSets" data-delta="-1"><text class="stepper-icon">-</text></view>
          <input class="qty-input" type="number" value="{{editForm.sets}}" data-field="sets" bindinput="onEditInput" />
          <view class="stepper-btn" bindtap="adjustEditSets" data-delta="1"><text class="stepper-icon">+</text></view>
        </view>
      </view>
      <view class="edit-row">
        <view class="edit-field edit-field-half">
          <text class="edit-label">重量 (kg)</text>
          <input class="edit-input" type="digit" value="{{editForm.weight}}" data-field="weight" bindinput="onEditInput" />
        </view>
        <view class="edit-field edit-field-half">
          <text class="edit-label">时长 (分钟)</text>
          <input class="edit-input" type="number" value="{{editForm.duration}}" data-field="duration" bindinput="onEditInput" />
        </view>
      </view>
      <view class="edit-field">
        <text class="edit-label">热量 (kcal)</text>
        <input class="edit-input" type="digit" value="{{editForm.calories}}" data-field="calories" bindinput="onEditInput" />
      </view>
    </block>

    <view class="edit-actions">
      <view class="edit-save-btn" bindtap="saveEdit"><text class="edit-save-text">保存</text></view>
      <view class="edit-delete-link" bindtap="deleteFromEdit"><text class="edit-delete-link-text">删除此记录</text></view>
    </view>
  </view>
</view>
```

- [ ] **Step 3: 更新 home.wxss — 添加弹窗和滑动样式**

在文件末尾添加：

```css
/* 左滑删除 */
.record-item-wrap {
  position: relative;
  overflow: hidden;
}

.record-item-swipe {
  transition: transform 0.2s;
}

.record-item-swiped {
  transform: translateX(-140rpx);
}

.record-delete-btn {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 140rpx;
  background: #c47a6c;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
}

.record-delete-text {
  font-size: 26rpx;
  color: #fff;
  font-weight: 600;
}

/* 编辑弹窗 */
.edit-mask {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4);
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: all 0.25s;
}

.edit-mask-show {
  opacity: 1;
  visibility: visible;
}

.edit-sheet {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  background: var(--bg-card);
  border-radius: 24rpx 24rpx 0 0;
  padding: 40rpx 32rpx;
  padding-bottom: calc(40rpx + env(safe-area-inset-bottom));
  transform: translateY(100%);
  transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  z-index: 1001;
  max-height: 80vh;
  overflow-y: auto;
}

.edit-sheet-show {
  transform: translateY(0);
}

.edit-field {
  margin-bottom: 24rpx;
}

.edit-label {
  font-size: 22rpx;
  color: var(--text-hint);
  margin-bottom: 10rpx;
  display: block;
}

.edit-input {
  background: var(--bg-input);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-sm);
  padding: 18rpx 22rpx;
  font-size: 28rpx;
  color: var(--text-primary);
}

.edit-row {
  display: flex;
  gap: 16rpx;
}

.edit-field-half {
  flex: 1;
}

/* 步进器（复用 log 页样式） */
.qty-stepper {
  display: flex;
  align-items: center;
  background: var(--bg-input);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.stepper-btn {
  width: 72rpx;
  height: 72rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stepper-icon {
  font-size: 32rpx;
  color: var(--text-secondary);
  font-weight: 600;
}

.qty-input {
  flex: 1;
  text-align: center;
  font-size: 32rpx;
  font-weight: 600;
  color: var(--text-primary);
  height: 72rpx;
  background: transparent;
  border: none;
}

/* 餐次选择（复用 log 页样式） */
.meal-types {
  display: flex;
  gap: 12rpx;
}

.meal-btn {
  flex: 1;
  text-align: center;
  padding: 14rpx 0;
  font-size: 24rpx;
  color: var(--text-hint);
  background: var(--bg-card);
  border: 1rpx solid var(--border);
  border-radius: var(--radius-sm);
}

.meal-active {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-dim);
}

/* 弹窗操作按钮 */
.edit-actions {
  margin-top: 32rpx;
}

.edit-save-btn {
  background: var(--text-primary);
  border-radius: var(--radius-sm);
  padding: 22rpx;
  text-align: center;
}

.edit-save-text {
  font-size: 28rpx;
  font-weight: 700;
  color: #fff;
  letter-spacing: 2rpx;
}

.edit-delete-link {
  text-align: center;
  padding: 24rpx 0 0;
}

.edit-delete-link-text {
  font-size: 24rpx;
  color: var(--danger);
}
```

- [ ] **Step 4: Commit**

```bash
git add miniprogram/pages/home/home.js miniprogram/pages/home/home.wxml miniprogram/pages/home/home.wxss
git commit -m "feat: 首页记录编辑弹窗 + 左滑删除"
```
