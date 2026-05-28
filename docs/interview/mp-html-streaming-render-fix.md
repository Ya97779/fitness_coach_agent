# mp-html 流式输出后不渲染 markdown 问题

## 现象

小程序 chat 页面，AI 流式输出完成后，显示的是纯文本而非渲染后的 markdown。需要切换到别的页面再回来才能渲染成功。有时还伴随页面跳动。

## 根因分析

### 背景：流式渲染分两阶段

1. **流式中**：`updateAiMessage` 设置 `_streaming: true`，只更新纯文本 `content`，不解析 markdown（避免高频 setData 卡顿）
2. **流式完成**：`finishAiMessage` 设置 `_streaming: false` + `html: parseMarkdown(content)`，切到 mp-html 渲染

### WXML 条件

```html
<mp-html wx:if="{{item.role !== 'user' && !item._streaming && item.html}}" content="{{item.html}}" />
<text wx:elif="{{item.role !== 'user'}}">{{item.content}}</text>
```

流式完成时 `_streaming=false` + `html` 有值 → 条件满足 → mp-html 组件通过 `wx:if` 创建。

### 核心问题：property observer 不触发初始值

微信小程序的 property `observer` **只在值变化时触发，初始值不触发**。当 `wx:if` 从 false 变 true 时组件新创建，`content` 的初始值不会调用 `observer`，所以 `setContent()` 不执行，组件不渲染。

切换页面再回来时，`loadMessagesFromCache` 重新 `setData`，此时 mp-html 已存在，`content` 值变化触发 observer，所以能渲染。

### 页面跳动问题

`sendMessage` 设置 `scrollToId` 滚动到 AI 消息位置，但 `finishAiMessage` 的 `setData` 没有清除 `scrollToId`，导致 `scroll-into-view` 重新触发滚动。

---

## 修复方案

### 最终方案：selectComponent 手动调用 setContent

分两步 setData，绕开 observer 问题：

```javascript
finishAiMessage(msgId) {
  // 第一步：关闭 streaming，清除 scrollToId，html 设为空
  const step1 = this.data.messages.map(m => {
    if (m.id === msgId) {
      return { ...m, loading: false, _streaming: false, html: '' }
    }
    return m
  })
  this.setData({ messages: step1, sending: false, scrollToId: '' })

  // 第二步：mp-html 创建后，手动调用 setContent
  wx.nextTick(() => {
    const comp = this.selectComponent(`#mp-${msgId}`)
    if (comp && htmlContent) {
      comp.setContent(htmlContent)
    }
    // 同步更新 data 中的 html（用于缓存）
    const step2 = this.data.messages.map(m => {
      if (m.id === msgId) return { ...m, html: htmlContent }
      return m
    })
    this.setData({ messages: step2 })
    this.saveMessagesToCache()
  })
}
```

**原理**：
1. 第一步 `html=''` + `_streaming=false` → `wx:if` 条件中 `!item.html` 为 true → 条件不满足 → text 显示
2. 第二步 `nextTick` 后 `html` 有值 → 条件满足 → mp-html 创建
3. `selectComponent` 直接获取组件实例调用 `setContent`，不依赖 observer

**同样的方案应用于**：
- `loadMessagesFromCache` — 从缓存恢复消息时
- `syncMessagesFromServer` — 从服务端同步消息时

---

## 已尝试的失败方案（供参考）

| 方案 | 思路 | 结果 |
|------|------|------|
| attached 直接调用 setContent | 组件挂载时检查 content 并初始化 | 失败：attached 时 content 可能未设置 |
| attached + setTimeout(0) | 延迟检查 properties.content | 失败：原因未知 |
| hidden 替代 wx:if | 组件始终存在，CSS 控制显隐 | 失败：mp-html 完全不渲染 |

## 关键教训

1. **微信小程序 property observer 不触发初始值** — `wx:if` 动态创建的组件，必须用 `selectComponent` 手动初始化
2. **scroll-into-view 会重复触发** — 修改消息内容的 `setData` 如果不清除 `scrollToId`，会导致页面跳动
3. **分两步 setData** — 第一步创建组件，第二步设置内容，避免组件创建时 content 初始值不触发 observer
4. **流式渲染要分阶段** — 流式中用纯文本，完成后一次性解析 markdown

## 涉及文件

- `miniprogram/components/mp-html/index.js` — 组件定义
- `miniprogram/pages/chat/chat.wxml:18` — mp-html 的 wx:if 条件
- `miniprogram/pages/chat/chat.js` — finishAiMessage、loadMessagesFromCache、syncMessagesFromServer
