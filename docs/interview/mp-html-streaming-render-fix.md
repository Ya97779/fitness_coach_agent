# mp-html 流式输出后不渲染 markdown 问题

## 现象

小程序 chat 页面，AI 流式输出完成后，显示的是纯文本而非渲染后的 markdown。需要切换到别的页面再回来才能渲染成功。

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

---

## 已尝试的修复方案（全部失败）

### 方案 1：attached 生命周期直接调用 setContent

```javascript
attached: function() {
  if (this.data.content) this.setContent(this.data.content)
}
```

**结果：失败。** 组件 attached 时 `this.data.content` 可能还未从外部设置。

### 方案 2：attached + setTimeout(0) + properties 检查

```javascript
attached: function() {
  var self = this;
  setTimeout(function() {
    var c = self.properties.content || self.data.content;
    if (c) self.setContent(c);
  }, 0)
}
```

**结果：失败。** 原因未知，可能 setTimeout(0) 时 properties 仍未设置完成。

### 方案 3：用 hidden 替代 wx:if

```html
<mp-html wx:if="{{item.role !== 'user'}}" hidden="{{item._streaming || !item.html}}" content="{{item.html}}" />
```

思路：组件始终存在（只要 role !== 'user'），用 CSS hidden 控制显隐。content 从空变有值时 observer 触发。

**结果：失败。** mp-html 完全不渲染，连"切页面回来"都不行了。hidden 属性可能干扰了 mp-html 组件的内部渲染机制。

---

## 待验证的假设

1. ~~observer 不触发初始值~~ — 已确认是根因，但 attached 兜底方案未能解决
2. mp-html 组件的 `setContent` 方法可能在组件 hidden 或刚创建时无法正确工作
3. 需要确认 `parseMarkdown` 返回的 HTML 是否正确传到了 mp-html 的 content 属性
4. 可能需要从 finishAiMessage 中用 `this.selectComponent` 直接调用组件方法

## 下一步排查方向

1. 在 finishAiMessage 中加 `console.log` 确认 `html` 值是否正确
2. 用小程序开发者工具的调试器检查 mp-html 组件实例的 data/properties
3. 尝试用 `this.selectComponent('#mp-xxx')` 手动调用 `setContent`
4. 考虑换用 `rich-text` 组件替代 mp-html（功能少但更可靠）

## 涉及文件

- `miniprogram/components/mp-html/index.js` — 组件定义，attached 生命周期
- `miniprogram/pages/chat/chat.wxml:18` — mp-html 的 wx:if 条件
- `miniprogram/pages/chat/chat.js:187` — finishAiMessage
