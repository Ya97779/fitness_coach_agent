# mp-html 流式输出后不渲染 markdown 问题

## 现象

小程序 chat 页面，AI 流式输出完成后，显示的是纯文本而非渲染后的 markdown。切换到其他页面再切回来才正常渲染。

## 根因

两层问题叠加：

### 1. 流式过程中不解析 markdown（设计如此）

`updateAiMessage` 在流式过程中只更新纯文本，设置 `_streaming: true`，此时 WXML 走 `text` 分支：

```html
<mp-html wx:if="{{item.role !== 'user' && !item._streaming && item.html}}" content="{{item.html}}" />
<text wx:elif="{{item.role !== 'user'}}">{{item.content}}</text>
```

这是正确的——高频 setData 中解析 markdown 会卡顿。

### 2. 流式完成后 mp-html 不渲染（bug）

`finishAiMessage` 同时设置 `_streaming: false` 和 `html: parseMarkdown(content)`，WXML 条件满足，`mp-html` 组件通过 `wx:if` 创建。

**但微信小程序的 property observer 只在值变化时触发，初始值不触发。** 当 `wx:if` 从 false 变 true 时，组件新创建，`content` 的初始值不会调用 `observer`，所以 `setContent()` 不执行，组件不渲染。

切换页面再回来时，`onShow` → `loadMessagesFromCache` 重新 `setData`，此时 `mp-html` 已存在，`content` 值变化触发 observer，所以能渲染。

## 修复

在 `miniprogram/components/mp-html/index.js` 添加 `attached` 生命周期：

```javascript
attached: function() {
  if (this.data.content) this.setContent(this.data.content)
}
```

组件挂载时主动检查 content 并初始化，不依赖 observer。

## 关键教训

1. **微信小程序 property observer 不触发初始值**——通过 `wx:if` 动态创建的组件，需要用 `attached` 生命周期兜底初始化
2. **"切页面回来才正常"是典型的组件生命周期问题**——说明数据没问题，但组件没有响应数据变化
3. **流式渲染要分阶段**——流式中用纯文本（避免高频 setData 卡顿），完成后一次性解析 markdown 并切到 mp-html

## 涉及文件

- `miniprogram/components/mp-html/index.js` — 添加 attached 生命周期
- `miniprogram/pages/chat/chat.wxml:18` — mp-html 的 wx:if 条件（未修改）
- `miniprogram/pages/chat/chat.js:187` — finishAiMessage（未修改）
