# mp-html 流式输出后不渲染 markdown 问题

## 现象

小程序 chat 页面，AI 流式输出完成后，显示的是纯文本而非渲染后的 markdown。需要切换到别的页面再回来才能渲染成功。有时还伴随页面跳动。

## 根因分析

### 背景：流式渲染分两阶段

1. **流式中**：`updateAiMessage` 设置 `_streaming: true`，只更新纯文本 `content`，不解析 markdown（避免高频 setData 卡顿）
2. **流式完成**：`finishAiMessage` 设置 `_streaming: false` + `html: parseMarkdown(content)`，切到富文本渲染

### WXML 条件

```html
<rich-text wx:if="{{item.role !== 'user' && !item._streaming && item.html}}" nodes="{{item.html}}" selectable />
<text wx:elif="{{item.role !== 'user'}}">{{item.content}}</text>
```

### 问题 1：property observer 不触发初始值（mp-html 特有）

微信小程序的 property `observer` **只在值变化时触发，初始值不触发**。当 `wx:if` 从 false 变 true 时组件新创建，`content` 的初始值不会调用 `observer`，所以 `setContent()` 不执行，组件不渲染。

### 问题 2：CSS 选择器不匹配

所有 markdown 样式选择器是 `.bubble-ai mp-html`，换成 `rich-text` 后选择器不匹配，导致样式完全不生效（不换行、无格式）。

### 问题 3：页面跳动

`sendMessage` 设置 `scrollToId` 滚动到 AI 消息位置，但 `finishAiMessage` 的 `setData` 没有清除 `scrollToId`，导致 `scroll-into-view` 重新触发滚动。

---

## 最终方案：用 rich-text 替代 mp-html

### 为什么放弃 mp-html

mp-html 是第三方组件，通过 `wx:if` 动态创建时 property observer 不触发初始值。已尝试的失败方案：

| 方案 | 思路 | 结果 |
|------|------|------|
| attached 直接调用 setContent | 组件挂载时检查 content 并初始化 | 失败：attached 时 content 可能未设置 |
| attached + setTimeout(0) | 延迟检查 properties.content | 失败：原因未知 |
| hidden 替代 wx:if | 组件始终存在，CSS 控制显隐 | 失败：mp-html 完全不渲染 |
| selectComponent 手动调用 | nextTick 后获取实例调用 setContent | 失败：引入重复消息等新问题 |

### rich-text 方案

小程序原生 `rich-text` 组件，`nodes` 属性直接接受 HTML 字符串，值变化时自动渲染，无需 observer。

**WXML**：
```html
<rich-text wx:if="{{item.role !== 'user' && !item._streaming && item.html}}" nodes="{{item.html}}" selectable />
<text wx:elif="{{item.role !== 'user'}}" class="msg-text" user-select>{{item.content}}</text>
```

**WXSS**：选择器从 `.bubble-ai mp-html` 改为 `.bubble-ai rich-text`

**JS**：`finishAiMessage` 保持原始逻辑，不需要 selectComponent 等 hack。

---

## 关键教训

1. **第三方组件 + wx:if 动态创建 = observer 不触发** — 微信小程序 property observer 不触发初始值，第三方组件无法通过生命周期兜底
2. **换组件时必须同步改 CSS 选择器** — `.bubble-ai mp-html` → `.bubble-ai rich-text`，否则样式完全失效
3. **scroll-into-view 会重复触发** — 修改消息内容的 `setData` 如果不清除 `scrollToId`，会导致页面跳动
4. **优先用原生组件** — `rich-text` 虽然功能比 `mp-html` 少，但没有 observer 问题，更可靠
5. **流式渲染要分阶段** — 流式中用纯文本（避免高频 setData 卡顿），完成后一次性解析 markdown

## 涉及文件

- `miniprogram/pages/chat/chat.wxml` — rich-text 组件
- `miniprogram/pages/chat/chat.wxss` — markdown 样式（选择器为 `.bubble-ai rich-text`）
- `miniprogram/pages/chat/chat.js` — finishAiMessage、loadMessagesFromCache、syncMessagesFromServer
- `miniprogram/utils/markdown.js` — parseMarkdown 解析器
