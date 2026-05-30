# Record1: GLM-4.7 流式输出两个典型问题

日期: 2026-05-30
状态: 已验证修复

---

## 问题一：流式输出为空（200+ 空 chunk）

### 现象

fitness agent 第二轮 LLM 流式调用（`bind_tools` + `stream`）返回 229 个 chunk，全部 `content=''`，`tool_calls=0`。最终触发兜底逻辑，输出"无法生成训练计划"。

```
[fitness_agent] chunk[1]: content='', tool_calls=[], tool_call_chunks=[]
[fitness_agent] chunk[2]: content='', tool_calls=[], tool_call_chunks=[]
...（共 229 个空 chunk）
[fitness_agent] 第二轮 LLM 流式完成: 229 chunks, has_content=False, tool_calls=0
```

### 根因

GLM-4.7 默认开启 thinking 模式，模型在流式输出时产生大量 thinking tokens，这些 tokens 不包含在 `content` 字段中，导致前端收到的全是空内容。`bind_tools` + `stream` 组合下该问题更容易触发。

### 解决方案

**根因修复**：在 `LLMManager` 创建 `ChatOpenAI` 实例时，通过 `extra_body` 注入 `thinking: {type: disabled}`，禁用 GLM thinking 模式。

```python
# backend/app/llm_manager.py
ChatOpenAI(
    model=os.getenv("LLM_MODEL", "glm-4.7"),
    ...
    extra_body={"thinking": {"type": "disabled"}}
)
```

**防御兜底**：在 `fitness_agent.py` 的流式和非流式路径中，如果 `bind_tools` 调用返回空内容且无工具调用，去掉 `bind_tools` 用普通 `llm.stream()` / `llm.invoke()` 重试。

### 涉及文件

- `backend/app/llm_manager.py` — 根因修复
- `backend/app/agents/fitness_agent.py` — 防御兜底

### 关键 commit

- `d1f38a1` — 最初发现并修复（extra_body 方案）
- `03bd66b` — 重新恢复该修复

---

## 问题二：Markdown 渲染无法分段分行（一整段文字）

### 现象

小程序聊天页面 AI 回复显示为一整段文字，没有段落分隔、列表换行、标题分级。消息原文中可见大量字面量 `\n`（两个字符，非真正换行）：

```
基于您的减脂目标...\nin推/拉/腿分化计划:\nn1.推日(胸、肩、三头)In-平板卧推4组x8-10次n-...
```

### 根因（两层问题叠加）

**第一层：LLM 输出字面量转义序列**

GLM-4.7 流式输出中，换行符以字面量 `\n`（反斜杠 + n，两个字符）而非真正的换行符（ASCII 0x0A）返回。这是 LLM API 的行为特征，不是代码 bug。

**第二层：SSE 传输丢失真正换行符**

即使 LLM 输出了真正的换行符，SSE 协议以 `\n` 作为消息分隔符，换行符会被当作消息边界丢弃：

```
后端发送:  data: 第一行\n第二行\n\n
前端解析:  "第一行"  ← 正确
           "第二行"  ← 不以"data: "开头，被丢弃
```

两层问题叠加：LLM 的字面量 `\n` 无法成为 markdown 换行，偶发的真正 `\n` 又在传输中丢失。

### 解决方案

在后端 SSE 发送前，统一处理两层问题：

```python
# backend/app/main.py
def _decode_llm_content(content: str) -> str:
    # 1. 解码 LLM 字面量转义序列 → 真正的控制字符
    decoded = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    # 2. SSE 转义：真正的换行符 → 传输安全的形式
    return decoded.replace('\\', '\\\\').replace('\n', '\\n')
```

前端接收后还原：

```javascript
// miniprogram/pages/chat/chat.js
const data = trimmed.slice(6).replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
```

处理链路：`LLM 字面量\n` → `真正换行` → `SSE 转义\\n` → `传输` → `前端还原真正换行` → `markdown 解析`

### 涉及文件

- `backend/app/main.py` — `_decode_llm_content()` 解码 + SSE 转义
- `miniprogram/pages/chat/chat.js` — SSE 接收还原

### 关键 commit

- `599196e` — SSE 传输转义换行符（第一层）
- `2ee184a` — LLM 字面量转义序列解码（第二层）

---

## 经验总结

| 问题 | 表象 | 根因层 | 修复层 |
|------|------|--------|--------|
| 流式空内容 | 200+ 空 chunk | LLM API（GLM thinking 模式） | LLM 配置（extra_body） |
| Markdown 不换行 | 一整段文字 | LLM 输出字面量 \n + SSE 传输丢失换行 | 后端解码+转义，前端还原 |

**排查思路**：流式输出问题需要逐层检查 — LLM 层（chunk 内容）→ 传输层（SSE 编码）→ 前端层（解析渲染）。日志是最有效的定位手段。

**GLM-4.7 流式输出注意事项**：
1. 必须用 `extra_body={"thinking": {"type": "disabled"}}` 禁用 thinking 模式
2. LLM 输出中的 `\n` 可能是字面量两字符，需要解码后再传输
3. SSE 传输中真正的换行符需要转义，否则被当作消息分隔符

## 验证记录

- 2026-05-30 10:14 — 问题一修复确认：禁用 thinking 后流式输出恢复正常，不再出现空 chunk
- 2026-05-30 10:30 — 问题二修复确认：小程序 markdown 正确渲染，标题、列表、段落分隔均正常显示
