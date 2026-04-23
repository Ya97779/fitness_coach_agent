# FitCoach AI 前端说明文档

## 📖 目录

1. [技术栈介绍](#1-技术栈介绍)
2. [项目结构](#2-项目结构)
3. [核心概念解析](#3-核心概念解析)
4. [代码详解](#4-代码详解)
5. [页面功能](#5-页面功能)
6. [API 通信](#6-api-通信)
7. [样式设计](#7-样式设计)
8. [状态管理](#8-状态管理)
9. [快速上手](#9-快速上手)

---

## 1. 技术栈介绍

### 1.1 Streamlit

**Streamlit** 是一个用于构建机器学习和数据科学 Web 应用的 Python 框架。

| 特点 | 说明 |
|------|------|
| 纯 Python | 无需 HTML/CSS/JavaScript |
| 实时更新 | 代码修改自动刷新 |
| 丰富组件 | 内置图表、表单、地图等 |
| 快速开发 | 几分钟就能搭一个 Web 应用 |

**对比传统前端框架：**

| 方面 | Streamlit | React/Vue |
|------|-----------|------------|
| 开发语言 | Python | JavaScript/TypeScript |
| 学习曲线 | 低 | 中高 |
| 定制程度 | 中 | 高 |
| 适合场景 | 数据应用、原型 | 复杂交互应用 |

### 1.2 第三方库

| 库名 | 用途 |
|------|------|
| `streamlit` | Web 框架 |
| `requests` | HTTP 请求 |
| `pandas` | 数据处理 |
| `plotly.express` | 数据可视化图表 |

---

## 2. 项目结构

```
frontend/
└── app.py      # 主应用文件（约 480 行）
```

**文件组成：**

```
app.py
├── 1. 依赖导入 (1-10行)
├── 2. 页面配置 (11-13行)
├── 3. 常量定义 (14行)
├── 4. CSS 样式 (15-94行)
├── 5. Session State 初始化 (95-108行)
├── 6. 侧边栏组件 (109-155行)
├── 7. 主内容区
│   ├── 7.1 聊天模式 (157-245行)
│   ├── 7.2 个人档案模式 (247-380行)
│   └── 7.3 数据统计模式 (382-483行)
```

---

## 3. 核心概念解析

### 3.1 Streamlit 组件

**常用组件：**

| 组件 | 函数 | 说明 |
|------|------|------|
| 标题 | `st.title()` | 页面主标题 |
| 文本 | `st.markdown()` | 支持 Markdown 格式 |
| 按钮 | `st.button()` | 可点击按钮 |
| 输入框 | `st.text_input()` | 单行文本输入 |
| 数字输入 | `st.number_input()` | 数字输入框 |
| 选择框 | `st.selectbox()` | 下拉选择 |
| 列布局 | `st.columns()` | 横向分列 |
| 容器 | `st.container()` | 内容分组 |
| 占位符 | `st.empty()` | 动态占位 |

### 3.2 Chat 组件（新版特性）

Streamlit 1.22+ 引入了原生 Chat 组件：

```python
# 显示聊天消息
with st.chat_message("user", avatar="👤"):
    st.markdown("用户消息")

with st.chat_message("assistant", avatar="🤖"):
    st.markdown("AI 回复")
```

### 3.3 Session State（会话状态）

Session State 用于在用户交互之间保持数据：

```python
# 初始化（只在第一次运行）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 读取
st.session_state.messages.append(...)

# 修改
st.session_state.user_id = 1
```

**生命周期：**
- 用户刷新页面 → 数据保留
- 用户关闭标签页 → 数据丢失
- 用户首次访问 → 初始化

---

## 4. 代码详解

### 4.1 页面配置

```python
st.set_page_config(
    page_title="FitCoach AI",  # 浏览器标签标题
    layout="wide"              # 宽布局（占满屏幕）
)
```

### 4.2 自定义 CSS

```python
st.markdown("""
<style>
    .chat-message-user {
        background-color: #4a5568;  /* 灰色背景 */
        border-radius: 18px 18px 4px 18px;  /* 圆角（左上、右上、右下、左下）*/
        padding: 12px 16px;  /* 内边距 */
        max-width: 75%;  /* 最大宽度 */
        margin-left: auto;  /* 靠右对齐 */
    }
</style>
""", unsafe_allow_html=True)
```

**`unsafe_allow_html=True` 的作用：**
- 允许渲染原始 HTML
- 默认 Streamlit 会转义 HTML 防止 XSS

### 4.3 侧边栏

```python
with st.sidebar:
    # 新对话按钮
    st.button("💬 新对话", on_click=clear_chat)

    # 导航按钮
    for mode, label in nav_items:
        if st.button(label, key=f"nav_{mode}"):
            st.session_state.app_mode = mode

    # 快捷操作
    for action, prompt in quick_actions:
        if st.button(action):
            st.session_state.messages.append({"role": "user", "content": prompt})
```

---

## 5. 页面功能

### 5.1 聊天模式（Chat Mode）

**功能：**
- 展示聊天历史
- 接收用户输入
- 调用后端 API 获取回复
- 流式显示响应

**核心流程：**

```
用户输入 → 点击发送 → 添加到 messages → 调用 API → 流式显示响应
```

**代码流程：**

```python
# 1. 用户输入
prompt = st.text_input("...", key="chat_input")

# 2. 点击发送
if send_btn and prompt:
    # 3. 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. 调用后端
    response = requests.post(f"{BACKEND_URL}/chat/stream", json=data, stream=True)

    # 5. 流式显示
    for chunk in response.iter_content():
        full_response += chunk
        placeholder.markdown(full_response + "▌")  # 打字机效果
```

### 5.2 个人档案模式（Profile Mode）

**功能：**
- 创建用户档案
- 显示 BMR/TDEE 等指标
- 更新用户信息

**用户档案表单：**

```python
with st.form("user_info_form"):
    height = st.number_input('身高 (cm)', ...)
    weight = st.number_input('体重 (kg)', ...)
    gender = st.selectbox('性别', ['男', '女'])
    submitted = st.form_submit_button("保存信息")
```

### 5.3 数据统计模式（Stats Mode）

**功能：**
- 展示今日热量摄入/消耗
- 显示历史趋势图表

**使用 Plotly 绑制图表：**

```python
# 折线图
fig = px.line(
    df,
    x='date',
    y=['intake_calories', 'burn_calories'],
    title='摄入 vs 消耗趋势'
)
st.plotly_chart(fig)

# 柱状图
fig_net = px.bar(
    df,
    x='date',
    y='net_calories',
    color='net_calories'  # 根据值着色
)
st.plotly_chart(fig_net)
```

---

## 6. API 通信

### 6.1 后端地址

```python
BACKEND_URL = "http://127.0.0.1:8000"
```

### 6.2 API 端点

| 方法 | 路径 | 功能 | 请求体 |
|------|------|------|--------|
| POST | `/user/` | 创建用户 | UserCreate |
| GET | `/user/{id}` | 获取用户信息 | - |
| GET | `/user/{id}/today` | 获取当日数据 | - |
| GET | `/user/{id}/logs` | 获取历史记录 | - |
| POST | `/chat/stream` | 流式聊天 | ChatRequest |

### 6.3 请求示例

```python
# 创建用户
response = requests.post(
    f"{BACKEND_URL}/user/",
    json={
        "height": 175,
        "weight": 70,
        "age": 25,
        "gender": "男"
    }
)

# 流式聊天
response = requests.post(
    f"{BACKEND_URL}/chat/stream",
    json={"message": "你好"},
    stream=True  # 关键：启用流式
)

# 处理流式响应
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        full_response += chunk
```

### 6.4 错误处理

```python
try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
    else:
        st.error(f"请求失败: {response.text}")
except requests.exceptions.ConnectionError:
    st.error("无法连接后端服务，请确保后端已启动")
except Exception as e:
    st.error(f"发生错误: {e}")
```

---

## 7. 样式设计

### 7.1 颜色主题

```css
/* 主色调：渐变紫 */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 深色背景 */
background-color: #1a202c;  /* 卡片背景 */
background-color: #2d3748;  /* 输入框背景 */
background-color: #4a5568;  /* 用户消息背景 */

/* 文字颜色 */
color: white;  /* 白色文字 */
```

### 7.2 组件样式

**聊天气泡：**

```css
/* 用户消息：靠右，浅灰背景 */
.chat-message-user {
    background-color: #4a5568;
    border-radius: 18px 18px 4px 18px;  /* 左下尖角 */
    margin-left: auto;
}

/* AI 消息：靠左，深灰背景 */
.chat-message-assistant {
    background-color: #2d3748;
    border-radius: 18px 18px 18px 4px;  /* 右下尖角 */
    margin-right: auto;
}
```

### 7.3 按钮样式

```css
/* 主按钮：渐变背景 */
.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
}

/* 禁用状态 */
.send-btn:disabled {
    opacity: 0.5;  /* 半透明 */
    cursor: not-allowed;  /* 显示禁止光标 */
}
```

---

## 8. 状态管理

### 8.1 Session State 变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `app_mode` | str | 当前页面模式：chat/profile/stats |
| `messages` | list | 聊天记录 |
| `user_id` | int | 当前用户 ID |
| `sidebar_collapsed` | bool | 侧边栏是否折叠 |

### 8.2 状态初始化

```python
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = None
```

### 8.3 状态更新

```python
# 更新 app_mode
st.session_state.app_mode = "profile"

# 清空聊天记录
st.session_state.messages = []

# 设置用户
st.session_state.user_id = user["id"]
```

---

## 9. 快速上手

### 9.1 运行前端

```bash
# 确保后端已启动（端口 8000）
cd backend
uvicorn app.main:app --reload

# 新终端启动前端
streamlit run frontend/app.py
```

### 9.2 访问地址

- 前端：http://localhost:8501
- 后端 API 文档：http://localhost:8000/docs

### 9.3 添加新页面

```python
# 1. 在 nav_items 中添加
nav_items = [
    ("chat", "🤖 智能教练"),
    ("profile", "👤 个人档案"),
    ("stats", "📊 数据统计"),
    ("new_page", "🆕 新页面")  # 新增
]

# 2. 添加路由逻辑
if app_mode == "new_page":
    st.title("🆕 新页面")
    st.write("这是新页面的内容")
```

### 9.4 添加新组件

```python
# 滑块
slider_value = st.slider("选择数值", 0, 100, 50)

# 日期选择
selected_date = st.date_input("选择日期")

# 文件上传
uploaded_file = st.file_uploader("上传文件")

# 进度条
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
```

---

## 📚 更多资源

- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Streamlit Components](https://streamlit.io/components)
- [Plotly Express 文档](https://plotly.com/python/plotly-express/)
