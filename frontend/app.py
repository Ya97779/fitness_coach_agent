import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import date

# Set page config
st.set_page_config(page_title="FitCoach AI", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# --- Custom CSS for ChatGPT style ---
st.markdown("""
<style>
    /* ChatGPT style theme */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .chat-message-user {
        background-color: #4a5568;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
    }
    
    .chat-message-assistant {
        background-color: #2d3748;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 75%;
        margin-right: auto;
    }
    
    .sidebar-button {
        border: none;
        background: transparent;
        width: 100%;
        text-align: left;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .sidebar-button:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    .sidebar-button.active {
        background-color: rgba(102, 126, 234, 0.3);
        border-left: 3px solid #667eea;
    }
    
    .new-chat-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .new-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .input-container {
        background-color: #2d3748;
        border-radius: 20px;
        padding: 4px;
        display: flex;
        align-items: center;
        border: 1px solid #4a5568;
        transition: border-color 0.2s;
    }
    
    .input-container:focus-within {
        border-color: #667eea;
    }
    
    .user-input {
        flex: 1;
        background: transparent;
        border: none;
        outline: none;
        color: white;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    .send-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 16px;
        padding: 10px 16px;
        color: white;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    
    .send-btn:hover:not(:disabled) {
        opacity: 0.9;
    }
    
    .send-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #4a5568;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #4a5568;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

# --- Sidebar ---
with st.sidebar:
    # New Chat Button
    st.button(
        "💬 新对话",
        key="new_chat",
        on_click=lambda: st.session_state.update(messages=[], app_mode="chat"),
        use_container_width=True,
        type="primary"
    )
    
    st.divider()
    
    # Navigation Buttons
    nav_items = [
        ("chat", "🤖 智能教练"),
        ("profile", "👤 个人档案"),
        ("stats", "📊 数据统计")
    ]
    
    for mode, label in nav_items:
        is_active = st.session_state.app_mode == mode
        button_key = f"nav_{mode}"
        
        # Use markdown button for better styling
        if st.button(
            label,
            key=button_key,
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.app_mode = mode
            st.rerun()
    
    st.divider()
    
    # Quick Actions
    st.subheader("快捷操作")
    quick_actions = [
        ("记录早餐", "帮我记录早餐"),
        ("记录运动", "记录今天的运动"),
        ("热量查询", "查一下苹果的热量"),
        ("训练建议", "给我制定训练计划")
    ]
    
    for action, prompt in quick_actions:
        if st.button(action, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.app_mode = "chat"
            st.rerun()

# --- Main Content ---
app_mode = st.session_state.app_mode

# --- Chat Mode ---
if app_mode == "chat":
    st.title("🤖 FitCoach AI")
    st.caption("您的智能健身与营养顾问")
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        # Welcome message if no messages
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                你好！我是你的私人健身与营养顾问 **FitCoach AI**。
                
                **我可以帮你做什么？**
                
                🥗 **饮食建议**：记录饮食、查询热量、制定食谱
                💪 **运动指导**：动作纠正、训练计划、恢复建议
                📊 **数据追踪**：热量缺口、运动记录、进度分析
                
                试试问我：
                - "今天中午吃了一碗兰州拉面，热量是多少？"
                - "如何正确做深蹲？"
                - "帮我制定一份增肌食谱"
                """)
        
        # Display messages
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
    # Input Area
    st.divider()
    
    with st.container():
        col_input, col_send = st.columns([12, 1])
        
        with col_input:
            prompt = st.text_input(
                "",
                placeholder="输入您的问题，例如：'如何正确做深蹲？' 或 '记录今天吃了苹果'",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_btn = st.button(
                "➤",
                key="send_button",
                disabled=not prompt,
                use_container_width=True,
                type="primary"
            )
    
    if send_btn and prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
        
        # Get response from backend
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    chat_data = {"message": prompt}
                    if st.session_state.user_id:
                        chat_data["user_id"] = st.session_state.user_id
                    
                    response = requests.post(
                        f"{BACKEND_URL}/chat/stream",
                        json=chat_data,
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"对话失败: {response.text}")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")
        
        # Auto-scroll to bottom
        st.rerun()

# --- Profile Mode ---
elif app_mode == "profile":
    st.title("👤 个人档案")
    st.caption("管理您的个人健康数据")
    
    profile_container = st.container()
    
    with profile_container:
        if st.session_state.user_id:
            # Fetch and display user info
            try:
                response = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}")
                if response.status_code == 200:
                    user = response.json()
                    
                    # Profile card layout
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("基本信息")
                        st.metric("身高", f"{user['height']} cm")
                        st.metric("体重", f"{user['weight']} kg")
                        st.metric("年龄", user['age'])
                    
                    with col2:
                        st.subheader("身体指标")
                        st.metric("BMR", f"{user['bmr']:.0f} kcal")
                        st.metric("TDEE", f"{user['tdee']:.0f} kcal")
                        bmi = user['weight'] / ((user['height'] / 100) ** 2)
                        st.metric("BMI", f"{bmi:.1f}")
                    
                    with col3:
                        st.subheader("目标设置")
                        st.metric("目标体重", f"{user['target_weight'] or '未设定'} kg")
                        st.info(f"性别: {user['gender']}")
                        st.info(f"过敏史: {user['allergies'] or '无'}")
                    
                    st.divider()
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("🔄 更新信息", use_container_width=True, type="primary"):
                            # Trigger form display
                            st.session_state.show_update_form = True
                    
                    with col_btn2:
                        if st.button("🚪 退出登录", use_container_width=True):
                            st.session_state.user_id = None
                            st.session_state.messages = []
                            st.rerun()
                
                else:
                    st.error("无法获取用户信息")
                    st.session_state.user_id = None
            except Exception as e:
                st.error(f"连接后端失败: {e}")
        
        # Show form if not logged in or update requested
        if not st.session_state.user_id:
            with st.form("user_info_form", border=False):
                st.subheader("创建个人档案")
                st.caption("填写以下信息以获得个性化建议")
                
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    height = st.number_input('身高 (cm)', min_value=100.0, max_value=250.0, value=170.0)
                    weight = st.number_input('体重 (kg)', min_value=30.0, max_value=200.0, value=70.0)
                    age = st.number_input('年龄', min_value=1, max_value=120, value=25)
                
                with col_f2:
                    gender = st.selectbox('性别', ['男', '女'])
                    target_weight = st.number_input('目标体重 (kg)', min_value=30.0, max_value=200.0, value=70.0)
                    allergies = st.text_input('过敏史 (可选)', placeholder='例如: 花生、海鲜')
                
                submitted = st.form_submit_button("保存信息", use_container_width=True, type="primary")
                
                if submitted:
                    user_data = {
                        "height": height,
                        "weight": weight,
                        "age": age,
                        "gender": gender,
                        "target_weight": target_weight,
                        "allergies": allergies
                    }
                    try:
                        response = requests.post(f"{BACKEND_URL}/user/", json=user_data)
                        if response.status_code == 200:
                            user = response.json()
                            st.session_state.user_id = user["id"]
                            st.success(f"✅ 用户创建成功！您的ID: {user['id']}")
                            st.rerun()
                        else:
                            st.error(f"保存失败: {response.text}")
                    except Exception as e:
                        st.error(f"连接后端失败: {e}")

# --- Stats Mode ---
elif app_mode == "stats":
    st.title("📊 数据统计")
    st.caption("追踪您的健康数据变化")
    
    if not st.session_state.user_id:
        st.warning("⚠️ 请先在个人档案中创建账号以查看数据统计")
        st.info("您可以直接使用智能教练进行咨询，无需登录")
    else:
        try:
            # Today's summary
            summary_resp = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}/today")
            
            if summary_resp.status_code == 200:
                summary = summary_resp.json()
                
                # Daily metrics
                st.subheader("今日概览")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    card = st.container()
                    with card:
                        st.metric("今日摄入", f"{summary['intake_calories']} kcal")
                
                with col2:
                    card = st.container()
                    with card:
                        st.metric("今日消耗", f"{summary['burn_calories']} kcal")
                
                with col3:
                    card = st.container()
                    with card:
                        net = summary['intake_calories'] - summary['burn_calories']
                        delta_color = "inverse" if net < 0 else "normal"
                        st.metric("热量缺口", f"{net} kcal", delta=net, delta_color=delta_color)
                
                # Historical trends
                st.subheader("历史趋势")
                logs_resp = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}/logs")
                
                if logs_resp.status_code == 200:
                    logs = logs_resp.json()
                    
                    if logs:
                        df = pd.DataFrame(logs)
                        df['date'] = pd.to_datetime(df['date'])
                        df['net_calories'] = df['intake_calories'] - df['burn_calories']
                        
                        # Calories trend chart
                        fig = px.line(
                            df,
                            x='date',
                            y=['intake_calories', 'burn_calories'],
                            labels={'value': '卡路里', 'date': '日期', 'variable': '类型'},
                            title='摄入 vs 消耗趋势',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Net calories chart
                        fig_net = px.bar(
                            df,
                            x='date',
                            y='net_calories',
                            labels={'net_calories': '热量缺口', 'date': '日期'},
                            title='每日热量缺口',
                            template='plotly_dark',
                            color='net_calories',
                            color_continuous_scale=['green', 'red']
                        )
                        st.plotly_chart(fig_net, use_container_width=True)
                    else:
                        st.info("暂无历史数据，开始记录您的饮食和运动吧！")
            
        except Exception as e:
            st.error(f"连接后端失败: {e}")
