import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import date

st.set_page_config(page_title="智能私人营养师与健身教练", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# Sidebar Navigation
st.sidebar.title("导航")
app_mode = st.sidebar.selectbox("选择页面", ["个人档案", "智能教练", "数据统计"])

# Helper to get user from session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if app_mode == "个人档案":
    st.title('👤 个人档案')
    
    if st.session_state.user_id:
        try:
            response = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}")
            if response.status_code == 200:
                user = response.json()
                st.success(f"当前用户 ID: {st.session_state.user_id}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("身高 (cm)", user["height"])
                    st.metric("体重 (kg)", user["weight"])
                    st.metric("年龄", user["age"])
                with col2:
                    st.metric("BMR (kcal)", user["bmr"])
                    st.metric("TDEE (kcal)", user["tdee"])
                    st.info(f"过敏史: {user['allergies'] or '无'}")
                
                if st.button("切换/重新录入"):
                    st.session_state.user_id = None
                    st.rerun()
            else:
                st.error("无法获取用户信息")
                st.session_state.user_id = None
        except Exception as e:
            st.error(f"连接后端失败: {e}")
    
    if not st.session_state.user_id:
        st.header('请输入您的基本信息')
        with st.form("user_info_form"):
            height = st.number_input('身高 (cm)', min_value=1.0, max_value=300.0, value=175.0)
            weight = st.number_input('体重 (kg)', min_value=1.0, max_value=500.0, value=70.0)
            age = st.number_input('年龄', min_value=1, max_value=150, value=30)
            gender = st.selectbox('性别', ['男', '女'])
            target_weight = st.number_input('目标体重 (kg)', min_value=1.0, max_value=500.0, value=65.0)
            allergies = st.text_input('过敏史 (可选, 如: 花生、海鲜)')

            submitted = st.form_submit_button("保存")
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
                        st.success(f"用户信息已保存！您的用户 ID 是: {user['id']}")
                        st.rerun()
                    else:
                        st.error(f"保存失败: {response.text}")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")

elif app_mode == "智能教练":
    st.title('🤖 智能教练')
    
    if not st.session_state.user_id:
        st.warning("请先在'个人档案'页面填写信息")
    else:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("今天吃了什么？做了什么运动？或者有什么健康疑问？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # 使用流式接口
                    response = requests.post(
                        f"{BACKEND_URL}/chat/stream", 
                        json={
                            "user_id": st.session_state.user_id,
                            "message": prompt
                        },
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        # 创建一个占位符用于流式显示
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        # 迭代流式响应
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_response += chunk
                                # 实时更新占位符内容
                                message_placeholder.markdown(full_response + "▌")
                        
                        # 完成后显示最终内容（去掉光标）
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"对话失败: {response.text}")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")

elif app_mode == "数据统计":
    st.title('📊 数据统计')
    
    if not st.session_state.user_id:
        st.warning("请先在'个人档案'页面填写信息")
    else:
        try:
            # Get today's summary
            summary_resp = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}/today")
            if summary_resp.status_code == 200:
                summary = summary_resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("今日摄入", f"{summary['intake_calories']} kcal")
                col2.metric("今日消耗", f"{summary['burn_calories']} kcal")
                deficit = summary['intake_calories'] - summary['burn_calories']
                col3.metric("热量缺口", f"{deficit} kcal", delta=-deficit, delta_color="inverse")
            
            # Get historical logs for charts
            logs_resp = requests.get(f"{BACKEND_URL}/user/{st.session_state.user_id}/logs")
            if logs_resp.status_code == 200:
                logs = logs_resp.json()
                if logs:
                    df = pd.DataFrame(logs)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    st.subheader("卡路里趋势")
                    fig = px.line(df, x='date', y=['intake_calories', 'burn_calories'], 
                                 labels={'value': '卡路里', 'date': '日期'},
                                 title='摄入 vs 消耗')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("暂无历史数据")
        except Exception as e:
            st.error(f"连接后端失败: {e}")
