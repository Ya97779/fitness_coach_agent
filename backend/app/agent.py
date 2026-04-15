from typing import Annotated, TypedDict, List, Union, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from . import models, database, rag_utils
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_id: int
    user_profile: Dict[str, Any]
    daily_stats: Dict[str, Any]

# Define tools
@tool
def get_user_info(user_id: int):
    """获取用户的身高、体重、BMR、TDEE、过敏史等基本信息。"""
    db = database.SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            return {
                "height": user.height,
                "weight": user.weight,
                "age": user.age,
                "gender": user.gender,
                "bmr": user.bmr,
                "tdee": user.tdee,
                "allergies": user.allergies
            }
        return "未找到用户信息"
    finally:
        db.close()

@tool
def log_food_intake(user_id: int, food_name: str, calories: float):
    """记录用户摄入的食物及其卡路里。"""
    db = database.SessionLocal()
    try:
        today = date.today()
        log = db.query(models.DailyLog).filter(models.DailyLog.user_id == user_id, models.DailyLog.date == today).first()
        if not log:
            log = models.DailyLog(user_id=user_id, date=today)
            db.add(log)
            db.commit()
            db.refresh(log)
        
        food_item = models.FoodItem(log_id=log.id, name=food_name, calories=calories)
        log.intake_calories += calories
        db.add(food_item)
        db.commit()
        return f"已记录: {food_name}, {calories} kcal"
    finally:
        db.close()

@tool
def log_exercise_burn(user_id: int, activity_type: str, duration: int, calories: float):
    """记录用户进行的运动、时长（分钟）及消耗的卡路里。"""
    db = database.SessionLocal()
    try:
        today = date.today()
        log = db.query(models.DailyLog).filter(models.DailyLog.user_id == user_id, models.DailyLog.date == today).first()
        if not log:
            log = models.DailyLog(user_id=user_id, date=today)
            db.add(log)
            db.commit()
            db.refresh(log)
        
        exercise_item = models.ExerciseItem(log_id=log.id, type=activity_type, duration=duration, calories=calories)
        log.burn_calories += calories
        db.add(exercise_item)
        db.commit()
        return f"已记录: {activity_type}, {duration} min, {calories} kcal"
    finally:
        db.close()

@tool
def get_daily_summary(user_id: int):
    """获取用户当日的摄入与消耗卡路里总结。"""
    db = database.SessionLocal()
    try:
        today = date.today()
        log = db.query(models.DailyLog).filter(models.DailyLog.user_id == user_id, models.DailyLog.date == today).first()
        if log:
            return {
                "intake_calories": log.intake_calories,
                "burn_calories": log.burn_calories,
                "net_calories": log.intake_calories - log.burn_calories
            }
        return {"intake_calories": 0, "burn_calories": 0, "net_calories": 0}
    finally:
        db.close()

@tool
def search_food_calories(food_name: str):
    """模拟搜索食物热量。"""
    # 实际应用中可以调用外部API或数据库
    mock_data = {
        "兰州拉面": 500,
        "可乐": 150,
        "鸡蛋": 70,
        "鸡胸肉": 165,
        "米饭": 130
    }
    return mock_data.get(food_name, 200) # 默认返回200

@tool
def estimate_exercise_burn(exercise_type: str, duration: int):
    """模拟估算运动消耗热量。"""
    # 实际应用中可以根据MET值计算
    mock_data = {
        "慢跑": 10, # kcal/min
        "游泳": 12,
        "步行": 4,
        "卧推": 6
    }
    return mock_data.get(exercise_type, 5) * duration

@tool
def rag_medical_search_tool(query: str):
    """当用户询问专业健康/医疗/解剖问题时，从本地知识库检索专业的建议。"""
    return rag_utils.rag_medical_search(query)

tools = [get_user_info, log_food_intake, log_exercise_burn, get_daily_summary, search_food_calories, estimate_exercise_burn, rag_medical_search_tool]
tool_node = ToolNode(tools)

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "glm-4.7"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Initialize memory with SqliteSaver
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

agent_app = workflow.compile(checkpointer=memory)
