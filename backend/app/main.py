from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from . import models, database, agent
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import json
import asyncio
from datetime import date
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

models.Base.metadata.create_all(bind=database.engine)

# Global agent app variable
agent_app = None
checkpointer_context = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app, checkpointer_context
    # Initialize AsyncSqliteSaver
    checkpointer_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
    checkpointer = await checkpointer_context.__aenter__()
    
    # Compile the agent app with the checkpointer
    agent_app = agent.workflow.compile(checkpointer=checkpointer)
    
    yield
    
    # Cleanup
    if checkpointer_context:
        await checkpointer_context.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class UserCreate(BaseModel):
    height: float
    weight: float
    age: int
    gender: str
    target_weight: Optional[float] = None
    allergies: Optional[str] = None

class UserResponse(UserCreate):
    id: int
    bmr: Optional[float] = None
    tdee: Optional[float] = None
    class Config:
        from_attributes = True

class FoodItemCreate(BaseModel):
    name: str
    calories: float

class ExerciseItemCreate(BaseModel):
    type: str
    duration: int
    calories: float

class DailyLogResponse(BaseModel):
    id: int
    date: date
    intake_calories: float
    burn_calories: float
    weight_log: Optional[float] = None
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    response: str

# Helper to calculate BMR and TDEE
def calculate_metrics(height, weight, age, gender):
    # Mifflin-St Jeor Equation
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    tdee = bmr * 1.375 # Assuming moderate activity for now
    return bmr, tdee

@app.post("/user/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    bmr, tdee = calculate_metrics(user.height, user.weight, user.age, user.gender)
    db_user = models.User(**user.dict(), bmr=bmr, tdee=tdee)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/user/{user_id}", response_model=UserResponse)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.get("/user/{user_id}/logs", response_model=List[DailyLogResponse])
def get_user_logs(user_id: int, db: Session = Depends(get_db)):
    logs = db.query(models.DailyLog).filter(models.DailyLog.user_id == user_id).all()
    return logs

@app.get("/user/{user_id}/today", response_model=DailyLogResponse)
def get_today_log(user_id: int, db: Session = Depends(get_db)):
    today = date.today()
    log = db.query(models.DailyLog).filter(models.DailyLog.user_id == user_id, models.DailyLog.date == today).first()
    if not log:
        log = models.DailyLog(user_id=user_id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    return log

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # ... existing implementation ...
    user = db.query(models.User).filter(models.User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    today = date.today()
    log = db.query(models.DailyLog).filter(models.DailyLog.user_id == request.user_id, models.DailyLog.date == today).first()
    if not log:
        log = models.DailyLog(user_id=request.user_id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "user_profile": {
            "height": user.height,
            "weight": user.weight,
            "age": user.age,
            "gender": user.gender,
            "bmr": user.bmr,
            "tdee": user.tdee,
            "allergies": user.allergies
        },
        "daily_stats": {
            "intake_calories": log.intake_calories,
            "burn_calories": log.burn_calories,
            "net_calories": log.intake_calories - log.burn_calories
        }
    }
    
    config = {"configurable": {"thread_id": str(request.user_id)}}
    final_state = await agent_app.ainvoke(initial_state, config=config)
    
    response_msg = final_state['messages'][-1]
    return ChatResponse(response=response_msg.content)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    today = date.today()
    log = db.query(models.DailyLog).filter(models.DailyLog.user_id == request.user_id, models.DailyLog.date == today).first()
    if not log:
        log = models.DailyLog(user_id=request.user_id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "user_profile": {
            "height": user.height,
            "weight": user.weight,
            "age": user.age,
            "gender": user.gender,
            "bmr": user.bmr,
            "tdee": user.tdee,
            "allergies": user.allergies
        },
        "daily_stats": {
            "intake_calories": log.intake_calories,
            "burn_calories": log.burn_calories,
            "net_calories": log.intake_calories - log.burn_calories
        }
    }
    
    async def event_generator():
        config = {"configurable": {"thread_id": str(request.user_id)}}
        # 使用 astream_events v2 来流式输出
        # 我们主要关注 on_chat_model_stream 事件来获取最终回答的 tokens
        async for event in agent_app.astream_events(initial_state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
            elif kind == "on_tool_start":
                # 可以选择性地将工具调用信息也发给前端（例如提示：正在查询...）
                # 这里为了简化先只流式输出最终文本
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")
