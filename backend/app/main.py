from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from . import models, database
from .agents.graph import process_user_message
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from datetime import date
from langchain_core.messages import HumanMessage

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

class DailyLogResponse(BaseModel):
    id: int
    date: date
    intake_calories: float
    burn_calories: float
    weight_log: Optional[float] = None
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    user_id: Optional[int] = None
    message: str

class ChatResponse(BaseModel):
    response: str
    agent: str
    nutrition_response: Optional[str] = None
    fitness_response: Optional[str] = None

class StreamChatRequest(BaseModel):
    user_id: Optional[int] = None
    message: str

def calculate_metrics(height, weight, age, gender):
    if not height or not weight or not age:
        return 0, 0
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    tdee = bmr * 1.375
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
    log = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == user_id,
        models.DailyLog.date == today
    ).first()
    if not log:
        log = models.DailyLog(user_id=user_id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    return log

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user_profile = {}
    daily_stats = {"intake_calories": 0, "burn_calories": 0, "net_calories": 0}

    if request.user_id:
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        if user:
            user_profile = {
                "height": user.height,
                "weight": user.weight,
                "age": user.age,
                "gender": user.gender,
                "bmr": user.bmr,
                "tdee": user.tdee,
                "allergies": user.allergies
            }

            today = date.today()
            log = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == request.user_id,
                models.DailyLog.date == today
            ).first()

            if log:
                daily_stats = {
                    "intake_calories": log.intake_calories,
                    "burn_calories": log.burn_calories,
                    "net_calories": log.intake_calories - log.burn_calories
                }

    result = process_user_message(
        user_message=request.message,
        user_id=request.user_id or 1,
        user_profile=user_profile,
        daily_stats=daily_stats
    )

    return ChatResponse(
        response=result["response"],
        agent=result["agent"],
        nutrition_response=result.get("nutrition_response"),
        fitness_response=result.get("fitness_response")
    )

@app.post("/chat/stream")
async def chat_stream(request: StreamChatRequest, db: Session = Depends(get_db)):
    user_profile = {
        "height": 0, "weight": 0, "age": 0, "gender": "未知",
        "bmr": 0, "tdee": 0, "allergies": "无"
    }
    daily_stats = {"intake_calories": 0, "burn_calories": 0, "net_calories": 0}

    if request.user_id:
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        if user:
            user_profile = {
                "height": user.height,
                "weight": user.weight,
                "age": user.age,
                "gender": user.gender,
                "bmr": user.bmr,
                "tdee": user.tdee,
                "allergies": user.allergies
            }

            today = date.today()
            log = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == request.user_id,
                models.DailyLog.date == today
            ).first()

            if log:
                daily_stats = {
                    "intake_calories": log.intake_calories,
                    "burn_calories": log.burn_calories,
                    "net_calories": log.intake_calories - log.burn_calories
                }

    async def event_generator():
        try:
            result = process_user_message(
                user_message=request.message,
                user_id=request.user_id or 1,
                user_profile=user_profile,
                daily_stats=daily_stats
            )

            response_text = result["response"]
            for char in response_text:
                yield char
                await asyncio.sleep(0.01)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "chat", "description": "闲聊助手 - 处理日常对话和寒暄"},
            {"name": "nutrition", "description": "营养师 - 饮食计划、热量计算、营养建议"},
            {"name": "fitness", "description": "健身教练 - 训练计划、动作指导、运动建议"},
            {"name": "expert", "description": "专家评审 - 评审营养师和教练的输出质量"}
        ]
    }
