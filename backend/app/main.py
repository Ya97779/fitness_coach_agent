from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, database, agent
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from langchain_core.messages import HumanMessage, AIMessage

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

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
    # 1. Fetch user profile
    user = db.query(models.User).filter(models.User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 2. Fetch daily stats
    today = date.today()
    log = db.query(models.DailyLog).filter(models.DailyLog.user_id == request.user_id, models.DailyLog.date == today).first()
    if not log:
        log = models.DailyLog(user_id=request.user_id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    
    # 3. Build initial state
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
    
    # 4. Run agent
    final_state = agent.agent_app.invoke(initial_state)
    
    # 5. Extract response
    response_msg = final_state['messages'][-1]
    return ChatResponse(response=response_msg.content)
