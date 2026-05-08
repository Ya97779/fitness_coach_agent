from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session
from . import models, database, auth
from .agents.graph import process_user_message, stream_user_message
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
from datetime import date
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# ========== CORS ==========
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 全局异常处理 ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误"},
    )

# ========== RAG 启动初始化 ==========
rag_initialized = False

@app.on_event("startup")
async def startup_event():
    global rag_initialized
    if rag_initialized:
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[RAG 启动] 未配置 OPENAI_API_KEY，跳过增量索引。请在项目根目录创建 .env 文件并配置 API Key。")
        rag_initialized = True
        return

    try:
        from .rag import get_rag_instance
        rag = get_rag_instance()
        result = rag.check_and_update_index()
        print(f"[RAG 启动] 增量索引结果: 新增 {result['new_files']} 文件, 更新 {result['updated_files']} 文件, 共索引 {result['total_indexed']} 文件")
        rag_initialized = True
    except Exception as e:
        print(f"[RAG 启动] 增量索引初始化失败: {e}")
        rag_initialized = True

# ========== Pydantic 模型 ==========
class UserCreate(BaseModel):
    height: float
    weight: float
    age: int
    gender: str
    target_weight: Optional[float] = None
    allergies: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    openid: Optional[str] = None
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    height: float
    weight: float
    age: int
    gender: str
    target_weight: Optional[float] = None
    allergies: Optional[str] = None
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
    message: str

class ChatResponse(BaseModel):
    response: str
    agent: str
    nutrition_response: Optional[str] = None
    fitness_response: Optional[str] = None

class StreamChatRequest(BaseModel):
    message: str

class WxLoginRequest(BaseModel):
    code: str

class WxLoginResponse(BaseModel):
    token: str
    user: UserResponse

class ErrorResponse(BaseModel):
    code: int
    message: str

# ========== 工具函数 ==========
def calculate_metrics(height, weight, age, gender):
    if not height or not weight or not age:
        return 0, 0
    if gender == "男":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    tdee = bmr * 1.375
    return bmr, tdee

# ========== API v1 路由 ==========
router = APIRouter(prefix="/api/v1")

# ----- 微信登录 -----
@router.post("/auth/wx-login", response_model=WxLoginResponse)
async def wx_login(req: WxLoginRequest, db: Session = Depends(database.get_db)):
    wx_session = await auth.wx_code_to_session(req.code)

    user = db.query(models.User).filter(
        models.User.openid == wx_session["openid"]
    ).first()

    if not user:
        user = models.User(
            openid=wx_session["openid"],
            unionid=wx_session.get("unionid"),
            session_key=wx_session["session_key"],
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.session_key = wx_session["session_key"]
        if wx_session.get("unionid"):
            user.unionid = wx_session["unionid"]
        db.commit()
        db.refresh(user)

    token = auth.create_access_token(user.id)
    return WxLoginResponse(token=token, user=UserResponse.model_validate(user))

# ----- 用户资料 -----
@router.post("/user/", response_model=UserResponse)
def create_or_update_user(
    user_data: UserCreate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    bmr, tdee = calculate_metrics(
        user_data.height, user_data.weight, user_data.age, user_data.gender
    )
    current_user.height = user_data.height
    current_user.weight = user_data.weight
    current_user.age = user_data.age
    current_user.gender = user_data.gender
    current_user.target_weight = user_data.target_weight
    current_user.allergies = user_data.allergies
    current_user.bmr = bmr
    current_user.tdee = tdee
    db.commit()
    db.refresh(current_user)
    return current_user

@router.get("/user/me", response_model=UserResponse)
def get_current_user_info(
    current_user: models.User = Depends(auth.get_current_user),
):
    return current_user

@router.get("/user/me/logs", response_model=List[DailyLogResponse])
def get_current_user_logs(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    return db.query(models.DailyLog).filter(
        models.DailyLog.user_id == current_user.id
    ).all()

@router.get("/user/me/today", response_model=DailyLogResponse)
def get_current_user_today(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    today = date.today()
    log = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == current_user.id,
        models.DailyLog.date == today,
    ).first()
    if not log:
        log = models.DailyLog(user_id=current_user.id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    return log

# ----- 对话 -----
def _build_user_context(user: models.User, db: Session):
    user_profile = {
        "height": user.height,
        "weight": user.weight,
        "age": user.age,
        "gender": user.gender,
        "bmr": user.bmr,
        "tdee": user.tdee,
        "allergies": user.allergies,
    }
    today = date.today()
    log = db.query(models.DailyLog).filter(
        models.DailyLog.user_id == user.id,
        models.DailyLog.date == today,
    ).first()
    daily_stats = {
        "intake_calories": log.intake_calories if log else 0,
        "burn_calories": log.burn_calories if log else 0,
        "net_calories": (log.intake_calories - log.burn_calories) if log else 0,
    }
    return user_profile, daily_stats

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    user_profile, daily_stats = _build_user_context(current_user, db)

    result = process_user_message(
        user_message=request.message,
        user_id=current_user.id,
        user_profile=user_profile,
        daily_stats=daily_stats,
    )

    return ChatResponse(
        response=result["response"],
        agent=result["agent"],
        nutrition_response=result.get("nutrition_response"),
        fitness_response=result.get("fitness_response"),
    )

@router.post("/chat/stream")
async def chat_stream(
    request: StreamChatRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    user_profile, daily_stats = _build_user_context(current_user, db)

    user_message = request.message.strip() if request.message else "你好"

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            response_generator = await loop.run_in_executor(
                None,
                stream_user_message,
                user_message,
                current_user.id,
                user_profile,
                daily_stats,
            )
            for chunk in response_generator:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ----- 工具列表（无需鉴权） -----
@router.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "chat", "description": "闲聊助手 - 处理日常对话和寒暄"},
            {"name": "nutrition", "description": "营养师 - 饮食计划、热量计算、营养建议"},
            {"name": "fitness", "description": "健身教练 - 训练计划、动作指导、运动建议"},
            {"name": "expert", "description": "专家评审 - 评审营养师和教练的输出质量"},
        ]
    }

# ========== 注册路由 ==========
app.include_router(router)

# 根路由健康检查（无需鉴权）
@app.get("/agents")
async def list_agents_root():
    return await list_agents()
