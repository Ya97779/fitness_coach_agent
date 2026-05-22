from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from . import models, database, auth
from .agents.graph import process_user_message, stream_user_message
from .agents.fitness_agent import estimate_exercise_calories
from .food_api import search_food_nutrient
from pydantic import BaseModel
from typing import List, Optional, Literal
import asyncio
import os
import uuid
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

# ========== 静态文件 ==========
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_guide_dir = os.path.join(_backend_dir, "static", "guide")
_avatars_dir = os.path.join(_backend_dir, "static", "avatars")
_feedback_dir = os.path.join(_backend_dir, "static", "feedback")
if os.path.isdir(_guide_dir):
    app.mount("/guide", StaticFiles(directory=_guide_dir), name="guide")
os.makedirs(_avatars_dir, exist_ok=True)
app.mount("/avatars", StaticFiles(directory=_avatars_dir), name="avatars")
os.makedirs(_feedback_dir, exist_ok=True)

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

class ProfileUpdate(BaseModel):
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None

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

class FoodLogResponse(BaseModel):
    id: int
    name: str
    calories: float
    meal_type: Optional[str] = None
    log_id: int
    class Config:
        from_attributes = True

class ExerciseLogResponse(BaseModel):
    id: int
    type: str
    name: Optional[str] = None
    sets: Optional[int] = None
    weight: Optional[float] = None
    duration: int
    calories: float
    log_id: int
    class Config:
        from_attributes = True

class DailyLogResponse(BaseModel):
    id: int
    date: date
    intake_calories: float
    burn_calories: float
    weight_log: Optional[float] = None
    food_items: list[FoodLogResponse] = []
    exercise_items: list[ExerciseLogResponse] = []
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

class FoodLogCreate(BaseModel):
    name: str
    calories: Optional[float] = None
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"]

class ExerciseLogCreate(BaseModel):
    type: str
    duration: int  # 分钟
    name: Optional[str] = None
    sets: Optional[int] = None
    weight: Optional[float] = None

class FeedbackCreate(BaseModel):
    content: str
    contact: Optional[str] = None

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

@router.post("/user/profile", response_model=UserResponse)
def update_profile(
    data: ProfileUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    if data.nickname is not None:
        current_user.nickname = data.nickname
    if data.avatar_url is not None:
        current_user.avatar_url = data.avatar_url
    db.commit()
    db.refresh(current_user)
    return current_user

@router.post("/user/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    ext = os.path.splitext(file.filename)[1] if file.filename else '.png'
    filename = f"avatar_{current_user.id}_{uuid.uuid4().hex[:8]}{ext}"
    avatar_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "static", "avatars"
    )
    os.makedirs(avatar_dir, exist_ok=True)
    filepath = os.path.join(avatar_dir, filename)
    content = await file.read()
    with open(filepath, 'wb') as f:
        f.write(content)

    base_url = os.getenv("API_BASE_URL", "https://gzyapi.gzyhm.xyz")
    avatar_url = f"{base_url}/avatars/{filename}"
    current_user.avatar_url = avatar_url
    db.commit()
    db.refresh(current_user)
    return {"avatar_url": avatar_url}

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
    from sqlalchemy.orm import joinedload
    today = date.today()
    log = db.query(models.DailyLog).options(
        joinedload(models.DailyLog.food_items),
        joinedload(models.DailyLog.exercise_items),
    ).filter(
        models.DailyLog.user_id == current_user.id,
        models.DailyLog.date == today,
    ).first()
    if not log:
        log = models.DailyLog(user_id=current_user.id, date=today)
        db.add(log)
        db.commit()
        db.refresh(log)
    return log

# ----- 快捷记录 -----
@router.post("/food-log", response_model=FoodLogResponse)
def create_food_log(
    data: FoodLogCreate,
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

    calories = data.calories
    if calories is None:
        result = search_food_nutrient(data.name)
        calories = result["calories"] if result else 0

    item = models.FoodItem(
        log_id=log.id,
        name=data.name,
        calories=calories,
        meal_type=data.meal_type,
    )
    db.add(item)
    log.intake_calories = (log.intake_calories or 0) + calories
    db.commit()
    db.refresh(item)
    return FoodLogResponse(
        id=item.id, name=item.name, calories=item.calories,
        meal_type=item.meal_type, log_id=item.log_id,
    )

@router.post("/exercise-log", response_model=ExerciseLogResponse)
def create_exercise_log(
    data: ExerciseLogCreate,
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

    body_weight = current_user.weight or 60
    calories = estimate_exercise_calories(data.type, data.duration, "medium", body_weight)

    item = models.ExerciseItem(
        log_id=log.id,
        type=data.type,
        name=data.name,
        sets=data.sets,
        weight=data.weight,
        duration=data.duration,
        calories=calories,
    )
    db.add(item)
    log.burn_calories = (log.burn_calories or 0) + calories
    db.commit()
    db.refresh(item)
    return ExerciseLogResponse(
        id=item.id, type=item.type, name=item.name, sets=item.sets,
        weight=item.weight, duration=item.duration,
        calories=item.calories, log_id=item.log_id,
    )

# ----- 对话 -----
def _build_user_context(user: models.User, db: Session):
    # 只包含用户实际填写的字段，跳过零值和默认值
    user_profile = {}
    if user.height: user_profile["height"] = user.height
    if user.weight: user_profile["weight"] = user.weight
    if user.age: user_profile["age"] = user.age
    if user.gender and user.gender != "未知": user_profile["gender"] = user.gender
    if user.bmr: user_profile["bmr"] = user.bmr
    if user.tdee: user_profile["tdee"] = user.tdee
    if user.allergies: user_profile["allergies"] = user.allergies

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
        import queue
        import threading

        q = queue.Queue()

        def run():
            try:
                print(f"[stream] 开始处理用户 {current_user.id} 的消息: {user_message[:50]}...", flush=True)
                for item in stream_user_message(
                    user_message, current_user.id, user_profile, daily_stats
                ):
                    # item 是 tuple: ("status", msg) 或 ("data", msg)
                    q.put(("chunk", item))
                q.put(("done", None))
                print(f"[stream] 用户 {current_user.id} 的消息处理完成", flush=True)
            except Exception as e:
                print(f"[stream] 用户 {current_user.id} 的消息处理异常: {e}", flush=True)
                import traceback
                traceback.print_exc()
                q.put(("error", str(e)))

        threading.Thread(target=run, daemon=True).start()

        while True:
            try:
                # 30 秒超时等待，超时发心跳保活
                msg_type, data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=30)
                )
                if msg_type == "chunk":
                    event_type, content = data
                    if event_type == "status":
                        yield f"event: status\ndata: {content}\n\n"
                    else:
                        yield f"data: {content}\n\n"
                elif msg_type == "done":
                    yield "data: [DONE]\n\n"
                    break
                elif msg_type == "error":
                    yield f"data: Error: {data}\n\n"
                    yield "data: [DONE]\n\n"
                    break
            except Exception:
                # queue.Empty 超时，发 SSE 注释心跳（客户端忽略）
                yield ": heartbeat\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ----- 反馈 -----
@router.post("/feedback")
def submit_feedback(
    data: FeedbackCreate,
    current_user: models.User = Depends(auth.get_current_user),
):
    from datetime import datetime
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    nickname = current_user.nickname or "未设置"
    contact = data.contact or "未提供"

    filename = f"{now.strftime('%Y-%m-%d')}_{current_user.id}_{ts}.md"
    filepath = os.path.join(_feedback_dir, filename)

    content = f"""# 用户反馈

- 时间: {date_str}
- 用户ID: {current_user.id}
- 昵称: {nickname}
- 联系方式: {contact}

## 反馈内容

{data.content}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return {"message": "反馈已提交", "filename": filename}


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
