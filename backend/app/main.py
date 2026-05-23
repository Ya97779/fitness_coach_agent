from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from . import models, database, auth
from .agents.graph import process_user_message, stream_user_message
from .agents.fitness_agent import estimate_exercise_calories
from .llm_manager import LLMManager
import json as _json
from .food_api import search_food_nutrient
from pydantic import BaseModel
from typing import List, Optional, Literal
import asyncio
import os
import threading
import uuid
import logging
from datetime import date
from langchain_core.messages import HumanMessage

logger = logging.getLogger("food_estimate")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
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

def init_exercise_calories():
    """初始化预置动作热量数据（补缺）"""
    db = database.SessionLocal()
    try:
        existing = {item.name for item in db.query(models.ExerciseCalorie.name).all()}
        presets = [
            # 胸
            ("平板卧推", 8, "胸", ["卧推", "杠铃卧推", "平板杠铃卧推"]),
            ("上斜卧推", 8, "胸", ["上斜杠铃卧推"]),
            ("哑铃卧推", 7.5, "胸", ["平板哑铃卧推"]),
            ("上斜哑铃卧推", 8, "胸", []),
            ("龙门架夹胸", 6, "胸", ["夹胸", "绳索夹胸", "飞鸟"]),
            ("蝴蝶机夹胸", 5.5, "胸", ["蝴蝶机"]),
            ("俯卧撑", 5, "胸", []),
            ("双杠臂屈伸", 8, "胸", ["双杠", "臂屈伸"]),
            # 背
            ("引体向上", 10, "背", ["引体", "引体向上", "正手引体"]),
            ("杠铃划船", 8, "背", ["俯身划船", "俯身杠铃划船"]),
            ("哑铃划船", 7, "背", ["单臂哑铃划船"]),
            ("坐姿划船", 6.5, "背", ["绳索坐姿划船", "坐姿绳索划船", "器械坐姿划船"]),
            ("高位下拉", 6, "背", ["下拉", "引体下拉"]),
            ("硬拉", 10, "背", ["传统硬拉", "杠铃硬拉"]),
            ("罗马尼亚硬拉", 9, "背", ["罗拉"]),
            # 肩
            ("杠铃推举", 8, "肩", ["推举", "站姿推举", "肩推"]),
            ("哑铃推举", 7, "肩", ["坐姿哑铃推举", "肩推"]),
            ("侧平举", 5, "肩", ["哑铃侧平举"]),
            ("前平举", 4.5, "肩", ["哑铃前平举"]),
            ("俯身飞鸟", 5, "肩", ["俯身侧平举", "反向飞鸟"]),
            ("面拉", 5, "肩", ["绳索面拉"]),
            # 腿
            ("深蹲", 10, "腿", ["杠铃深蹲", "杠铃深蹲", "颈后深蹲"]),
            ("前蹲", 9.5, "腿", ["杠铃前蹲"]),
            ("腿举", 7, "腿", ["腿举机"]),
            ("箭步蹲", 8, "腿", ["弓步蹲", "保加利亚深蹲"]),
            ("腿弯举", 5, "腿", ["俯卧腿弯举"]),
            ("腿屈伸", 5, "腿", ["坐姿腿屈伸", "腿举"]),
            ("小腿提踵", 4, "腿", ["提踵", "站姿提踵"]),
            # 手臂
            ("杠铃弯举", 5, "手臂", ["弯举", "二头弯举"]),
            ("哑铃弯举", 4.5, "手臂", ["交替弯举"]),
            ("锤式弯举", 4.5, "手臂", ["锤式"]),
            ("三头下压", 4.5, "手臂", ["绳索下压", "三头绳索下压"]),
            ("窄距卧推", 7, "手臂", ["窄握卧推"]),
            ("仰卧臂屈伸", 5, "手臂", ["碎颅者"]),
            # 核心
            ("平板支撑", 3, "核心", ["plank"]),
            ("卷腹", 3, "核心", ["仰卧起坐"]),
            ("悬垂举腿", 5, "核心", ["举腿"]),
            ("俄罗斯转体", 3.5, "核心", []),
        ]
        added = 0
        for name, cal, cat, aliases in presets:
            if name not in existing:
                db.add(models.ExerciseCalorie(
                    name=name,
                    calories_per_set=cal,
                    category=cat,
                    aliases=_json.dumps(aliases, ensure_ascii=False)
                ))
                added += 1
            else:
                # 预设优先：覆盖 LLM 缓存的空别名条目
                item = db.query(models.ExerciseCalorie).filter(
                    models.ExerciseCalorie.name == name
                ).first()
                if item and (not item.aliases or item.aliases == '[]'):
                    item.calories_per_set = cal
                    item.aliases = _json.dumps(aliases, ensure_ascii=False)
                    added += 1
        if added:
            db.commit()
            print(f"[热量表] 补充 {added} 个预置动作")
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    init_exercise_calories()
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
    goal: Optional[str] = None

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
    goal: Optional[str] = None
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
    estimating: bool = False
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
    portion_qty: Optional[float] = None
    portion_unit: Optional[str] = None

class ExerciseLogCreate(BaseModel):
    type: str
    duration: int  # 分钟
    name: Optional[str] = None
    sets: Optional[int] = None
    weight: Optional[float] = None
    calories: Optional[float] = None

class FoodLogUpdate(BaseModel):
    name: Optional[str] = None
    calories: Optional[float] = None
    meal_type: Optional[str] = None

class ExerciseLogUpdate(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None
    sets: Optional[int] = None
    weight: Optional[float] = None
    duration: Optional[int] = None
    calories: Optional[float] = None

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
    current_user.goal = user_data.goal
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
    logger.info(f"[food-log] 收到请求: name='{data.name}', calories={data.calories}, qty={data.portion_qty}, unit={data.portion_unit}, meal={data.meal_type}")
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
    need_llm = False
    qty = data.portion_qty
    unit = data.portion_unit

    if calories is None:
        if unit == '克' and qty:
            # 克模式：查 API 每100g热量，按克数换算
            result = search_food_nutrient(data.name)
            if result:
                calories = round(result["calories"] * qty / 100)
            else:
                need_llm = True
        elif qty and unit:
            # 非克单位（份/碗/个）：API 返回的是每100g，无法换算，直接 LLM
            need_llm = True
        else:
            # 没填份量：走原有逻辑
            result = search_food_nutrient(data.name)
            if result:
                calories = result["calories"]
            else:
                need_llm = True

    if need_llm:
        calories = 0  # 先存 0，后台 LLM 算完再更新
        logger.info(f"[food-log] 需要 LLM 估算: '{data.name}'")
    else:
        logger.info(f"[food-log] 直接计算热量: '{data.name}' → {calories} kcal")

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

    if need_llm:
        # 后台线程调 LLM 估算，算完更新记录
        _item_id = item.id
        _food_name = data.name
        logger.info(f"[食物热量估算] 启动后台线程: item_id={_item_id}, name='{_food_name}', qty={qty}, unit={unit}")

        def _bg_estimate():
            try:
                logger.info(f"[食物热量估算] 开始 LLM 调用: '{_food_name}'")
                estimated = _estimate_food_calories_via_llm(_food_name, qty, unit)
                logger.info(f"[食物热量估算] LLM 返回: '{_food_name}' → {estimated} kcal")
                if estimated:
                    _db = database.SessionLocal()
                    try:
                        _item = _db.query(models.FoodItem).get(_item_id)
                        if _item:
                            _log = _db.query(models.DailyLog).get(_item.log_id)
                            _old = _item.calories or 0
                            _item.calories = estimated
                            if _log:
                                _log.intake_calories = (_log.intake_calories or 0) - _old + estimated
                            _db.commit()
                            logger.info(f"[食物热量估算] DB 更新成功: '{_food_name}' → {estimated} kcal (旧值: {_old})")
                        else:
                            logger.warning(f"[食物热量估算] item_id={_item_id} 不存在，可能已被删除")
                    finally:
                        _db.close()
                else:
                    logger.warning(f"[食物热量估算] LLM 返回 0，跳过更新: '{_food_name}'")
            except Exception as e:
                logger.error(f"[食物热量估算] 后台估算失败: '{_food_name}' → {e}", exc_info=True)
        threading.Thread(target=_bg_estimate, daemon=True).start()

    return FoodLogResponse(
        id=item.id, name=item.name, calories=item.calories,
        meal_type=item.meal_type, log_id=item.log_id,
        estimating=need_llm,
    )

@router.patch("/food-log/{item_id}", response_model=FoodLogResponse)
def update_food_log(
    item_id: int,
    data: FoodLogUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.FoodItem).join(models.DailyLog).filter(
        models.FoodItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    old_calories = item.calories or 0
    if data.name is not None:
        item.name = data.name
    if data.calories is not None:
        item.calories = data.calories
    if data.meal_type is not None:
        item.meal_type = data.meal_type

    log = db.query(models.DailyLog).get(item.log_id)
    if log and data.calories is not None:
        log.intake_calories = (log.intake_calories or 0) - old_calories + data.calories

    db.commit()
    db.refresh(item)
    return FoodLogResponse(
        id=item.id, name=item.name, calories=item.calories,
        meal_type=item.meal_type, log_id=item.log_id,
    )

@router.delete("/food-log/{item_id}", status_code=204)
def delete_food_log(
    item_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.FoodItem).join(models.DailyLog).filter(
        models.FoodItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    log = db.query(models.DailyLog).get(item.log_id)
    if log:
        log.intake_calories = (log.intake_calories or 0) - (item.calories or 0)

    db.delete(item)
    db.commit()

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
    calories = data.calories if data.calories else estimate_exercise_calories.invoke({"exercise_type": data.type, "duration": data.duration, "intensity": "medium", "user_weight": body_weight}).get("calories", 0)

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

@router.patch("/exercise-log/{item_id}", response_model=ExerciseLogResponse)
def update_exercise_log(
    item_id: int,
    data: ExerciseLogUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.ExerciseItem).join(models.DailyLog).filter(
        models.ExerciseItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    old_calories = item.calories or 0
    if data.type is not None:
        item.type = data.type
    if data.name is not None:
        item.name = data.name
    if data.sets is not None:
        item.sets = data.sets
    if data.weight is not None:
        item.weight = data.weight
    if data.duration is not None:
        item.duration = data.duration
    if data.calories is not None:
        item.calories = data.calories

    log = db.query(models.DailyLog).get(item.log_id)
    if log and data.calories is not None:
        log.burn_calories = (log.burn_calories or 0) - old_calories + data.calories

    db.commit()
    db.refresh(item)
    return ExerciseLogResponse(
        id=item.id, type=item.type, name=item.name, sets=item.sets,
        weight=item.weight, duration=item.duration,
        calories=item.calories, log_id=item.log_id,
    )

@router.delete("/exercise-log/{item_id}", status_code=204)
def delete_exercise_log(
    item_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    item = db.query(models.ExerciseItem).join(models.DailyLog).filter(
        models.ExerciseItem.id == item_id,
        models.DailyLog.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")

    log = db.query(models.DailyLog).get(item.log_id)
    if log:
        log.burn_calories = (log.burn_calories or 0) - (item.calories or 0)

    db.delete(item)
    db.commit()

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

# ========== 动作热量估算 ==========

def _find_exercise(db: Session, query_name: str):
    """查找动作热量数据：精确 → 别名 → 包含"""
    # 1. 精确匹配
    item = db.query(models.ExerciseCalorie).filter(models.ExerciseCalorie.name == query_name).first()
    if item:
        return item
    # 2. 别名匹配
    all_items = db.query(models.ExerciseCalorie).all()
    for item in all_items:
        if item.aliases:
            aliases = _json.loads(item.aliases)
            if query_name in aliases:
                return item
    # 3. 包含匹配（query_name 包含库中的名称，或库中的名称包含 query_name）
    best = None
    best_len = 0
    for item in all_items:
        if item.name in query_name or query_name in item.name:
            if len(item.name) > best_len:
                best = item
                best_len = len(item.name)
    return best

def _estimate_via_llm(exercises: list, user_weight: float) -> dict:
    """调用 LLM 估算热量"""
    exercise_desc = "\n".join(
        f"{i+1}. {e['name']} - {e['sets']}组, {e.get('weight', 0)}kg, 约{e.get('duration', 5)}分钟"
        for i, e in enumerate(exercises)
    )
    prompt = f"""你是运动热量估算专家。根据以下运动数据估算消耗的热量（kcal）。
用户体重：{user_weight}kg

运动数据：
{exercise_desc}

请返回 JSON 格式：
{{"details": [{{"name": "动作名", "calories": 数字}}]}}
只返回 JSON，不要其他内容。"""
    try:
        llm = LLMManager.get_llm(temperature=0.1)
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return _json.loads(text)
    except Exception as e:
        print(f"[热量估算] LLM 调用失败: {e}")
        return None

def _estimate_food_calories_via_llm(food_name: str, portion_qty: float = None, portion_unit: str = None) -> float:
    """调用 LLM 估算食物总热量（带超时保护）"""
    if portion_qty and portion_unit:
        desc = f"{portion_qty}{portion_unit}{food_name}"
    else:
        desc = food_name
    prompt = f"""你是食物营养专家。估算以下食物的总热量（kcal）。

食物："{desc}"

规则：
- 根据食物名称和份量估算总热量
- 如果是具体份量（如2份、200克、1碗），按该份量估算
- 如果没有明确份量，默认估算1份的热量
- 只返回一个整数数字，不要其他内容

示例：
鸡腿拌面 → 650
2份鸡腿拌面 → 1300
200克鸡胸肉 → 330
1个苹果 → 95
1碗米饭 → 230"""
    try:
        logger.info(f"[LLM] 获取 LLM 实例 (temperature=0.1)")
        llm = LLMManager.get_llm(temperature=0.1)
        logger.info(f"[LLM] 开始调用 invoke, prompt 长度={len(prompt)}")
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        logger.info(f"[LLM] 原始返回: '{text}'")
        import re
        match = re.search(r'[\d.]+', text)
        if match:
            result = float(match.group())
            logger.info(f"[LLM] 解析结果: {result}")
            return result
        else:
            logger.warning(f"[LLM] 无法从返回中解析数字: '{text}'")
    except Exception as e:
        logger.error(f"[LLM] 调用异常: {e}", exc_info=True)
    return 0

class CalorieEstimateRequest(BaseModel):
    exercises: List[dict]

class CalorieEstimateDetail(BaseModel):
    name: str
    calories: int

class CalorieEstimateResponse(BaseModel):
    total_calories: int
    details: List[CalorieEstimateDetail]

@router.post("/estimate-calories", response_model=CalorieEstimateResponse)
def estimate_calories(
    data: CalorieEstimateRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    user_weight = current_user.weight or 70
    details = []
    unknown_exercises = []
    unknown_indices = []

    # 第一轮：查表
    for i, ex in enumerate(data.exercises):
        name = ex.get("name", "")
        sets = ex.get("sets", 1)
        item = _find_exercise(db, name)
        if item:
            cal = round(item.calories_per_set * sets * (user_weight / 70))
            details.append({"name": name, "calories": cal})
        else:
            details.append({"name": name, "calories": 0})
            unknown_exercises.append(ex)
            unknown_indices.append(i)

    # 第二轮：LLM 估算未命中动作
    if unknown_exercises:
        llm_result = _estimate_via_llm(unknown_exercises, user_weight)
        if llm_result and "details" in llm_result:
            llm_map = {d["name"]: d["calories"] for d in llm_result["details"]}
            for idx, ex in zip(unknown_indices, unknown_exercises):
                cal = llm_map.get(ex["name"], round(ex.get("sets", 1) * 5 * (user_weight / 70)))
                details[idx]["calories"] = cal
                # 缓存到 DB
                existing = db.query(models.ExerciseCalorie).filter(
                    models.ExerciseCalorie.name == ex["name"]
                ).first()
                if not existing:
                    cal_per_set = round(cal / max(ex.get("sets", 1), 1) / (user_weight / 70), 1)
                    db.add(models.ExerciseCalorie(
                        name=ex["name"],
                        calories_per_set=cal_per_set,
                        aliases=_json.dumps([], ensure_ascii=False)
                    ))
            db.commit()
        else:
            # LLM 失败，用通用公式兜底
            for idx, ex in zip(unknown_indices, unknown_exercises):
                details[idx]["calories"] = round(ex.get("sets", 1) * 5 * (user_weight / 70))

    total = sum(d["calories"] for d in details)
    return CalorieEstimateResponse(total_calories=total, details=details)

# ========== 注册路由 ==========
app.include_router(router)

# 根路由健康检查（无需鉴权）
@app.get("/agents")
async def list_agents_root():
    return await list_agents()
