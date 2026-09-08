from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, DateTime, Boolean, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # 微信登录字段
    openid = Column(String, unique=True, index=True, nullable=True)
    unionid = Column(String, index=True, nullable=True)
    session_key = Column(String, nullable=True)
    nickname = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    # 身体数据
    height = Column(Float, nullable=False, default=0)
    weight = Column(Float, nullable=False, default=0)
    age = Column(Integer, nullable=False, default=0)
    gender = Column(String, nullable=False, default="未知")
    target_weight = Column(Float, nullable=True)
    allergies = Column(String, nullable=True)
    training_preference = Column(String, nullable=True)  # 训练偏好
    dietary_preference = Column(String, nullable=True)  # 饮食偏好
    goal = Column(String, nullable=True)
    calorie_adjustment = Column(Float, nullable=True)  # 热量缺口(负)或盈余(正)
    bmr = Column(Float, nullable=True)
    tdee = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("DailyLog", back_populates="user")

class DailyLog(Base):
    __tablename__ = "daily_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_daily_log_user_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date, default=datetime.utcnow().date())
    intake_calories = Column(Float, default=0.0)
    burn_calories = Column(Float, default=0.0)
    weight_log = Column(Float, nullable=True)
    notes = Column(String, nullable=True)

    user = relationship("User", back_populates="logs")
    food_items = relationship("FoodItem", back_populates="log")
    exercise_items = relationship("ExerciseItem", back_populates="log")

class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(Integer, ForeignKey("daily_logs.id"))
    name = Column(String, nullable=False)
    calories = Column(Float, nullable=False)
    meal_type = Column(String, nullable=True)  # breakfast/lunch/dinner/snack
    portion_qty = Column(Float, nullable=True)
    portion_unit = Column(String, nullable=True)

    log = relationship("DailyLog", back_populates="food_items")

class ExerciseItem(Base):
    __tablename__ = "exercise_items"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(Integer, ForeignKey("daily_logs.id"))
    type = Column(String, nullable=False)
    name = Column(String, nullable=True)
    sets = Column(Integer, nullable=True)
    reps = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True)
    duration = Column(Integer, nullable=False)
    calories = Column(Float, nullable=False)

    log = relationship("DailyLog", back_populates="exercise_items")

class ExerciseCalorie(Base):
    __tablename__ = "exercise_calories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    calories_per_set = Column(Float, nullable=False)
    category = Column(String, nullable=True)
    aliases = Column(String, nullable=True)  # JSON 数组：["卧推", "平板卧推"]

class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, nullable=True)
    agent_type = Column(String, nullable=False)
    user_message = Column(String, nullable=False)
    agent_response = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class ConversationSession(Base):
    """会话级工作记忆。

    ConversationLog 继续作为逐轮情景记忆的事实来源；本表只保存会话
    状态、路由和幂等游标，避免再复制一份完整聊天记录。
    """

    __tablename__ = "conversation_sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "session_id", name="uq_conversation_session_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    summary = Column(Text, nullable=True)
    last_agent = Column(String, nullable=True)
    last_route_json = Column(Text, nullable=True)
    working_memory_json = Column(Text, nullable=True)
    last_request_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class UserMemory(Base):
    """用户确认或高可信的语义记忆，不重复存储业务日志事实。"""

    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "memory_key", name="uq_user_memory_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    memory_key = Column(String, nullable=False)
    memory_value = Column(Text, nullable=False)
    memory_type = Column(String, nullable=False, default="semantic")
    source = Column(String, nullable=True)
    confidence = Column(Float, nullable=False, default=0.5)
    confirmed = Column(Boolean, nullable=False, default=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")

class FoodCalorieCache(Base):
    __tablename__ = "food_calorie_cache"
    __table_args__ = (
        UniqueConstraint("name", name="uq_food_calorie_cache_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    portion_qty = Column(Float, nullable=True)
    portion_unit = Column(String, nullable=True)
    calories = Column(Float, nullable=False)
    source = Column(String, nullable=False, default="llm")  # "api" or "llm"
    created_at = Column(DateTime, default=datetime.utcnow)
