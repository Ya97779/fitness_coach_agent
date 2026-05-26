from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, DateTime
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

class FoodCalorieCache(Base):
    __tablename__ = "food_calorie_cache"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    portion_qty = Column(Float, nullable=True)
    portion_unit = Column(String, nullable=True)
    calories = Column(Float, nullable=False)
    source = Column(String, nullable=False, default="llm")  # "api" or "llm"
    created_at = Column(DateTime, default=datetime.utcnow)
