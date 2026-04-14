from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    target_weight = Column(Float, nullable=True)
    allergies = Column(String, nullable=True)
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

    log = relationship("DailyLog", back_populates="food_items")

class ExerciseItem(Base):
    __tablename__ = "exercise_items"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(Integer, ForeignKey("daily_logs.id"))
    type = Column(String, nullable=False)
    duration = Column(Integer, nullable=False) # in minutes
    calories = Column(Float, nullable=False)

    log = relationship("DailyLog", back_populates="exercise_items")
