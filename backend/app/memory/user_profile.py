"""用户画像加载模块 - 从数据库加载用户信息并格式化为 Agent 上下文"""

from typing import Dict, Any, Optional
from datetime import date
from .. import models, database


class UserProfileLoader:
    """用户画像加载器

    从数据库加载用户信息，并格式化为适合 Agent 使用的上下文格式。
    """

    @staticmethod
    def load_user_profile(user_id: int) -> Dict[str, Any]:
        """加载用户完整画像

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 用户画像信息
        """
        db = database.SessionLocal()
        try:
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if not user:
                return UserProfileLoader._get_default_profile()

            profile = {
                "user_id": user.id,
                "basic_info": {
                    "height": user.height,
                    "weight": user.weight,
                    "age": user.age,
                    "gender": user.gender,
                },
                "body_metrics": {
                    "bmr": user.bmr,
                    "tdee": user.tdee,
                },
                "goal": {
                    "target_weight": user.target_weight,
                    "current_weight": user.weight,
                    "weight_diff": user.weight - (user.target_weight or user.weight)
                },
                "constraints": {
                    "allergies": user.allergies or "无",
                },
                "created_at": user.created_at.isoformat() if user.created_at else None
            }

            return profile
        finally:
            db.close()

    @staticmethod
    def load_compact_profile(user_id: int) -> Dict[str, Any]:
        """加载紧凑型用户画像（用于上下文注入）

        只包含最关键的信息，控制 token 消耗。

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 紧凑型用户画像
        """
        db = database.SessionLocal()
        try:
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if not user:
                return {
                    "身高": "未知",
                    "体重": "未知",
                    "BMR": "未知",
                    "TDEE": "未知"
                }

            return {
                "身高": f"{user.height} cm",
                "体重": f"{user.weight} kg",
                "年龄": f"{user.age} 岁",
                "性别": user.gender,
                "BMR": f"{user.bmr:.0f} kcal" if user.bmr else "未知",
                "TDEE": f"{user.tdee:.0f} kcal" if user.tdee else "未知",
                "目标体重": f"{user.target_weight} kg" if user.target_weight else "未设定",
                "过敏史": user.allergies or "无"
            }
        finally:
            db.close()

    @staticmethod
    def get_user_goal(user_id: int) -> str:
        """根据目标体重和当前体重判断用户目标

        Args:
            user_id: 用户 ID

        Returns:
            str: 目标描述（增肌/减脂/维持）
        """
        db = database.SessionLocal()
        try:
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if not user or not user.target_weight:
                return "维持现状"

            diff = user.weight - user.target_weight
            if diff > 2:
                return "减脂"
            elif diff < -2:
                return "增肌"
            else:
                return "维持"
        finally:
            db.close()

    @staticmethod
    def format_profile_for_agent(profile: Dict[str, Any], goal: str) -> str:
        """格式化用户画像为 Agent 可读的字符串

        Args:
            profile: 用户画像字典
            goal: 用户目标

        Returns:
            str: 格式化后的字符串
        """
        if not profile or "basic_info" not in profile:
            return "用户信息未找到"

        basic = profile.get("basic_info", {})
        metrics = profile.get("body_metrics", {})
        constraints = profile.get("constraints", {})

        return f"""【用户基本信息】
- 身高: {basic.get('height', '未知')} cm
- 体重: {basic.get('weight', '未知')} kg
- 年龄: {basic.get('age', '未知')} 岁
- 性别: {basic.get('gender', '未知')}

【身体指标】
- 基础代谢率(BMR): {metrics.get('bmr', '未知')} kcal
- 每日总消耗(TDEE): {metrics.get('tdee', '未知')} kcal

【用户目标】
- 目标: {goal}
- 过敏史: {constraints.get('allergies', '无')}"""

    @staticmethod
    def _get_default_profile() -> Dict[str, Any]:
        """获取默认用户画像"""
        return {
            "user_id": None,
            "basic_info": {
                "height": 0,
                "weight": 0,
                "age": 0,
                "gender": "未知"
            },
            "body_metrics": {
                "bmr": 0,
                "tdee": 0
            },
            "goal": {
                "target_weight": None,
                "current_weight": 0,
                "weight_diff": 0
            },
            "constraints": {
                "allergies": "无"
            }
        }
