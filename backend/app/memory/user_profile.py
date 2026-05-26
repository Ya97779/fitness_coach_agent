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
                    "type": user.goal or None,
                    "target_weight": user.target_weight,
                    "current_weight": user.weight,
                    "weight_diff": user.weight - (user.target_weight or user.weight)
                },
                "constraints": {
                    "allergies": user.allergies or "无",
                },
                "preferences": {
                    "training": user.training_preference or None,
                    "dietary": user.dietary_preference or None,
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

            result = {}
            if user.height: result["身高"] = f"{user.height} cm"
            if user.weight: result["体重"] = f"{user.weight} kg"
            if user.age: result["年龄"] = f"{user.age} 岁"
            if user.gender and user.gender != "未知": result["性别"] = user.gender
            if user.bmr: result["BMR"] = f"{user.bmr:.0f} kcal"
            if user.tdee: result["TDEE"] = f"{user.tdee:.0f} kcal"
            if user.target_weight: result["目标体重"] = f"{user.target_weight} kg"
            if user.goal: result["健身目标"] = user.goal
            if user.allergies: result["过敏史"] = user.allergies
            if user.training_preference: result["训练偏好"] = user.training_preference
            if user.dietary_preference: result["饮食偏好"] = user.dietary_preference
            return result
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
            if not user:
                return "维持现状"
            if user.goal:
                return user.goal
            if not user.target_weight:
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
        """格式化用户画像为 Agent 可读的字符串。只包含用户实际填写的字段。"""
        if not profile or "basic_info" not in profile:
            return "【用户状态】新用户，尚未填写身体数据"

        basic = profile.get("basic_info", {})
        metrics = profile.get("body_metrics", {})
        constraints = profile.get("constraints", {})

        lines = []

        # 基本信息：只输出非零值
        basic_lines = []
        h = basic.get("height")
        w = basic.get("weight")
        a = basic.get("age")
        g = basic.get("gender")
        if h: basic_lines.append(f"- 身高: {h} cm")
        if w: basic_lines.append(f"- 体重: {w} kg")
        if a: basic_lines.append(f"- 年龄: {a} 岁")
        if g and g != "未知": basic_lines.append(f"- 性别: {g}")
        if basic_lines:
            lines.append("【用户基本信息】")
            lines.extend(basic_lines)

        # 身体指标：只输出非空值
        bmr = metrics.get("bmr")
        tdee = metrics.get("tdee")
        metric_lines = []
        if bmr: metric_lines.append(f"- 基础代谢率(BMR): {bmr:.0f} kcal")
        if tdee: metric_lines.append(f"- 每日总消耗(TDEE): {tdee:.0f} kcal")
        if metric_lines:
            lines.append("【身体指标】")
            lines.extend(metric_lines)

        # 目标和约束
        target_w = (profile.get("goal") or {}).get("target_weight")
        allergies = constraints.get("allergies")
        if goal and goal != "维持现状" or target_w or (allergies and allergies != "无"):
            lines.append("【用户目标】")
            if goal and goal != "维持现状":
                lines.append(f"- 目标: {goal}")
            if target_w:
                lines.append(f"- 目标体重: {target_w} kg")
            if allergies and allergies != "无":
                lines.append(f"- 过敏史: {allergies}")

        # 偏好设置
        preferences = profile.get("preferences", {})
        training_pref = preferences.get("training")
        dietary_pref = preferences.get("dietary")
        if training_pref or dietary_pref:
            lines.append("【偏好设置】")
            if training_pref:
                lines.append(f"- 训练偏好: {training_pref}")
            if dietary_pref:
                lines.append(f"- 饮食偏好: {dietary_pref}")

        return "\n".join(lines) if lines else "【用户状态】新用户，尚未填写身体数据"

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
