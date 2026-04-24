"""统计汇总模块 - 每日/每周营养和运动数据汇总"""

from typing import Dict, Any, Optional, List
from datetime import date, timedelta
from sqlalchemy import func
from .. import models, database


class StatsSummarizer:
    """统计汇总器

    功能：
    1. 获取用户当日营养/运动统计
    2. 获取本周统计趋势
    3. 格式化统计数据为 Agent 可读格式
    """

    @staticmethod
    def get_today_stats(user_id: int) -> Dict[str, Any]:
        """获取用户当日统计

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 当日统计数据
        """
        db = database.SessionLocal()
        try:
            today = date.today()
            log = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date == today
            ).first()

            user = db.query(models.User).filter(models.User.id == user_id).first()
            tdee = user.tdee if user else 2000

            if not log:
                return {
                    "date": today.isoformat(),
                    "intake_calories": 0,
                    "burn_calories": 0,
                    "net_calories": 0,
                    "tdee": tdee,
                    "calorie_balance": -tdee,
                    "food_count": 0,
                    "exercise_count": 0,
                    "food_items": [],
                    "exercise_items": []
                }

            food_items = [{
                "name": item.name,
                "calories": item.calories
            } for item in log.food_items]

            exercise_items = [{
                "type": item.type,
                "duration": item.duration,
                "calories": item.calories,
                "notes": item.notes
            } for item in log.exercise_items]

            return {
                "date": today.isoformat(),
                "intake_calories": log.intake_calories,
                "burn_calories": log.burn_calories,
                "net_calories": log.intake_calories - log.burn_calories,
                "tdee": tdee,
                "calorie_balance": log.intake_calories - log.burn_calories - tdee,
                "food_count": len(log.food_items),
                "exercise_count": len(log.exercise_items),
                "food_items": food_items,
                "exercise_items": exercise_items,
                "weight_log": log.weight_log
            }
        finally:
            db.close()

    @staticmethod
    def get_week_stats(user_id: int) -> Dict[str, Any]:
        """获取用户本周统计

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 本周统计数据
        """
        db = database.SessionLocal()
        try:
            today = date.today()
            week_start = today - timedelta(days=today.weekday())
            week_end = today

            logs = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date >= week_start,
                models.DailyLog.date <= week_end
            ).all()

            user = db.query(models.User).filter(models.User.id == user_id).first()
            tdee = user.tdee if user else 2000

            if not logs:
                return {
                    "week_start": week_start.isoformat(),
                    "week_end": week_end.isoformat(),
                    "days_logged": 0,
                    "total_intake": 0,
                    "total_burn": 0,
                    "avg_intake": 0,
                    "avg_burn": 0,
                    "days_below_tdee": 0,
                    "days_above_tdee": 0,
                    "daily_logs": []
                }

            daily_logs = []
            total_intake = 0
            total_burn = 0
            days_below = 0
            days_above = 0

            for log in logs:
                intake = log.intake_calories
                burn = log.burn_calories
                total_intake += intake
                total_burn += burn

                daily_logs.append({
                    "date": log.date.isoformat(),
                    "intake": intake,
                    "burn": burn,
                    "net": intake - burn
                })

                if intake < tdee:
                    days_below += 1
                else:
                    days_above += 1

            days_count = len(logs) if logs else 1

            return {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "days_logged": len(logs),
                "total_intake": total_intake,
                "total_burn": total_burn,
                "avg_intake": total_intake / days_count,
                "avg_burn": total_burn / days_count,
                "days_below_tdee": days_below,
                "days_above_tdee": days_above,
                "daily_logs": daily_logs
            }
        finally:
            db.close()

    @staticmethod
    def get_recent_food_history(user_id: int, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近的饮食记录

        Args:
            user_id: 用户 ID
            days: 天数

        Returns:
            List: 最近的饮食记录
        """
        db = database.SessionLocal()
        try:
            start_date = date.today() - timedelta(days=days)

            logs = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date >= start_date
            ).order_by(models.DailyLog.date.desc()).all()

            result = []
            for log in logs:
                foods = [{
                    "name": item.name,
                    "calories": item.calories
                } for item in log.food_items]

                result.append({
                    "date": log.date.isoformat(),
                    "foods": foods,
                    "total_calories": log.intake_calories
                })

            return result
        finally:
            db.close()

    @staticmethod
    def get_recent_exercise_history(user_id: int, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近的运动记录

        Args:
            user_id: 用户 ID
            days: 天数

        Returns:
            List: 最近的运动记录
        """
        db = database.SessionLocal()
        try:
            start_date = date.today() - timedelta(days=days)

            logs = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == user_id,
                models.DailyLog.date >= start_date
            ).order_by(models.DailyLog.date.desc()).all()

            result = []
            for log in logs:
                exercises = [{
                    "type": item.type,
                    "duration": item.duration,
                    "calories": item.calories,
                    "notes": item.notes
                } for item in log.exercise_items]

                result.append({
                    "date": log.date.isoformat(),
                    "exercises": exercises,
                    "total_calories": log.burn_calories
                })

            return result
        finally:
            db.close()

    @staticmethod
    def format_today_for_agent(stats: Dict[str, Any]) -> str:
        """格式化当日统计为 Agent 可读格式

        Args:
            stats: 当日统计数据

        Returns:
            str: 格式化后的字符串
        """
        if not stats:
            return "【今日统计】暂无数据"

        date_str = stats.get("date", "未知")
        intake = stats.get("intake_calories", 0)
        burn = stats.get("burn_calories", 0)
        net = stats.get("net_calories", 0)
        tdee = stats.get("tdee", 2000)
        balance = stats.get("calorie_balance", -tdee)
        food_count = stats.get("food_count", 0)
        exercise_count = stats.get("exercise_count", 0)

        food_items = stats.get("food_items", [])
        exercise_items = stats.get("exercise_items", [])

        result = f"""【今日统计】{date_str}
- 摄入热量: {intake:.0f} kcal
- 消耗热量: {burn:.0f} kcal
- 净热量: {net:.0f} kcal
- 目标TDEE: {tdee:.0f} kcal
- 热量平衡: {'+' if balance > 0 else ''}{balance:.0f} kcal
- 记录食物: {food_count} 种
- 记录运动: {exercise_count} 项"""

        if food_items:
            food_list = ", ".join([f"{f['name']}({f['calories']:.0f}kcal)" for f in food_items[:5]])
            if len(food_items) > 5:
                food_list += f" 等{food_count}种"
            result += f"\n- 已摄入食物: {food_list}"

        if exercise_items:
            ex_list = ", ".join([f"{e['type']}({e['duration']}分钟)" for e in exercise_items[:3]])
            result += f"\n- 已完成运动: {ex_list}"

        return result

    @staticmethod
    def format_week_for_agent(stats: Dict[str, Any]) -> str:
        """格式化本周统计为 Agent 可读格式

        Args:
            stats: 本周统计数据

        Returns:
            str: 格式化后的字符串
        """
        if not stats or stats.get("days_logged", 0) == 0:
            return "【本周统计】暂无数据"

        week_start = stats.get("week_start", "未知")
        week_end = stats.get("week_end", "未知")
        days_logged = stats.get("days_logged", 0)
        avg_intake = stats.get("avg_intake", 0)
        avg_burn = stats.get("avg_burn", 0)
        total_intake = stats.get("total_intake", 0)
        total_burn = stats.get("total_burn", 0)
        days_below = stats.get("days_below_tdee", 0)
        days_above = stats.get("days_above_tdee", 0)

        return f"""【本周统计】{week_start} 至 {week_end}
- 记录天数: {days_logged} 天
- 总摄入: {total_intake:.0f} kcal
- 总消耗: {total_burn:.0f} kcal
- 日均摄入: {avg_intake:.0f} kcal
- 日均消耗: {avg_burn:.0f} kcal
- 达标天数(摄入<TDEE): {days_below} 天
- 超标天数(摄入≥TDEE): {days_above} 天"""

    @staticmethod
    def get_context_for_nutrition(stats: Dict[str, Any]) -> str:
        """获取营养相关的上下文信息

        Args:
            stats: 当日统计数据

        Returns:
            str: 营养上下文
        """
        if not stats:
            return "今日暂无饮食记录"

        intake = stats.get("intake_calories", 0)
        tdee = stats.get("tdee", 2000)
        remaining = tdee - intake
        food_count = stats.get("food_count", 0)

        if remaining > 0:
            return f"今日已摄入 {intake:.0f} kcal，还剩 {remaining:.0f} kcal 可摄入（目标TDEE: {tdee:.0f} kcal）"
        else:
            return f"今日已摄入 {intake:.0f} kcal，已超出目标 {abs(remaining):.0f} kcal（目标TDEE: {tdee:.0f} kcal）"

    @staticmethod
    def get_context_for_fitness(stats: Dict[str, Any]) -> str:
        """获取运动相关的上下文信息

        Args:
            stats: 当日统计数据

        Returns:
            str: 运动上下文
        """
        if not stats:
            return "今日暂无运动记录"

        burn = stats.get("burn_calories", 0)
        exercise_count = stats.get("exercise_count", 0)
        exercise_items = stats.get("exercise_items", [])

        if exercise_items:
            recent = exercise_items[-1]
            return f"今日已运动 {exercise_count} 项，消耗 {burn:.0f} kcal。最近完成: {recent['type']} {recent['duration']}分钟"
        else:
            return f"今日已消耗 {burn:.0f} kcal（来自 {exercise_count} 项运动）"
