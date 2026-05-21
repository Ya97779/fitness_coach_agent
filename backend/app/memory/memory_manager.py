"""记忆管理器 - 整合用户画像、对话历史、统计数据的管理"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .user_profile import UserProfileLoader
from .conversation_summary import ConversationSummarizer
from .stats_summary import StatsSummarizer
from .. import models, database


class MemoryManager:
    """记忆管理器

    整合管理用户的所有记忆信息：
    1. 用户画像
    2. 对话历史（带摘要功能）
    3. 每日/每周统计

    使用方式：
        memory = MemoryManager(user_id=1)
        context = memory.get_full_context()
        enhanced_prompt = memory.enhance_system_prompt(base_prompt, "nutrition")
    """

    def __init__(
        self,
        user_id: int,
        max_messages_before_summary: int = 10
    ):
        """初始化记忆管理器

        Args:
            user_id: 用户 ID
            max_messages_before_summary: 摘要前的最大消息数
        """
        self.user_id = user_id
        self.profile_loader = UserProfileLoader()
        self.summarizer = ConversationSummarizer(
            max_messages=max_messages_before_summary
        )
        self.stats_summarizer = StatsSummarizer()

        self._profile: Optional[Dict[str, Any]] = None
        self._goal: Optional[str] = None
        self._today_stats: Optional[Dict[str, Any]] = None
        self._week_stats: Optional[Dict[str, Any]] = None

    def load_profile(self) -> Dict[str, Any]:
        """加载用户画像（带缓存）

        Returns:
            Dict: 用户画像
        """
        if self._profile is None:
            self._profile = self.profile_loader.load_user_profile(self.user_id)
        return self._profile

    def get_goal(self) -> str:
        """获取用户目标（带缓存）

        Returns:
            str: 用户目标（增肌/减脂/维持）
        """
        if self._goal is None:
            self._goal = self.profile_loader.get_user_goal(self.user_id)
        return self._goal

    def get_today_stats(self) -> Dict[str, Any]:
        """获取当日统计（带缓存）

        Returns:
            Dict: 当日统计数据
        """
        if self._today_stats is None:
            self._today_stats = self.stats_summarizer.get_today_stats(self.user_id)
        return self._today_stats

    def get_week_stats(self) -> Dict[str, Any]:
        """获取本周统计（带缓存）

        Returns:
            Dict: 本周统计数据
        """
        if self._week_stats is None:
            self._week_stats = self.stats_summarizer.get_week_stats(self.user_id)
        return self._week_stats

    def get_full_context(self) -> Dict[str, Any]:
        """获取完整上下文

        Returns:
            Dict: 包含用户画像、目标、当日统计、本周统计
        """
        return {
            "profile": self.load_profile(),
            "goal": self.get_goal(),
            "today_stats": self.get_today_stats(),
            "week_stats": self.get_week_stats()
        }

    def format_profile_for_agent(self) -> str:
        """格式化用户画像为 Agent 可读格式

        Returns:
            str: 格式化的用户画像
        """
        profile = self.load_profile()
        goal = self.get_goal()
        return self.profile_loader.format_profile_for_agent(profile, goal)

    def format_today_stats_for_agent(self) -> str:
        """格式化当日统计为 Agent 可读格式

        Returns:
            str: 格式化的当日统计
        """
        return self.stats_summarizer.format_today_for_agent(self.get_today_stats())

    def format_week_stats_for_agent(self) -> str:
        """格式化本周统计为 Agent 可读格式

        Returns:
            str: 格式化的本周统计
        """
        return self.stats_summarizer.format_week_for_agent(self.get_week_stats())

    def enhance_system_prompt(
        self,
        base_prompt: str,
        agent_type: str,
        messages: Optional[List[BaseMessage]] = None
    ) -> str:
        """增强 System Prompt，注入用户记忆信息

        Args:
            base_prompt: 原始 System Prompt
            agent_type: Agent 类型（chat/nutrition/fitness）
            messages: 当前对话历史（可选，用于摘要）

        Returns:
            str: 增强后的 System Prompt
        """
        enhanced_parts = [base_prompt]

        profile_section = self.format_profile_for_agent()
        enhanced_parts.append(f"\n{profile_section}")

        today_stats = self.format_today_stats_for_agent()
        enhanced_parts.append(f"\n{today_stats}")

        if agent_type in ["nutrition", "fitness"]:
            week_stats = self.format_week_stats_for_agent()
            enhanced_parts.append(f"\n{week_stats}")

        if messages and len(messages) > 1:
            if self.summarizer.should_summarize(messages):
                profile = self.load_profile()
                summarized_messages = self.summarizer.summarize_messages(
                    messages, profile
                )
                key_info = self.summarizer.extract_key_info(messages)
                if key_info["topics"] or key_info["goals"]:
                    enhanced_parts.append("\n【对话要点】")
                    if key_info["topics"]:
                        enhanced_parts.append(f"讨论话题: {', '.join(key_info['topics'])}")
                    if key_info["goals"]:
                        enhanced_parts.append(f"用户目标: {', '.join(key_info['goals'])}")

        return "\n".join(enhanced_parts)

    def get_nutrition_context(self) -> str:
        """获取营养相关的上下文

        Returns:
            str: 营养上下文
        """
        return self.stats_summarizer.get_context_for_nutrition(self.get_today_stats())

    def get_fitness_context(self) -> str:
        """获取运动相关的上下文

        Returns:
            str: 运动上下文
        """
        return self.stats_summarizer.get_context_for_fitness(self.get_today_stats())

    def summarize_conversation(
        self,
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """对对话历史进行摘要

        Args:
            messages: 原始消息列表

        Returns:
            List[BaseMessage]: 摘要后的消息列表
        """
        profile = self.load_profile()
        return self.summarizer.summarize_messages(messages, profile)

    def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """判断是否需要摘要

        Args:
            messages: 消息列表

        Returns:
            bool: 是否需要摘要
        """
        return self.summarizer.should_summarize(messages)

    def load_all_memory(self) -> None:
        """一次性加载所有记忆数据到缓存（单次 DB 连接）

        替代分别调用 load_profile / get_goal / get_today_stats / get_week_stats，
        将 8-10 次独立 DB 连接合并为 1 次，大幅减少流式响应的首字延迟。
        """
        from datetime import date, timedelta

        db = database.SessionLocal()
        try:
            # ── User 表：一次查询，供 profile / goal / tdee 共用 ──
            user = db.query(models.User).filter(models.User.id == self.user_id).first()

            if user:
                self._profile = {
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
                # goal
                if user.target_weight:
                    diff = user.weight - user.target_weight
                    if diff > 2:
                        self._goal = "减脂"
                    elif diff < -2:
                        self._goal = "增肌"
                    else:
                        self._goal = "维持"
                else:
                    self._goal = "维持现状"
                tdee = user.tdee
            else:
                from .user_profile import UserProfileLoader
                self._profile = UserProfileLoader._get_default_profile()
                self._goal = "维持现状"
                tdee = None

            # ── DailyLog 表：一次查本周所有记录，供 today_stats / week_stats 共用 ──
            today = date.today()
            week_start = today - timedelta(days=today.weekday())

            logs = db.query(models.DailyLog).filter(
                models.DailyLog.user_id == self.user_id,
                models.DailyLog.date >= week_start,
                models.DailyLog.date <= today
            ).all()

            log_by_date = {log.date: log for log in logs}
            today_log = log_by_date.get(today)

            # today_stats
            if today_log:
                food_items = [{"name": i.name, "calories": i.calories} for i in today_log.food_items]
                exercise_items = [{"type": i.type, "duration": i.duration, "calories": i.calories, "notes": i.notes} for i in today_log.exercise_items]
                net = today_log.intake_calories - today_log.burn_calories
                self._today_stats = {
                    "date": today.isoformat(),
                    "intake_calories": today_log.intake_calories,
                    "burn_calories": today_log.burn_calories,
                    "net_calories": net,
                    "tdee": tdee,
                    "calorie_balance": net - tdee if tdee else None,
                    "food_count": len(today_log.food_items),
                    "exercise_count": len(today_log.exercise_items),
                    "food_items": food_items,
                    "exercise_items": exercise_items,
                    "weight_log": today_log.weight_log
                }
            else:
                self._today_stats = {
                    "date": today.isoformat(),
                    "intake_calories": 0, "burn_calories": 0, "net_calories": 0,
                    "tdee": tdee,
                    "calorie_balance": -tdee if tdee else None,
                    "food_count": 0, "exercise_count": 0,
                    "food_items": [], "exercise_items": []
                }

            # week_stats
            if logs:
                total_intake = sum(l.intake_calories for l in logs)
                total_burn = sum(l.burn_calories for l in logs)
                days_count = len(logs)
                days_below = sum(1 for l in logs if tdee and l.intake_calories < tdee)
                days_above = sum(1 for l in logs if tdee and l.intake_calories >= tdee)
                daily_logs = [{"date": l.date.isoformat(), "intake": l.intake_calories, "burn": l.burn_calories, "net": l.intake_calories - l.burn_calories} for l in logs]
                self._week_stats = {
                    "week_start": week_start.isoformat(),
                    "week_end": today.isoformat(),
                    "days_logged": days_count,
                    "total_intake": total_intake, "total_burn": total_burn,
                    "avg_intake": total_intake / days_count, "avg_burn": total_burn / days_count,
                    "days_below_tdee": days_below, "days_above_tdee": days_above,
                    "daily_logs": daily_logs
                }
            else:
                self._week_stats = {
                    "week_start": week_start.isoformat(), "week_end": today.isoformat(),
                    "days_logged": 0, "total_intake": 0, "total_burn": 0,
                    "avg_intake": 0, "avg_burn": 0,
                    "days_below_tdee": 0, "days_above_tdee": 0, "daily_logs": []
                }
        finally:
            db.close()

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要

        Returns:
            Dict: 记忆摘要信息
        """
        return {
            "user_id": self.user_id,
            "goal": self.get_goal(),
            "today_intake": self.get_today_stats().get("intake_calories", 0),
            "today_burn": self.get_today_stats().get("burn_calories", 0),
            "week_avg_intake": self.get_week_stats().get("avg_intake", 0)
        }

    def save_conversation(
        self,
        user_message: str,
        agent_response: str,
        agent_type: str,
        session_id: Optional[str] = None
    ) -> bool:
        """保存单轮对话到数据库

        Args:
            user_message: 用户消息
            agent_response: Agent 回复
            agent_type: Agent 类型
            session_id: 会话 ID（可选）

        Returns:
            bool: 是否保存成功
        """
        db = database.SessionLocal()
        try:
            log = models.ConversationLog(
                user_id=self.user_id,
                session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                agent_type=agent_type,
                user_message=user_message,
                agent_response=agent_response,
                created_at=datetime.now()
            )
            db.add(log)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"保存对话失败: {e}")
            return False
        finally:
            db.close()

    def load_conversation_history(
        self,
        days: int = 7,
        limit: int = 50,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """加载历史对话记录

        Args:
            days: 加载最近 N 天的对话
            limit: 最多加载 N 条记录
            session_id: 指定会话 ID（可选）

        Returns:
            List[Dict]: 对话历史列表
        """
        db = database.SessionLocal()
        try:
            start_date = datetime.now() - timedelta(days=days)

            query = db.query(models.ConversationLog).filter(
                models.ConversationLog.user_id == self.user_id,
                models.ConversationLog.created_at >= start_date
            )

            if session_id:
                query = query.filter(models.ConversationLog.session_id == session_id)

            logs = query.order_by(
                models.ConversationLog.created_at.desc()
            ).limit(limit).all()

            result = []
            for log in reversed(logs):
                result.append({
                    "role": "user",
                    "content": log.user_message,
                    "agent_type": log.agent_type,
                    "created_at": log.created_at.isoformat() if log.created_at else None
                })
                result.append({
                    "role": "assistant",
                    "content": log.agent_response,
                    "agent_type": log.agent_type,
                    "created_at": log.created_at.isoformat() if log.created_at else None
                })

            return result
        finally:
            db.close()

    def format_conversation_history_for_agent(
        self,
        days: int = 7,
        limit: int = 10
    ) -> str:
        """格式化对话历史为 Agent 可读格式

        Args:
            days: 加载最近 N 天的对话
            limit: 最多加载 N 轮对话

        Returns:
            str: 格式化的对话历史
        """
        history = self.load_conversation_history(days=days, limit=limit)

        if not history:
            return "（无历史对话）"

        parts = ["【近期对话历史】"]
        for i, msg in enumerate(history):
            role = "用户" if msg["role"] == "user" else "AI"
            agent = msg.get("agent_type", "")
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            time = msg.get("created_at", "")[:16] if msg.get("created_at") else ""

            if i % 2 == 0:
                parts.append(f"\n[{time}] {role}: {content}")
            else:
                parts.append(f"→ {agent}回复: {content}")

        return "\n".join(parts)

    def get_conversation_summary_for_agent(self, days: int = 7) -> str:
        """获取对话摘要（简化版，用于 System Prompt）

        Args:
            days: 加载最近 N 天的对话

        Returns:
            str: 对话摘要
        """
        history = self.load_conversation_history(days=days, limit=20)

        if not history:
            return ""

        topics = set()
        for msg in history:
            if msg["role"] == "user":
                content = msg["content"]
                if len(content) > 50:
                    topics.add(content[:50] + "...")
                else:
                    topics.add(content)

        if not topics:
            return ""

        topic_list = list(topics)[:5]
        return f"\n\n【用户近期咨询话题】\n" + "\n".join([f"- {t}" for t in topic_list])
