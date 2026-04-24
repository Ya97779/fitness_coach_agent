"""记忆管理器 - 整合用户画像、对话历史、统计数据的管理"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage
from .user_profile import UserProfileLoader
from .conversation_summary import ConversationSummarizer
from .stats_summary import StatsSummarizer


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

    def load_profile(self) -> Dict[str, Any]:
        """加载用户画像

        Returns:
            Dict: 用户画像
        """
        if self._profile is None:
            self._profile = self.profile_loader.load_user_profile(self.user_id)
        return self._profile

    def get_goal(self) -> str:
        """获取用户目标

        Returns:
            str: 用户目标（增肌/减脂/维持）
        """
        if self._goal is None:
            self._goal = self.profile_loader.get_user_goal(self.user_id)
        return self._goal

    def get_full_context(self) -> Dict[str, Any]:
        """获取完整上下文

        Returns:
            Dict: 包含用户画像、目标、当日统计、本周统计
        """
        profile = self.load_profile()
        goal = self.get_goal()
        today_stats = self.stats_summarizer.get_today_stats(self.user_id)
        week_stats = self.stats_summarizer.get_week_stats(self.user_id)

        return {
            "profile": profile,
            "goal": goal,
            "today_stats": today_stats,
            "week_stats": week_stats
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
        stats = self.stats_summarizer.get_today_stats(self.user_id)
        return self.stats_summarizer.format_today_for_agent(stats)

    def format_week_stats_for_agent(self) -> str:
        """格式化本周统计为 Agent 可读格式

        Returns:
            str: 格式化的本周统计
        """
        stats = self.stats_summarizer.get_week_stats(self.user_id)
        return self.stats_summarizer.format_week_for_agent(stats)

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
        today_stats = self.stats_summarizer.get_today_stats(self.user_id)
        return self.stats_summarizer.get_context_for_nutrition(today_stats)

    def get_fitness_context(self) -> str:
        """获取运动相关的上下文

        Returns:
            str: 运动上下文
        """
        today_stats = self.stats_summarizer.get_today_stats(self.user_id)
        return self.stats_summarizer.get_context_for_fitness(today_stats)

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

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要

        Returns:
            Dict: 记忆摘要信息
        """
        return {
            "user_id": self.user_id,
            "goal": self.get_goal(),
            "today_intake": self.stats_summarizer.get_today_stats(self.user_id).get("intake_calories", 0),
            "today_burn": self.stats_summarizer.get_today_stats(self.user_id).get("burn_calories", 0),
            "week_avg_intake": self.stats_summarizer.get_week_stats(self.user_id).get("avg_intake", 0)
        }
