"""Memory 模块单元测试"""

import sys
import os
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.memory.user_profile import UserProfileLoader
from app.memory.conversation_summary import ConversationSummarizer
from app.memory.stats_summary import StatsSummarizer
from app.memory.memory_manager import MemoryManager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestUserProfileLoader(unittest.TestCase):
    """UserProfileLoader 单元测试"""

    @patch('app.memory.user_profile.database.SessionLocal')
    def test_load_user_profile_returns_dict(self, mock_db):
        """测试 load_user_profile 返回字典类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = UserProfileLoader.load_user_profile(user_id=999)
        self.assertIsInstance(result, dict)

    @patch('app.memory.user_profile.database.SessionLocal')
    def test_load_user_profile_has_required_keys(self, mock_db):
        """测试 load_user_profile 返回的字典包含必要键"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = UserProfileLoader.load_user_profile(user_id=999)

        self.assertIn('basic_info', result)
        self.assertIn('body_metrics', result)
        self.assertIn('goal', result)
        self.assertIn('constraints', result)

    @patch('app.memory.user_profile.database.SessionLocal')
    def test_load_compact_profile_returns_dict(self, mock_db):
        """测试 load_compact_profile 返回字典类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = UserProfileLoader.load_compact_profile(user_id=999)
        self.assertIsInstance(result, dict)

    @patch('app.memory.user_profile.database.SessionLocal')
    def test_get_user_goal_returns_str(self, mock_db):
        """测试 get_user_goal 返回字符串类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = UserProfileLoader.get_user_goal(user_id=999)
        self.assertIsInstance(result, str)

    @patch('app.memory.user_profile.database.SessionLocal')
    def test_format_profile_for_agent_returns_str(self, mock_db):
        """测试 format_profile_for_agent 返回字符串类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        profile = UserProfileLoader.load_user_profile(user_id=999)
        goal = UserProfileLoader.get_user_goal(user_id=999)
        result = UserProfileLoader.format_profile_for_agent(profile, goal)
        self.assertIsInstance(result, str)


class TestConversationSummarizer(unittest.TestCase):
    """ConversationSummarizer 单元测试"""

    def test_should_summarize_below_threshold(self):
        """测试消息数未超过阈值时不需摘要"""
        summarizer = ConversationSummarizer(max_messages=10)
        messages = [HumanMessage(content=f"message {i}") for i in range(5)]

        self.assertFalse(summarizer.should_summarize(messages))

    def test_should_summarize_above_threshold(self):
        """测试消息数超过阈值时需摘要"""
        summarizer = ConversationSummarizer(max_messages=10)
        messages = [HumanMessage(content=f"message {i}") for i in range(15)]

        self.assertTrue(summarizer.should_summarize(messages))

    def test_should_summarize_at_threshold(self):
        """测试消息数正好等于阈值时不需摘要"""
        summarizer = ConversationSummarizer(max_messages=10)
        messages = [HumanMessage(content=f"message {i}") for i in range(10)]

        self.assertFalse(summarizer.should_summarize(messages))

    def test_summarize_messages_preserves_recent(self):
        """测试 summarize_messages 保留最近消息"""
        summarizer = ConversationSummarizer(max_messages=10)
        messages = [HumanMessage(content=f"old message {i}") for i in range(15)]
        recent = [AIMessage(content="recent message")]
        all_messages = messages + recent

        result = summarizer.summarize_messages(all_messages)

        self.assertTrue(any("recent message" in str(m.content) for m in result))

    def test_extract_key_info_returns_dict(self):
        """测试 extract_key_info 返回字典"""
        summarizer = ConversationSummarizer(max_messages=10)
        messages = [
            HumanMessage(content="我想减脂"),
            HumanMessage(content="今天吃了鸡胸肉"),
            AIMessage(content="鸡胸肉是不错的选择")
        ]

        result = summarizer.extract_key_info(messages)
        self.assertIsInstance(result, dict)
        self.assertIn('topics', result)
        self.assertIn('goals', result)


class TestStatsSummarizer(unittest.TestCase):
    """StatsSummarizer 单元测试"""

    @patch('app.memory.stats_summary.database.SessionLocal')
    def test_get_today_stats_returns_dict(self, mock_db):
        """测试 get_today_stats 返回字典类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = StatsSummarizer.get_today_stats(user_id=999)
        self.assertIsInstance(result, dict)

    @patch('app.memory.stats_summary.database.SessionLocal')
    def test_get_today_stats_has_required_keys(self, mock_db):
        """测试 get_today_stats 返回的字典包含必要键"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = StatsSummarizer.get_today_stats(user_id=999)

        self.assertIn('intake_calories', result)
        self.assertIn('burn_calories', result)
        self.assertIn('net_calories', result)
        self.assertIn('tdee', result)

    @patch('app.memory.stats_summary.database.SessionLocal')
    def test_get_today_stats_defaults_to_zero(self, mock_db):
        """测试无日志时 get_today_stats 返回零值"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = StatsSummarizer.get_today_stats(user_id=999)

        self.assertEqual(result['intake_calories'], 0)
        self.assertEqual(result['burn_calories'], 0)

    @patch('app.memory.stats_summary.database.SessionLocal')
    def test_get_week_stats_returns_dict(self, mock_db):
        """测试 get_week_stats 返回字典类型"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.all.return_value = []

        result = StatsSummarizer.get_week_stats(user_id=999)
        self.assertIsInstance(result, dict)

    def test_format_today_for_agent_returns_str(self):
        """测试 format_today_for_agent 返回字符串"""
        stats = {
            'intake_calories': 1500,
            'burn_calories': 500,
            'net_calories': 1000,
            'tdee': 2000,
            'calorie_balance': -500
        }

        result = StatsSummarizer.format_today_for_agent(stats)
        self.assertIsInstance(result, str)

    def test_format_week_for_agent_returns_str(self):
        """测试 format_week_for_agent 返回字符串"""
        stats = {
            'avg_intake': 1800,
            'avg_burn': 400,
            'days_logged': 5
        }

        result = StatsSummarizer.format_week_for_agent(stats)
        self.assertIsInstance(result, str)


class TestMemoryManager(unittest.TestCase):
    """MemoryManager 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.mock_profile_loader_patcher = patch('app.memory.memory_manager.UserProfileLoader')
        self.mock_stats_summarizer_patcher = patch('app.memory.memory_manager.StatsSummarizer')
        self.mock_conversation_summarizer_patcher = patch('app.memory.memory_manager.ConversationSummarizer')

        self.mock_profile_loader = self.mock_profile_loader_patcher.start()
        self.mock_stats_summarizer = self.mock_stats_summarizer_patcher.start()
        self.mock_conversation_summarizer = self.mock_conversation_summarizer_patcher.start()

    def tearDown(self):
        """测试后清理"""
        self.mock_profile_loader_patcher.stop()
        self.mock_stats_summarizer_patcher.stop()
        self.mock_conversation_summarizer_patcher.stop()

    def test_init_initializes_attributes(self):
        """测试 __init__ 正确初始化属性"""
        mm = MemoryManager(user_id=1)

        self.assertEqual(mm.user_id, 1)
        self.assertIsNotNone(mm.profile_loader)
        self.assertIsNotNone(mm.summarizer)
        self.assertIsNotNone(mm.stats_summarizer)

    def test_init_with_custom_max_messages(self):
        """测试 __init__ 支持自定义 max_messages 参数"""
        with patch('app.memory.memory_manager.ConversationSummarizer') as MockSummarizer:
            MockSummarizer_instance = MagicMock()
            MockSummarizer_instance.max_messages = 20
            MockSummarizer.return_value = MockSummarizer_instance

            mm = MemoryManager(user_id=1, max_messages_before_summary=20)

            self.assertEqual(mm.summarizer.max_messages, 20)

    def test_load_profile_caches_result(self):
        """测试 load_profile 缓存机制"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'test': 'data'}) as mock_load:
            result1 = mm.load_profile()
            result2 = mm.load_profile()

            self.assertEqual(result1, result2)
            mock_load.assert_called_once()

    def test_get_goal_caches_result(self):
        """测试 get_goal 缓存机制"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂') as mock_goal:
            result1 = mm.get_goal()
            result2 = mm.get_goal()

            self.assertEqual(result1, result2)
            mock_goal.assert_called_once()

    def test_get_full_context_returns_dict(self):
        """测试 get_full_context 返回字典"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake': 100}):
                    with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg': 1500}):
                        result = mm.get_full_context()

        self.assertIsInstance(result, dict)
        self.assertIn('profile', result)
        self.assertIn('goal', result)
        self.assertIn('today_stats', result)
        self.assertIn('week_stats', result)

    def test_format_profile_for_agent_returns_str(self):
        """测试 format_profile_for_agent 返回字符串"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.profile_loader, 'format_profile_for_agent', return_value='用户目标: 减脂'):
                    result = mm.format_profile_for_agent()

        self.assertIsInstance(result, str)

    def test_format_today_stats_for_agent_returns_str(self):
        """测试 format_today_stats_for_agent 返回字符串"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake': 100}):
            with patch.object(mm.stats_summarizer, 'format_today_for_agent', return_value='今日摄入: 100 kcal'):
                result = mm.format_today_stats_for_agent()

        self.assertIsInstance(result, str)

    def test_format_week_stats_for_agent_returns_str(self):
        """测试 format_week_stats_for_agent 返回字符串"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg': 1500}):
            with patch.object(mm.stats_summarizer, 'format_week_for_agent', return_value='本周日均: 1500 kcal'):
                result = mm.format_week_stats_for_agent()

        self.assertIsInstance(result, str)

    def test_enhance_system_prompt_returns_str(self):
        """测试 enhance_system_prompt 返回字符串"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.profile_loader, 'format_profile_for_agent', return_value='用户目标: 减脂'):
                    with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake': 100}):
                        with patch.object(mm.stats_summarizer, 'format_today_for_agent', return_value='今日摄入: 100 kcal'):
                            with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg': 1500}):
                                with patch.object(mm.stats_summarizer, 'format_week_for_agent', return_value='本周日均: 1500 kcal'):
                                    result = mm.enhance_system_prompt("base prompt", "nutrition")

        self.assertIsInstance(result, str)
        self.assertIn("base prompt", result)
        self.assertIn("用户目标", result)

    def test_enhance_system_prompt_includes_week_stats_for_nutrition(self):
        """测试 enhance_system_prompt 为 nutrition 类型包含本周统计"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.profile_loader, 'format_profile_for_agent', return_value='用户目标: 减脂'):
                    with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake': 100}):
                        with patch.object(mm.stats_summarizer, 'format_today_for_agent', return_value='今日摄入: 100 kcal'):
                            with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg': 1500}):
                                with patch.object(mm.stats_summarizer, 'format_week_for_agent', return_value='本周日均: 1500 kcal'):
                                    result = mm.enhance_system_prompt("base prompt", "nutrition")

        self.assertIn("本周日均", result)

    def test_enhance_system_prompt_does_not_include_week_stats_for_chat(self):
        """测试 enhance_system_prompt 为 chat 类型不包含本周统计"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.profile_loader, 'format_profile_for_agent', return_value='用户目标: 减脂'):
                    with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake': 100}):
                        with patch.object(mm.stats_summarizer, 'format_today_for_agent', return_value='今日摄入: 100 kcal'):
                            with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg': 1500}):
                                with patch.object(mm.stats_summarizer, 'format_week_for_agent', return_value='本周日均: 1500 kcal'):
                                    result = mm.enhance_system_prompt("base prompt", "chat")

        self.assertNotIn("本周日均", result)

    def test_get_memory_summary_returns_dict(self):
        """测试 get_memory_summary 返回字典"""
        mm = MemoryManager(user_id=1)

        with patch.object(mm.profile_loader, 'load_user_profile', return_value={'user_id': 1}):
            with patch.object(mm.profile_loader, 'get_user_goal', return_value='减脂'):
                with patch.object(mm.stats_summarizer, 'get_today_stats', return_value={'intake_calories': 1500, 'burn_calories': 500}):
                    with patch.object(mm.stats_summarizer, 'get_week_stats', return_value={'avg_intake': 1800}):
                        result = mm.get_memory_summary()

        self.assertIsInstance(result, dict)
        self.assertIn('user_id', result)
        self.assertIn('goal', result)
        self.assertIn('today_intake', result)
        self.assertIn('today_burn', result)
        self.assertIn('week_avg_intake', result)

    def test_should_summarize_delegates_to_summarizer(self):
        """测试 should_summarize 委托给 summarizer"""
        mm = MemoryManager(user_id=1)

        messages = [HumanMessage(content=f"message {i}") for i in range(15)]

        with patch.object(mm.summarizer, 'should_summarize', return_value=True):
            result = mm.should_summarize(messages)
        self.assertTrue(result)

    @patch('app.memory.memory_manager.database.SessionLocal')
    def test_save_conversation_returns_bool(self, mock_db):
        """测试 save_conversation 返回布尔值"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session

        mm = MemoryManager(user_id=1)
        result = mm.save_conversation(
            user_message="你好",
            agent_response="你好！",
            agent_type="chat"
        )

        self.assertIsInstance(result, bool)

    @patch('app.memory.memory_manager.database.SessionLocal')
    def test_load_conversation_history_returns_list(self, mock_db):
        """测试 load_conversation_history 返回列表"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mm = MemoryManager(user_id=1)
        result = mm.load_conversation_history(days=7, limit=50)

        self.assertIsInstance(result, list)


class TestMemoryIntegration(unittest.TestCase):
    """Memory 模块集成测试"""

    @patch('app.memory.memory_manager.UserProfileLoader')
    @patch('app.memory.memory_manager.StatsSummarizer')
    @patch('app.memory.memory_manager.ConversationSummarizer')
    def test_enhance_prompt_chain(self, MockConvSummarizer, MockStatsSummarizer, MockProfileLoader):
        """测试 enhance_system_prompt 完整链路"""
        MockProfileLoader_instance = MagicMock()
        MockStatsSummarizer_instance = MagicMock()
        MockConvSummarizer_instance = MagicMock()

        MockProfileLoader.return_value = MockProfileLoader_instance
        MockStatsSummarizer.return_value = MockStatsSummarizer_instance
        MockConvSummarizer.return_value = MockConvSummarizer_instance

        MockProfileLoader_instance.load_user_profile.return_value = {'user_id': 1}
        MockProfileLoader_instance.get_user_goal.return_value = '减脂'
        MockProfileLoader_instance.format_profile_for_agent.return_value = '用户目标: 减脂'
        MockStatsSummarizer_instance.get_today_stats.return_value = {'intake_calories': 1500}
        MockStatsSummarizer_instance.format_today_for_agent.return_value = '今日摄入: 1500 kcal'
        MockStatsSummarizer_instance.get_week_stats.return_value = {'avg_intake': 1800}
        MockStatsSummarizer_instance.format_week_for_agent.return_value = '本周日均: 1800 kcal'
        MockConvSummarizer_instance.should_summarize.return_value = False

        mm = MemoryManager(user_id=1)
        messages = [HumanMessage(content="今天吃什么好？")]

        result = mm.enhance_system_prompt("base", "nutrition", messages)

        self.assertIsInstance(result, str)
        self.assertIn("base", result)
        self.assertIn("减脂", result)


if __name__ == '__main__':
    unittest.main()