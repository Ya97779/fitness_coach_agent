"""Agents 模块单元测试"""

import sys
import os
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.agents.base import (
    AgentConfig, AgentResponse, MultiAgentState,
    AGENT_SYSTEM_PROMPTS
)
from app.agents.router import route_with_context
from app.agents.chat_agent import chat_with_user, format_memory_context
from app.agents.nutrition_agent import (
    get_user_nutrition_info, log_food_intake, get_daily_nutrition_summary,
    search_food_nutrition, nutrition_tools, nutrition_with_user, format_nutrition_memory
)
from app.agents.fitness_agent import (
    get_user_fitness_info, log_exercise, estimate_exercise_calories,
    search_fitness_knowledge, fitness_tools, fitness_with_user, format_fitness_memory
)
from app.agents.expert_agent import review_output
from app.agents.graph import (
    AgentState, create_llm, router, chat, nutrition, fitness,
    expert_review, should_continue_nutrition, should_continue_fitness,
    route_after_router, build_graph, MAX_RETRIES, MIN_APPROVAL_SCORE
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestAgentConfig(unittest.TestCase):
    """AgentConfig 单元测试"""

    def test_agent_config_creation(self):
        """测试 AgentConfig 创建"""
        config = AgentConfig(
            name="test_agent",
            role="测试角色",
            description="测试描述",
            system_prompt="测试 prompt",
            tools=["tool1", "tool2"]
        )
        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.role, "测试角色")
        self.assertEqual(len(config.tools), 2)


class TestAgentResponse(unittest.TestCase):
    """AgentResponse 单元测试"""

    def test_agent_response_default_values(self):
        """测试 AgentResponse 默认值"""
        response = AgentResponse(
            agent_name="chat",
            response="你好"
        )
        self.assertEqual(response.agent_name, "chat")
        self.assertEqual(response.response, "你好")
        self.assertFalse(response.needs_review)
        self.assertIsNone(response.review_feedback)

    def test_agent_response_with_review(self):
        """测试 AgentResponse 带评审信息"""
        response = AgentResponse(
            agent_name="nutrition",
            response="饮食建议",
            needs_review=True,
            review_feedback="需要优化"
        )
        self.assertTrue(response.needs_review)
        self.assertEqual(response.review_feedback, "需要优化")


class TestAgentSystemPrompts(unittest.TestCase):
    """AGENT_SYSTEM_PROMPTS 单元测试"""

    def test_prompts_contain_required_keys(self):
        """测试 prompts 包含所有必需的 key"""
        required_keys = ["chat", "nutrition", "fitness"]
        for key in required_keys:
            self.assertIn(key, AGENT_SYSTEM_PROMPTS)
            self.assertIsInstance(AGENT_SYSTEM_PROMPTS[key], str)
            self.assertTrue(len(AGENT_SYSTEM_PROMPTS[key]) > 0)

    def test_chat_prompt_contains_role(self):
        """测试 chat prompt 包含角色定义"""
        self.assertIn("闲聊助手", AGENT_SYSTEM_PROMPTS["chat"])

    def test_nutrition_prompt_contains_role(self):
        """测试 nutrition prompt 包含角色定义"""
        self.assertIn("营养师", AGENT_SYSTEM_PROMPTS["nutrition"])

    def test_fitness_prompt_contains_role(self):
        """测试 fitness prompt 包含角色定义"""
        self.assertIn("健身教练", AGENT_SYSTEM_PROMPTS["fitness"])


class TestRouter(unittest.TestCase):
    """Router 单元测试"""

    @patch('app.agents.router.ChatOpenAI')
    def test_route_with_context_nutrition(self, mock_llm_class):
        """测试路由到 nutrition"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="2")

        result = route_with_context("我想减肥，应该吃什么")

        self.assertEqual(result["agent"], "nutrition")
        self.assertIn("饮食", result["reason"])

    @patch('app.agents.router.ChatOpenAI')
    def test_route_with_context_fitness(self, mock_llm_class):
        """测试路由到 fitness"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="3")

        result = route_with_context("今天想跑步训练")

        self.assertEqual(result["agent"], "fitness")
        self.assertIn("运动", result["reason"])

    @patch('app.agents.router.ChatOpenAI')
    def test_route_with_context_chat(self, mock_llm_class):
        """测试路由到 chat"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="1")

        result = route_with_context("今天天气不错")

        self.assertEqual(result["agent"], "chat")
        self.assertIn("闲聊", result["reason"])

    @patch('app.agents.router.ChatOpenAI')
    def test_route_with_context_exception(self, mock_llm_class):
        """测试路由异常处理"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API Error")

        result = route_with_context("测试消息")

        self.assertEqual(result["agent"], "chat")
        self.assertIn("默认", result["reason"])


class TestFormatMemoryContext(unittest.TestCase):
    """format_memory_context 单元测试"""

    def test_format_memory_context_empty(self):
        """测试空记忆返回空字符串"""
        result = format_memory_context({})
        self.assertEqual(result, "")

    def test_format_memory_context_none(self):
        """测试 None 记忆返回空字符串"""
        result = format_memory_context(None)
        self.assertEqual(result, "")

    def test_format_memory_context_with_goal(self):
        """测试包含目标的记忆格式化"""
        memory = {"goal": "减脂"}
        result = format_memory_context(memory, "chat")
        self.assertIn("减脂", result)

    def test_format_memory_context_with_today_stats(self):
        """测试包含今日统计的记忆格式化"""
        memory = {"today_intake": 1500, "today_burn": 500}
        result = format_memory_context(memory, "chat")
        self.assertIn("1500", result)
        self.assertIn("500", result)

    def test_format_memory_context_with_conversation_history(self):
        """测试包含对话历史的记忆格式化"""
        memory = {
            "conversation_history": [
                {"role": "user", "content": "今天吃什么好？"},
                {"role": "assistant", "content": "推荐鸡胸肉"}
            ]
        }
        result = format_memory_context(memory, "chat")
        self.assertIn("近期对话", result)


class TestChatAgent(unittest.TestCase):
    """Chat Agent 单元测试"""

    @patch('app.agents.chat_agent.ChatOpenAI')
    def test_chat_with_user_returns_str(self, mock_llm_class):
        """测试 chat_with_user 返回字符串"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="你好！有什么可以帮你的？")

        messages = [HumanMessage(content="你好")]
        result = chat_with_user(messages, user_id=1)

        self.assertIsInstance(result, str)

    @patch('app.agents.chat_agent.ChatOpenAI')
    def test_chat_with_user_with_enhanced_prompt(self, mock_llm_class):
        """测试使用增强 prompt 的 chat_with_user"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="回答")

        messages = [HumanMessage(content="你好")]
        enhanced_prompt = "你是一个助手，用户目标是减脂"
        result = chat_with_user(messages, user_id=1, enhanced_prompt=enhanced_prompt)

        self.assertIsInstance(result, str)
        mock_llm.invoke.assert_called_once()


class TestNutritionTools(unittest.TestCase):
    """Nutrition Agent Tools 单元测试"""

    @patch('app.agents.nutrition_agent.database.SessionLocal')
    def test_get_user_nutrition_info_returns_dict(self, mock_db):
        """测试 get_user_nutrition_info 返回字典"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_user = MagicMock()
        mock_user.height = 175
        mock_user.weight = 70
        mock_user.age = 25
        mock_user.gender = "男"
        mock_user.bmr = 1700
        mock_user.tdee = 2200
        mock_user.allergies = "无"
        mock_user.target_weight = 65
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        result = get_user_nutrition_info.invoke({"user_id": 1})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["height"], 175)
        self.assertEqual(result["weight"], 70)

    @patch('app.agents.nutrition_agent.database.SessionLocal')
    def test_get_user_nutrition_info_not_found(self, mock_db):
        """测试用户不存在时返回字符串"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = get_user_nutrition_info.invoke({"user_id": 999})

        self.assertEqual(result, "未找到用户信息")

    @patch('app.agents.nutrition_agent.database.SessionLocal')
    def test_log_food_intake_returns_str(self, mock_db):
        """测试 log_food_intake 返回字符串"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.intake_calories = 0

        with patch('app.agents.nutrition_agent.models.DailyLog') as MockDailyLog:
            MockDailyLog.return_value = mock_log

            result = log_food_intake.invoke({
                "user_id": 1,
                "food_name": "鸡胸肉",
                "calories": 165,
                "protein": 31,
                "fat": 3.6,
                "carbs": 0
            })

        self.assertIsInstance(result, str)
        self.assertIn("鸡胸肉", result)

    @patch('app.agents.nutrition_agent.database.SessionLocal')
    def test_get_daily_nutrition_summary_returns_dict(self, mock_db):
        """测试 get_daily_nutrition_summary 返回字典"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_log = MagicMock()
        mock_log.intake_calories = 1500
        mock_log.burn_calories = 500
        mock_session.query.return_value.filter.return_value.first.side_effect = [mock_log, None]

        result = get_daily_nutrition_summary.invoke({"user_id": 1})

        self.assertIsInstance(result, dict)
        self.assertIn("intake_calories", result)
        self.assertIn("net_calories", result)

    def test_nutrition_tools_list_contains_required_tools(self):
        """测试 nutrition_tools 包含必需的 tools"""
        tool_names = [t.name for t in nutrition_tools]
        self.assertIn("get_user_nutrition_info", tool_names)
        self.assertIn("log_food_intake", tool_names)
        self.assertIn("get_daily_nutrition_summary", tool_names)
        self.assertIn("search_food_nutrition", tool_names)


class TestFormatNutritionMemory(unittest.TestCase):
    """format_nutrition_memory 单元测试"""

    def test_format_nutrition_memory_empty(self):
        """测试空记忆返回空字符串"""
        result = format_nutrition_memory({})
        self.assertEqual(result, "")

    def test_format_nutrition_memory_none(self):
        """测试 None 记忆返回空字符串"""
        result = format_nutrition_memory(None)
        self.assertEqual(result, "")

    def test_format_nutrition_memory_with_goal(self):
        """测试包含目标的记忆格式化"""
        memory = {"goal": "减脂"}
        result = format_nutrition_memory(memory)
        self.assertIn("减脂", result)

    def test_format_nutrition_memory_with_today_intake(self):
        """测试包含今日摄入的记忆格式化"""
        memory = {"today_intake": 1500}
        result = format_nutrition_memory(memory)
        self.assertIn("1500", result)


class TestFitnessTools(unittest.TestCase):
    """Fitness Agent Tools 单元测试"""

    @patch('app.agents.fitness_agent.database.SessionLocal')
    def test_get_user_fitness_info_returns_dict(self, mock_db):
        """测试 get_user_fitness_info 返回字典"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_user = MagicMock()
        mock_user.height = 175
        mock_user.weight = 70
        mock_user.age = 25
        mock_user.gender = "男"
        mock_user.bmr = 1700
        mock_user.tdee = 2200
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        result = get_user_fitness_info.invoke({"user_id": 1})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["height"], 175)

    @patch('app.agents.fitness_agent.database.SessionLocal')
    def test_get_user_fitness_info_not_found(self, mock_db):
        """测试用户不存在时返回字符串"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = get_user_fitness_info.invoke({"user_id": 999})

        self.assertEqual(result, "未找到用户信息")

    @patch('app.agents.fitness_agent.models')
    @patch('app.agents.fitness_agent.database.SessionLocal')
    def test_log_exercise_returns_str(self, mock_db, mock_models):
        """测试 log_exercise 返回字符串"""
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.burn_calories = 0
        mock_models.DailyLog.return_value = mock_log
        mock_models.ExerciseItem = MagicMock()

        result = log_exercise.invoke({
            "user_id": 1,
            "exercise_type": "跑步",
            "duration": 30,
            "calories": 300
        })

        self.assertIsInstance(result, str)
        self.assertIn("跑步", result)

    def test_estimate_exercise_calories_returns_dict(self):
        """测试 estimate_exercise_calories 返回字典"""
        result = estimate_exercise_calories.invoke({
            "exercise_type": "跑步",
            "duration": 30,
            "intensity": "medium",
            "user_weight": 70
        })

        self.assertIsInstance(result, dict)
        self.assertIn("calories", result)
        self.assertIn("exercise_type", result)
        self.assertEqual(result["exercise_type"], "跑步")

    def test_estimate_exercise_calories_unknown_type(self):
        """测试未知运动类型使用默认 MET 值"""
        result = estimate_exercise_calories.invoke({
            "exercise_type": "未知运动",
            "duration": 30,
            "intensity": "medium",
            "user_weight": 70
        })

        self.assertIsInstance(result, dict)
        self.assertIn("calories", result)

    def test_fitness_tools_list_contains_required_tools(self):
        """测试 fitness_tools 包含必需的 tools"""
        tool_names = [t.name for t in fitness_tools]
        self.assertIn("get_user_fitness_info", tool_names)
        self.assertIn("log_exercise", tool_names)
        self.assertIn("estimate_exercise_calories", tool_names)
        self.assertIn("search_fitness_knowledge", tool_names)


class TestFormatFitnessMemory(unittest.TestCase):
    """format_fitness_memory 单元测试"""

    def test_format_fitness_memory_empty(self):
        """测试空记忆返回空字符串"""
        result = format_fitness_memory({})
        self.assertEqual(result, "")

    def test_format_fitness_memory_none(self):
        """测试 None 记忆返回空字符串"""
        result = format_fitness_memory(None)
        self.assertEqual(result, "")

    def test_format_fitness_memory_with_goal(self):
        """测试包含目标的记忆格式化"""
        memory = {"goal": "增肌"}
        result = format_fitness_memory(memory)
        self.assertIn("增肌", result)

    def test_format_fitness_memory_with_today_burn(self):
        """测试包含今日消耗的记忆格式化"""
        memory = {"today_burn": 500}
        result = format_fitness_memory(memory)
        self.assertIn("500", result)


class TestExpertAgent(unittest.TestCase):
    """Expert Agent 单元测试"""

    @patch('app.agents.expert_agent.ChatOpenAI')
    def test_review_output_returns_dict(self, mock_llm_class):
        """测试 review_output 返回字典"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="评分: 4\n内容良好")

        result = review_output("营养师输出", "健身教练输出")

        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("approved", result)
        self.assertIn("feedback", result)
        self.assertIn("needs_revision", result)

    @patch('app.agents.expert_agent.ChatOpenAI')
    def test_review_output_high_score_approved(self, mock_llm_class):
        """测试高分通过评审"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="评分: 4\n内容良好")

        result = review_output("营养师输出", "健身教练输出")

        self.assertTrue(result["approved"])
        self.assertFalse(result["needs_revision"])

    @patch('app.agents.expert_agent.ChatOpenAI')
    def test_review_output_low_score_needs_revision(self, mock_llm_class):
        """测试低分需要修改"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="评分: 2\n内容需要改进")

        result = review_output("营养师输出", "健身教练输出")

        self.assertFalse(result["approved"])
        self.assertTrue(result["needs_revision"])

    @patch('app.agents.expert_agent.ChatOpenAI')
    def test_review_output_exception_handling(self, mock_llm_class):
        """测试异常处理"""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API Error")

        result = review_output("营养师输出", "健身教练输出")

        self.assertEqual(result["score"], 3)
        self.assertTrue(result["approved"])


class TestGraphConstants(unittest.TestCase):
    """Graph 常量单元测试"""

    def test_max_retries_value(self):
        """测试 MAX_RETRIES 值"""
        self.assertEqual(MAX_RETRIES, 3)

    def test_min_approval_score_value(self):
        """测试 MIN_APPROVAL_SCORE 值"""
        self.assertEqual(MIN_APPROVAL_SCORE, 3)


class TestAgentState(unittest.TestCase):
    """AgentState 单元测试"""

    def test_agent_state_has_required_keys(self):
        """测试 AgentState 包含所有必需的 key"""
        state = AgentState(
            messages=[],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="chat",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        self.assertIn("messages", state)
        self.assertIn("user_id", state)
        self.assertIn("user_profile", state)
        self.assertIn("daily_stats", state)
        self.assertIn("current_agent", state)
        self.assertIn("retry_count", state)
        self.assertIn("review_history", state)
        self.assertIn("memory_summary", state)
        self.assertIn("enhanced_prompts", state)


class TestRouterNode(unittest.TestCase):
    """Router 节点单元测试"""

    @patch('app.agents.graph.create_llm')
    def test_router_returns_chat_for_empty_message(self, mock_create_llm):
        """测试空消息路由到 chat"""
        state = AgentState(
            messages=[HumanMessage(content="")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="chat",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = router(state)

        self.assertEqual(result["current_agent"], "chat")
        self.assertEqual(result["retry_count"], 0)

    @patch('app.agents.graph.create_llm')
    def test_router_returns_nutrition(self, mock_create_llm):
        """测试饮食相关消息路由到 nutrition"""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="2")

        state = AgentState(
            messages=[HumanMessage(content="我想减肥吃什么好")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="chat",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = router(state)

        self.assertEqual(result["current_agent"], "nutrition")

    @patch('app.agents.graph.create_llm')
    def test_router_returns_fitness(self, mock_create_llm):
        """测试运动相关消息路由到 fitness"""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="3")

        state = AgentState(
            messages=[HumanMessage(content="今天想跑步训练")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="chat",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = router(state)

        self.assertEqual(result["current_agent"], "fitness")


class TestChatNode(unittest.TestCase):
    """Chat 节点单元测试"""

    @patch('app.agents.graph.chat_with_user')
    def test_chat_returns_messages(self, mock_chat):
        """测试 chat 节点返回消息"""
        mock_chat.return_value = "你好！有什么可以帮助你的？"

        state = AgentState(
            messages=[HumanMessage(content="你好")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="chat",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = chat(state)

        self.assertIn("messages", result)
        self.assertEqual(result["current_agent"], "chat")


class TestNutritionNode(unittest.TestCase):
    """Nutrition 节点单元测试"""

    @patch('app.agents.graph.nutrition_with_user')
    def test_nutrition_returns_messages(self, mock_nutrition):
        """测试 nutrition 节点返回消息"""
        mock_nutrition.return_value = "推荐你吃鸡胸肉"

        state = AgentState(
            messages=[HumanMessage(content="我想减肥")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="nutrition",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = nutrition(state)

        self.assertIn("messages", result)
        self.assertEqual(result["current_agent"], "nutrition")


class TestFitnessNode(unittest.TestCase):
    """Fitness 节点单元测试"""

    @patch('app.agents.graph.fitness_with_user')
    def test_fitness_returns_messages(self, mock_fitness):
        """测试 fitness 节点返回消息"""
        mock_fitness.return_value = "今天推荐跑步训练"

        state = AgentState(
            messages=[HumanMessage(content="今天想运动")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="fitness",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = fitness(state)

        self.assertIn("messages", result)
        self.assertEqual(result["current_agent"], "fitness")


class TestExpertReviewNode(unittest.TestCase):
    """Expert Review 节点单元测试"""

    @patch('app.agents.graph.review_output')
    def test_expert_review_high_score_no_retry(self, mock_review):
        """测试高分不需要重试"""
        mock_review.return_value = {
            "score": 4,
            "approved": True,
            "feedback": "内容良好",
            "needs_revision": False
        }

        state = AgentState(
            messages=[AIMessage(content="推荐内容")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="nutrition",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = expert_review(state)

        self.assertEqual(result["retry_count"], 1)
        self.assertFalse(result["should_retry"])

    @patch('app.agents.graph.review_output')
    def test_expert_review_low_score_retry(self, mock_review):
        """测试低分需要重试"""
        mock_review.return_value = {
            "score": 2,
            "approved": False,
            "feedback": "内容需要改进",
            "needs_revision": True
        }

        state = AgentState(
            messages=[AIMessage(content="推荐内容")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="nutrition",
            retry_count=0,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = expert_review(state)

        self.assertEqual(result["retry_count"], 1)
        self.assertTrue(result["should_retry"])

    @patch('app.agents.graph.review_output')
    def test_expert_review_max_retries_no_retry(self, mock_review):
        """测试达到最大重试次数时不重试"""
        mock_review.return_value = {
            "score": 2,
            "approved": False,
            "feedback": "内容需要改进",
            "needs_revision": True
        }

        state = AgentState(
            messages=[AIMessage(content="推荐内容")],
            user_id=1,
            user_profile={},
            daily_stats={},
            current_agent="nutrition",
            retry_count=MAX_RETRIES - 1,
            review_history=[],
            memory_summary={},
            enhanced_prompts={}
        )

        result = expert_review(state)

        self.assertFalse(result["should_retry"])


class TestShouldContinue(unittest.TestCase):
    """Should Continue 条件判断单元测试"""

    def test_should_continue_nutrition_retry(self):
        """测试 nutrition 需要重试"""
        state = {"should_retry": True}
        result = should_continue_nutrition(state)
        self.assertEqual(result, "nutrition")

    def test_should_continue_nutrition_end(self):
        """测试 nutrition 结束"""
        state = {"should_retry": False}
        result = should_continue_nutrition(state)
        self.assertEqual(result, "__end__")

    def test_should_continue_fitness_retry(self):
        """测试 fitness 需要重试"""
        state = {"should_retry": True}
        result = should_continue_fitness(state)
        self.assertEqual(result, "fitness")

    def test_should_continue_fitness_end(self):
        """测试 fitness 结束"""
        state = {"should_retry": False}
        result = should_continue_fitness(state)
        self.assertEqual(result, "__end__")


class TestRouteAfterRouter(unittest.TestCase):
    """Route After Router 单元测试"""

    def test_route_after_router_chat(self):
        """测试路由到 chat"""
        state = {"current_agent": "chat"}
        result = route_after_router(state)
        self.assertEqual(result, "chat")

    def test_route_after_router_nutrition(self):
        """测试路由到 nutrition"""
        state = {"current_agent": "nutrition"}
        result = route_after_router(state)
        self.assertEqual(result, "nutrition")

    def test_route_after_router_fitness(self):
        """测试路由到 fitness"""
        state = {"current_agent": "fitness"}
        result = route_after_router(state)
        self.assertEqual(result, "fitness")


class TestBuildGraph(unittest.TestCase):
    """Build Graph 单元测试"""

    def test_build_graph_returns_state_graph(self):
        """测试 build_graph 返回 StateGraph"""
        graph = build_graph()
        self.assertIsNotNone(graph)


if __name__ == '__main__':
    unittest.main()