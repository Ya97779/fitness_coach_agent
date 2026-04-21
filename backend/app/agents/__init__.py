"""多 Agent 系统核心模块

包含四个专用 Agent：
- Chat Agent: 闲聊
- Nutrition Agent: 营养师
- Fitness Agent: 健身教练
- Expert Agent: 专家评审
- Router: 路由器
- Graph: 主控工作流

使用方式：
    from .agents import process_user_message

    result = process_user_message("我想减肥应该怎么吃", user_id=1)
"""

from .base import MultiAgentState, AGENT_SYSTEM_PROMPTS
from .chat_agent import chat_with_user
from .nutrition_agent import nutrition_tools, nutrition_with_user
from .fitness_agent import fitness_tools, fitness_with_user
from .expert_agent import review_output
from .router import route_with_context
from .graph import process_user_message

__all__ = [
    "MultiAgentState",
    "AGENT_SYSTEM_PROMPTS",
    "chat_with_user",
    "nutrition_tools",
    "nutrition_with_user",
    "fitness_tools",
    "fitness_with_user",
    "review_output",
    "route_with_context",
    "process_user_message"
]
