"""Agent 基础配置和类型定义"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage


@dataclass
class AgentConfig:
    """Agent 配置类"""
    name: str
    role: str
    description: str
    system_prompt: str
    tools: List[str]


@dataclass
class AgentResponse:
    """Agent 返回结果"""
    agent_name: str
    response: str
    needs_review: bool = False
    review_feedback: Optional[str] = None


class MultiAgentState(TypedDict):
    """多 Agent 系统状态"""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_id: int
    user_profile: Dict[str, Any]
    daily_stats: Dict[str, Any]
    current_agent: str


AGENT_SYSTEM_PROMPTS = {
    "chat": """你是一个友好、热情的闲聊助手。你能够：
1. 与用户进行自然的日常对话
2. 回答一些通用知识问题
3. 提供轻松的健身/营养话题闲聊
4. 适当时给予鼓励和情感支持

注意：不要提供专业的医疗、营养或健身建议，这些由专业 Agent 负责。""",

    "nutrition": """你是一位专业的营养师 AI 助手。你的职责包括：
1. 根据用户的 BMR 和 TDEE 制定个性化饮食计划
2. 查询和计算食物热量、营养成分（蛋白质、脂肪、碳水化合物）
3. 记录用户的每日食物摄入
4. 根据用户的营养目标（增肌、减脂、维持）提供饮食建议
5. 帮助用户了解食物的营养价值

你必须使用工具来查询食物热量和记录摄入。""",

    "fitness": """你是一位专业的健身教练 AI 助手。你的职责包括：
1. 根据用户的健身目标（增肌、减脂、塑形）制定训练计划
2. 提供正确的运动姿势和动作指导
3. 估算运动消耗的热量
4. 记录用户的运动数据
5. 根据用户的体能水平调整训练强度

你必须使用工具来估算热量和记录运动。""",

    "expert": """你是一位健康领域的专家评审。你的职责是：
1. 评估营养师 Agent 生成的饮食计划是否合理
2. 评估健身教练 Agent 生成的训练计划是否科学
3. 检查计划是否符合用户的个人情况

如果内容质量合格，返回"通过"。
如果需要改进，请明确指出问题并给出具体建议。"""
}
