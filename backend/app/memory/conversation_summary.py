"""对话历史摘要模块 - 管理对话历史的压缩和摘要"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class ConversationSummarizer:
    """对话历史摘要器

    功能：
    1. 检测对话长度，决定是否需要摘要
    2. 对早期对话进行摘要，保留关键信息
    3. 维护压缩后的对话历史
    """

    MAX_MESSAGES_BEFORE_SUMMARY = 10
    MAX_SUMMARY_LENGTH = 5

    def __init__(
        self,
        max_messages: int = 10,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """初始化摘要器

        Args:
            max_messages: 摘要前的最大消息数
            model: LLM 模型名称
            api_key: API 密钥
            api_base: API 基础 URL
        """
        self.max_messages = max_messages
        self.model = model or os.getenv("LLM_MODEL", "glm-4.7")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")

    def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """判断是否需要对对话进行摘要

        Args:
            messages: 消息列表

        Returns:
            bool: 是否需要摘要
        """
        return len(messages) > self.max_messages

    def summarize_messages(
        self,
        messages: List[BaseMessage],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[BaseMessage]:
        """对早期消息进行分层摘要，保留关键信息

        分层策略：
        - 最近 10 条：完整保留
        - 11-20 条：简要摘要（每5条→1条，~50字）
        - 20 条以上：精简摘要（每10条→1条，~30字）

        Args:
            messages: 原始消息列表
            user_profile: 用户画像（用于识别关键信息）

        Returns:
            List[BaseMessage]: 压缩后的消息列表，顺序为 [精简摘要] → [简要摘要] → [最近10条原文]
        """
        if not self.should_summarize(messages):
            return messages

        keep_recent = self.MAX_MESSAGES_BEFORE_SUMMARY  # 10
        recent_messages = messages[-keep_recent:]
        older_messages = messages[:-keep_recent]

        # 分层处理
        system_messages = [m for m in older_messages if isinstance(m, SystemMessage)]
        non_system = [m for m in older_messages if not isinstance(m, SystemMessage)]

        summary_parts = []
        if len(non_system) > 10:
            # 精简摘要层：20条以上的部分
            deep_old = non_system[:-10]
            summary_parts.append(self._generate_layered_summary(deep_old, "compact"))
            # 简要摘要层：11-20条的部分
            mid_old = non_system[-10:]
            summary_parts.append(self._generate_layered_summary(mid_old, "brief"))
        else:
            # 只有简要摘要层
            summary_parts.append(self._generate_layered_summary(non_system, "brief"))

        result = []
        result.extend(system_messages)
        result.append(AIMessage(content="\n".join(summary_parts)))
        result.extend(recent_messages)
        return result

    def _generate_layered_summary(
        self,
        messages: List[BaseMessage],
        level: str = "brief"
    ) -> str:
        """按层级生成摘要

        Args:
            messages: 需要摘要的消息
            level: "brief"（简要，~50字）或 "compact"（精简，~30字）

        Returns:
            str: 生成的摘要
        """
        if not messages:
            return ""

        message_texts = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            message_texts.append(f"{role}: {content}")

        conversation_text = "\n".join(message_texts)

        if level == "compact":
            prompt = f"用30字以内精简概括以下对话的核心结论和用户偏好：\n{conversation_text}"
        else:
            prompt = f"用50字以内概括以下对话的主要话题和结论：\n{conversation_text}"

        try:
            from ..llm_manager import LLMManager
            llm = LLMManager.get_llm(temperature=0.3)
            response = llm.invoke([HumanMessage(content=prompt)])
            return f"【历史摘要】{response.content}"
        except Exception as e:
            return f"【历史摘要】（摘要生成失败）"

    def _generate_summary(
        self,
        messages: List[BaseMessage],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """使用 LLM 生成对话摘要

        Args:
            messages: 需要摘要的消息列表
            user_profile: 用户画像

        Returns:
            str: 生成的摘要
        """
        if not messages:
            return "（无历史对话）"

        message_texts = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            message_texts.append(f"{role}: {content}")

        conversation_text = "\n".join(message_texts)

        user_info = ""
        if user_profile:
            goal = user_profile.get("goal", {})
            target = goal.get("target_weight", "未设定")
            current = goal.get("current_weight", "未知")
            allergies = user_profile.get("constraints", {}).get("allergies", "无")
            user_info = f"\n用户目标体重: {target} kg, 当前: {current} kg\n过敏史: {allergies}"

        prompt = f"""请对以下对话历史进行简短摘要，保留关键信息：

{conversation_text}

{user_info}

请按以下格式生成摘要（100字以内）：
1. 用户主要讨论的话题/问题：
2. 达成的共识或建议：
3. 用户的特殊需求或限制（如有）："""

        try:
            from ..llm_manager import LLMManager
            llm = LLMManager.get_llm(temperature=0.3)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"（早期对话摘要失败: {str(e)[:50]}）"

    def extract_key_info(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """从对话中提取关键信息

        Args:
            messages: 消息列表

        Returns:
            Dict: 提取的关键信息
        """
        key_info = {
            "topics": [],
            "goals": [],
            "preferences": [],
            "constraints": []
        }

        all_content = " ".join([
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in messages
        ])

        if "增肌" in all_content or "肌肉" in all_content:
            key_info["goals"].append("增肌")
        if "减脂" in all_content or "减肥" in all_content or "瘦身" in all_content:
            key_info["goals"].append("减脂")
        if "饮食" in all_content or "营养" in all_content or "食物" in all_content:
            key_info["topics"].append("饮食营养")
        if "运动" in all_content or "训练" in all_content or "健身" in all_content:
            key_info["topics"].append("运动训练")

        return key_info
