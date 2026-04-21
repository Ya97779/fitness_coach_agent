"""专家 Agent - 评审营养师和健身教练的输出"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import re
from dotenv import load_dotenv
from .base import AGENT_SYSTEM_PROMPTS

load_dotenv()


def review_output(nutrition_output: str, fitness_output: str) -> dict:
    """评审营养师和健身教练的输出

    评估维度：
    - 专业性：内容是否科学准确
    - 实用性：是否易于执行
    - 个性化：是否考虑用户个人情况
    - 安全性：是否存在潜在风险

    Args:
        nutrition_output: 营养师的输出内容
        fitness_output: 健身教练的输出内容

    Returns:
        dict: {
            "score": int,             # 评分 1-5
            "approved": bool,         # 是否通过（score >= 3）
            "feedback": str,          # 评审意见
            "needs_revision": bool    # 是否需要修改（score <= 2）
        }
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "glm-4.7"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    review_prompt = f"""{AGENT_SYSTEM_PROMPTS["expert"]}

请对以下内容进行评分（1-5分）：

=== 营养师输出 ===
{nutrition_output if nutrition_output else "（无营养师输出）"}

=== 健身教练输出 ===
{fitness_output if fitness_output else "（无健身教练输出）"}

评分标准：
- 1分：内容严重不足、错误或不相关
- 2分：内容有较大问题，需要显著改进
- 3分：内容基本合格，有小问题
- 4分：内容良好，专业且实用
- 5分：内容优秀，完美符合要求

请先给出评分（格式：评分: X），然后详细说明评审理由。
评分必须是一个1-5的数字。"""

    system_msg = SystemMessage(content=review_prompt)
    response = llm.invoke([system_msg])

    content = response.content

    # 提取评分
    score = extract_score(content)

    return {
        "score": score,
        "approved": score >= 3,
        "feedback": content,
        "needs_revision": score <= 2
    }


def extract_score(text: str) -> int:
    """从评审文本中提取评分

    优先查找 "评分: X" 或 "评分:X" 格式，
    其次查找纯数字 X（1-5）。

    Args:
        text: 评审文本

    Returns:
        int: 评分（1-5），默认3
    """
    patterns = [
        r"评分[:：]\s*(\d)",
        r"分数[:：]\s*(\d)",
        r"^(\d)$",
        r"(\d)\s*分"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

    # 默认返回3分（及格）
    return 3
