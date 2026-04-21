"""主 Agent 工作流 - 整合路由器和各个 Agent

工作流程：
1. 用户输入 → 路由器分析意图
2. 路由器选择合适的 Agent（chat/nutrition/fitness）
3. 被选中的 Agent 处理请求并返回结果
4. 如果是 nutrition/fitness，Expert Agent 评审输出
5. 评分 <= 2 时打回重试，最多重试 MAX_RETRIES 次
"""

from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

from .router import route_with_context
from .chat_agent import chat_with_user
from .nutrition_agent import nutrition_tools, nutrition_with_user
from .fitness_agent import fitness_tools, fitness_with_user
from .expert_agent import review_output

load_dotenv()

# Expert 评审配置
MAX_RETRIES = 3  # 最多重试次数
MIN_APPROVAL_SCORE = 3  # 通过分数阈值


def process_user_message(
    user_message: str,
    user_id: int = 1,
    user_profile: dict = None,
    daily_stats: dict = None
) -> dict:
    """处理用户消息的入口函数

    工作流程：
    1. 路由决策
    2. 调用对应 Agent
    3. Expert 评审（仅 nutrition/fitness）
    4. 评分 <= 2 时重试，最多 MAX_RETRIES 次

    Args:
        user_message: 用户输入的消息
        user_id: 用户ID，默认1
        user_profile: 用户信息字典（可选）
        daily_stats: 当日统计数据（可选）

    Returns:
        dict: {
            "response": str,              # 最终回复内容
            "agent": str,                 # 处理的 Agent 类型
            "nutrition_response": str,     # 营养师回复
            "fitness_response": str,      # 健身教练回复
            "expert_review": {            # 专家评审结果
                "score": int,             # 评分 1-5
                "approved": bool,         # 是否通过
                "feedback": str,          # 评审意见
                "retries": int,           # 重试次数
                "review_history": list    # 评审历史
            }
        }
    """
    # Step 1: 路由决策
    routing_result = route_with_context(user_message, user_id)
    selected_agent = routing_result["agent"]

    # 构建消息列表
    messages = [HumanMessage(content=user_message)]

    # Step 2: Chat Agent 直接返回，不经过 Expert 评审
    if selected_agent == "chat":
        try:
            response = chat_with_user(messages, user_id)
            return {
                "response": response,
                "agent": "chat",
                "nutrition_response": "",
                "fitness_response": "",
                "expert_review": {
                    "score": 0,
                    "approved": True,
                    "feedback": "闲聊不需要评审",
                    "retries": 0,
                    "review_history": []
                }
            }
        except Exception as e:
            return {
                "response": f"闲聊处理出错: {str(e)}",
                "agent": "chat",
                "nutrition_response": "",
                "fitness_response": "",
                "expert_review": {}
            }

    # Step 3: Nutrition/Fitness Agent 需要 Expert 评审
    nutrition_response = ""
    fitness_response = ""
    review_history = []

    if selected_agent == "nutrition":
        for attempt in range(MAX_RETRIES):
            try:
                response = nutrition_with_user(messages, user_id)
                nutrition_response = response

                # Expert 评审
                review = review_output(nutrition_response, "")
                review_history.append({
                    "attempt": attempt + 1,
                    "score": review["score"],
                    "feedback": review["feedback"]
                })

                if review["approved"]:
                    # 评分 >= 3，通过
                    return {
                        "response": nutrition_response,
                        "agent": "nutrition",
                        "nutrition_response": nutrition_response,
                        "fitness_response": "",
                        "expert_review": {
                            "score": review["score"],
                            "approved": True,
                            "feedback": review["feedback"],
                            "retries": attempt,
                            "review_history": review_history
                        }
                    }
                else:
                    # 评分 <= 2，需要重试
                    if attempt < MAX_RETRIES - 1:
                        continue
            except Exception as e:
                return {
                    "response": f"营养师处理出错: {str(e)}",
                    "agent": "nutrition",
                    "nutrition_response": "",
                    "fitness_response": "",
                    "expert_review": {
                        "score": 0,
                        "approved": False,
                        "feedback": f"处理异常: {str(e)}",
                        "retries": attempt,
                        "review_history": review_history
                    }
                }

        # 超过最大重试次数，返回最后一次结果
        last_review = review_history[-1] if review_history else {"score": 0, "feedback": "未获得评审"}
        return {
            "response": nutrition_response,
            "agent": "nutrition",
            "nutrition_response": nutrition_response,
            "fitness_response": "",
            "expert_review": {
                "score": last_review["score"],
                "approved": False,
                "feedback": f"已达最大重试次数({MAX_RETRIES})，{last_review['feedback']}",
                "retries": MAX_RETRIES,
                "review_history": review_history
            }
        }

    elif selected_agent == "fitness":
        for attempt in range(MAX_RETRIES):
            try:
                response = fitness_with_user(messages, user_id)
                fitness_response = response

                # Expert 评审
                review = review_output("", fitness_response)
                review_history.append({
                    "attempt": attempt + 1,
                    "score": review["score"],
                    "feedback": review["feedback"]
                })

                if review["approved"]:
                    return {
                        "response": fitness_response,
                        "agent": "fitness",
                        "nutrition_response": "",
                        "fitness_response": fitness_response,
                        "expert_review": {
                            "score": review["score"],
                            "approved": True,
                            "feedback": review["feedback"],
                            "retries": attempt,
                            "review_history": review_history
                        }
                    }
                else:
                    if attempt < MAX_RETRIES - 1:
                        continue
            except Exception as e:
                return {
                    "response": f"健身教练处理出错: {str(e)}",
                    "agent": "fitness",
                    "nutrition_response": "",
                    "fitness_response": "",
                    "expert_review": {
                        "score": 0,
                        "approved": False,
                        "feedback": f"处理异常: {str(e)}",
                        "retries": attempt,
                        "review_history": review_history
                    }
                }

        last_review = review_history[-1] if review_history else {"score": 0, "feedback": "未获得评审"}
        return {
            "response": fitness_response,
            "agent": "fitness",
            "nutrition_response": "",
            "fitness_response": fitness_response,
            "expert_review": {
                "score": last_review["score"],
                "approved": False,
                "feedback": f"已达最大重试次数({MAX_RETRIES})，{last_review['feedback']}",
                "retries": MAX_RETRIES,
                "review_history": review_history
            }
        }

    # 默认返回 Chat
    try:
        response = chat_with_user(messages, user_id)
        return {
            "response": response,
            "agent": "chat",
            "nutrition_response": "",
            "fitness_response": "",
            "expert_review": {
                "score": 0,
                "approved": True,
                "feedback": "闲聊不需要评审",
                "retries": 0,
                "review_history": []
            }
        }
    except Exception as e:
        return {
            "response": f"闲聊处理出错: {str(e)}",
            "agent": "chat",
            "nutrition_response": "",
            "fitness_response": "",
            "expert_review": {}
        }
