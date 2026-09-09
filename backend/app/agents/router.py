"""意图路由器。

路由仍然复用现有 ``hybrid_route`` 入口，但决策顺序改为：

1. 高精度规则识别明确的记录/查询/安全场景；
2. 使用当前会话的上一轮状态承接省略句；
3. 只有模糊或跨领域请求才调用 LLM，并要求结构化 JSON。

``agent`` 字段和旧的 ``route_with_context`` 返回值保留，避免影响现有
LangGraph、测试和小程序；新增字段用于后续主教练协调和观测。
"""

from typing import Any, Dict, Iterable, List, Optional, TypedDict
import json
import re

from langchain_core.messages import HumanMessage


class RouteDecision(TypedDict, total=False):
    """稳定的路由协议；旧调用方只需读取 agent/reason。"""

    agent: str
    mode: str
    primary_domain: str
    domains: List[str]
    intent: str
    needs_tools: bool
    needs_clarification: bool
    safety_flags: List[str]
    confidence: float
    reason: str
    reason_code: str
    method: str


_NUTRITION_TERMS = {
    "营养师", "营养专家", "营养助手", "早餐", "午餐", "晚餐", "早饭", "午饭",
    "晚饭", "加餐", "零食", "夜宵", "食物", "饮食", "营养", "热量", "卡路里",
    "蛋白质", "脂肪", "碳水", "碳水化合物", "膳食纤维", "维生素", "矿物质",
    "补剂", "蛋白粉", "摄入", "代谢", "基础代谢", "TDEE", "BMR", "食谱",
    "怎么吃", "吃什么", "多少克", "鸡胸肉", "鸡蛋", "蔬菜", "水果", "米饭",
    "面条", "面包", "包子", "馒头", "饺子", "牛奶", "豆浆", "苹果", "香蕉",
    "鸡腿", "鸡翅", "猪肉", "牛肉", "羊肉", "鱼", "虾", "豆腐", "汉堡",
    "披萨", "沙拉", "酸奶", "麦片",
}

_FITNESS_TERMS = {
    "健身教练", "教练", "运动", "训练", "健身", "跑步", "卧推", "深蹲", "硬拉",
    "俯卧撑", "有氧", "无氧", "HIIT", "瑜伽", "普拉提", "游泳", "骑行", "力量",
    "耐力", "柔韧性", "拉伸", "热身", "肌肉", "手臂", "腿部", "背部", "胸部",
    "肩部", "腹部", "臀部", "次数", "组数", "重量", "RM", "健身房", "怎么练",
    "动作", "姿势", "发力", "肌肉酸痛", "恢复", "器械", "哑铃", "杠铃",
}

_FOOD_RECORD_VERBS = ("吃了", "喝了", "吃的", "喝的", "食用了", "摄入了")
_EXERCISE_RECORD_VERBS = (
    "跑了", "游了", "骑了", "练了", "做了运动", "做了训练", "完成了训练",
)
_RECORD_HINTS = ("帮我记录", "记录一下", "帮我记", "记一下", "记到日志", "记入日志")
_SHARED_GOALS = ("增肌", "减脂", "减肥", "瘦身", "保持体重")
_CONTINUATION_HINTS = (
    "第二个", "第一个", "上一个", "下一个", "继续", "换成", "改成", "就这个",
    "按这个", "这个呢", "那这个", "还有吗", "怎么选", "可以",
)
_SAFETY_TERMS = (
    "疼", "痛", "受伤", "伤病", "肿", "麻", "头晕", "胸闷", "呼吸困难",
    "心慌", "膝盖", "腰疼", "腰痛", "肩疼", "肩痛",
)


def _normalize(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def _context_text(context_messages: Optional[Iterable[Any]]) -> str:
    """Extract a small amount of prior message text for continuation detection."""

    if not context_messages:
        return ""
    parts: List[str] = []
    for message in list(context_messages)[-6:]:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", message)
        if content:
            parts.append(_normalize(str(content))[:500])
    return "\n".join(parts)


def _domain_score(text: str, terms: Iterable[str]) -> int:
    score = 0
    for term in sorted(terms, key=len, reverse=True):
        if term in text:
            # 长实体比“运动/食物”这类宽泛词更可靠。
            score += 2 if len(term) >= 3 else 1
    return score


def _safety_flags(text: str) -> List[str]:
    flags: List[str] = []
    if any(term in text for term in _SAFETY_TERMS):
        flags.append("pain_or_symptom")
    if any(term in text for term in ("怀孕", "孕期", "术后", "高血压", "糖尿病")):
        flags.append("special_population")
    return flags


def _rule_decision(
    user_message: str,
    context_messages: Optional[Iterable[Any]] = None,
    context_route: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    text = _normalize(user_message)
    if not text:
        return {
            "agent": "chat", "mode": "clarify", "primary_domain": "general",
            "domains": ["general"], "intent": "clarify", "needs_tools": False,
            "confidence": 1.0, "reason": "空输入", "reason_code": "EMPTY_INPUT",
            "safety_flags": [], "needs_clarification": True,
        }

    nutrition_score = _domain_score(text, _NUTRITION_TERMS)
    fitness_score = _domain_score(text, _FITNESS_TERMS)
    safety_flags = _safety_flags(text)
    food_record = (
        any(verb in text for verb in _FOOD_RECORD_VERBS)
        or (any(hint in text for hint in _RECORD_HINTS) and nutrition_score > 0)
    )
    exercise_record = (
        any(verb in text for verb in _EXERCISE_RECORD_VERBS)
        or (
            any(hint in text for hint in _RECORD_HINTS)
            and fitness_score > 0
            and not food_record
        )
    )

    # 明确包含两类动作时，不再强行二选一。当前实现先交给 chat
    # 主入口，结构化字段供后续协调器拆分任务。
    if food_record and exercise_record:
        return {
            "agent": "chat", "mode": "multi_domain", "primary_domain": "general",
            "domains": ["nutrition", "fitness"], "intent": "record",
            "needs_tools": True, "confidence": 0.95,
            "reason": "同时检测到饮食和运动记录", "reason_code": "MULTI_RECORD",
            "safety_flags": safety_flags, "needs_clarification": False,
        }

    # 明确的训练记录优先于“热量/消耗”类通用营养词；带运动实体的
    # “跑步消耗多少”仍然属于运动热量查询。
    if exercise_record or (
        fitness_score > nutrition_score
        and any(term in text for term in ("消耗", "训练", "动作", "组", "次数"))
    ) or (
        fitness_score > 0
        and any(term in text for term in ("消耗", "热量", "卡路里"))
    ):
        return {
            "agent": "fitness", "mode": "safety" if safety_flags else "fast_action",
            "primary_domain": "fitness", "domains": ["fitness"],
            "intent": "record" if exercise_record else "query",
            "needs_tools": True, "confidence": 0.94 if exercise_record else 0.88,
            "reason": "明确运动/训练语义", "reason_code": "EXPLICIT_EXERCISE",
            "safety_flags": safety_flags, "needs_clarification": False,
        }

    if food_record or (
        nutrition_score > 0
        and any(term in text for term in ("热量", "卡路里", "营养", "吃什么", "怎么吃", "食谱"))
    ):
        return {
            "agent": "nutrition", "mode": "fast_action" if food_record else "single_domain",
            "primary_domain": "nutrition", "domains": ["nutrition"],
            "intent": "record" if food_record else "query",
            "needs_tools": True, "confidence": 0.94 if food_record else 0.87,
            "reason": "明确饮食/营养语义", "reason_code": "EXPLICIT_NUTRITION",
            "safety_flags": safety_flags, "needs_clarification": False,
        }

    has_both_domains = (
        nutrition_score > 0 and fitness_score > 0
    ) or (
        any(goal in text for goal in _SHARED_GOALS)
        and any(term in text for term in ("训练", "运动", "饮食", "吃"))
    )
    if has_both_domains:
        return {
            "agent": "chat", "mode": "safety" if safety_flags else "multi_domain",
            "primary_domain": "general", "domains": ["fitness", "nutrition"],
            "intent": "advise", "needs_tools": True, "confidence": 0.86,
            "reason": "跨运动和饮食的综合问题", "reason_code": "MULTI_DOMAIN",
            "safety_flags": safety_flags, "needs_clarification": False,
        }

    context = _context_text(context_messages)
    is_continuation = any(hint in text for hint in _CONTINUATION_HINTS) or len(text) <= 8
    if is_continuation:
        previous_agent = (context_route or {}).get("agent") if context_route else None
        if previous_agent in {"nutrition", "fitness"}:
            return {
                "agent": previous_agent, "mode": "single_domain",
                "primary_domain": previous_agent, "domains": [previous_agent],
                "intent": "follow_up", "needs_tools": True, "confidence": 0.82,
                "reason": "承接当前会话的上一领域", "reason_code": "SESSION_CONTINUATION",
                "safety_flags": safety_flags, "needs_clarification": False,
            }
        if context:
            prior_fitness = _domain_score(context, _FITNESS_TERMS)
            prior_nutrition = _domain_score(context, _NUTRITION_TERMS)
            if prior_fitness > prior_nutrition and prior_fitness > 0:
                inherited = "fitness"
            elif prior_nutrition > 0:
                inherited = "nutrition"
            else:
                inherited = None
            if inherited:
                return {
                    "agent": inherited, "mode": "single_domain",
                    "primary_domain": inherited, "domains": [inherited],
                    "intent": "follow_up", "needs_tools": True, "confidence": 0.75,
                    "reason": "根据近期对话承接领域", "reason_code": "SESSION_CONTEXT",
                    "safety_flags": safety_flags, "needs_clarification": False,
                }

    # 有安全信号时优先进入安全模式；如果同时提到目标/运动，保留
    # fitness 作为主领域，让已有健身提示词给出动作边界和就医提醒。
    if safety_flags:
        safety_agent = (
            "fitness"
            if fitness_score > 0 or any(goal in text for goal in _SHARED_GOALS)
            else "chat"
        )
        return {
            "agent": safety_agent, "mode": "safety",
            "primary_domain": safety_agent if safety_agent != "chat" else "general",
            "domains": [safety_agent if safety_agent != "chat" else "general"],
            "intent": "safety_advice", "needs_tools": safety_agent == "fitness",
            "confidence": 0.72, "reason": "检测到疼痛或特殊情况",
            "reason_code": "SAFETY_SIGNAL", "safety_flags": safety_flags,
            "needs_clarification": False,
        }

    return None


def _keyword_match(text: str) -> Optional[str]:
    """Compatibility helper returning only the primary legacy domain."""

    decision = _rule_decision(text)
    if not decision:
        return None
    if decision.get("mode") == "multi_domain":
        return "mixed"
    agent = decision.get("agent")
    return agent if agent in {"nutrition", "fitness"} else None


def _safe_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if "```" in raw:
        raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _llm_route(
    user_message: str,
    context_messages: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Use the existing LLM manager for ambiguous structured routing."""

    from ..llm_manager import LLMManager

    context = _context_text(context_messages)
    prompt = f"""你是健身助手的意图路由器，只负责分类，不回答用户问题。
当前输入：{user_message}
最近会话（可能为空）：
{context[-1800:]}

返回严格 JSON，不要 Markdown：
{{
  "agent": "chat|nutrition|fitness",
  "mode": "single_domain|multi_domain|clarify|safety",
  "domains": ["general|nutrition|fitness"],
  "intent": "record|query|advise|create_plan|modify_plan|follow_up|clarify|safety_advice",
  "needs_tools": true,
  "needs_clarification": false,
  "safety_flags": [],
  "confidence": 0.0,
  "reason_code": "..."
}}
规则：跨饮食和训练的问题使用 chat + multi_domain；疼痛、伤病或特殊人群使用 safety；短省略句优先继承最近领域。"""

    try:
        from ..llm_manager import LLMManager
        llm = LLMManager.get_llm(temperature=0.1)
        # 路由结果最终仍需完整 JSON 才能做决定，但模型调用本身使用
        # stream，避免流式请求链路中再出现一个隐含的阻塞 invoke。
        streamed_parts = []
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            value = getattr(chunk, "content", chunk)
            if isinstance(value, str):
                streamed_parts.append(value)
            elif value:
                streamed_parts.append(str(value))
        content = "".join(streamed_parts)

        # 兼容旧 Provider 或测试桩不提供 stream 内容的情况；正常生产
        # Provider 会在上面的流式路径返回至少一个文本 chunk。
        if not content:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = getattr(response, "content", "")
        parsed = _safe_json_from_text(content)
        if parsed:
            agent = parsed.get("agent")
            if agent not in {"chat", "nutrition", "fitness"}:
                agent = "chat"
            mode = parsed.get("mode") or "single_domain"
            if mode not in {"single_domain", "multi_domain", "clarify", "safety", "fast_action"}:
                mode = "single_domain"
            domains = parsed.get("domains")
            if not isinstance(domains, list) or not domains:
                domains = [agent if agent != "chat" else "general"]
            try:
                confidence = max(0.0, min(float(parsed.get("confidence", 0.65)), 1.0))
            except (TypeError, ValueError):
                confidence = 0.65
            safety_flags = parsed.get("safety_flags", [])
            if not isinstance(safety_flags, list):
                safety_flags = []
            return {
                "agent": agent,
                "mode": mode,
                "primary_domain": agent if agent != "chat" else "general",
                "domains": domains[:3],
                "intent": parsed.get("intent", "advise"),
                "needs_tools": bool(parsed.get("needs_tools", False)),
                "needs_clarification": bool(parsed.get("needs_clarification", False)),
                "safety_flags": safety_flags,
                "confidence": confidence,
                "reason": "LLM结构化语义判断",
                "reason_code": parsed.get("reason_code", "LLM_CLASSIFIED"),
            }

        # 兼容旧提示词/测试桩返回的 1、2、3。
        match = re.search(r"\b([123])\b", content or "")
        if match:
            agent = {"1": "chat", "2": "nutrition", "3": "fitness"}[match.group(1)]
            return {
                "agent": agent, "mode": "single_domain" if agent != "chat" else "clarify",
                "primary_domain": agent if agent != "chat" else "general",
                "domains": [agent if agent != "chat" else "general"],
                "intent": "advise", "needs_tools": agent != "chat",
                "needs_clarification": agent == "chat", "safety_flags": [],
                "confidence": 0.8, "reason": "闲聊/通用（LLM语义判断）", "reason_code": "LLM_LEGACY_CLASSIFIED",
            }

        lowered = (content or "").lower()
        if "营养" in lowered or "饮食" in lowered:
            agent, reason = "nutrition", "LLM语义判断：营养"
        elif "健身" in lowered or "运动" in lowered or "训练" in lowered:
            agent, reason = "fitness", "LLM语义判断：健身"
        else:
            agent, reason = "chat", "闲聊/通用"
        return {
            "agent": agent, "mode": "single_domain" if agent != "chat" else "clarify",
            "primary_domain": agent if agent != "chat" else "general",
            "domains": [agent if agent != "chat" else "general"], "intent": "advise",
            "needs_tools": agent != "chat", "needs_clarification": agent == "chat",
            "safety_flags": [], "confidence": 0.6, "reason": reason,
            "reason_code": "LLM_TEXT_FALLBACK",
        }
    except Exception as e:
        print(f"路由出错: {e}")
        return {
            "agent": "chat", "mode": "clarify", "primary_domain": "general",
            "domains": ["general"], "intent": "clarify", "needs_tools": False,
            "needs_clarification": True, "safety_flags": [], "confidence": 0.3,
            "reason": "默认闲聊", "reason_code": "ROUTER_FALLBACK",
        }


def hybrid_route(
    user_message: str,
    require_llm_confirm: bool = True,
    context_messages: Optional[Iterable[Any]] = None,
    context_route: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a backward-compatible but structured route decision."""

    rule_result = _rule_decision(user_message, context_messages, context_route)
    if rule_result is not None:
        rule_result["method"] = "rule"
        return rule_result

    if not require_llm_confirm:
        return {
            "agent": "chat", "mode": "clarify", "primary_domain": "general",
            "domains": ["general"], "intent": "clarify", "needs_tools": False,
            "needs_clarification": True, "safety_flags": [], "confidence": 0.4,
            "reason": "规则未识别，等待澄清", "reason_code": "RULE_UNCERTAIN",
            "method": "rule",
        }

    result = _llm_route(user_message, context_messages)
    result["method"] = "llm"
    return result


def route_with_context(
    user_message: str,
    user_id: int = None,
    context_messages: Optional[Iterable[Any]] = None,
    context_route: Optional[Dict[str, Any]] = None,
) -> dict:
    """Compatibility wrapper used by existing callers and tests."""

    result = hybrid_route(
        user_message,
        require_llm_confirm=True,
        context_messages=context_messages,
        context_route=context_route,
    )
    return {
        "agent": result["agent"],
        "reason": result["reason"],
        "mode": result.get("mode"),
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "reason_code": result.get("reason_code"),
    }
