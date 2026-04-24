"""Agentic RAG - 自主决策的 RAG Agent

核心思想：让 LLM 作为大脑，自主决定：
1. 是否需要检索
2. 使用哪种检索策略
3. 是否需要多步推理
4. 回答质量是否需要反思纠正

工作流程：
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  Router Agent                            │
│  - 分析问题类型（事实性/观点性/分析性/闲聊）             │
│  - 决定检索策略（no_retrieval/hybrid/hyde/CoT）         │
│  - 决定是否需要反思（Self-RAG）                         │
└─────────────────────┬───────────────────────────────────┘
                      │
     ┌────────────────┼────────────────┐
     │                │                │
     ▼                ▼                ▼
┌─────────┐    ┌───────────┐    ┌──────────┐
│ 无需检索 │    │  混合检索  │    │  HyDE    │
│ (直接答) │    │ + Self-RAG│    │ + CoT    │
└─────────┘    └───────────┘    └──────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  质量评估        │
             │  (Self-Reflect) │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   最终回答       │
             └─────────────────┘
"""

from typing import List, Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import json

load_dotenv()


ROUTER_SYSTEM_PROMPT = """你是一个专业的 RAG 策略规划助手。你的任务是根据用户问题，自主决定最佳的信息获取和回答策略。

## 可用策略

### 检索策略
1. **no_retrieval**: 不需要检索
   - 适用：闲聊、观点性讨论、个人经验分享
   - 特征：问题不涉及具体事实、数据、知识点

2. **basic**: 基础检索（混合向量+BM25）
   - 适用：一般事实性问题
   - 特征：询问具体知识、定义、做法

3. **hyde**: 假设性文档嵌入
   - 适用：查询与文档表述差异大的问题
   - 特征：问题简短、模糊、或与知识库表述方式不同

4. **query_expansion**: 多查询扩展
   - 适用：复杂问题、多角度问题
   - 特征：需要从多个角度检索才能全面回答

5. **cot**: 思维链推理
   - 适用：分析性、推理性问题
   - 特征：需要逐步分析、因果推断

6. **self_rag**: 自我反思（最高质量）
   - 适用：重要问题、需要高质量回答
   - 特征：对准确性要求高，愿意等待更长处理时间

### 生成策略
1. **direct**: 直接生成，基于自身知识
2. **rag_based**: 基于检索内容生成
3. **cot**: 思维链推理后生成
4. **self_reflect**: 自我反思后生成

## 决策规则

1. **必须检索**：涉及具体知识点、数据、科学事实的问题
2. **推荐检索**：询问"如何"、"为什么"、"是什么"类问题
3. **可选检索**：观点讨论、闲聊类问题
4. **高质量优先**：重要决策类问题建议启用 self_rag
5. **效率优先**：日常闲聊可用 no_retrieval

## 输出格式

请严格按以下 JSON 格式输出（不要输出其他内容）：

```json
{
  "need_retrieval": true/false,
  "retrieval_strategy": "no_retrieval/basic/hyde/query_expansion/cot/self_rag",
  "generation_strategy": "direct/rag_based/cot/self_reflect",
  "reasoning": "决策理由（1-2句话）",
  "suggested_top_k": 3-5,
  "priority": "high/medium/low"
}
```"""


ROUTER_USER_PROMPT = """用户问题：{query}

请分析并决定最佳策略。"""


EXECUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的健身与营养顾问。

根据以下信息生成高质量回答：

问题：{query}
检索策略：{retrieval_strategy}
生成策略：{generation_strategy}

检索结果（如果有）：
{context}

要求：
1. 优先使用检索信息
2. 检索信息不足时结合自身知识
3. 保持回答准确、清晰、有条理
4. 如果不确定，明确说明
5. 标注信息来源（如有）"""),
    ("human", "请生成回答：")
])


REFLECTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的回答质量评估师。分析给定的回答，判断是否需要修正。

评估标准：
1. 准确性：回答是否正确、是否有事实错误
2. 完整性：是否完整回答了问题的各个方面
3. 相关性：是否紧密围绕问题展开
4. 清晰度：表达是否清晰、结构良好

原始问题：{query}

原始回答：
{answer}

检索内容：
{context}

请分析并决定是否需要修正。如果回答已经很好，输出"通过"；如果需要修正，简要说明问题。"""),
    ("human", "评估结果：")
])


class QueryClassifier:
    """查询分类器 - 分析问题类型"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.0
        )

    def classify(self, query: str) -> Dict[str, Any]:
        """分类查询

        Returns:
            {
                "type": str,  # factual/opinional/analytical/conversational
                "needs_knowledge": bool,
                "complexity": str  # simple/complex
            }
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """分析问题类型：
- factual: 事实性问题（who/what/when/where/how many）
- opinionial: 观点性问题（你认为/你觉得）
- analytical: 分析性问题（为什么/分析/比较）
- conversational: 闲聊（你好/在吗/谢谢）

同时判断：
- needs_knowledge: 是否需要外部知识
- complexity: 简单/复杂"""),
            ("human", "问题：{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"query": query})
            return self._parse_classification(result, query)
        except Exception as e:
            print(f"分类失败: {e}")
            return {"type": "factual", "needs_knowledge": True, "complexity": "simple"}

    def _parse_classification(self, result: str, query: str) -> Dict[str, Any]:
        """解析分类结果"""
        result_lower = result.lower()

        needs_knowledge = True
        if any(kw in result_lower for kw in ["不需要", "闲聊", "个人", "观点"]):
            needs_knowledge = False

        complexity = "simple"
        if any(kw in result_lower for kw in ["复杂", "多角度", "分析"]):
            complexity = "complex"

        q_type = "factual"
        if "观点" in result or "opinion" in result_lower:
            q_type = "opinionial"
        elif "分析" in result or "为什么" in query:
            q_type = "analytical"
        elif "你好" in query or "在吗" in query or "谢谢" in query:
            q_type = "conversational"

        return {
            "type": q_type,
            "needs_knowledge": needs_knowledge,
            "complexity": complexity
        }


class RouterAgent:
    """路由 Agent - 自主决策检索和生成策略"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4.7"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_USER_PROMPT)
        ])

        self.classifier = QueryClassifier(self.llm)

    def decide(self, query: str) -> Dict[str, Any]:
        """决定策略

        Args:
            query: 用户问题

        Returns:
            {
                "need_retrieval": bool,
                "retrieval_strategy": str,
                "generation_strategy": str,
                "reasoning": str,
                "suggested_top_k": int,
                "priority": str
            }
        """
        chain = self.prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"query": query})
            return self._parse_decision(result)
        except Exception as e:
            print(f"路由决策失败: {e}, 使用默认策略")
            return self._default_decision(query)

    def _parse_decision(self, result: str) -> Dict[str, Any]:
        """解析 LLM 决策结果"""
        try:
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            else:
                json_str = result

            data = json.loads(json_str.strip())

            return {
                "need_retrieval": data.get("need_retrieval", True),
                "retrieval_strategy": data.get("retrieval_strategy", "basic"),
                "generation_strategy": data.get("generation_strategy", "rag_based"),
                "reasoning": data.get("reasoning", ""),
                "suggested_top_k": data.get("suggested_top_k", 3),
                "priority": data.get("priority", "medium")
            }
        except json.JSONDecodeError:
            print(f"JSON 解析失败: {result[:100]}")
            return self._default_decision("")

    def _default_decision(self, query: str) -> Dict[str, Any]:
        """默认策略"""
        if len(query) < 5:
            return {
                "need_retrieval": False,
                "retrieval_strategy": "no_retrieval",
                "generation_strategy": "direct",
                "reasoning": "简短查询，默认为闲聊",
                "suggested_top_k": 0,
                "priority": "low"
            }

        return {
            "need_retrieval": True,
            "retrieval_strategy": "basic",
            "generation_strategy": "rag_based",
            "reasoning": "默认使用基础检索",
            "suggested_top_k": 3,
            "priority": "medium"
        }


class AgenticRAG:
    """Agentic RAG - 自主决策的 RAG 系统

    让 LLM 作为大脑，自主决定：
    - 是否检索
    - 使用什么检索策略
    - 使用什么生成策略
    - 是否需要自我反思
    """

    def __init__(
        self,
        modern_rag: Any,
        llm: Optional[ChatOpenAI] = None
    ):
        """初始化 Agentic RAG

        Args:
            modern_rag: ModernRAG 实例
            llm: LLM 模型（可选）
        """
        self.rag = modern_rag
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.7
        )

        self.router = RouterAgent(self.llm)
        self.classifier = QueryClassifier(self.llm)

    def query(self, query: str, auto_reflect: bool = True) -> Dict[str, Any]:
        """自主查询

        让大模型自己决定使用什么策略。

        Args:
            query: 用户问题
            auto_reflect: 是否自动自我反思

        Returns:
            {
                "answer": str,              # 最终回答
                "retrieval_used": bool,    # 是否使用了检索
                "retrieval_strategy": str, # 使用的检索策略
                "generation_strategy": str, # 使用的生成策略
                "reflection_used": bool,   # 是否使用了自我反思
                "reasoning": str,          # 决策理由
                "sources": List[Dict],     # 检索来源
                "confidence": float        # 置信度
            }
        """
        decision = self.router.decide(query)
        print(f"[Router] 决策: {decision['retrieval_strategy']} | {decision['generation_strategy']} | {decision['reasoning']}")

        strategy = decision["retrieval_strategy"]
        top_k = decision["suggested_top_k"]

        results = []
        context = ""
        hyde_doc = None

        if decision["need_retrieval"]:
            if strategy == "hyde":
                hyde_result = self.rag.search(query, top_k, mode="hyde")
                results = hyde_result if isinstance(hyde_result, list) else hyde_result.get("results", [])
                hyde_doc = hyde_result.get("hyde_doc") if isinstance(hyde_result, dict) else None
            elif strategy == "query_expansion":
                results = self.rag.search(query, top_k, mode="query_expansion")
            elif strategy == "cot":
                results = self.rag.search(query, top_k, mode="hybrid")
            elif strategy == "self_rag":
                results = self.rag.search(query, top_k, mode="hybrid")
            else:
                results = self.rag.search(query, top_k, mode="hybrid")

            context = self._build_context(results)

        generation_strategy = decision["generation_strategy"]

        if generation_strategy == "direct" or not decision["need_retrieval"]:
            answer = self._direct_generate(query)
        elif generation_strategy == "cot":
            answer = self._cot_generate(query, context)
        else:
            answer = self._rag_based_generate(query, context)

        reflection_used = False
        if auto_reflect and decision["priority"] == "high":
            reflection_result = self._reflect(query, answer, context)
            if reflection_result["needs_correction"]:
                print(f"[Reflection] 需要修正: {reflection_result['reason']}")
                answer = reflection_result["corrected_answer"]
                reflection_used = True

        confidence = self._estimate_confidence(decision, results, reflection_used)

        return {
            "answer": answer,
            "retrieval_used": decision["need_retrieval"],
            "retrieval_strategy": strategy,
            "generation_strategy": generation_strategy,
            "reflection_used": reflection_used,
            "reasoning": decision["reasoning"],
            "sources": results,
            "confidence": confidence,
            "hyde_doc": hyde_doc
        }

    def _build_context(self, results: List[Dict]) -> str:
        """构建上下文"""
        if not results:
            return "无相关检索结果。"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[来源{i}] {r.get('content', '')}")

        return "\n\n".join(parts)

    def _direct_generate(self, query: str) -> str:
        """直接生成（不检索）"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的健身与营养顾问。直接回答用户问题。

要求：
1. 回答准确、清晰、有条理
2. 如果不确定，明确说明
3. 可以提供额外的相关建议"""),
            ("human", "问题：{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def _rag_based_generate(self, query: str, context: str) -> str:
        """基于检索生成"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的健身与营养顾问。基于检索信息回答用户问题。

要求：
1. 优先使用检索信息
2. 检索信息不足时结合自身知识
3. 保持回答准确、清晰、有条理
4. 如果不确定，明确说明
5. 标注信息来源"""),
            ("human", "问题：{query}\n\n检索内容：\n{context}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context})

    def _cot_generate(self, query: str, context: str) -> str:
        """思维链生成"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严谨的健身与营养专家。

请采用逐步推理的方式回答问题：

工作流程：
1. 理解问题 → 提取关键信息
2. 分析检索内容 → 找到相关信息
3. 链式推理 → 基于事实逐步推导
4. 验证结论 → 检查是否符合检索内容
5. 给出答案 → 清晰、准确、有依据

问题：{query}

检索内容：
{context}

请进行链式推理并给出最终答案。"""),
            ("human", "推理过程：")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context})

    def _reflect(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """自我反思"""
        chain = REFLECTION_ANALYSIS_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "answer": answer,
                "context": context or "无"
            })

            if "通过" in result and "修正" not in result:
                return {
                    "needs_correction": False,
                    "reason": "",
                    "corrected_answer": answer
                }

            if result and len(result) > 5:
                corrected = self._cot_generate(query, context + f"\n\n[反思建议] {result}")
                return {
                    "needs_correction": True,
                    "reason": result,
                    "corrected_answer": corrected
                }

            return {
                "needs_correction": False,
                "reason": "",
                "corrected_answer": answer
            }

        except Exception as e:
            print(f"反思失败: {e}")
            return {
                "needs_correction": False,
                "reason": "",
                "corrected_answer": answer
            }

    def _estimate_confidence(
        self,
        decision: Dict,
        results: List[Dict],
        reflection_used: bool
    ) -> float:
        """估算置信度"""
        confidence = 0.5

        if decision["need_retrieval"] and results:
            confidence += 0.2

        if len(results) >= 3:
            confidence += 0.1

        if decision["priority"] == "high":
            confidence -= 0.1

        if reflection_used:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))


class AutoRAG:
    """自动 RAG - 简化的自主决策接口

    一句话使用：
        auto_rag = AutoRAG(rag)
        result = auto_rag.query("如何科学增肌？")
    """

    def __init__(self, modern_rag: Any):
        self.agentic_rag = AgenticRAG(modern_rag)

    def query(self, query: str) -> str:
        """查询，直接返回回答

        Args:
            query: 用户问题

        Returns:
            生成的回答
        """
        result = self.agentic_rag.query(query)
        return result["answer"]
