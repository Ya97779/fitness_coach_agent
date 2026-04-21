"""CoT - Chain of Thought 思维链推理

核心思想：
在生成最终答案之前，先让 LLM 进行逐步推理。
支持：
- 简单思维链（直接推理）
- 链式思维（Step-by-Step）
- 检索增强思维（RAG-enhanced CoT）

适用场景：
- 复杂问题分析
- 多步骤问题求解
- 需要推理的问题
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


SIMPLE_COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的健身与营养顾问。请在回答用户问题时采用逐步推理的方式。

要求：
1. 先理解问题的核心要点
2. 逐步分析问题，每个推理步骤清晰可见
3. 最终给出明确结论
4. 如果不确定，明确指出

输出格式：
### 理解问题
[对问题的理解]

### 逐步分析
1. [第一个分析点]
2. [第二个分析点]
3. ...

### 最终结论
[结论]

### 参考来源
[如果基于检索内容，标注来源]"""),
    ("human", "{query}\n\n检索上下文：\n{context}")
])


CHAIN_COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的健身与营养专家。在回答问题时，必须：

1. 明确区分"已知事实"和"推理结论"
2. 每个推理步骤都要有依据
3. 如果检索到的信息与推理矛盾，以检索信息为准
4. 最终答案要回溯到具体检索内容

工作流程：
1. 理解问题 → 提取关键信息
2. 检索信息 → 找到相关内容
3. 链式推理 → 基于事实逐步推导
4. 验证结论 → 检查是否符合检索内容
5. 给出答案 → 清晰、准确、有依据"""),
    ("human", """问题：{query}

检索到的上下文：
{context}

请进行链式推理并给出最终答案。""")
])


class CoTReasoner:
    """思维链推理器

    支持多种推理模式，提高复杂问题的回答质量。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        mode: str = "simple"
    ):
        """初始化推理器

        Args:
            llm: LLM 模型
            mode: 推理模式 ("simple" | "chain")
        """
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.7
        )

        self.mode = mode

        if mode == "chain":
            self.prompt = CHAIN_COT_PROMPT
        else:
            self.prompt = SIMPLE_COT_PROMPT

    def reason(
        self,
        query: str,
        context: str = ""
    ) -> str:
        """推理

        Args:
            query: 用户问题
            context: 检索上下文（可选）

        Returns:
            推理后的回答
        """
        if not context:
            context = "无检索上下文，依赖自身知识回答。"

        chain = self.prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "context": context
            })
            return result
        except Exception as e:
            print(f"CoT 推理失败: {e}")
            return f"推理失败: {str(e)}"

    def reason_with_sources(
        self,
        query: str,
        context: str = "",
        sources: List[Dict] = None
    ) -> Dict[str, Any]:
        """带来源标注的推理

        Args:
            query: 用户问题
            context: 检索上下文
            sources: 检索来源列表

        Returns:
            {
                "answer": str,       # 最终答案
                "reasoning": str,    # 推理过程
                "sources": List,     # 参考来源
                "confidence": float  # 可信度
            }
        """
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严谨的健身与营养专家。
首先进行逐步推理（放在 <reasoning> 标签中），
然后给出最终答案（放在 <answer> 标签中）。

推理要求：
1. 明确标注每个推理步骤
2. 说明推理依据（来自检索还是自身知识）
3. 指明信息来源"""),
            ("human", """问题：{query}

检索上下文：
{context}

请先推理后回答。""")
        ])

        chain = reasoning_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "context": context or "无"
            })

            answer = ""
            reasoning = ""

            if "<answer>" in result and "</answer>" in result:
                parts = result.split("<answer>")
                reasoning = parts[0].replace("<reasoning>", "").replace("</reasoning>", "").strip()
                answer = parts[1].replace("</answer>", "").strip()
            else:
                reasoning = "（未使用结构化输出）"
                answer = result

            confidence = self._estimate_confidence(context, sources)

            return {
                "answer": answer,
                "reasoning": reasoning,
                "sources": sources or [],
                "confidence": confidence
            }

        except Exception as e:
            print(f"推理失败: {e}")
            return {
                "answer": f"推理过程中出错: {str(e)}",
                "reasoning": "",
                "sources": sources or [],
                "confidence": 0.0
            }

    def _estimate_confidence(
        self,
        context: str,
        sources: List[Dict] = None
    ) -> float:
        """估算答案可信度

        Args:
            context: 检索上下文
            sources: 检索来源

        Returns:
            可信度分数 (0-1)
        """
        confidence = 0.5

        if context and context != "无检索上下文，依赖自身知识回答。":
            confidence += 0.2

        if sources:
            confidence += min(0.2, len(sources) * 0.05)

        if "不确定" in context or "未知" in context:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))


class RAGCoT:
    """检索增强思维链

    将 RAG 检索与 CoT 推理结合：
    1. 检索相关知识
    2. 基于知识进行链式推理
    3. 生成最终答案
    """

    def __init__(
        self,
        retriever: Callable,
        reasoner: Optional[CoTReasoner] = None
    ):
        """初始化 RAG-CoT

        Args:
            retriever: 检索函数 (query, top_k) -> List[Dict]
            reasoner: 推理器（可选）
        """
        self.retriever = retriever
        self.reasoner = reasoner or CoTReasoner(mode="chain")

    def query(
        self,
        query: str,
        top_k: int = 5,
        show_reasoning: bool = True
    ) -> Dict[str, Any]:
        """查询

        Args:
            query: 用户问题
            top_k: 检索数量
            show_reasoning: 是否显示推理过程

        Returns:
            {
                "answer": str,
                "reasoning": str,
                "sources": List[Dict],
                "context": str
            }
        """
        results = self.retriever(query, top_k)

        if not results:
            reasoning_result = self.reasoner.reason_with_sources(
                query,
                context="无相关检索结果",
                sources=[]
            )
            return {
                "answer": reasoning_result["answer"],
                "reasoning": reasoning_result["reasoning"] if show_reasoning else "",
                "sources": [],
                "context": ""
            }

        context = self._build_context(results)

        reasoning_result = self.reasoner.reason_with_sources(
            query,
            context=context,
            sources=results
        )

        return {
            "answer": reasoning_result["answer"],
            "reasoning": reasoning_result["reasoning"] if show_reasoning else "",
            "sources": reasoning_result["sources"],
            "context": context,
            "confidence": reasoning_result.get("confidence", 0.5)
        }

    def _build_context(self, results: List[Dict]) -> str:
        """构建上下文

        Args:
            results: 检索结果

        Returns:
            上下文字符串
        """
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[来源{i}]\n{r.get('content', '')}")

        return "\n\n".join(parts)
