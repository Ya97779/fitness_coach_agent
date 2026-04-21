"""Self-RAG - 自我反思与纠正机制

核心思想：
- 不只依赖检索结果，评估每个检索片段的相关性
- 判断回答是否需要检索
- 自我反思回答质量，决定是否需要修正
- 标记回答类型：检索型 / 非检索型 / 混合型

引用标记格式：
[Retrieval: Yes/No, Relevance: 0-4, Utility: 0-4]
"""

from typing import List, Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


ISRETRIEVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的 RAG 评估助手。判断给定问题是否需要检索外部知识来回答。

评估标准：
- 事实性问题（who, what, when, where）→ 需要检索
- 需要具体数据/统计的问题 → 需要检索
- 观点性/开放性问题 → 可能不需要检索
- 个人经验/闲聊 → 不需要检索

输出格式（只输出一个词）：
- 如果需要检索：Yes
- 如果不需要检索：No"""),
    ("human", "问题：{query}")
])


RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的 RAG 评估助手。评估检索到的文档片段对问题的相关性。

评分标准（0-4分）：
- 0分：不相关，完全答非所问
- 1分：略微相关，有一点帮助但不直接
- 2分：相关，可以提供部分信息
- 3分：高度相关，可以回答大部分问题
- 4分：完全相关，精确匹配

评估要求：
1. 只考虑文档内容与问题的相关性
2. 不要猜测文档可能包含什么，只评估实际给出的内容
3. 简短输出评分和简要理由"""),
    ("human", "问题：{query}\n\n文档：{content}")
])


UTILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的回答质量评估助手。评估生成的回答在以下方面的表现：

评估维度：
1. 准确性 - 回答是否正确、事实性错误少
2. 完整性 - 是否完整回答了问题的各个方面
3. 清晰度 - 表达是否清晰、结构良好
4. 相关性 - 是否紧密围绕问题展开

评分标准：
- 0分：质量差，完全不符合要求
- 1分：质量较差，有明显问题
- 2分：质量一般，勉强可用
- 3分：质量良好，满足要求
- 4分：质量优秀，超出预期

输出格式：
评分: [分数]
理由: [简要说明]"""),
    ("human", "问题：{query}\n\n回答：{answer}")
])


CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的回答优化助手。基于给定的检索内容和评估反馈，优化原始回答。

优化要求：
1. 修正事实性错误
2. 补充遗漏的重要信息
3. 改进表达清晰度
4. 确保回答与问题高度相关
5. 保持回答简洁、有条理

如果原始回答已经很好，直接返回原回答。"""),
    ("human", "原始问题：{query}\n\n原始回答：{answer}\n\n评估反馈：{feedback}\n\n检索内容：\n{context}\n\n请优化回答。")
])


class SelfRAGScorer:
    """Self-RAG 评分器

    评估：
    - 是否需要检索
    - 检索内容相关性
    - 回答质量
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """初始化评分器

        Args:
            llm: LLM 模型
        """
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.0
        )

        self.isretrieval_prompt = ISRETRIEVAL_PROMPT
        self.relevance_prompt = RELEVANCE_PROMPT
        self.utility_prompt = UTILITY_PROMPT
        self.correction_prompt = CORRECTION_PROMPT

    def should_retrieve(self, query: str) -> bool:
        """判断是否需要检索

        Args:
            query: 用户问题

        Returns:
            True 表示需要检索
        """
        chain = self.isretrieval_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"query": query})
            return "yes" in result.lower().strip()
        except Exception as e:
            print(f"检索判断失败: {e}")
            return True

    def score_relevance(
        self,
        query: str,
        content: str
    ) -> Dict[str, Any]:
        """评估检索内容相关性

        Args:
            query: 用户问题
            content: 检索到的文档内容

        Returns:
            {
                "score": int,       # 0-4 分
                "reasoning": str    # 评估理由
            }
        """
        chain = self.relevance_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "content": content[:1000]
            })

            score = 2
            reasoning = ""

            lines = result.strip().split('\n')
            for line in lines:
                if line and line[0].isdigit():
                    try:
                        score = int(line[0])
                        reasoning = line[1:].strip()
                    except:
                        reasoning = line

            return {"score": score, "reasoning": reasoning}

        except Exception as e:
            print(f"相关性评估失败: {e}")
            return {"score": 1, "reasoning": "评估失败"}

    def score_utility(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """评估回答质量

        Args:
            query: 用户问题
            answer: 生成的回答

        Returns:
            {
                "score": int,       # 0-4 分
                "reasoning": str    # 评估理由
            }
        """
        chain = self.utility_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "answer": answer
            })

            score = 2
            reasoning = ""

            lines = result.strip().split('\n')
            for line in lines:
                if '评分' in line or '分数' in line:
                    try:
                        score = int(line.split(':')[1].strip())
                    except:
                        pass
                elif '理由' in line or '说明' in line:
                    reasoning = line.split(':')[1].strip()

            return {"score": score, "reasoning": reasoning}

        except Exception as e:
            print(f"质量评估失败: {e}")
            return {"score": 2, "reasoning": "评估失败"}

    def correct_answer(
        self,
        query: str,
        answer: str,
        feedback: str,
        context: str
    ) -> str:
        """纠正回答

        Args:
            query: 用户问题
            answer: 原始回答
            feedback: 评估反馈
            context: 检索上下文

        Returns:
            优化后的回答
        """
        chain = self.correction_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "answer": answer,
                "feedback": feedback,
                "context": context
            })
            return result
        except Exception as e:
            print(f"回答纠正失败: {e}")
            return answer


class SelfRAG:
    """自我反思 RAG

    工作流程：
    1. 判断是否需要检索
    2. 执行检索
    3. 评估检索内容相关性，过滤低质量片段
    4. 生成回答
    5. 评估回答质量，必要时自我纠正
    6. 输出最终回答（带反思标记）
    """

    def __init__(
        self,
        retriever: Any,
        llm: Optional[ChatOpenAI] = None,
        retrieval_threshold: float = 2.0,
        utility_threshold: float = 2.5,
        max_corrections: int = 2
    ):
        """初始化 Self-RAG

        Args:
            retriever: 检索器
            llm: LLM 模型
            retrieval_threshold: 相关性阈值，低于此值过滤
            utility_threshold: 质量阈值，低于此值纠正
            max_corrections: 最大纠正次数
        """
        self.retriever = retriever
        self.scorer = SelfRAGScorer(llm)
        self.retrieval_threshold = retrieval_threshold
        self.utility_threshold = utility_threshold
        self.max_corrections = max_corrections

        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL", "glm-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.7
        )

    def query(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """查询

        Args:
            query: 用户问题
            top_k: 检索数量

        Returns:
            {
                "answer": str,              # 最终回答
                "is_retrieval": bool,       # 是否使用了检索
                "retrieval_used": List[Dict], # 使用的高质量检索
                "rejected": List[Dict],     # 过滤掉的相关性低的检索
                "utility_score": float,     # 质量评分
                "num_corrections": int,      # 纠正次数
                "reflection": str           # 反思标记
            }
        """
        is_retrieval = self.scorer.should_retrieve(query)

        retrieval_used = []
        rejected = []
        context = ""

        if is_retrieval:
            results = self.retriever(query, top_k)

            for r in results:
                relevance = self.scorer.score_relevance(query, r.get("content", ""))
                r["relevance_score"] = relevance["score"]
                r["relevance_reasoning"] = relevance["reasoning"]

                if relevance["score"] >= self.retrieval_threshold:
                    retrieval_used.append(r)
                else:
                    rejected.append(r)

            context = "\n\n".join([
                f"[相关度:{r['relevance_score']}] {r['content']}"
                for r in retrieval_used
            ])

        answer = self._generate_answer(query, context, is_retrieval)

        utility = self.scorer.score_utility(query, answer)
        num_corrections = 0
        final_answer = answer

        while utility["score"] < self.utility_threshold and num_corrections < self.max_corrections:
            feedback = f"质量评分: {utility['score']}/4\n理由: {utility['reasoning']}"
            final_answer = self.scorer.correct_answer(
                query, final_answer, feedback, context
            )
            utility = self.scorer.score_utility(query, final_answer)
            num_corrections += 1

        reflection = self._generate_reflection(
            is_retrieval, retrieval_used, rejected, utility, num_corrections
        )

        return {
            "answer": final_answer,
            "is_retrieval": is_retrieval,
            "retrieval_used": retrieval_used,
            "rejected": rejected,
            "utility_score": utility["score"],
            "num_corrections": num_corrections,
            "reflection": reflection
        }

    def _generate_answer(
        self,
        query: str,
        context: str,
        is_retrieval: bool
    ) -> str:
        """生成回答

        Args:
            query: 用户问题
            context: 检索上下文
            is_retrieval: 是否使用了检索

        Returns:
            生成的回答
        """
        if is_retrieval and context:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的健身与营养顾问。基于检索到的信息回答用户问题。

要求：
1. 优先使用检索信息
2. 如果检索信息不足，可以结合自身知识
3. 标注信息来源
4. 保持回答准确、清晰、有条理

引用格式：[来源: 相关度分数]"""),
                ("human", "问题：{query}\n\n检索内容：\n{context}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的健身与营养顾问。直接回答用户问题。

要求：
1. 回答准确、清晰、有条理
2. 如果不确定，明确说明
3. 可以提供额外的相关建议"""),
                ("human", "问题：{query}")
            ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            return chain.invoke({
                "query": query,
                "context": context or "无"
            })
        except Exception as e:
            print(f"回答生成失败: {e}")
            return f"生成回答时出错: {str(e)}"

    def _generate_reflection(
        self,
        is_retrieval: bool,
        retrieval_used: List[Dict],
        rejected: List[Dict],
        utility: Dict[str, Any],
        num_corrections: int
    ) -> str:
        """生成反思标记

        Args:
            is_retrieval: 是否使用检索
            retrieval_used: 使用的高质量检索
            rejected: 过滤的低质量检索
            utility: 质量评分
            num_corrections: 纠正次数

        Returns:
            反思标记字符串
        """
        parts = []

        if is_retrieval:
            parts.append(f"[检索: Yes, 使用: {len(retrieval_used)}, 过滤: {len(rejected)}]")
        else:
            parts.append("[检索: No]")

        parts.append(f"[质量: {utility['score']}/4]")

        if num_corrections > 0:
            parts.append(f"[纠正: {num_corrections}次]")

        if rejected:
            parts.append(f"[过滤低相关: {len(rejected)}条]")

        return " ".join(parts)
