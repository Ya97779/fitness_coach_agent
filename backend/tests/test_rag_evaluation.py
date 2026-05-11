"""RAG 效果评估测试

使用 RAGAS 框架对 RAG 系统进行自动化评估。

评估指标：
- Faithfulness（忠实度）：回答是否忠于检索内容
- Answer Relevancy（回答相关性）：回答与问题的相关程度
- Context Precision（上下文精确度）：检索结果的排序质量
- Context Recall（上下文召回率）：正确信息被检索到的比例

使用方法：
    cd backend
    python -m tests.test_rag_evaluation                    # 运行完整评估
    python -m tests.test_rag_evaluation --quick            # 快速模式（仅 5 条）
    python -m tests.test_rag_evaluation --output report.json  # 输出报告到文件
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到 path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv()


class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self, llm=None, embeddings=None):
        """初始化评估器

        Args:
            llm: RAGAS 用的 LLM（LangChain ChatOpenAI 实例）
            embeddings: RAGAS 用的嵌入模型
        """
        self.llm = llm
        self.embeddings = embeddings
        self._rag_instance = None

    def _get_rag(self):
        """延迟初始化 RAG 实例"""
        if self._rag_instance is None:
            from app.rag import ModernRAG
            self._rag_instance = ModernRAG()
        return self._rag_instance

    def _get_ragas_llm(self):
        """获取 RAGAS 兼容的 LLM（通过 llm_factory）"""
        if self.llm is None:
            from openai import OpenAI
            from ragas.llms import llm_factory
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
            # RAGAS 评估用非推理模型，避免 reasoning_tokens 耗尽 token 预算
            eval_model = os.getenv("RAGAS_EVAL_MODEL", "glm-4-flash")
            print(f"RAGAS 评估模型: {eval_model}")
            self.llm = llm_factory(
                model=eval_model,
                client=client,
                max_tokens=4096
            )
        return self.llm

    def _get_ragas_embeddings(self):
        """获取 RAGAS 兼容的嵌入模型（LangChain wrapper）"""
        if self.embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import _LangchainEmbeddingsWrapper
            lc_embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "embedding-2"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
            self.embeddings = _LangchainEmbeddingsWrapper(lc_embeddings)
        return self.embeddings

    def load_dataset(self, path: str = None) -> List[Dict]:
        """加载评估数据集"""
        if path is None:
            path = Path(__file__).parent / "eval_dataset.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_rag_query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """对单个问题运行 RAG 检索 + 生成

        Returns:
            {"answer": str, "contexts": [str, ...]}
        """
        rag = self._get_rag()
        results = rag.search(question, top_k=top_k)
        contexts = [r["content"] for r in results]

        # 使用 RAG 的 query 方法获取带上下文的回答
        try:
            query_result = rag.query(question, top_k=top_k)
            answer = query_result.get("answer", "")
        except Exception:
            # fallback：用 search 结果手动拼接
            context_text = "\n\n".join(contexts)
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4.7"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=0.0
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的健身与营养顾问。基于检索信息回答用户问题。"),
                ("human", "问题：{query}\n\n检索内容：\n{context}")
            ])
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"query": question, "context": context_text})

        return {"answer": answer, "contexts": contexts}

    def build_ragas_dataset(self, test_cases: List[Dict]) -> Any:
        """构建 RAGAS 评估数据集

        对每个测试用例运行 RAG，收集 answer 和 contexts，
        然后构建 HuggingFace Dataset 格式的数据。
        """
        from datasets import Dataset

        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        total = len(test_cases)
        for i, case in enumerate(test_cases):
            print(f"  [{i+1}/{total}] 处理: {case['question'][:40]}...")
            result = self.run_rag_query(case["question"])

            questions.append(case["question"])
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            ground_truths.append(case["ground_truth"])

            # 避免 API 限流
            time.sleep(0.5)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })
        return dataset

    def evaluate(
        self,
        dataset: Any,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """运行 RAGAS 评估

        Args:
            dataset: RAGAS 格式的数据集
            metrics: 要评估的指标列表，默认全部

        Returns:
            评估结果字典
        """
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import _Faithfulness, _AnswerRelevancy, _ContextPrecision, _ContextRecall

        ragas_llm = self._get_ragas_llm()
        ragas_embeddings = self._get_ragas_embeddings()

        metric_map = {
            "faithfulness": _Faithfulness(),
            "answer_relevancy": _AnswerRelevancy(),
            "context_precision": _ContextPrecision(),
            "context_recall": _ContextRecall(),
        }

        if metrics is None:
            metrics = list(metric_map.keys())

        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

        print(f"\n开始 RAGAS 评估，指标: {', '.join(metrics)}")
        result = ragas_evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=True,
            raise_exceptions=False
        )
        return result

    def format_report(
        self,
        eval_result: Any,
        test_cases: List[Dict],
        elapsed: float
    ) -> Dict[str, Any]:
        """格式化评估报告"""
        # 获取各指标分数
        scores_dict = getattr(eval_result, '_scores_dict', {})
        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in scores_dict:
                values = scores_dict[metric_name]
                valid = [v for v in values if v is not None and v == v]  # filter NaN
                scores[metric_name] = sum(valid) / len(valid) if valid else 0

        # 综合分数：各指标平均值
        valid_scores = [v for v in scores.values() if v > 0]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(test_cases),
            "elapsed_seconds": round(elapsed, 1),
            "overall_score": round(overall, 4),
            "metric_scores": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in scores.items()
            },
            "per_query_scores": []
        }

        # 逐条结果
        df = eval_result.to_pandas() if hasattr(eval_result, 'to_pandas') else None
        for i, case in enumerate(test_cases):
            row = {
                "id": case["id"],
                "question": case["question"],
            }
            if df is not None:
                for metric_name in scores:
                    if metric_name in df.columns and i < len(df):
                        val = df[metric_name].iloc[i]
                        if val is not None and val == val:  # not NaN
                            row[metric_name] = round(float(val), 4)
            report["per_query_scores"].append(row)

        return report

    def print_report(self, report: Dict):
        """打印评估报告到控制台"""
        print("\n" + "=" * 60)
        print("RAG 效果评估报告")
        print("=" * 60)
        print(f"时间: {report['timestamp']}")
        print(f"测试用例数: {report['total_queries']}")
        print(f"耗时: {report['elapsed_seconds']}s")
        print(f"\n综合分数: {report['overall_score']}")
        print("\n各指标分数:")
        for name, score in report["metric_scores"].items():
            filled = int(score * 20)
            bar = "#" * filled + "-" * (20 - filled)
            print(f"  {name:<25} {bar} {score:.4f}")

        print("\n逐题详情:")
        print("-" * 60)
        for row in report["per_query_scores"]:
            q = row["question"][:35]
            metrics_str = " | ".join(
                f"{k}={v:.3f}" for k, v in row.items()
                if k not in ("id", "question") and isinstance(v, float)
            )
            print(f"  [{row['id']:>2}] {q:<35} {metrics_str}")

        print("\n" + "=" * 60)

    def run(
        self,
        dataset_path: str = None,
        quick: bool = False,
        output_path: str = None,
        metrics: List[str] = None
    ) -> Dict:
        """运行完整评估流程

        Args:
            dataset_path: 评估数据集路径
            quick: 快速模式（仅前 5 条）
            output_path: 报告输出路径
            metrics: 评估指标列表

        Returns:
            评估报告字典
        """
        print("加载评估数据集...")
        test_cases = self.load_dataset(dataset_path)
        if quick:
            test_cases = test_cases[:5]
            print(f"快速模式: 仅评估前 {len(test_cases)} 条")

        print(f"共 {len(test_cases)} 条测试用例\n")

        print("运行 RAG 检索和生成...")
        start_time = time.time()
        dataset = self.build_ragas_dataset(test_cases)

        eval_result = self.evaluate(dataset, metrics)
        elapsed = time.time() - start_time

        report = self.format_report(eval_result, test_cases, elapsed)

        # 先保存 JSON，再打印（避免打印崩溃导致报告丢失）
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存到: {output_path}")

        self.print_report(report)

        return report


def main():
    parser = argparse.ArgumentParser(description="RAG 效果评估")
    parser.add_argument("--dataset", type=str, help="评估数据集路径")
    parser.add_argument("--quick", action="store_true", help="快速模式（仅 5 条）")
    parser.add_argument("--output", type=str, help="报告输出路径")
    parser.add_argument("--metrics", type=str, nargs="+",
                        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                        help="指定评估指标")
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    evaluator.run(
        dataset_path=args.dataset,
        quick=args.quick,
        output_path=args.output,
        metrics=args.metrics
    )


if __name__ == "__main__":
    main()
