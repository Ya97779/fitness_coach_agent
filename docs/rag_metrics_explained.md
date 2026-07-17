# RAGAS 评测指标详解

## 数据集字段的实际用途

| 字段 | 实际被谁用 | 作用 |
|------|-----------|------|
| `question` | 全部 4 个指标 | 原始用户查询 |
| `answer` | Faithfulness + Answer Relevancy | RAG 系统生成的回答（由 `run_rag_query` 动态产生） |
| `contexts` | Faithfulness + Context Precision + Context Recall | RAG 检索到的文档片段列表（由 `rag.search` 动态产生） |
| `ground_truth` | Context Recall（主要） | 标准参考答案 |
| `expected_topics` | **无人使用** | 死数据，代码中从未引用 |

注意：`answer` 和 `contexts` 不在 `eval_dataset.json` 文件里，它们是在评测运行时由 `run_rag_query()` 动态生成后填入 HuggingFace Dataset 的。

## RAGAS 的评判方式

每个指标内部都是 **LLM 当裁判**，不是规则匹配。RAGAS 拿到 dataset 后，对每条记录用 `llm`（glm-4-flash）做声明拆解和评分，用 `embeddings`（embedding-2）计算语义相似度，最后返回各指标分数。项目代码本身不参与评分逻辑，只负责喂数据和收结果。

### RAGAS 调用位置

集中在 `backend/tests/test_rag_evaluation.py`：

| 行号 | import | 用途 |
|------|--------|------|
| 59 | `ragas.llms.llm_factory` | 把 OpenAI 客户端包装成 RAGAS 兼容的 LLM |
| 78 | `ragas.embeddings._LangchainEmbeddingsWrapper` | 把 LangChain Embedding 包装成 RAGAS 兼容格式 |
| 178-179 | `ragas.evaluate` + 4 个 `ragas.metrics._*` | 核心评估函数 + 4 个指标类 |

核心调用：

```python
ragas_evaluate(
    dataset=dataset,        # HuggingFace Dataset，含 question/answer/contexts/ground_truth
    metrics=[_Faithfulness(), _AnswerRelevancy(), _ContextPrecision(), _ContextRecall()],
    llm=ragas_llm,          # glm-4-flash，RAGAS 用它做评判（拆解声明、打分等）
    embeddings=ragas_embeddings,  # embedding-2，计算余弦相似度用
    show_progress=True,
    raise_exceptions=False
)
```

---

## 四个指标的评判方式

### 1. Faithfulness（忠实度）

**使用字段：** `answer` + `contexts`

**回答的问题：** 回答有没有瞎编？

**评判过程：**

```
LLM 把 answer 拆成独立声明：
  声明1: "蛋白质推荐摄入量为每公斤体重0.8-1.0克"  ← 能从 contexts 推导？ ✓
  声明2: "蛋白质过量会伤肾"                       ← 能从 contexts 推导？ ✗（幻觉）

得分 = 1/2 = 0.5
```

**公式：** `Faithfulness = |supported claims| / |total claims|`

**依赖 ground_truth：** 否。只看回答是否忠于检索内容，不看标准答案。

---

### 2. Answer Relevancy（回答相关性）

**使用字段：** `question` + `answer`

**回答的问题：** 回答跑题了没？

**评判过程：**

```
原始问题: "蛋白质的每日推荐摄入量是多少？"

LLM 从 answer 反推可能的问题：
  Q1: "每天应该吃多少蛋白质？"   ← 与原始问题余弦相似度 0.92
  Q2: "蛋白质有什么营养价值？"   ← 与原始问题余弦相似度 0.45
  Q3: "哪些食物含蛋白质？"       ← 与原始问题余弦相似度 0.38

得分 = mean(0.92, 0.45, 0.38) = 0.58
```

**公式：** `Answer Relevancy = mean(cos_sim(q_gen, q_orig))`

**依赖 ground_truth：** 否。用 embedding 算语义相似度，完全基于 question 和 answer 的关系。

---

### 3. Context Precision（上下文精确率）

**使用字段：** `question` + `contexts` + `ground_truth`

**回答的问题：** 好文档排前面了没？

**评判过程：**

```
问题: "深蹲的常见错误有哪些？"
检索到 5 个 contexts：
  排名1: "深蹲时膝盖不应内扣..."          ← LLM判定：相关 ✓
  排名2: "卧推是最常见的胸部训练动作..."   ← LLM判定：不相关 ✗
  排名3: "深蹲时脚跟容易离地..."          ← LLM判定：相关 ✓
  排名4: "硬拉需要保持背部挺直..."        ← LLM判定：不相关 ✗
  排名5: "深蹲下蹲深度应低于膝盖..."      ← LLM判定：相关 ✓
```

加权精确率公式（相关文档排得越靠前，权重越高）：

```
Precision@1 = 1/1 = 1.0   (第1个相关)
Precision@3 = 2/3 = 0.67  (前3个里2个相关)
Precision@5 = 3/5 = 0.6   (前5个里3个相关)

得分 = (1.0×1 + 0.67×1 + 0.6×1) / 3个相关文档 = 0.756
```

**公式：** `Context Precision = Σ(precision@k × rel_k) / |relevant|`

**依赖 ground_truth：** 辅助参考。`ground_truth` 帮助 LLM 判断相关性，但核心评判对象是「question 和 context 的关系」。

**关于健身场景：** Context Precision 衡量的是**检索排序质量**，不是答案正确性。它回答的问题是：「相关的文档是不是排在前面？」。LLM 只需要判断每个 context 是否和 question 相关，不需要标准答案。

---

### 4. Context Recall（上下文召回率）

**使用字段：** `ground_truth` + `contexts`

**回答的问题：** 该查到的都查到了没？

**评判过程：**

```
ground_truth: "深蹲常见错误包括：膝盖内扣、脚跟离地、腰部过度弯曲"

LLM 拆成声明：
  声明1: "膝盖内扣是深蹲常见错误"     ← contexts 里有？ ✓
  声明2: "脚跟离地是深蹲常见错误"     ← contexts 里有？ ✓
  声明3: "腰部过度弯曲是深蹲常见错误" ← contexts 里有？ ✗

得分 = 2/3 = 0.67
```

**公式：** `Context Recall = |recovered claims| / |gt claims|`

**依赖 ground_truth：** **是**。这是唯一强依赖 `ground_truth` 的指标，要求检索结果覆盖标准答案的所有要点。

---

## 总结

| 指标 | 回答了什么问题 | 依赖 ground_truth？ | 评分维度 |
|------|--------------|-------------------|---------|
| Faithfulness | 回答有没有瞎编 | 否 | 生成质量 |
| Answer Relevancy | 回答跑题了没 | 否 | 生成质量 |
| Context Precision | 好文档排前面了没 | 辅助参考 | 检索质量 |
| Context Recall | 该查到的都查到了没 | **是** | 检索质量 |

- **Faithfulness + Answer Relevancy** 衡量生成质量，完全不依赖标准答案，是通用评估维度
- **Context Precision** 衡量检索排序质量，`ground_truth` 仅辅助判断相关性
- **Context Recall** 衡量检索覆盖度，强依赖标准答案，对健身场景最「苛刻」
