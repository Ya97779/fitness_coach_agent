# RAG 系统优化建议

## 概述

本文档记录 RAG 系统中需要改进和优化的地方，基于对 `backend/app/rag` 模块的深度分析。

---

## 一、文档处理流程分析

### 1.1 当前处理流程

| 阶段 | 模块 | 功能 |
|------|------|------|
| 1 | DocumentLoader | 支持 PDF/DOCX/TXT/HTML/图片，带重试机制 |
| 2 | AdvancedDocumentProcessor | 语义分割、表格提取、结构分析 |
| 3 | IntelligentSplitter | 500 字符块 / 50 重叠分割 |
| 4 | TextPreprocessor | 去噪、标准化、去重 |
| 5 | OpenAIEmbeddings | 向量化 |
| 6 | HybridSearch | 向量 + BM25 混合检索 |

### 1.2 优点

- 模块化设计良好，各职责分离清晰
- 支持多种高级特性：HyDE、Self-RAG、CoT、Query Expansion
- 增量索引机制（`check_and_update_index()`）
- 文档加载有重试装饰器

---

## 二、需要改进的地方

### 2.1 高优先级改进

#### ① 中文分词优化

**位置**：`backend/app/rag/modules/bm25.py:40`

**问题**：当前使用简单字符级分词，无法处理中文词语边界

```python
# 当前代码
tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
```

**建议**：使用 jieba 分词

```python
import jieba

def _tokenize(self, text: str) -> List[str]:
    text = text.lower()
    tokens = list(jieba.cut(text))
    return [t for t in tokens if t.strip()]
```

**预期效果**：BM25 检索质量提升 20-30%

---

#### ② 动态 Chunk Size

**位置**：`backend/app/rag/modules/splitter.py:20`

**问题**：固定 500 字符无法适应健身/营养文档的长度差异

**建议**：

```python
# 根据内容类型动态调整 chunk_size
def get_adaptive_chunk_size(self, text: str, metadata: dict = None) -> int:
    source = metadata.get("source", "") if metadata else ""

    if "表格" in text or "营养成分" in text:
        return 800  # 表格内容需要更大块
    elif len(text) < 200:
        return 300  # 短内容减小块
    elif len(text) > 2000:
        return 600  # 长内容增大块
    return 500  # 默认
```

---

#### ③ 检索结果相关性阈值过滤

**位置**：`backend/app/rag/modules/hybrid_search.py:search()`

**问题**：当前返回所有结果，未过滤低相关性内容

**建议**：

```python
def search(
    self,
    query: str,
    top_k: int = 5,
    min_score: float = 0.3  # 新增参数
) -> List[dict]:
    # ... 现有检索逻辑 ...

    # 过滤低相关性结果
    results = [r for r in results if r["score"] >= min_score]
    return results[:top_k]
```

---

### 2.2 中优先级改进

#### ④ RRF 融合参数调优

**位置**：`backend/app/rag/modules/hybrid_search.py:30`

**问题**：`rrf_k=60` 为经验值，未针对本项目调优

**建议**：通过评估集实验确定最优值，范围可试 [40, 60, 80]

---

#### ⑤ HyDE 生成内容置信度过滤

**位置**：`backend/app/rag/modules/hyde.py:generate()`

**问题**：假设性文档可能包含幻觉

**建议**：

```python
def generate(self, query: str) -> Optional[str]:
    content = self._generate_impl(query)

    # 简单置信度检查
    if content and len(content) > 50:
        # 检查是否包含明确的不确定性表达
        uncertainty_phrases = ["不确定", "可能", "或许", "大概"]
        if not any(phrase in content for phrase in uncertainty_phrases):
            return content
    return None
```

---

#### ⑥ 多语言混合检索优化

**问题**：健身领域专业词汇多为中英混合（如 BMI、蛋白质等）

**建议**：在分词时保留英文术语的完整性

```python
def _tokenize(self, text: str) -> List[str]:
    text = text.lower()
    # 保留英文术语
    tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
    # 补充提取纯英文词组
    english_terms = re.findall(r'[a-z]+(?:\s+[a-z]+)*', text, re.I)
    return list(set(tokens + english_terms))
```

---

## 三、效果评估体系

### 3.1 核心评价指标

| 指标类别 | 指标名称 | 说明 |
|----------|----------|------|
| 检索指标 | Precision@K | 检索结果中相关文档的比例 |
| 检索指标 | Recall@K | 正确答案被召回的比例 |
| 检索指标 | MRR | 第一个相关结果排名的倒数 |
| 检索指标 | NDCG@K | 考虑排名权重的召回率 |
| 生成指标 | Faithfulness | 生成回答对检索内容的忠实度 |
| 生成指标 | Answer Relevance | 回答与问题的相关程度 |
| 生成指标 | Context Precision | 上下文排序质量 |

### 3.2 评估方法

#### 方法一：人工评估（Ground Truth）

准备测试集：

```python
test_queries = [
    {
        "query": "如何科学增肌？",
        "relevant_docs": ["增肌原理", "蛋白质摄入", "力量训练"]
    },
    {
        "query": "上斜卧推的正确姿势",
        "relevant_topics": ["卧推", "角度", "发力", "肩部"]
    }
]
```

#### 方法二：LLM 自动评估

```python
EVAL_PROMPT = """评估检索结果与查询的相关性，输出 1-5 分。
1 = 完全不相关，5 = 高度相关。只输出分数。

查询：{query}
检索结果：{content}
"""
```

#### 方法三：RAGAS 框架（推荐）

```python
from ragas import EvaluateDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
```

---

## 四、测试用例建议

### 4.1 新增测试文件

建议在 `backend/tests/` 下创建 `test_rag_effectiveness.py`：

```python
class TestRAGEffectiveness:
    """RAG 效果评估测试"""

    def test_retrieval_recall(self):
        """测试检索召回率"""
        test_cases = [
            {
                "query": "上斜卧推的正确姿势",
                "expected_topics": ["卧推", "角度", "发力", "肩部"]
            }
        ]
        # 验证检索内容覆盖预期主题

    def test_retrieval_diversity(self):
        """测试检索多样性"""
        results = rag.search("健身计划", top_k=10)
        # 验证内容重复度 < 30%

    def test_context_precision(self):
        """测试上下文排序精确度"""
        # 验证第一个结果得分明显高于第二个
```

### 4.2 评估流程

```
1. 数据准备
   ├── 准备测试集（query + expected_answer）
   └── 可选：人工标注 relevant_docs

2. 检索评估
   ├── Precision@K / Recall@K / MRR / NDCG
   └── 检索结果内容覆盖度检查

3. 生成评估
   ├── Faithfulness（忠实度）
   ├── Answer Relevance（回答相关性）
   └── Context Utilization（上下文利用率）

4. 端到端评估
   └── 使用 RAGAS 框架一次性评估多个指标
```

---

## 五、改进优先级总结

| 优先级 | 改进项 | 预期收益 | 难度 |
|--------|--------|----------|------|
| 高 | 中文分词优化（jieba） | 检索召回率 +20-30% | 低 |
| 高 | 添加检索相关性阈值 | 减少低质量结果 | 低 |
| 高 | 构建评估测试集 | 量化优化效果 | 中 |
| 中 | 动态 Chunk Size | 更好处理不同类型内容 | 中 |
| 中 | RRF 参数调优 | 提升混合检索效果 | 中 |
| 低 | HyDE 置信度过滤 | 减少幻觉 | 低 |

---

## 六、快速生效的改动

如需快速提升效果，建议按以下顺序修改：

1. **立即生效**：在 `bm25.py` 中添加 jieba 分词
2. **立即生效**：在 `hybrid_search.py` 中添加 `min_score` 过滤
3. **1 天内**：准备 20 条测试 query 用于评估
4. **1 周内**：完成效果对比测试并调优参数
