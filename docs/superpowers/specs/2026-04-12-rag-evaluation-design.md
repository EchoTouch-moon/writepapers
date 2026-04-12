# thesis_library RAG 评估系统设计文档

> 设计日期: 2026-04-12
> 项目路径: /Users/v/new-idea/writepapers
> 演进顺序: D → C → A → B（本文档为 Phase D）

---

## 1. 设计背景

### 1.1 问题陈述

当前 `thesis_library` 系统已完成基础检索功能验证，但缺乏量化评估手段。每次修改 chunker、indexer 或 retriever 后，无法确认是否发生性能退化。

### 1.2 设计目标

建立自动化 RAG 评估流程：
- **建立基准线**：10 个黄金测试案例，覆盖四种查询场景
- **量化指标**：Recall@5 + MRR@5 双指标组合
- **防止退化**：baseline.json 记录历史基线，每次运行显示 Diff
- **精准定位**：按 QueryType 分类统计，快速识别薄弱环节

---

## 2. 系统架构

```
thesis_library/
├── evaluator/
│   ├── __init__.py           # 模块入口
│   ├── evaluator.py          # 评估器核心逻辑
│   ├── test_cases.py         # TestCase 数据类定义
│   ├── metrics.py            # Recall@K, MRR@K 计算
│   └── report.py             # 报告生成 + Diff 显示
│   └── cli_add.py            # eval-add 交互式录入
│
├── cli.py                    # 扩展：eval 命令
│
thesis/library/
└── eval/
    ├── test_cases.json       # 黄金测试集（人工构造）
    ├── baseline.json         # 历史最佳基线
    └── last_run.json         # 最近一次运行结果
```

---

## 3. 数据结构设计

### 3.1 TestCase 定义

```python
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """查询类型分类"""
    EXACT_TERM = "exact_term"        # 精准术语检索
    FUZZY_CONCEPT = "fuzzy_concept"  # 概念模糊匹配
    MULTI_CONDITION = "multi_cond"   # 多条件约束（带 chapter_type）
    CROSS_PARAGRAPH = "cross_para"   # 跨段落逻辑（测试分块质量）

@dataclass
class TestCase:
    """单个测试案例"""
    id: str                          # TC-001, TC-002...
    query: str                       # 用户输入的查询
    query_type: QueryType            # 查询类型
    expected_chunk_ids: list[str]    # 期望返回的 chunk_id 列表
    chapter_type: str | None         # 章节约束（multi_cond 类型必填）
    threshold: float | None = None   # 相似度阈值（可选覆盖默认值）
    notes: str | None = None         # 构造说明（人工标注原因）
```

### 3.2 JSON 存储格式

```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "query": "大语言模型",
      "query_type": "exact_term",
      "expected_chunk_ids": ["Paper5462_para1"],
      "chapter_type": null,
      "threshold": null,
      "notes": "精准术语，期望返回定义段落"
    },
    {
      "id": "TC-002",
      "query": "如何解决对话上下文遗忘的问题",
      "query_type": "fuzzy_concept",
      "expected_chunk_ids": ["Paper4721_para23", "Paper4721_para24"],
      "chapter_type": null,
      "threshold": null,
      "notes": "原文无'遗忘'词，测试 Embedding 泛化能力"
    }
  ],
  "metadata": {
    "created": "2026-04-12",
    "last_updated": "2026-04-12",
    "total_cases": 10
  }
}
```

---

## 4. 核心指标实现

### 4.1 Recall@K

```python
def recall_at_k(results: list[str], expected: list[str], k: int) -> float:
    """计算 Recall@K
    
    Args:
        results: 检索返回的 chunk_id 列表
        expected: 期望的 chunk_id 列表
        k: 截断位置
    
    Returns:
       召回率（0.0 - 1.0）
    
    Logic:
        前 K 结果中包含正确答案的比例
        Recall = hits / len(expected)
    """
    top_k = results[:k]
    hits = len(set(top_k) & set(expected))
    return hits / len(expected) if expected else 0.0
```

### 4.2 MRR@K

```python
def mrr_at_k(results: list[str], expected: list[str], k: int) -> float:
    """计算 MRR@K (Mean Reciprocal Rank)
    
    Args:
        results: 检索返回的 chunk_id 列表
        expected: 期望的 chunk_id 列表
        k: 截断位置
    
    Returns:
        第一个正确答案排名的倒数（0.0 - 1.0）
    
    Logic:
        - 第1位命中: 1.0
        - 第2位命中: 0.5
        - 第3位命中: 0.33
        - 未命中: 0.0
    
    Purpose:
        对抗"迷失在中间"效应，确保正确答案排在前面
    """
    for i, chunk_id in enumerate(results[:k]):
        if chunk_id in expected:
            return 1.0 / (i + 1)
    return 0.0
```

---

## 5. 评估流程

### 5.1 Evaluator 核心逻辑

```python
class Evaluator:
    """RAG 评估器"""
    
    def __init__(self, library: Library, test_cases_path: str):
        self.library = library
        self.test_cases = self._load_test_cases(test_cases_path)
    
    def run(self, k: int = 5) -> EvalResult:
        """运行完整评估
        
        Returns:
            EvalResult 包含：
            - overall_recall: 总体 Recall@K
            - overall_mrr: 总体 MRR@K
            - by_type: 按 QueryType 分组统计
            - failed_cases: 失败案例详情
        """
        results = []
        
        for tc in self.test_cases:
            # 执行检索
            search_results = self.library.search(
                query=tc.query,
                chapter_type=tc.chapter_type,
                threshold=tc.threshold or self.library.config.similarity_threshold,
                top_k=k
            )
            
            # 提取 chunk_ids
            result_ids = [r.chunk.id for r in search_results]
            
            # 计算指标
            recall = recall_at_k(result_ids, tc.expected_chunk_ids, k)
            mrr = mrr_at_k(result_ids, tc.expected_chunk_ids, k)
            
            results.append(TestCaseResult(
                test_case=tc,
                result_ids=result_ids,
                recall=recall,
                mrr=mrr,
                hit=(recall > 0)
            ))
        
        return self._aggregate(results)
```

### 5.2 报告生成 + Diff 显示

```python
def generate_report(result: EvalResult, baseline_path: str | None) -> str:
    """生成评估报告，包含与基线的 Diff
    
    Diff 显示规则：
        - 与 baseline.json 对比
        - +0.05 ▲ 绿色：改进
        - -0.02 ▼ 红色：退化
        - 无变化：不显示箭头
    """
    baseline = load_baseline(baseline_path) if baseline_path else None
    
    report = []
    report.append("=" * 50)
    report.append("RAG Evaluation Report")
    report.append("=" * 50)
    
    # Overall Metrics with Diff
    recall_diff = _compute_diff(result.overall_recall, baseline?.overall_recall)
    mrr_diff = _compute_diff(result.overall_mrr, baseline?.overall_mrr)
    
    report.append(f"Overall Metrics (K={result.k}):")
    report.append(f"  Recall@{result.k}: {result.overall_recall:.2f} {recall_diff}")
    report.append(f"  MRR@{result.k}:    {result.overall_mrr:.2f} {mrr_diff}")
    
    # By QueryType
    report.append("\nBy Query Type:")
    for qtype, metrics in result.by_type.items():
        report.append(f"  {qtype.value:15} Recall@{result.k}: {metrics.recall:.2f}  MRR@{result.k}: {metrics.mrr:.2f}")
    
    # Failed Cases
    if result.failed_cases:
        report.append("\nFailed Cases:")
        for fc in result.failed_cases:
            report.append(f"  {fc.test_case.id}: \"{fc.test_case.query}\"")
            report.append(f"    Expected: {fc.test_case.expected_chunk_ids}")
            report.append(f"    Got: {fc.result_ids}")
    
    return "\n".join(report)
```

---

## 6. CLI 命令

### 6.1 eval 命令

```bash
uv run thesis-library eval [--k 5] [--save-baseline]

# 示例输出：
# =====================================================
# RAG Evaluation Report
# =====================================================
# Test Cases: 10
# 
# Overall Metrics (K=5):
#   Recall@5: 0.73 (+0.05) ▲
#   MRR@5:    0.58 (-0.02) ▼
#
# By Query Type:
#   exact_term     Recall@5: 1.00  MRR@5: 0.95
#   fuzzy_concept  Recall@5: 0.60  MRR@5: 0.40
#   multi_cond     Recall@5: 0.80  MRR@5: 0.65
#   cross_para     Recall@5: 0.50  MRR@5: 0.25
#
# Failed Cases:
#   TC-007: "长程依赖问题解决方案"
#     Expected: ["Paper4721_para23"]
#     Got: [] (no matches)
# =====================================================
```

### 6.2 eval-add 命令

```bash
uv run thesis-library eval-add

# 交互式录入流程：
# Query: 大语言模型的参数量对比
# Searching...
# Found 5 results:
#   1. [Paper4721_para15] "LLaMA-7B 参数量约 7B..."
#   2. [Paper5462_para8] "GPT-3 参数量..."
# 
# Select expected chunks (comma-separated): 1,2
# Chapter type constraint (optional, press Enter to skip): 系统设计
# Query type [exact_term/fuzzy_concept/multi_cond/cross_para]: multi_cond
# Notes (optional): 用户想找参数对比表格
# 
# ✓ Saved as TC-003
```

---

## 7. 测试案例构造指南

### 7.1 四种场景覆盖建议

| QueryType | 示例 Query | 期望结果特征 | 数量建议 |
|-----------|------------|--------------|----------|
| exact_term | "大语言模型" | 包含术语定义的段落 | 2-3 个 |
| fuzzy_concept | "如何解决上下文遗忘" | 原文用"长程依赖"等不同表述 | 2-3 个 |
| multi_cond | "评价指标" + chapter_type="实验" | 仅实验章节的指标段落 | 2-3 个 |
| cross_para | "公式推导过程" | 连续多段落，测试分块是否切断 | 2 个 |

### 7.2 测试案例来源

- **人工构造**：基于真实写作场景的查询意图
- **Bad Case 录入**：实际使用中搜不到的情况，随时用 `eval-add` 补充
- **迭代生长**：从 10 个开始，随论文写作自然扩展

---

## 8. Baseline 管理

### 8.1 baseline.json 结构

```json
{
  "overall": {
    "recall@5": 0.73,
    "mrr@5": 0.58
  },
  "by_type": {
    "exact_term": {"recall@5": 1.00, "mrr@5": 0.95},
    "fuzzy_concept": {"recall@5": 0.60, "mrr@5": 0.40},
    "multi_cond": {"recall@5": 0.80, "mrr@5": 0.65},
    "cross_para": {"recall@5": 0.50, "mrr@5": 0.25}
  },
  "timestamp": "2026-04-12T20:10:00",
  "git_commit": "abc123",
  "config_snapshot": {
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "chunk_size": 500,
    "threshold": 0.7
  }
}
```

### 8.2 Baseline 更新策略

- **手动更新**：`--save-baseline` 标志显式保存
- **自动记录**：每次 eval 运行后写入 `last_run.json`
- **对比提示**：若新结果优于 baseline，提示用户是否更新

---

## 9. 后续演进路径

本设计为架构演进的第一阶段（Phase D）。后续阶段：

| Phase | 目标 | 前置依赖 |
|-------|------|----------|
| **C** | Chunking 策略改进（AST/段落级分块） | 评估系统验证改进效果 |
| **A** | Embedding 模型升级（bge-m3） | 评估系统对比模型性能 |
| **B** | 向量库替换（ChromaDB/LanceDB） | 大规模验证后迁移 |

---

## 10. Edge Cases 处理

### 10.1 Recall 数学天花板

**问题**：当 TestCase 的 `len(expected_chunk_ids) > K` 时，即使完美召回前 K 位，Recall 也不会达到 1.0。

**示例**：
- cross_para 类型，公式推导横跨 8 个 Chunks
- K=5 时，即使前 5 位全对，Recall = 5/8 = 0.625

**应对**：
```python
def generate_report(result: EvalResult) -> str:
    # 预警标记
    for tc in result.test_cases:
        if len(tc.expected_chunk_ids) > result.k:
            report.append(f"  {tc.id}: ⚠️ expected={len(tc.expected_chunk_ids)} > K={result.k}")
```

在 Failed Cases 输出中标记 `⚠️`，明确是截断瓶颈而非检索问题。

### 10.2 eval-add 冷启动悖论

**问题**：当系统检索能力差时，正确段落不在 top-K 结果中，无法通过选择数字完成录入。

**应对**：
```bash
Select expected chunks (comma-separated, or type 'manual' to input IDs directly): manual
Enter target chunk IDs: Paper4721_para23, Paper4721_para24

# 或扩大检索范围
Select expected chunks (comma-separated, or type 'search' to expand): search
Search with larger top_k (20): ...
```

CLI 流程增加 `manual` 和 `search` 后备选项。

### 10.3 噪声监控（Precision 底层埋点）

**问题**：召回结果中包含大量无关 Chunk，占用 Token 并诱发 LLM 幻觉。

**应对**：
```python
@dataclass
class TestCaseResult:
    # 主指标
    recall: float
    mrr: float
    
    # 底层埋点（不显示在主报告，用于排查）
    hit_ratio: float = 0.0  # hits / len(result_ids)，噪声指标
    noise_count: int = 0    # 无关 Chunk 数量
```

未来若 LLM 写作质量下降，可调出此指标排查。

---

## 11. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 测试案例主观偏差 | 评估结果不反映真实场景 | Bad Case 持续录入，自然生长 |
| 评估过于频繁 | 测试集膨胀导致运行慢 | 限制核心集 ≤ 30 个，额外案例单独管理 |
| Baseline 过时 | 误判改进为退化 | 每次重大架构变更后重置 baseline |
| expected > K 截断 | Recall 天花板误判 | ⚠️ 标记预警，区分检索问题与截断问题 |
| eval-add 检索死角 | 无法录入正确答案 | `manual` 后备选项，强制输入 ID |
| 噪声 Chunk 过多 | LLM 幻觉风险 | 底层埋点 hit_ratio，随时可调出排查 |

---

## 12. 实施清单

### 12.1 核心模块（P0）

1. 创建 `thesis_library/evaluator/` 模块目录
2. 实现 `test_cases.py` TestCase 数据类（含 QueryType 枚举）
3. 实现 `metrics.py` Recall@K + MRR@K 计算
4. 实现 `evaluator.py` 核心评估逻辑（含 Recall 天花板检测）
5. 实现 `report.py` 报告生成 + Diff 显示 + ⚠️ 预警标记
6. 实现 `cli_add.py` eval-add 交互式录入（含 `manual` 后备选项）
7. 扩展 `cli.py` 添加 eval 命令

### 12.2 测试数据（P1）

8. 构造初始 10 个黄金测试案例（覆盖四种 QueryType）
9. 运行首次评估，建立 baseline.json
10. 验证 Diff 显示功能
11. 验证 ⚠️ 预警标记（构造一个 expected > K 的测试案例）

### 12.3 噪声埋点（P2）

12. 在 TestCaseResult 中添加 hit_ratio 底层埋点
13. 添加 `eval --verbose` 选项显示详细噪声指标