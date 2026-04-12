# Phase C: Metadata Filtering Optimization Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement metadata filtering to achieve Multi_cond Recall@5 ≥ 0.80 and hit_ratio ≥ 0.80, fixing the structural constraint bug and enabling chapter-type-based retrieval.

**Architecture:** Two-phase approach - Phase 1 uses Faiss oversampling with Python post-filtering for immediate validation; Phase 2 migrates to ChromaDB for native metadata filtering.

**Tech Stack:** Qwen API for batch classification, sliding window smoothing, oversample_multiplier configuration.

---

## Problem Analysis

### Current State

| Metric | Value | Issue |
|--------|-------|-------|
| Multi_cond Recall@5 | 0.50 | Worst performing query type |
| Multi_cond MRR@5 | 0.50 | First match at position 2 |
| hit_ratio | 0.20-0.50 | High noise in results |
| `_apply_structural_constraint` | Bug | Retains all chunks (lines 175-177) |

### Root Causes

1. **Faiss IndexFlatIP limitation**: No native metadata filtering support
2. **Structural constraint bug**: `_apply_structural_constraint` never actually filters
3. **Metadata quality gap**: 40% chunks have section_title = "公式符号" (parse failure)
4. **No chapter_type field**: Only raw section_title, not standardized classification

---

## Design Decisions

### D1: Optimization Targets

**Primary:** Multi_cond Recall@5 ≥ 0.80
**Secondary:** hit_ratio ≥ 0.80 (noise reduction)

Rationale: Multi_cond is the query type that explicitly requires metadata filtering. Fixing this validates the filtering logic. High hit_ratio ensures LLM receives clean context.

---

### D2: Implementation Strategy

**Phase 1: Faiss Oversampling + Post-filtering**

```
Query → Faiss search(top_k × oversample_multiplier) → Python filter → Return top_k
```

- `oversample_multiplier: int = 10` (configurable in LibraryConfig)
- Filter by `chapter_type` in Python memory
- Minimal code changes, immediate validation
- Risk: If top 50 results all mismatch, return count may be < top_k

**Phase 2: ChromaDB Migration (future)**

- Replace Faiss Indexer with ChromaDB collection
- Native `where={"chapter_type": "METHODOLOGY"}` pre-filtering
- Delete Python post-filtering code (temporary patch removed)
- Not implemented in this phase

---

### D3: Metadata Classification Pipeline

**Architecture:**

```
opendataloader-pdf → Chunker → ChapterClassifier → Indexer
                              ↓
                         Qwen API Batch
                              ↓
                     SlidingWindowSmoothing
                              ↓
                  chunk_map.json (with chapter_type)
```

**Classification Approach: Option C (Batch Processing)**

Batch 5 chunks per API call to minimize rate limits.

**System Prompt (Optimized):**
```
你是学术论文章节分类器。阅读以下按顺序排列的文本块（Chunk 1 到 Chunk N）。
将每个文本块归类到以下类型之一：[ABSTRACT, INTRODUCTION, METHODOLOGY, EXPERIMENT, CONCLUSION, REFERENCE, OTHER]。
你必须且只能输出一个严格的 JSON 数组，数组长度必须与输入的 Chunk 数量一致。不要包含任何其他文字。

示例输出：
["ABSTRACT", "INTRODUCTION", "INTRODUCTION", "METHODOLOGY", "METHODOLOGY"]
```

**User Prompt:**
```
Chunk 1: {chunk_1_content[:300]}
Chunk 2: {chunk_2_content[:300]}
...
Chunk 5: {chunk_5_content[:300]}
```

**Cost Analysis:**
- 827 chunks / 5 per batch = 165 API calls
- ~330 tokens/chunk × 827 = ~274K tokens
- Qwen Coding Plan: ¥0 (free)

---

### D4: ChapterType Taxonomy

**Option A: English 7-category Enum (selected)**

```python
class ChapterType(Enum):
    ABSTRACT = "abstract"       # 摘要/概述
    INTRODUCTION = "introduction"   # 引言/背景/相关工作
    METHODOLOGY = "methodology"     # 方法/系统设计/模型架构
    EXPERIMENT = "experiment"       # 实验/评估/结果分析
    CONCLUSION = "conclusion"       # 结论/总结/未来展望
    REFERENCE = "reference"         # 参考文献
    OTHER = "other"                 # 附录/致谢/其他
```

**Rationale:**
- Compatible with existing SECTION_TYPE_MAPPING (config.py)
- Clear boundaries, LLM不易混淆
- Works for both Chinese and English papers

---

### D5: Sliding Window Smoothing

**Option A: Majority Voting (selected)**

```python
def sliding_window_smoothing(
    chapter_types: list[str],
    window_size: int = 3
) -> list[str]:
    """Fix isolated classification errors."""
    smoothed = chapter_types.copy()
    half_window = window_size // 2

    for i in range(len(chapter_types)):
        # Boundary protection
        if chapter_types[i] in ["ABSTRACT", "REFERENCE"]:
            continue

        # Get window
        start = max(0, i - half_window)
        end = min(len(chapter_types), i + half_window + 1)
        window = chapter_types[start:end]

        # Majority vote
        majority = max(set(window), key=window.count)
        if window.count(majority) > window.count(chapter_types[i]):
            smoothed[i] = majority

    return smoothed
```

**Boundary Protection:**
- `ABSTRACT`: Typically only 1-2 chunks, vulnerable to "majority tyranny"
- `REFERENCE`: Must never be changed to other types

**Window Size: 3**
- Conservative, preserves chapter boundaries
- Only fixes isolated errors

---

### D6: Code Architecture Changes

**Approach 1: Retriever Internal Modification (selected)**

Changes in `thesis_library/core/retriever.py`:

1. Add `oversample_multiplier` parameter
2. Modify `_semantic_search` to fetch top_k × multiplier
3. Add `chapter_type` filtering loop
4. Return filtered results[:top_k]

**Why Not Approach 2 (Independent Filter Module):**
- ChromaDB Phase 2 uses native pre-filtering
- Python filter module becomes dead code after migration
- Minimal changes now = easier migration later

---

## File Structure

```
thesis_library/
├── core/
│   ├── retriever.py        # Modify: oversampling + post-filtering
│   ├── indexer.py          # Modify: add chapter_type to chunk_map
│   ├── chapter_classifier.py  # Create: LLM batch classification
│   └── smoother.py         # Create: sliding window smoothing
├── config.py               # Modify: add ChapterType enum, oversample_multiplier
├── cli.py                  # Modify: add --chapter-type flag to search
└── evaluator/
    └── test_cases.py       # Modify: update TC-004 with chapter_type constraint
```

---

## Success Criteria

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Multi_cond Recall@5 | 0.50 | ≥ 0.80 | `thesis-library eval` |
| Multi_cond MRR@5 | 0.50 | ≥ 0.70 | `thesis-library eval` |
| hit_ratio (TC-004) | 0.33 | ≥ 0.80 | `thesis-library eval --verbose` |
| Classification accuracy | N/A | ≥ 85% | Manual sample check |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Qwen API rate limits | Batch 5 chunks, retry with exponential backoff |
| JSON parse failure | Mark batch as OTHER, let smoothing fix |
| Oversample exhausted | Increase multiplier to 20, warn user |
| Chapter boundary drift | window_size=3, boundary protection |

---

## Implementation Order

1. Add ChapterType enum to config.py
2. Create chapter_classifier.py (Qwen API batch call)
3. Create smoother.py (sliding window)
4. Modify retriever.py (oversampling + post-filtering)
5. Update chunk_map.json with chapter_type field
6. Re-run eval to validate metrics improvement
7. Add --chapter-type CLI flag

---

## Phase 2 Preview (ChromaDB Migration)

Not implemented now, but design keeps migration path clean:

```python
# ChromaDB usage (future)
collection.query(
    query_embeddings=[embedding],
    n_results=5,
    where={"chapter_type": "METHODOLOGY"}  # Native pre-filter
)

# Retriever._semantic_search will be simplified:
# - Remove oversample_multiplier
# - Remove Python post-filter loop
# - Just call collection.query with where clause
```