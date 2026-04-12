# RAG Evaluation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build automated RAG evaluation system with Recall@5 + MRR@5 metrics, baseline comparison, and interactive test case creation.

**Architecture:** Create `thesis_library/evaluator/` module with test_cases.py (data classes), metrics.py (indicator calculations), evaluator.py (core logic), report.py (Diff display), and extend cli.py with eval/eval-add commands.

**Tech Stack:** Python 3.11, dataclasses, argparse, existing thesis_library Library API

---

## File Structure

```
thesis_library/
├── evaluator/
│   ├── __init__.py           # Module exports
│   ├── test_cases.py         # TestCase, QueryType, TestCaseResult data classes
│   ├── metrics.py            # recall_at_k, mrr_at_k functions
│   ├── evaluator.py          # Evaluator class + EvalResult
│   ├── report.py             # generate_report, load/save baseline
│   └── cli_add.py            # run_eval_add interactive flow
│
├── cli.py                    # Extend: add cmd_eval, cmd_eval_add
│
thesis/library/eval/
├── test_cases.json           # Created by user via eval-add
├── baseline.json             # Created after first eval --save-baseline
└── last_run.json             # Auto-created after each eval
```

---

### Task 1: Create QueryType Enum and TestCase Data Class

**Files:**
- Create: `thesis_library/evaluator/test_cases.py`

- [ ] **Step 1: Write test_cases.py with QueryType enum and TestCase dataclass**

```python
"""Test case definitions for RAG evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Query type classification for test cases."""
    EXACT_TERM = "exact_term"        # Precise term retrieval
    FUZZY_CONCEPT = "fuzzy_concept"  # Fuzzy concept matching
    MULTI_CONDITION = "multi_cond"   # Multi-condition with chapter_type
    CROSS_PARAGRAPH = "cross_para"   # Cross-paragraph logic test


@dataclass
class TestCase:
    """Single test case for RAG evaluation.
    
    Attributes:
        id: Unique identifier (TC-001, TC-002...)
        query: User search query
        query_type: Classification of query type
        expected_chunk_ids: List of expected chunk IDs
        chapter_type: Chapter constraint (for multi_cond)
        threshold: Override similarity threshold
        notes: Human annotation notes
    """
    id: str
    query: str
    query_type: QueryType
    expected_chunk_ids: list[str]
    chapter_type: str | None = None
    threshold: float | None = None
    notes: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "query": self.query,
            "query_type": self.query_type.value,
            "expected_chunk_ids": self.expected_chunk_ids,
            "chapter_type": self.chapter_type,
            "threshold": self.threshold,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        """Create from dict loaded from JSON."""
        return cls(
            id=data["id"],
            query=data["query"],
            query_type=QueryType(data["query_type"]),
            expected_chunk_ids=data["expected_chunk_ids"],
            chapter_type=data.get("chapter_type"),
            threshold=data.get("threshold"),
            notes=data.get("notes"),
        )


@dataclass
class TestCaseResult:
    """Result of evaluating a single test case.
    
    Attributes:
        test_case: The original test case
        result_ids: Retrieved chunk IDs (from search)
        recall: Recall@K score
        mrr: MRR@K score
        hit: Whether any expected chunk was found
        hit_ratio: Noise metric (hits / len(result_ids))
        ceiling_warning: Whether expected > K (Recall ceiling)
    """
    test_case: TestCase
    result_ids: list[str]
    recall: float
    mrr: float
    hit: bool
    hit_ratio: float = 0.0
    ceiling_warning: bool = False
    
    def __post_init__(self) -> None:
        # Calculate hit_ratio if result_ids not empty
        if self.result_ids:
            hits = len(set(self.result_ids) & set(self.test_case.expected_chunk_ids))
            self.hit_ratio = hits / len(self.result_ids)


@dataclass
class TypeMetrics:
    """Aggregated metrics for a single QueryType."""
    recall: float
    mrr: float
    count: int


@dataclass
class EvalResult:
    """Complete evaluation result.
    
    Attributes:
        k: The K value used for Recall@K and MRR@K
        overall_recall: Aggregate Recall@K
        overall_mrr: Aggregate MRR@K
        by_type: Metrics grouped by QueryType
        case_results: Individual TestCaseResult list
        failed_cases: Cases where recall == 0
        total_cases: Number of test cases evaluated
    """
    k: int
    overall_recall: float
    overall_mrr: float
    by_type: dict[QueryType, TypeMetrics]
    case_results: list[TestCaseResult]
    failed_cases: list[TestCaseResult]
    total_cases: int
```

- [ ] **Step 2: Commit test_cases.py**

```bash
git add thesis_library/evaluator/test_cases.py
git commit -m "feat(evaluator): add TestCase, QueryType, TestCaseResult data classes"
```

---

### Task 2: Implement Metrics Functions (Recall@K and MRR@K)

**Files:**
- Create: `thesis_library/evaluator/metrics.py`

- [ ] **Step 1: Write metrics.py with recall_at_k and mrr_at_k functions**

```python
"""Metrics calculation for RAG evaluation."""


def recall_at_k(results: list[str], expected: list[str], k: int) -> float:
    """Calculate Recall@K.
    
    Recall = (hits in top K) / (total expected)
    
    Args:
        results: Retrieved chunk IDs (ordered by similarity)
        expected: Expected chunk IDs
        k: Truncation position
    
    Returns:
        Recall score (0.0 - 1.0)
    """
    if not expected:
        return 0.0
    
    top_k = results[:k]
    hits = len(set(top_k) & set(expected))
    return hits / len(expected)


def mrr_at_k(results: list[str], expected: list[str], k: int) -> float:
    """Calculate MRR@K (Mean Reciprocal Rank).
    
    MRR = 1 / (rank of first correct answer)
    
    Args:
        results: Retrieved chunk IDs (ordered by similarity)
        expected: Expected chunk IDs
        k: Truncation position
    
    Returns:
        MRR score (0.0 - 1.0)
        - 1.0 if first position hit
        - 0.5 if second position hit
        - 0.33 if third position hit
        - 0.0 if not found in top K
    """
    for i, chunk_id in enumerate(results[:k]):
        if chunk_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_metrics(results: list[float], weights: list[int] | None = None) -> float:
    """Calculate weighted average of metrics.
    
    Args:
        results: List of individual metric values
        weights: Optional weights (default: uniform)
    
    Returns:
        Weighted average
    """
    if not results:
        return 0.0
    
    if weights is None:
        return sum(results) / len(results)
    
    total_weight = sum(weights)
    weighted_sum = sum(r * w for r, w in zip(results, weights))
    return weighted_sum / total_weight if total_weight else 0.0
```

- [ ] **Step 2: Commit metrics.py**

```bash
git add thesis_library/evaluator/metrics.py
git commit -m "feat(evaluator): add recall_at_k and mrr_at_k metrics functions"
```

---

### Task 3: Implement Evaluator Core Logic

**Files:**
- Create: `thesis_library/evaluator/evaluator.py`

- [ ] **Step 1: Write evaluator.py with Evaluator class**

```python
"""Core evaluation logic for RAG system."""

import json
import logging
from pathlib import Path

from thesis_library import Library
from thesis_library.evaluator.test_cases import (
    EvalResult,
    TestCase,
    TestCaseResult,
    TypeMetrics,
)
from thesis_library.evaluator.metrics import recall_at_k, mrr_at_k, aggregate_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """RAG evaluation engine.
    
    Evaluates retrieval quality against golden test cases.
    """
    
    def __init__(
        self,
        library: Library,
        test_cases_path: str,
    ) -> None:
        self.library = library
        self.test_cases_path = test_cases_path
        self.test_cases: list[TestCase] = []
        self._load_test_cases()
    
    def _load_test_cases(self) -> None:
        """Load test cases from JSON file."""
        path = Path(self.test_cases_path)
        if not path.exists():
            logger.warning(f"Test cases file not found: {self.test_cases_path}")
            self.test_cases = []
            return
        
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        cases_data = data.get("test_cases", [])
        self.test_cases = [TestCase.from_dict(c) for c in cases_data]
        logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    def run(self, k: int = 5) -> EvalResult:
        """Run full evaluation.
        
        Args:
            k: Truncation position for Recall@K and MRR@K
        
        Returns:
            EvalResult with aggregated metrics
        """
        case_results: list[TestCaseResult] = []
        
        for tc in self.test_cases:
            # Execute search
            threshold = tc.threshold or self.library.config.similarity_threshold
            search_results = self.library.search(
                query=tc.query,
                chapter_type=tc.chapter_type,
                threshold=threshold,
                top_k=k,
            )
            
            # Extract chunk IDs
            result_ids = [r.chunk.id for r in search_results]
            
            # Calculate metrics
            recall = recall_at_k(result_ids, tc.expected_chunk_ids, k)
            mrr = mrr_at_k(result_ids, tc.expected_chunk_ids, k)
            
            # Check for ceiling warning
            ceiling_warning = len(tc.expected_chunk_ids) > k
            
            case_results.append(TestCaseResult(
                test_case=tc,
                result_ids=result_ids,
                recall=recall,
                mrr=mrr,
                hit=(recall > 0),
                ceiling_warning=ceiling_warning,
            ))
        
        return self._aggregate_results(case_results, k)
    
    def _aggregate_results(
        self,
        case_results: list[TestCaseResult],
        k: int,
    ) -> EvalResult:
        """Aggregate individual results into EvalResult."""
        if not case_results:
            return EvalResult(
                k=k,
                overall_recall=0.0,
                overall_mrr=0.0,
                by_type={},
                case_results=[],
                failed_cases=[],
                total_cases=0,
            )
        
        # Overall metrics (uniform weights)
        recalls = [cr.recall for cr in case_results]
        mrrs = [cr.mrr for cr in case_results]
        
        overall_recall = aggregate_metrics(recalls)
        overall_mrr = aggregate_metrics(mrrs)
        
        # Group by QueryType
        by_type: dict[TypeMetrics] = {}
        for qtype in TestCase.query_type.__class__.__members__.values():
            type_results = [cr for cr in case_results if cr.test_case.query_type == qtype]
            if type_results:
                type_recall = aggregate_metrics([cr.recall for cr in type_results])
                type_mrr = aggregate_metrics([cr.mrr for cr in type_results])
                by_type[qtype] = TypeMetrics(
                    recall=type_recall,
                    mrr=type_mrr,
                    count=len(type_results),
                )
        
        # Failed cases (recall == 0)
        failed_cases = [cr for cr in case_results if cr.recall == 0]
        
        return EvalResult(
            k=k,
            overall_recall=overall_recall,
            overall_mrr=overall_mrr,
            by_type=by_type,
            case_results=case_results,
            failed_cases=failed_cases,
            total_cases=len(case_results),
        )
```

- [ ] **Step 2: Fix TypeMetrics import issue (use QueryType directly)**

The `by_type` dict needs to use QueryType from test_cases.py. Edit evaluator.py:

```python
# Fix: Import QueryType from test_cases and use it in by_type dict
from thesis_library.evaluator.test_cases import (
    EvalResult,
    QueryType,  # Add this
    TestCase,
    TestCaseResult,
    TypeMetrics,
)

# In _aggregate_results, change the loop:
for qtype in QueryType:  # Simplified iteration over enum
    type_results = [cr for cr in case_results if cr.test_case.query_type == qtype]
```

- [ ] **Step 3: Commit evaluator.py**

```bash
git add thesis_library/evaluator/evaluator.py
git commit -m "feat(evaluator): add Evaluator class with run() and aggregation logic"
```

---

### Task 4: Implement Report Generation with Diff Display

**Files:**
- Create: `thesis_library/evaluator/report.py`

- [ ] **Step 1: Write report.py with generate_report and baseline functions**

```python
"""Report generation and baseline management."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from thesis_library.evaluator.test_cases import EvalResult, TestCaseResult

logger = logging.getLogger(__name__)


def load_baseline(baseline_path: str) -> dict | None:
    """Load baseline metrics from JSON file."""
    path = Path(baseline_path)
    if not path.exists():
        return None
    
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(
    result: EvalResult,
    baseline_path: str,
    library_config: dict,
) -> None:
    """Save current result as baseline."""
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get git commit if available
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown"
    
    # Build by_type dict
    by_type_dict = {}
    for qtype, metrics in result.by_type.items():
        by_type_dict[qtype.value] = {
            f"recall@{result.k}": metrics.recall,
            f"mrr@{result.k}": metrics.mrr,
        }
    
    baseline = {
        "overall": {
            f"recall@{result.k}": result.overall_recall,
            f"mrr@{result.k}": result.overall_mrr,
        },
        "by_type": by_type_dict,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config_snapshot": library_config,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    
    logger.info(f"Saved baseline to {baseline_path}")


def save_last_run(result: EvalResult, last_run_path: str) -> None:
    """Save last run result for comparison."""
    path = Path(last_run_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "overall": {
            f"recall@{result.k}": result.overall_recall,
            f"mrr@{result.k}": result.overall_mrr,
        },
        "k": result.k,
        "total_cases": result.total_cases,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def format_diff(current: float, baseline: float | None) -> str:
    """Format diff string with arrow indicator.
    
    Returns:
        - "+0.05 ▲" if improvement
        - "-0.02 ▼" if regression
        - "" if no change or no baseline
    """
    if baseline is None:
        return ""
    
    diff = current - baseline
    if abs(diff) < 0.001:  # No significant change
        return ""
    
    if diff > 0:
        return f" (+{diff:.2f}) ▲"
    else:
        return f" ({diff:.2f}) ▼"


def generate_report(
    result: EvalResult,
    baseline: dict | None = None,
    verbose: bool = False,
) -> str:
    """Generate evaluation report with Diff display.
    
    Args:
        result: Evaluation result
        baseline: Optional baseline dict for comparison
        verbose: Show detailed noise metrics
    
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 55)
    lines.append("RAG Evaluation Report")
    lines.append("=" * 55)
    lines.append(f"Test Cases: {result.total_cases}")
    lines.append("")
    
    # Overall metrics with Diff
    baseline_overall = baseline.get("overall") if baseline else None
    recall_base = baseline_overall.get(f"recall@{result.k}") if baseline_overall else None
    mrr_base = baseline_overall.get(f"mrr@{result.k}") if baseline_overall else None
    
    recall_diff = format_diff(result.overall_recall, recall_base)
    mrr_diff = format_diff(result.overall_mrr, mrr_base)
    
    lines.append(f"Overall Metrics (K={result.k}):")
    lines.append(f"  Recall@{result.k}: {result.overall_recall:.2f}{recall_diff}")
    lines.append(f"  MRR@{result.k}:    {result.overall_mrr:.2f}{mrr_diff}")
    lines.append("")
    
    # By QueryType
    lines.append("By Query Type:")
    for qtype, metrics in result.by_type.items():
        lines.append(
            f"  {qtype.value:15} Recall@{result.k}: {metrics.recall:.2f}  "
            f"MRR@{result.k}: {metrics.mrr:.2f}  ({metrics.count} cases)"
        )
    lines.append("")
    
    # Failed cases with ceiling warning
    if result.failed_cases:
        lines.append("Failed Cases:")
        for cr in result.failed_cases:
            warning = " ⚠️" if cr.ceiling_warning else ""
            lines.append(f"  {cr.test_case.id}: \"{cr.test_case.query}\"{warning}")
            lines.append(f"    Expected: {cr.test_case.expected_chunk_ids}")
            lines.append(f"    Got: {cr.result_ids if cr.result_ids else '(no matches)'}")
        lines.append("")
    
    # Ceiling warnings (expected > K but still hit)
    ceiling_cases = [
        cr for cr in result.case_results
        if cr.ceiling_warning and cr.hit and cr not in result.failed_cases
    ]
    if ceiling_cases:
        lines.append("Ceiling Warnings (expected > K):")
        for cr in ceiling_cases:
            lines.append(
                f"  {cr.test_case.id}: expected={len(cr.test_case.expected_chunk_ids)} "
                f"> K={result.k}, recall capped at {cr.recall:.2f}"
            )
        lines.append("")
    
    # Verbose: show noise metrics
    if verbose:
        lines.append("Noise Metrics (verbose):")
        for cr in result.case_results:
            if cr.hit_ratio < 1.0 and cr.hit:
                lines.append(
                    f"  {cr.test_case.id}: hit_ratio={cr.hit_ratio:.2f} "
                    f"(noise in results)"
                )
    
    lines.append("=" * 55)
    return "\n".join(lines)
```

- [ ] **Step 2: Commit report.py**

```bash
git add thesis_library/evaluator/report.py
git commit -m "feat(evaluator): add report generation with Diff display and baseline management"
```

---

### Task 5: Create evaluator Module __init__.py

**Files:**
- Create: `thesis_library/evaluator/__init__.py`

- [ ] **Step 1: Write __init__.py with module exports**

```python
"""RAG evaluation module for thesis_library."""

from thesis_library.evaluator.test_cases import (
    EvalResult,
    QueryType,
    TestCase,
    TestCaseResult,
    TypeMetrics,
)
from thesis_library.evaluator.metrics import (
    aggregate_metrics,
    mrr_at_k,
    recall_at_k,
)
from thesis_library.evaluator.evaluator import Evaluator
from thesis_library.evaluator.report import (
    generate_report,
    load_baseline,
    save_baseline,
    save_last_run,
)

__all__ = [
    "EvalResult",
    "Evaluator",
    "QueryType",
    "TestCase",
    "TestCaseResult",
    "TypeMetrics",
    "aggregate_metrics",
    "generate_report",
    "load_baseline",
    "mrr_at_k",
    "recall_at_k",
    "save_baseline",
    "save_last_run",
]
```

- [ ] **Step 2: Commit __init__.py**

```bash
git add thesis_library/evaluator/__init__.py
git commit -m "feat(evaluator): add module exports in __init__.py"
```

---

### Task 6: Implement eval-add Interactive CLI

**Files:**
- Create: `thesis_library/evaluator/cli_add.py`

- [ ] **Step 1: Write cli_add.py with interactive test case creation**

```python
"""Interactive CLI for adding test cases."""

import json
import logging
from pathlib import Path

from thesis_library import Library
from thesis_library.evaluator.test_cases import QueryType, TestCase

logger = logging.getLogger(__name__)


def run_eval_add(
    library: Library,
    test_cases_path: str,
) -> int:
    """Interactive test case creation.
    
    Flow:
    1. User enters query
    2. System searches and shows top results
    3. User selects expected chunks (or types 'manual' for fallback)
    4. User specifies query_type, chapter_type, notes
    5. Save to test_cases.json
    
    Args:
        library: Library instance
        test_cases_path: Path to test_cases.json
    
    Returns:
        0 on success, 1 on failure
    """
    # Load existing test cases to determine next ID
    path = Path(test_cases_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_cases: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        existing_cases = data.get("test_cases", [])
    
    next_id = f"TC-{len(existing_cases) + 1:03d}"
    
    print("\n=== Add Test Case ===")
    print(f"ID: {next_id}")
    
    # Step 1: Enter query
    query = input("Query: ").strip()
    if not query:
        print("Query required. Aborted.")
        return 1
    
    # Step 2: Search with default params
    print("\nSearching...")
    results = library.search(query, top_k=5)
    
    if results:
        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            content_preview = r.chunk.content[:80].replace("\n", " ")
            print(f"  {i}. [{r.chunk.id}] \"{content_preview}...\"")
    else:
        print("No results found.")
    
    # Step 3: Select expected chunks
    print("\nSelect expected chunks:")
    print("  - Enter numbers (comma-separated): 1,2,3")
    print("  - Or type 'manual' to enter chunk IDs directly")
    print("  - Or type 'search' to search with larger top_k")
    
    selection = input("Selection: ").strip().lower()
    
    expected_chunk_ids: list[str] = []
    
    if selection == "manual":
        # Fallback: manual chunk ID entry
        manual_ids = input("Enter chunk IDs (comma-separated): ").strip()
        expected_chunk_ids = [id.strip() for id in manual_ids.split(",") if id.strip()]
    
    elif selection == "search":
        # Expand search
        top_k = input("Search with larger top_k (default 20): ").strip()
        top_k = int(top_k) if top_k else 20
        results = library.search(query, top_k=top_k)
        
        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            content_preview = r.chunk.content[:60].replace("\n", " ")
            print(f"  {i}. [{r.chunk.id}] \"{content_preview}...\"")
        
        selection = input("Selection (numbers): ").strip()
        indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
        expected_chunk_ids = [results[i - 1].chunk.id for i in indices if 0 < i <= len(results)]
    
    elif selection:
        # Parse selection as numbers
        indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
        expected_chunk_ids = [results[i - 1].chunk.id for i in indices if 0 < i <= len(results)]
    
    if not expected_chunk_ids:
        print("No expected chunks selected. Aborted.")
        return 1
    
    print(f"\nExpected chunks: {expected_chunk_ids}")
    
    # Step 4: Query type
    print("\nQuery type:")
    print("  1. exact_term")
    print("  2. fuzzy_concept")
    print("  3. multi_cond")
    print("  4. cross_para")
    
    type_selection = input("Select (1-4): ").strip()
    query_type_map = {
        "1": QueryType.EXACT_TERM,
        "2": QueryType.FUZZY_CONCEPT,
        "3": QueryType.MULTI_CONDITION,
        "4": QueryType.CROSS_PARAGRAPH,
    }
    query_type = query_type_map.get(type_selection, QueryType.EXACT_TERM)
    
    # Step 5: Chapter type (optional)
    chapter_type = input("Chapter type constraint (optional, press Enter to skip): ").strip()
    chapter_type = chapter_type if chapter_type else None
    
    # Step 6: Notes (optional)
    notes = input("Notes (optional): ").strip()
    notes = notes if notes else None
    
    # Step 7: Threshold override (optional)
    threshold_input = input(
        f"Threshold override (current: {library.config.similarity_threshold}, press Enter to keep): "
    ).strip()
    threshold = float(threshold_input) if threshold_input else None
    
    # Create test case
    test_case = TestCase(
        id=next_id,
        query=query,
        query_type=query_type,
        expected_chunk_ids=expected_chunk_ids,
        chapter_type=chapter_type,
        threshold=threshold,
        notes=notes,
    )
    
    # Save to file
    existing_cases.append(test_case.to_dict())
    
    data = {
        "test_cases": existing_cases,
        "metadata": {
            "created": existing_cases[0].get("id", "unknown") if existing_cases else "unknown",
            "last_updated": test_case.id,
            "total_cases": len(existing_cases),
        },
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved as {next_id}")
    return 0
```

- [ ] **Step 2: Commit cli_add.py**

```bash
git add thesis_library/evaluator/cli_add.py
git commit -m "feat(evaluator): add interactive eval-add CLI with manual fallback"
```

---

### Task 7: Extend cli.py with eval and eval-add Commands

**Files:**
- Modify: `thesis_library/cli.py`

- [ ] **Step 1: Add import for evaluator module in cli.py**

Add at line 10 after existing imports:

```python
from thesis_library.evaluator import Evaluator, generate_report, load_baseline, save_baseline, save_last_run
from thesis_library.evaluator.cli_add import run_eval_add
```

- [ ] **Step 2: Add cmd_eval function (lines after cmd_terms)**

```python
def cmd_eval(args: argparse.Namespace) -> int:
    """Run RAG evaluation."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)
    
    eval_dir = Path(args.library_dir) / "eval"
    test_cases_path = str(eval_dir / "test_cases.json")
    baseline_path = str(eval_dir / "baseline.json")
    last_run_path = str(eval_dir / "last_run.json")
    
    # Check test cases exist
    if not Path(test_cases_path).exists():
        print(f"No test cases found at {test_cases_path}")
        print("Run 'thesis-library eval-add' to create test cases first.")
        return 1
    
    # Check index is built
    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1
    
    # Run evaluation
    evaluator = Evaluator(library, test_cases_path)
    result = evaluator.run(k=args.k)
    
    # Load baseline for comparison
    baseline = load_baseline(baseline_path) if Path(baseline_path).exists() else None
    
    # Generate report
    report = generate_report(result, baseline, verbose=args.verbose)
    print(report)
    
    # Save last run
    save_last_run(result, last_run_path)
    
    # Save baseline if requested
    if args.save_baseline:
        library_config = {
            "embedding_model": config.embedding_model,
            "chunk_size": config.max_chunk_size,
            "threshold": config.similarity_threshold,
        }
        save_baseline(result, baseline_path, library_config)
        print(f"\nBaseline saved to {baseline_path}")
    
    return 0


def cmd_eval_add(args: argparse.Namespace) -> int:
    """Add test case interactively."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)
    
    # Check index is built
    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1
    
    eval_dir = Path(args.library_dir) / "eval"
    test_cases_path = str(eval_dir / "test_cases.json")
    
    return run_eval_add(library, test_cases_path)
```

- [ ] **Step 3: Add subparser definitions in main() function**

Add after the `terms_parser` definition (around line 308):

```python
    # eval command
    eval_parser = subparsers.add_parser("eval", help="Run RAG evaluation")
    eval_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value for Recall@K and MRR@K",
    )
    eval_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save result as new baseline",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed noise metrics",
    )
    
    # eval-add command
    eval_add_parser = subparsers.add_parser("eval-add", help="Add test case interactively")
```

- [ ] **Step 4: Add command routing in main() function**

Add after `elif args.command == "terms":` (around line 327):

```python
    elif args.command == "eval":
        return cmd_eval(args)
    elif args.command == "eval-add":
        return cmd_eval_add(args)
```

- [ ] **Step 5: Commit cli.py changes**

```bash
git add thesis_library/cli.py
git commit -m "feat(cli): add eval and eval-add commands with baseline comparison"
```

---

### Task 8: Verify Full System Integration

**Files:**
- Verify: `thesis_library/evaluator/` module
- Verify: `thesis_library/cli.py`

- [ ] **Step 1: Test module import**

```bash
uv run python -c "from thesis_library.evaluator import Evaluator, QueryType, TestCase; print('Evaluator module OK')"
```

Expected: `Evaluator module OK`

- [ ] **Step 2: Test CLI help**

```bash
uv run thesis-library eval --help
uv run thesis-library eval-add --help
```

Expected: Help text showing `--k`, `--save-baseline`, `--verbose` options

- [ ] **Step 3: Verify eval-add flow (manual mode)**

```bash
# This is a manual test - run in interactive terminal
uv run thesis-library eval-add
# Enter query, type 'manual', enter chunk IDs
```

- [ ] **Step 4: Commit verification**

```bash
git add -A
git commit -m "test(evaluator): verify module import and CLI integration"
```

---

## Spec Coverage Check

| Spec Section | Task Coverage |
|--------------|---------------|
| 3.1 TestCase definition | Task 1: test_cases.py |
| 4.1 Recall@K | Task 2: metrics.py |
| 4.2 MRR@K | Task 2: metrics.py |
| 5.1 Evaluator class | Task 3: evaluator.py |
| 5.2 Report + Diff | Task 4: report.py |
| 6.1 eval command | Task 7: cli.py extension |
| 6.2 eval-add command | Task 6 + Task 7 |
| 10.1 Recall ceiling | Task 1: TestCaseResult.ceiling_warning + Task 4: report |
| 10.2 Manual fallback | Task 6: cli_add.py 'manual' option |
| 10.3 Noise metrics | Task 1: hit_ratio + Task 4: verbose option |

---

## Placeholder Scan Results

| Pattern | Status |
|---------|--------|
| TBD/TODO | ✅ None found |
| "implement later" | ✅ None found |
| "add appropriate error handling" | ✅ None found |
| "similar to Task N" | ✅ None found |
| Missing code in steps | ✅ All steps have complete code |

---

## Type Consistency Check

| Type/Function | Definition | Usage | Status |
|---------------|------------|-------|--------|
| TestCase | Task 1 test_cases.py | evaluator.py, cli_add.py | ✅ Consistent |
| TestCaseResult | Task 1 test_cases.py | evaluator.py, report.py | ✅ Consistent |
| EvalResult | Task 1 test_cases.py | evaluator.py, report.py | ✅ Consistent |
| QueryType | Task 1 test_cases.py | evaluator.py, cli_add.py | ✅ Consistent |
| recall_at_k | Task 2 metrics.py | evaluator.py | ✅ Consistent |
| mrr_at_k | Task 2 metrics.py | evaluator.py | ✅ Consistent |
| Evaluator.run() | Task 3 evaluator.py | cli.py | ✅ Returns EvalResult |