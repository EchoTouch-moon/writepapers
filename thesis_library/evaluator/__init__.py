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