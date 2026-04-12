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