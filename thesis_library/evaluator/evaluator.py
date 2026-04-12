"""Core evaluation logic for RAG system."""

import json
import logging
from pathlib import Path

from thesis_library import Library
from thesis_library.evaluator.test_cases import (
    EvalResult,
    QueryType,
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
        by_type: dict[QueryType, TypeMetrics] = {}
        for qtype in QueryType:
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