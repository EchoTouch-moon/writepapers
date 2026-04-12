"""Report generation and baseline management."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from thesis_library.evaluator.test_cases import EvalResult

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