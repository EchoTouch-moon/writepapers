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