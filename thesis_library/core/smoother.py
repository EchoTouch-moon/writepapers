"""Sliding window smoothing for chapter type classification."""

import logging
from thesis_library.config import ChapterType

logger = logging.getLogger(__name__)


def sliding_window_smoothing(
    chapter_types: list[ChapterType],
    window_size: int = 3,
) -> list[ChapterType]:
    """Fix isolated classification errors using majority voting.

    Args:
        chapter_types: List of classified chapter types
        window_size: Size of sliding window (default 3)

    Returns:
        Smoothed list of chapter types
    """
    if len(chapter_types) <= window_size:
        return chapter_types  # Too short to smooth

    smoothed = chapter_types.copy()
    half_window = window_size // 2

    # Boundary protection - these types must never be changed
    protected_types = {ChapterType.ABSTRACT, ChapterType.REFERENCE}

    for i in range(len(chapter_types)):
        current_type = chapter_types[i]

        # Skip protected types
        if current_type in protected_types:
            continue

        # Get window bounds
        start = max(0, i - half_window)
        end = min(len(chapter_types), i + half_window + 1)
        window = chapter_types[start:end]

        # Count occurrences in window
        type_counts = {}
        for t in window:
            type_counts[t] = type_counts.get(t, 0) + 1

        # Find majority
        majority_type = max(type_counts, key=type_counts.get)
        majority_count = type_counts[majority_type]
        current_count = type_counts[current_type]

        # Only change if majority is strictly greater
        if majority_count > current_count:
            smoothed[i] = majority_type
            logger.debug(
                f"Smoothed position {i}: {current_type.value} → {majority_type.value}"
            )

    # Count changes
    changes = sum(1 for i in range(len(chapter_types)) if smoothed[i] != chapter_types[i])
    if changes > 0:
        logger.info(f"Sliding window smoothed {changes}/{len(chapter_types)} chunks")

    return smoothed