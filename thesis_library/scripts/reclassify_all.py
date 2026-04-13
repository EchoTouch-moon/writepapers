#!/usr/bin/env python3
"""Re-classify all papers with updated model (qwen3.5-plus)."""

import json
import logging
from pathlib import Path

from thesis_library.config import LibraryConfig, ChapterType
from thesis_library.core.chunker import Chunk, Chunker
from thesis_library.core.chapter_classifier import create_classifier
from thesis_library.core.smoother import sliding_window_smoothing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def reclassify_all():
    """Re-classify all paper chunks."""
    config = LibraryConfig()
    chunker = Chunker()

    # Load metadata
    metadata_path = Path(config.metadata_file)
    metadata = json.loads(metadata_path.read_text())

    classifier = create_classifier(config)

    for cite_key in metadata.keys():
        chunks_path = Path(config.papers_dir) / f"{cite_key}_chunks.json"

        if not chunks_path.exists():
            logger.warning(f"Chunks file not found: {chunks_path}")
            continue

        # Load existing chunks
        chunks_data = json.loads(chunks_path.read_text())
        chunks = [Chunk(**c) for c in chunks_data]

        # Check if already classified (not all OTHER)
        non_other_count = sum(1 for c in chunks if c.chapter_type != "OTHER")
        if non_other_count > len(chunks) * 0.5:
            logger.info(f"Skipping {cite_key}: already classified ({non_other_count}/{len(chunks)} non-OTHER)")
            continue

        logger.info(f"Re-classifying {cite_key} with {len(chunks)} chunks...")

        try:
            # Classify all chunks
            chapter_types = classifier.classify_all(chunks)
            smoothed_types = sliding_window_smoothing(chapter_types)

            # Update chunks
            for chunk, chapter_type in zip(chunks, smoothed_types):
                chunk.chapter_type = chapter_type.value

            # Save updated chunks
            chunker.save_chunks(chunks, str(chunks_path))

            # Log distribution
            type_dist = {}
            for c in chunks:
                type_dist[c.chapter_type] = type_dist.get(c.chapter_type, 0) + 1

            logger.info(f"Classified {cite_key}: {type_dist}")

        except Exception as e:
            logger.error(f"Classification failed for {cite_key}: {e}")
            # Continue with next paper

if __name__ == "__main__":
    reclassify_all()