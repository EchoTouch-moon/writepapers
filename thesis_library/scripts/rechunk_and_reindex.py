#!/usr/bin/env python3
"""Re-chunk all papers with fixed chunker and rebuild index."""

import json
import logging
from pathlib import Path

from thesis_library.config import LibraryConfig
from thesis_library.core.chunker import Chunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rechunk_all():
    """Re-chunk all papers with fixed chunker."""
    config = LibraryConfig()
    chunker = Chunker(config.max_chunk_size, config.min_chunk_size)

    # Load metadata
    metadata_path = Path(config.metadata_file)
    metadata = json.loads(metadata_path.read_text())

    total_before = 0
    total_after = 0

    for cite_key in metadata.keys():
        json_path = Path(config.papers_dir) / f"{cite_key}.json"
        chunks_path = Path(config.papers_dir) / f"{cite_key}_chunks.json"

        if not json_path.exists():
            logger.warning(f"JSON not found: {json_path}")
            continue

        # Load parsed PDF data
        data = json.loads(json_path.read_text())
        before = len(data)

        # Re-chunk with fixed logic
        chunks = chunker.chunk_paper(data, cite_key)
        after = len(chunks)

        # Save updated chunks
        chunker.save_chunks(chunks, str(chunks_path))

        total_before += before
        total_after += after

        logger.info(f"{cite_key}: {before} elements -> {after} chunks")

    logger.info(f"TOTAL: {total_before} elements -> {total_after} chunks (reduction: {(total_before - total_after) / total_before * 100:.1f}%)")


def rebuild_index():
    """Rebuild Faiss index from updated chunks."""
    config = LibraryConfig()
    chunker = Chunker()

    # Load all chunks
    all_chunks = []
    metadata = json.loads(Path(config.metadata_file).read_text())

    for cite_key in metadata.keys():
        chunks_path = Path(config.papers_dir) / f"{cite_key}_chunks.json"
        if chunks_path.exists():
            chunks = chunker.load_chunks(str(chunks_path))
            all_chunks.extend(chunks)

    logger.info(f"Loaded {len(all_chunks)} chunks total")

    # Build index
    from thesis_library.core.indexer import Indexer
    indexer = Indexer(
        model_name=config.embedding_model,
        index_type=config.index_type,
        min_term_freq=config.min_term_freq,
        max_term_length=config.max_term_length,
    )

    indexer.add_chunks(all_chunks)
    indexer.save_index(config.index_dir)

    # Build term index
    term_index = indexer.build_term_index(all_chunks)

    # Save term index
    indexer.save_term_index(term_index, config.index_dir)

    logger.info(f"Index built with {indexer.index.ntotal} vectors, {len(term_index)} terms")


if __name__ == "__main__":
    logger.info("Step 1: Re-chunking all papers...")
    rechunk_all()

    logger.info("Step 2: Rebuilding index...")
    rebuild_index()

    logger.info("Done!")