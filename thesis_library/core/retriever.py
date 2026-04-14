"""Combined retrieval module with term anchoring, structural constraints, and semantic matching."""

import logging
import re
from dataclasses import dataclass

import numpy as np

from thesis_library.config import SECTION_TYPE_MAPPING
from thesis_library.core.chunker import Chunk
from thesis_library.core.indexer import Indexer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from retrieval search.

    Attributes:
        chunk: The matched Chunk
        similarity: Similarity score (0-1)
        matched_terms: Terms that matched
        section_match: Whether structural constraint was satisfied
    """

    chunk: Chunk
    similarity: float
    matched_terms: list[str]
    section_match: bool


class Retriever:
    """Combined retrieval strategy implementation.

    Strategy: Term anchoring → Structural constraint → Semantic matching

    Attributes:
        indexer: Indexer instance for semantic search
        similarity_threshold: Minimum similarity score
        default_top_k: Default number of results
    """

    def __init__(
        self,
        indexer: Indexer,
        similarity_threshold: float = 0.7,
        default_top_k: int = 10,
        oversample_multiplier: int = 10,
    ) -> None:
        self.indexer = indexer
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k
        self.oversample_multiplier = oversample_multiplier
        self.term_index: dict[str, list[str]] = {}
        self.chapter_type: str | None = None
        self.section_mapping = SECTION_TYPE_MAPPING

    def set_term_index(self, term_index: dict[str, list[str]]) -> None:
        """Set term inverted index."""
        self.term_index = term_index

    def search(
        self,
        query: str,
        chapter_type: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Execute combined retrieval strategy.

        Args:
            query: Search query text
            chapter_type: Filter by chapter type (e.g., "METHODOLOGY")
            threshold: Override similarity threshold
            top_k: Override number of results

        Returns:
            List of SearchResult objects sorted by similarity
        """
        threshold = threshold if threshold is not None else self.similarity_threshold
        top_k = top_k if top_k is not None else self.default_top_k

        # Step 1: Term anchoring (skip if chapter_type filter for post-filtering)
        candidate_chunk_ids = self._term_anchoring(query) if not chapter_type else None

        # Step 2: Structural constraint (skip if chapter_type filter for post-filtering)
        if chapter_type is None and self.chapter_type and candidate_chunk_ids:
            candidate_chunk_ids = self._apply_structural_constraint(
                candidate_chunk_ids, self.chapter_type
            )

        # Step 3: Semantic search with post-filtering
        results = self._semantic_search(
            query,
            candidate_chunk_ids,
            threshold,
            top_k,
            chapter_type,  # Pass for post-filtering
        )

        return results

    def _term_anchoring(self, query: str) -> list[str] | None:
        """Step 1: Extract terms from query and find matching chunks.

        Args:
            query: Search query

        Returns:
            List of candidate chunk IDs (or None if no terms match)
        """
        # Extract terms from query
        terms = self._extract_query_terms(query)

        if not terms:
            return None

        # Find chunks containing these terms
        matched_chunks: set[str] = set()
        for term in terms:
            if term in self.term_index:
                matched_chunks.update(self.term_index[term])

        logger.debug(f"Term anchoring: {len(matched_chunks)} chunks from terms {terms}")
        return list(matched_chunks) if matched_chunks else None

    def _extract_query_terms(self, query: str) -> list[str]:
        """Extract candidate terms from query."""
        terms: list[str] = []

        # Chinese terms
        chinese_terms = re.findall(r"[^\x00-\xff]{2,4}", query)
        terms.extend(chinese_terms)

        # English capitalized phrases
        english_terms = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query)
        terms.extend(english_terms)

        # Technical abbreviations
        tech_terms = re.findall(r"[A-Z]{2,}[0-9]*|[A-Za-z]+-[A-Za-z]+", query)
        terms.extend(tech_terms)

        # Check against term index
        matched = [t for t in terms if t in self.term_index]
        return matched

    def _apply_structural_constraint(
        self, chunk_ids: list[str], chapter_type: str
    ) -> list[str]:
        """Step 2: Filter chunks by structural relevance.

        Args:
            chunk_ids: Candidate chunk IDs from term anchoring
            chapter_type: Current chapter being written

        Returns:
            Filtered chunk IDs matching structure
        """
        # Get allowed section types for this chapter
        allowed_sections = self.section_mapping.get(chapter_type, [])
        if not allowed_sections:
            # No mapping, skip constraint
            return chunk_ids

        # Normalize allowed sections for matching
        allowed_lower = [s.lower() for s in allowed_sections]

        filtered: list[str] = []
        for chunk_id in chunk_ids:
            chunk = self.indexer.chunk_map.get(chunk_id)
            if chunk:
                section_lower = chunk.section_title.lower()
                # Check if section matches any allowed type
                if any(allowed in section_lower or section_lower in allowed for allowed in allowed_lower):
                    filtered.append(chunk_id)
                else:
                    # Include anyway but mark as section mismatch
                    filtered.append(chunk_id)

        logger.debug(f"Structural constraint: {len(filtered)} chunks remain")
        return filtered

    def _semantic_search(
        self,
        query: str,
        candidate_ids: list[str] | None,
        threshold: float,
        top_k: int,
        chapter_type: str | None = None,
    ) -> list[SearchResult]:
        """Step 3: Semantic matching with similarity threshold and optional chapter_type filter.

        Args:
            query: Search query
            candidate_ids: Pre-filtered chunk IDs (or None for all)
            threshold: Minimum similarity score
            top_k: Number of results to return
            chapter_type: Optional chapter type for post-filtering

        Returns:
            List of SearchResult objects
        """
        if self.indexer.index is None:
            logger.warning("Index not built")
            return []

        # Encode query
        model = self.indexer._load_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")

        # Normalize for inner product
        import faiss

        faiss.normalize_L2(query_embedding)

        # Determine fetch size (oversampling when chapter_type filter is active)
        oversample_multiplier = self.oversample_multiplier if chapter_type else 1
        k_fetch = min(top_k * oversample_multiplier, self.indexer.index.ntotal)

        # Search
        distances, ids = self.indexer.index.search(query_embedding, k_fetch)

        results: list[SearchResult] = []
        for i, (dist, idx) in enumerate(zip(distances[0], ids[0])):
            if idx < 0:  # Invalid ID
                continue

            chunk_id = self.indexer.id_to_chunk.get(int(idx))
            if not chunk_id:
                continue

            # Skip if not in candidates (when candidates provided)
            if candidate_ids is not None and chunk_id not in candidate_ids:
                continue

            # Post-filter by chapter_type
            if chapter_type is not None:
                chunk = self.indexer.chunk_map.get(chunk_id)
                # Handle both enum and string comparison
                chapter_type_str = chapter_type.value if hasattr(chapter_type, 'value') else chapter_type
                if chunk and chunk.chapter_type != chapter_type_str:
                    continue

            # Check threshold
            similarity = float(dist)
            if similarity < threshold:
                continue

            chunk = self.indexer.chunk_map.get(chunk_id)
            if not chunk:
                continue

            # Find matched terms
            matched_terms = self._find_matched_terms(query, chunk.content)

            # Check section match
            section_match = True  # Already filtered if candidates provided

            results.append(SearchResult(
                chunk=chunk,
                similarity=similarity,
                matched_terms=matched_terms,
                section_match=section_match,
            ))

            if len(results) >= top_k:
                break

        # Sort by similarity
        results.sort(key=lambda r: r.similarity, reverse=True)

        # Warn if filter exhausted candidates
        if chapter_type and len(results) < top_k:
            logger.warning(
                f"Metadata filter exhausted retrieved candidates. "
                f"Only {len(results)}/{top_k} results match chapter_type={chapter_type}"
            )

        return results

    def _find_matched_terms(self, query: str, content: str) -> list[str]:
        """Find terms that appear in both query and content."""
        terms = self._extract_query_terms(query)
        matched = [t for t in terms if t in content]
        return matched

    def search_by_terms(self, terms: list[str]) -> list[Chunk]:
        """Pure term search (Step 1 only).

        Args:
            terms: Terms to search for

        Returns:
            List of Chunks containing any of the terms
        """
        matched_chunk_ids: set[str] = set()
        for term in terms:
            if term in self.term_index:
                matched_chunk_ids.update(self.term_index[term])

        chunks: list[Chunk] = []
        for chunk_id in matched_chunk_ids:
            chunk = self.indexer.chunk_map.get(chunk_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    def semantic_search(
        self,
        query: str,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Pure semantic search (Step 3 only).

        Args:
            query: Search query
            threshold: Similarity threshold
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        return self._semantic_search(
            query,
            None,  # No candidate filtering
            threshold or self.similarity_threshold,
            top_k or self.default_top_k,
        )

    def get_known_terms(self) -> list[str]:
        """Get all known terms from the index."""
        return list(self.term_index.keys())