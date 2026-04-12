"""Vector index and term index building module."""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from thesis_library.core.chunker import Chunk

logger = logging.getLogger(__name__)


class IndexBuildError(Exception):
    """Index building failed."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Index build error: {reason}")


class Indexer:
    """Build vector index and term inverted index.

    Attributes:
        model_name: Sentence-transformers model for embeddings
        index_type: Faiss index type
        min_term_freq: Minimum frequency to consider a term
        max_term_length: Maximum term length
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        index_type: str = "FlatIP",
        min_term_freq: int = 2,
        max_term_length: int = 10,
    ) -> None:
        self.model_name = model_name
        self.index_type = index_type
        self.min_term_freq = min_term_freq
        self.max_term_length = max_term_length

        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunk_map: dict[str, Chunk] = {}
        self.id_to_chunk: dict[int, str] = {}  # Faiss ID → chunk_id

    def _load_model(self) -> SentenceTransformer:
        """Lazy load embedding model.

        Raises:
            IndexBuildError: If model loading fails
        """
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise IndexBuildError(
                    f"Failed to load model '{self.model_name}': {e}"
                ) from e
        return self.model

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build vector index from chunks.

        Args:
            chunks: List of Chunk objects

        Raises:
            IndexBuildError: If index building fails
        """
        if not chunks:
            raise IndexBuildError("No chunks to index")

        model = self._load_model()

        # Build chunk map
        self.chunk_map = {chunk.id: chunk for chunk in chunks}

        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(contents)} chunks")

        try:
            embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype("float32")

            # Normalize for inner product similarity
            faiss.normalize_L2(embeddings)

            # Build Faiss index
            n = len(chunks)
            ids = np.arange(n, dtype=np.int64)

            if self.index_type == "FlatIP":
                base_index = faiss.IndexFlatIP(embeddings.shape[1])
            else:
                # Fallback to L2 distance
                base_index = faiss.IndexFlatL2(embeddings.shape[1])

            # For ID mapping, use IndexIDMap
            self.index = faiss.IndexIDMap(base_index)
            self.index.add_with_ids(embeddings, ids)

            # Build ID mapping
            for i, chunk in enumerate(chunks):
                self.id_to_chunk[i] = chunk.id

            logger.info(f"Built index with {self.index.ntotal} vectors")

        except Exception as e:
            raise IndexBuildError(str(e)) from e

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks to existing index.

        Args:
            chunks: New chunks to add
        """
        if not chunks:
            return

        if self.index is None:
            # No existing index, build new one
            self.build_index(chunks)
            return

        model = self._load_model()

        # Update chunk map
        for chunk in chunks:
            self.chunk_map[chunk.id] = chunk

        # Generate embeddings for new chunks
        contents = [chunk.content for chunk in chunks]
        embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Get new IDs starting from current count
        start_id = len(self.id_to_chunk)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        # Add to index
        self.index.add_with_ids(embeddings, ids)

        # Update ID mapping
        for i, chunk in enumerate(chunks):
            self.id_to_chunk[start_id + i] = chunk.id

        logger.info(f"Added {len(chunks)} chunks, total: {self.index.ntotal}")

    def build_term_index(self, chunks: list[Chunk]) -> dict[str, list[str]]:
        """Build term inverted index from chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Dictionary mapping terms to chunk IDs
        """
        term_counts: dict[str, int] = defaultdict(int)
        term_chunks: dict[str, list[str]] = defaultdict(list)

        for chunk in chunks:
            # Extract terms from content
            terms = self._extract_terms(chunk.content)

            for term in terms:
                term_counts[term] += 1
                term_chunks[term].append(chunk.id)

        # Filter by minimum frequency
        filtered_term_index = {
            term: chunk_ids
            for term, chunk_ids in term_chunks.items()
            if term_counts[term] >= self.min_term_freq
        }

        logger.info(f"Built term index with {len(filtered_term_index)} terms")
        return filtered_term_index

    def _extract_terms(self, content: str) -> list[str]:
        """Extract candidate terms from content.

        Strategy:
            - Chinese: Extract 2-4 character sequences
            - English: Extract capitalized phrases and noun patterns
        """
        terms: list[str] = []

        # Chinese terms (2-4 character sequences that aren't common words)
        chinese_pattern = re.findall(r"[^\x00-\xff]{2,4}", content)
        for term in chinese_pattern:
            if len(term) <= self.max_term_length:
                terms.append(term)

        # English terms (capitalized phrases)
        english_pattern = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", content)
        for term in english_pattern:
            if len(term) <= self.max_term_length:
                terms.append(term)

        # Technical terms with numbers/letters
        tech_pattern = re.findall(r"[A-Z]{2,}[0-9]*|[A-Za-z]+-[A-Za-z]+", content)
        for term in tech_pattern:
            if len(term) <= self.max_term_length:
                terms.append(term)

        return terms

    def save_index(self, index_dir: str) -> None:
        """Save index and mappings to files.

        Args:
            index_dir: Directory to save index files
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save Faiss index
        faiss.write_index(self.index, str(path / "embeddings.faiss"))

        # Save chunk map
        chunk_map_data = {
            chunk_id: {
                "id": chunk.id,
                "cite_key": chunk.cite_key,
                "content": chunk.content[:200],  # Truncate for storage
                "chunk_type": chunk.chunk_type,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
                "chapter_type": chunk.chapter_type,
            }
            for chunk_id, chunk in self.chunk_map.items()
        }
        with open(path / "chunk_map.json", "w", encoding="utf-8") as f:
            json.dump(chunk_map_data, f, ensure_ascii=False, indent=2)

        # Save ID mapping
        with open(path / "id_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.id_to_chunk, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved index to {index_dir}")

    def save_term_index(self, term_index: dict[str, list[str]], index_dir: str) -> None:
        """Save term index to file.

        Args:
            term_index: Term inverted index
            index_dir: Directory to save
        """
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "term_index.json", "w", encoding="utf-8") as f:
            json.dump(term_index, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved term index to {index_dir}")

    def load_index(self, index_dir: str) -> None:
        """Load index and mappings from files.

        Args:
            index_dir: Directory containing index files
        """
        path = Path(index_dir)

        # Load Faiss index
        index_path = path / "embeddings.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")

        # Load chunk map
        chunk_map_path = path / "chunk_map.json"
        if chunk_map_path.exists():
            with open(chunk_map_path, encoding="utf-8") as f:
                chunk_map_data = json.load(f)
            self.chunk_map = {
                chunk_id: Chunk(
                    id=data["id"],
                    cite_key=data["cite_key"],
                    content=data["content"],
                    chunk_type=data["chunk_type"],
                    section_title=data["section_title"],
                    page_number=data["page_number"],
                    bounding_box=[0.0, 0.0, 0.0, 0.0],  # Placeholder
                    chapter_type=data.get("chapter_type", "OTHER"),
                )
                for chunk_id, data in chunk_map_data.items()
            }

        # Load ID mapping
        id_mapping_path = path / "id_mapping.json"
        if id_mapping_path.exists():
            with open(id_mapping_path, encoding="utf-8") as f:
                self.id_to_chunk = {int(k): v for k, v in json.load(f).items()}

    def load_term_index(self, index_dir: str) -> dict[str, list[str]]:
        """Load term index from file.

        Args:
            index_dir: Directory containing term_index.json

        Returns:
            Term inverted index dictionary
        """
        path = Path(index_dir) / "term_index.json"
        if not path.exists():
            return {}

        with open(path, encoding="utf-8") as f:
            return json.load(f)