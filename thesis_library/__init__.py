"""Thesis Library - Local literature management for LLM-assisted thesis writing."""

import json
import logging
from pathlib import Path

from thesis_library.config import LibraryConfig
from thesis_library.core import (
    Chunk,
    Chunker,
    ClassificationError,
    Indexer,
    MetadataExtractor,
    PaperMetadata,
    PDFProcessor,
    Retriever,
    SearchResult,
    sliding_window_smoothing,
)
from thesis_library.core.chapter_classifier import create_classifier

logger = logging.getLogger(__name__)


class Library:
    """Main interface for thesis literature library.

    Integrates PDF parsing, chunking, indexing, and retrieval.
    """

    def __init__(self, config: LibraryConfig | None = None) -> None:
        self.config = config or LibraryConfig()
        self.config.ensure_dirs()

        self.pdf_processor = PDFProcessor()
        self.chunker = Chunker(
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
        )
        self.metadata_extractor = MetadataExtractor()
        self.indexer = Indexer(
            model_name=self.config.embedding_model,
            index_type=self.config.index_type,
            min_term_freq=self.config.min_term_freq,
            max_term_length=self.config.max_term_length,
        )
        self.retriever = Retriever(
            indexer=self.indexer,
            similarity_threshold=self.config.similarity_threshold,
            default_top_k=self.config.default_top_k,
        )

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load existing index and metadata."""
        # Load metadata registry
        self.metadata_registry = self.metadata_extractor.load_metadata_registry(
            self.config.metadata_file
        )

        # Load index if exists
        index_path = Path(self.config.index_dir) / "embeddings.faiss"
        if index_path.exists():
            self.indexer.load_index(self.config.index_dir)
            term_index = self.indexer.load_term_index(self.config.index_dir)
            self.retriever.set_term_index(term_index)

    def ingest(
        self,
        pdf_paths: list[str],
        use_hybrid: bool = False,
        interactive: bool = True,
    ) -> list[PaperMetadata]:
        """Ingest PDF files into the library.

        Args:
            pdf_paths: List of PDF file paths
            use_hybrid: Use hybrid mode for complex content
            interactive: Prompt user to confirm/edit metadata

        Returns:
            List of PaperMetadata for ingested papers
        """
        self.pdf_processor.use_hybrid = use_hybrid

        results = self.pdf_processor.process_batch(pdf_paths, use_hybrid)
        metadata_list: list[PaperMetadata] = []

        for result in results:
            pdf_path = result["pdf_path"]
            md_content = result["md_content"]
            json_data = result["json_content"]

            # Extract metadata
            metadata = self.metadata_extractor.extract(
                json_data, pdf_path, self.config.papers_dir
            )

            # Interactive metadata editing (skip in non-interactive mode)
            if interactive:
                metadata = self._interactive_metadata_edit(metadata)

            # Save markdown content
            self._save_paper_md(metadata.md_path, md_content)

            # Save JSON data
            self._save_paper_json(metadata.json_path, json_data)

            # Chunk paper
            chunks = self.chunker.chunk_paper(json_data, metadata.cite_key)

            # Classify chapter types
            try:
                classifier = create_classifier(self.config)
                chapter_types = classifier.classify_all(chunks)
                smoothed_types = sliding_window_smoothing(chapter_types)

                for chunk, chapter_type in zip(chunks, smoothed_types):
                    chunk.chapter_type = chapter_type.value

                logger.info(f"Classified {len(chunks)} chunks into chapter types")
            except ClassificationError as e:
                logger.warning(f"Classification skipped: {e}")
                # Chunks retain default "OTHER" chapter_type

            self.chunker.save_chunks(chunks, metadata.chunks_path)

            # Add to index
            self.indexer.add_chunks(chunks)
            term_index = self.indexer.build_term_index(chunks)

            # Merge term index with existing
            self._merge_term_index(term_index)

            # Save metadata
            self.metadata_extractor.save_metadata(metadata, self.config.metadata_file)

            metadata_list.append(metadata)
            logger.info(f"Ingested: {metadata.cite_key} - {metadata.title}")

        # Rebuild and save full index
        self._save_index()

        return metadata_list

    def _interactive_metadata_edit(self, metadata: PaperMetadata) -> PaperMetadata:
        """Prompt user to edit metadata (placeholder for CLI interaction)."""
        # In CLI mode, this would use rich/prompt toolkit
        # For now, return unchanged
        return metadata

    def _save_paper_md(self, path: str, content: str) -> None:
        """Save paper markdown content."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_paper_json(self, path: str, data: list) -> None:
        """Save paper JSON data."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _merge_term_index(self, new_terms: dict[str, list[str]]) -> None:
        """Merge new term index with existing."""
        current = self.retriever.term_index
        for term, chunk_ids in new_terms.items():
            if term in current:
                # Merge chunk IDs
                current[term] = list(set(current[term] + chunk_ids))
            else:
                current[term] = chunk_ids
        self.retriever.set_term_index(current)

    def _save_index(self) -> None:
        """Save index and term index to disk."""
        self.indexer.save_index(self.config.index_dir)
        self.indexer.save_term_index(self.retriever.term_index, self.config.index_dir)

    def rebuild_index(self) -> None:
        """Rebuild index from all ingested papers."""
        all_chunks: list[Chunk] = []

        for cite_key, metadata in self.metadata_registry.items():
            chunks_path = metadata.chunks_path
            if Path(chunks_path).exists():
                chunks = self.chunker.load_chunks(chunks_path)
                all_chunks.extend(chunks)

        if all_chunks:
            self.indexer.build_index(all_chunks)
            term_index = self.indexer.build_term_index(all_chunks)
            self.retriever.set_term_index(term_index)
            self._save_index()
            logger.info(f"Rebuilt index with {len(all_chunks)} chunks")

    def search(
        self,
        query: str,
        chapter_type: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search the library for relevant content.

        Args:
            query: Search query
            chapter_type: Current chapter type for structural constraint
            threshold: Minimum similarity score
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        return self.retriever.search(query, chapter_type, threshold, top_k)

    def search_by_terms(self, terms: list[str]) -> list[Chunk]:
        """Search by specific terms only."""
        return self.retriever.search_by_terms(terms)

    def list_papers(self) -> list[PaperMetadata]:
        """List all papers in the library."""
        return list(self.metadata_registry.values())

    def get_paper(self, cite_key: str) -> PaperMetadata | None:
        """Get paper metadata by cite key."""
        return self.metadata_registry.get(cite_key)

    def edit_metadata(self, cite_key: str, updates: dict) -> PaperMetadata | None:
        """Edit paper metadata.

        Args:
            cite_key: Citation key
            updates: Dictionary of field updates

        Returns:
            Updated metadata or None if not found
        """
        metadata = self.metadata_registry.get(cite_key)
        if not metadata:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(metadata, key):
                object.__setattr__(metadata, key, value)

        # Handle cite_key change (requires renaming files)
        if "cite_key" in updates and updates["cite_key"] != cite_key:
            metadata = self.metadata_extractor.update_cite_key(
                metadata, updates["cite_key"]
            )
            # Remove old entry
            del self.metadata_registry[cite_key]

        # Save updated metadata
        self.metadata_extractor.save_metadata(metadata, self.config.metadata_file)
        self.metadata_registry[metadata.cite_key] = metadata

        return metadata

    def status(self) -> dict:
        """Get library status."""
        return {
            "papers_count": len(self.metadata_registry),
            "index_status": "built" if self.indexer.index else "pending",
            "chunks_count": len(self.indexer.chunk_map),
            "terms_count": len(self.retriever.term_index),
            "library_dir": self.config.library_dir,
        }

    def get_known_terms(self) -> list[str]:
        """Get all known technical terms."""
        return self.retriever.get_known_terms()

    def detect_terms_in_content(self, content: str) -> list[dict]:
        """Detect technical terms in content and find potential citations.

        Args:
            content: Text content to analyze

        Returns:
            List of detected terms with citation suggestions
        """
        known_terms = self.get_known_terms()
        detected: list[dict] = []

        for term in known_terms:
            if term in content:
                chunks = self.search_by_terms([term])
                if chunks:
                    detected.append({
                        "term": term,
                        "position": content.find(term),
                        "potential_citations": [
                            c.cite_key for c in chunks[:3]
                        ],
                    })

        return detected


# Convenience exports
__all__ = [
    "Library",
    "LibraryConfig",
    "Chunk",
    "Chunker",
    "Indexer",
    "MetadataExtractor",
    "PaperMetadata",
    "PDFProcessor",
    "Retriever",
    "SearchResult",
]