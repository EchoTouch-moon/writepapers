"""Core module exports."""

from thesis_library.core.chapter_classifier import ChapterClassifier, ClassificationError
from thesis_library.core.chunker import Chunk, Chunker
from thesis_library.core.indexer import IndexBuildError, Indexer
from thesis_library.core.metadata_extractor import MetadataExtractor, PaperMetadata
from thesis_library.core.pdf_processor import PDFParseError, PDFProcessor
from thesis_library.core.retriever import Retriever, SearchResult
from thesis_library.core.smoother import sliding_window_smoothing

__all__ = [
    "ChapterClassifier",
    "Chunk",
    "Chunker",
    "ClassificationError",
    "IndexBuildError",
    "Indexer",
    "MetadataExtractor",
    "PaperMetadata",
    "PDFParseError",
    "PDFProcessor",
    "Retriever",
    "SearchResult",
    "sliding_window_smoothing",
]