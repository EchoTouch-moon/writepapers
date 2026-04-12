"""Library configuration management."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union


class ChapterType(Enum):
    """Standardized chapter type classification for academic papers."""

    ABSTRACT = "ABSTRACT"  # 摘要/概述
    INTRODUCTION = "INTRODUCTION"  # 引言/背景/相关工作
    METHODOLOGY = "METHODOLOGY"  # 方法/系统设计/模型架构
    EXPERIMENT = "EXPERIMENT"  # 实验/评估/结果分析
    CONCLUSION = "CONCLUSION"  # 结论/总结/未来展望
    REFERENCE = "REFERENCE"  # 参考文献
    OTHER = "OTHER"  # 附录/致谢/其他


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LibraryConfig:
    """Configuration for the thesis library.

    Attributes:
        library_dir: Root directory for library data
        papers_dir: Directory for parsed paper files
        index_dir: Directory for vector and term indices
        metadata_file: Path to metadata registry file
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        embedding_model: Sentence-transformers model name
        index_type: Faiss index type
        similarity_threshold: Minimum similarity score for retrieval
        default_top_k: Default number of results to return
        min_term_freq: Minimum frequency to consider a term
        max_term_length: Maximum characters for a term
        oversample_multiplier: Oversample factor for post-filtering
        qwen_api_key: Qwen API key for chapter classification
        classifier_batch_size: Number of chunks per API call
        classifier_model: Qwen model name for classification
    """

    library_dir: str = "thesis/library"
    papers_dir: str = field(default="thesis/library/papers", init=False)
    index_dir: str = field(default="thesis/library/index", init=False)
    metadata_file: str = field(default="thesis/library/metadata.json", init=False)

    # Chunking configuration
    max_chunk_size: int = 500
    min_chunk_size: int = 100

    # Index configuration
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    index_type: str = "FlatIP"

    # Retrieval configuration
    similarity_threshold: float = 0.7
    default_top_k: int = 10

    # Term configuration
    min_term_freq: int = 2
    max_term_length: int = 10

    # Metadata filtering configuration
    oversample_multiplier: int = 10  # Oversample factor for post-filtering

    # Chapter classifier configuration
    qwen_api_key: str | None = None  # Set via environment variable
    classifier_batch_size: int = 5  # Chunks per API call
    classifier_model: str = "qwen-plus"  # Qwen model name

    def __post_init__(self) -> None:
        # Compute derived paths based on library_dir
        base = Path(self.library_dir)
        object.__setattr__(self, "papers_dir", str(base / "papers"))
        object.__setattr__(self, "index_dir", str(base / "index"))
        object.__setattr__(self, "metadata_file", str(base / "metadata.json"))

    def ensure_dirs(self) -> None:
        """Create library directories if they don't exist."""
        Path(self.papers_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_file).parent.mkdir(parents=True, exist_ok=True)


# Section type mapping for structural constraints
SECTION_TYPE_MAPPING: dict[str, list[str]] = {
    "绪论": ["Introduction", "Background", "Related Work"],
    "相关技术": ["Methodology", "Approach", "Technical Background"],
    "需求分析": ["Requirements", "Use Case", "Functional Analysis"],
    "系统设计": ["Architecture", "Design", "Database Design", "System Design"],
    "系统实现": ["Implementation", "Implementation Details", "Technical Implementation"],
    "系统测试": ["Testing", "Experiment", "Evaluation", "Results"],
    "总结与展望": ["Conclusion", "Future Work", "Discussion"],
}


def sanitize_cite_key(cite_key: str) -> str:
    """Sanitize cite_key to prevent path traversal.

    Args:
        cite_key: Raw cite key from user input or PDF filename

    Returns:
        Cleaned cite key safe for file path construction

    Raises:
        ValueError: If cite_key is invalid after sanitization
    """
    # Remove any path separators and special characters
    clean = re.sub(r"[^\w\-]", "", cite_key)
    if not clean or len(clean) < 2:
        raise ValueError(f"Invalid cite_key: {cite_key}")
    return clean


def validate_path(base_dir: Path, target_path: Path) -> Path:
    """Ensure target_path is within base_dir to prevent path traversal.

    Args:
        base_dir: Base directory that paths should be contained within
        target_path: Path to validate

    Returns:
        Resolved target_path if valid

    Raises:
        ValueError: If path traversal is detected
    """
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    if not str(resolved_target).startswith(str(resolved_base)):
        raise ValueError(f"Path traversal detected: {target_path}")
    return resolved_target