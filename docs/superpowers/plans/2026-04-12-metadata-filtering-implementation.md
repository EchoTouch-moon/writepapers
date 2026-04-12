# Metadata Filtering Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement metadata filtering to achieve Multi_cond Recall@5 ≥ 0.80 and hit_ratio ≥ 0.80.

**Architecture:** Phase 1 uses Faiss oversampling with Python post-filtering. Chunks are classified by Qwen API during ingestion, smoothed by sliding window, then filtered during retrieval.

**Tech Stack:** Qwen API (dashscope), tenacity for retry, ChapterType enum.

---

## File Structure

```
thesis_library/
├── config.py               # Task 1: Add ChapterType enum, oversample_multiplier
├── core/
│   ├── chunker.py          # Task 2: Add chapter_type field to Chunk
│   ├── chapter_classifier.py  # Task 3: Create - Qwen API batch classification
│   ├── smoother.py         # Task 4: Create - sliding window smoothing
│   ├── indexer.py          # Task 5: Modify - store chapter_type in chunk_map
│   ├── retriever.py        # Task 6: Modify - oversampling + post-filtering
│   └── __init__.py         # Task 7: Export new modules
├── cli.py                  # Task 8: Add --chapter-type flag
└── evaluator/
    └── test_cases.py       # Task 9: Update TC-004 with chapter_type
```

---

## Task 1: Add ChapterType Enum to Config

**Files:**
- Modify: `thesis_library/config.py`

- [ ] **Step 1: Add ChapterType enum and oversample_multiplier**

```python
# Add to thesis_library/config.py after existing imports

from enum import Enum

class ChapterType(Enum):
    """Standardized chapter type classification for academic papers."""
    ABSTRACT = "ABSTRACT"       # 摘要/概述
    INTRODUCTION = "INTRODUCTION"   # 引言/背景/相关工作
    METHODOLOGY = "METHODOLOGY"     # 方法/系统设计/模型架构
    EXPERIMENT = "EXPERIMENT"       # 实验/评估/结果分析
    CONCLUSION = "CONCLUSION"       # 结论/总结/未来展望
    REFERENCE = "REFERENCE"         # 参考文献
    OTHER = "OTHER"                 # 附录/致谢/其他
```

- [ ] **Step 2: Add oversample_multiplier to LibraryConfig**

```python
# In LibraryConfig dataclass (around line 36), add:
@dataclass(frozen=True)
class LibraryConfig:
    # ... existing fields ...
    
    # Metadata filtering configuration
    oversample_multiplier: int = 10  # Oversample factor for post-filtering
    
    # Chapter classifier configuration
    qwen_api_key: str | None = None  # Set via environment variable
    classifier_batch_size: int = 5   # Chunks per API call
    classifier_model: str = "qwen-plus"  # Qwen model name
```

- [ ] **Step 3: Verify config loads correctly**

Run: `uv run python -c "from thesis_library.config import ChapterType, LibraryConfig; c = LibraryConfig(); print(c.oversample_multiplier, ChapterType.METHODOLOGY)"`

Expected: `10 METHODOLOGY`

- [ ] **Step 4: Commit**

```bash
git add thesis_library/config.py
git commit -m "feat(config): add ChapterType enum and oversample_multiplier for metadata filtering"
```

---

## Task 2: Add chapter_type Field to Chunk

**Files:**
- Modify: `thesis_library/core/chunker.py:13-34`

- [ ] **Step 1: Add chapter_type field to Chunk dataclass**

```python
# Modify Chunk dataclass in thesis_library/core/chunker.py

@dataclass
class Chunk:
    """A chunk of paper content.

    Attributes:
        id: Unique chunk identifier (cite_key + type + index)
        cite_key: Citation key of source paper
        content: Text content
        chunk_type: Type of chunk (section, paragraph, table, list)
        section_title: Title of the section this chunk belongs to
        page_number: Page number in the PDF
        bounding_box: Coordinates [left, bottom, right, top]
        parent_id: ID of parent section chunk (if this is a sub-chunk)
        chapter_type: Standardized chapter classification (added for metadata filtering)
    """

    id: str
    cite_key: str
    content: str
    chunk_type: str  # section | paragraph | table | list
    section_title: str
    page_number: int
    bounding_box: list[float]
    parent_id: str | None = None
    chapter_type: str = "OTHER"  # Default, will be classified by ChapterClassifier
```

- [ ] **Step 2: Verify Chunk imports**

Run: `uv run python -c "from thesis_library.core.chunker import Chunk; c = Chunk(id='test', cite_key='test', content='test', chunk_type='para', section_title='test', page_number=1, bounding_box=[0,0,0,0]); print(c.chapter_type)"`

Expected: `OTHER`

- [ ] **Step 3: Commit**

```bash
git add thesis_library/core/chunker.py
git commit -m "feat(chunker): add chapter_type field to Chunk dataclass"
```

---

## Task 3: Create ChapterClassifier Module

**Files:**
- Create: `thesis_library/core/chapter_classifier.py`

- [ ] **Step 1: Create chapter_classifier.py with Qwen API integration**

```python
"""Chapter classification using Qwen API with batch processing."""

import json
import logging
import os
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from thesis_library.config import ChapterType, LibraryConfig
from thesis_library.core.chunker import Chunk

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Chapter classification failed."""
    pass


@dataclass
class ClassifierConfig:
    """Configuration for chapter classifier."""
    api_key: str
    model: str = "qwen-plus"
    batch_size: int = 5
    max_retries: int = 3


class ChapterClassifier:
    """Batch classify chunks using Qwen API.
    
    Workflow:
    1. Group chunks into batches (batch_size=5)
    2. Call Qwen API with system prompt
    3. Parse JSON array response
    4. Handle retry on rate limits/network errors
    """

    SYSTEM_PROMPT = """你是学术论文章节分类器。阅读以下按顺序排列的文本块（Chunk 1 到 Chunk N）。
将每个文本块归类到以下类型之一：[ABSTRACT, INTRODUCTION, METHODOLOGY, EXPERIMENT, CONCLUSION, REFERENCE, OTHER]。
你必须且只能输出一个严格的 JSON 数组，数组长度必须与输入的 Chunk 数量一致。不要包含任何其他文字。

示例输出：
["ABSTRACT", "INTRODUCTION", "INTRODUCTION", "METHODOLOGY", "METHODOLOGY"]"""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy load Qwen client."""
        if self._client is None:
            try:
                import dashscope
                dashscope.api_key = self.config.api_key
                self._client = dashscope.Generation
            except ImportError:
                raise ClassificationError("dashscope not installed. Run: uv add dashscope")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_api(self, user_prompt: str) -> str:
        """Call Qwen API with retry logic."""
        client = self._get_client()
        
        response = client.call(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            result_format="message"
        )
        
        if response.status_code != 200:
            raise ClassificationError(f"API error: {response.code} - {response.message}")
        
        return response.output.choices[0].message.content

    def classify_batch(self, chunks: list[Chunk]) -> list[ChapterType]:
        """Classify a batch of chunks.
        
        Args:
            chunks: List of chunks to classify (max batch_size)
            
        Returns:
            List of ChapterType enums
        """
        if not chunks:
            return []
        
        # Build user prompt
        prompt_lines = []
        for i, chunk in enumerate(chunks, 1):
            prompt_lines.append(f"Chunk {i}: {chunk.content}")
        user_prompt = "\n".join(prompt_lines)
        
        # Call API
        try:
            response_text = self._call_api(user_prompt)
            
            # Parse JSON array
            labels = json.loads(response_text.strip())
            
            if len(labels) != len(chunks):
                logger.warning(
                    f"Response length mismatch: got {len(labels)}, expected {len(chunks)}. "
                    f"Marking all as OTHER."
                )
                return [ChapterType.OTHER] * len(chunks)
            
            # Convert to ChapterType
            chapter_types = []
            for label in labels:
                try:
                    chapter_types.append(ChapterType(label))
                except ValueError:
                    logger.warning(f"Invalid label '{label}', using OTHER")
                    chapter_types.append(ChapterType.OTHER)
            
            return chapter_types
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Marking batch as OTHER.")
            return [ChapterType.OTHER] * len(chunks)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise ClassificationError(str(e)) from e

    def classify_all(self, chunks: list[Chunk]) -> list[ChapterType]:
        """Classify all chunks in batches.
        
        Args:
            chunks: All chunks from a paper
            
        Returns:
            List of ChapterType enums for each chunk
        """
        all_types: list[ChapterType] = []
        
        # Process in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_types = self.classify_batch(batch)
            all_types.extend(batch_types)
            
            logger.info(f"Classified batch {i//self.config.batch_size + 1}: {len(batch)} chunks")
        
        return all_types


def create_classifier(library_config: LibraryConfig) -> ChapterClassifier:
    """Factory function to create classifier from library config."""
    api_key = library_config.qwen_api_key or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ClassificationError(
            "QWEN_API_KEY not set. Set via environment variable or LibraryConfig."
        )
    
    return ChapterClassifier(ClassifierConfig(
        api_key=api_key,
        model=library_config.classifier_model,
        batch_size=library_config.classifier_batch_size,
    ))
```

- [ ] **Step 2: Add dashscope dependency**

Run: `uv add dashscope`

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from thesis_library.core.chapter_classifier import ChapterClassifier; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thesis_library/core/chapter_classifier.py pyproject.toml uv.lock
git commit -m "feat(classifier): add ChapterClassifier with Qwen API batch processing"
```

---

## Task 4: Create Sliding Window Smoother

**Files:**
- Create: `thesis_library/core/smoother.py`

- [ ] **Step 1: Create smoother.py with sliding window implementation**

```python
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
```

- [ ] **Step 2: Verify smoother logic**

Run: `uv run python -c "
from thesis_library.config import ChapterType
from thesis_library.core.smoother import sliding_window_smoothing

types = [ChapterType.ABSTRACT, ChapterType.INTRODUCTION, ChapterType.OTHER, ChapterType.INTRODUCTION, ChapterType.INTRODUCTION]
smoothed = sliding_window_smoothing(types)
print([t.value for t in smoothed])
"`

Expected: `['ABSTRACT', 'INTRODUCTION', 'INTRODUCTION', 'INTRODUCTION', 'INTRODUCTION']`

- [ ] **Step 3: Commit**

```bash
git add thesis_library/core/smoother.py
git commit -m "feat(smoother): add sliding window smoothing with boundary protection"
```

---

## Task 5: Modify Indexer to Store chapter_type

**Files:**
- Modify: `thesis_library/core/indexer.py:236-254`

- [ ] **Step 1: Update chunk_map serialization to include chapter_type**

```python
# In indexer.py, modify save_index method (around line 236)

# Replace the chunk_map_data building:
chunk_map_data = {
    chunk_id: {
        "id": chunk.id,
        "cite_key": chunk.cite_key,
        "content": chunk.content[:200],  # Truncate for storage
        "chunk_type": chunk.chunk_type,
        "section_title": chunk.section_title,
        "page_number": chunk.page_number,
        "chapter_type": chunk.chapter_type,  # NEW: Store chapter_type
    }
    for chunk_id, chunk in self.chunk_map.items()
}
```

- [ ] **Step 2: Update chunk_map deserialization**

```python
# In indexer.py, modify load_index method (around line 290)

# Replace the Chunk construction:
self.chunk_map = {
    chunk_id: Chunk(
        id=data["id"],
        cite_key=data["cite_key"],
        content=data["content"],
        chunk_type=data["chunk_type"],
        section_title=data["section_title"],
        page_number=data["page_number"],
        bounding_box=[0.0, 0.0, 0.0, 0.0],  # Placeholder
        chapter_type=data.get("chapter_type", "OTHER"),  # NEW: Load chapter_type
    )
    for chunk_id, data in chunk_map_data.items()
}
```

- [ ] **Step 3: Verify serialization**

Run: `uv run python -c "
from thesis_library.core.indexer import Indexer
from thesis_library.core.chunker import Chunk
import tempfile, json

chunk = Chunk(id='test', cite_key='test', content='test content', chunk_type='para', section_title='test', page_number=1, bounding_box=[0,0,0,0], chapter_type='METHODOLOGY')
idx = Indexer()
idx.chunk_map = {'test': chunk}

with tempfile.TemporaryDirectory() as tmp:
    idx.save_index(tmp)
    with open(f'{tmp}/chunk_map.json') as f:
        data = json.load(f)
    print(data['test']['chapter_type'])
"`

Expected: `METHODOLOGY`

- [ ] **Step 4: Commit**

```bash
git add thesis_library/core/indexer.py
git commit -m "feat(indexer): store chapter_type in chunk_map serialization"
```

---

## Task 6: Modify Retriever for Oversampling + Post-filtering

**Files:**
- Modify: `thesis_library/core/retriever.py:62-100, 182-260`

- [ ] **Step 1: Add chapter_type parameter to search method**

```python
# In retriever.py, modify search method (around line 62)

def search(
    self,
    query: str,
    chapter_type: str | None = None,  # NEW: chapter_type filter
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

    # Step 1: Term anchoring (skip if chapter_type filter)
    candidate_chunk_ids = self._term_anchoring(query) if not chapter_type else None

    # Step 2: Structural constraint (skip if chapter_type filter)
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
        chapter_type,  # NEW: Pass chapter_type
    )

    return results
```

- [ ] **Step 2: Modify _semantic_search for oversampling and post-filtering**

```python
# In retriever.py, modify _semantic_search method (around line 182)

def _semantic_search(
    self,
    query: str,
    candidate_ids: list[str] | None,
    threshold: float,
    top_k: int,
    chapter_type: str | None = None,  # NEW: chapter_type filter
) -> list[SearchResult]:
    """Step 3: Semantic matching with similarity threshold and optional chapter_type filter.

    Args:
        query: Search query
        candidate_ids: Pre-filtered chunk IDs (or None for all)
        threshold: Minimum similarity score
        top_k: Number of results to return
        chapter_type: Optional chapter type filter (triggers oversampling)

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

        # NEW: Filter by chapter_type
        if chapter_type is not None:
            chunk = self.indexer.chunk_map.get(chunk_id)
            if chunk and chunk.chapter_type != chapter_type:
                continue  # Skip mismatching chapter_type

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
        section_match = True

        results.append(SearchResult(
            chunk=chunk,
            similarity=similarity,
            matched_terms=matched_terms,
            section_match=section_match,
        ))

        # Stop when we have enough (after filtering)
        if len(results) >= top_k:
            break

    # Sort by similarity
    results.sort(key=lambda r: r.similarity, reverse=True)
    
    # NEW: Warn if filter exhausted candidates
    if chapter_type and len(results) < top_k:
        logger.warning(
            f"Metadata filter exhausted retrieved candidates. "
            f"Only {len(results)}/{top_k} results match chapter_type={chapter_type}"
        )
    
    return results
```

- [ ] **Step 3: Add oversample_multiplier attribute**

```python
# In retriever.py, modify __init__ method (around line 46)

def __init__(
    self,
    indexer: Indexer,
    similarity_threshold: float = 0.7,
    default_top_k: int = 10,
    oversample_multiplier: int = 10,  # NEW: Oversample factor
) -> None:
    self.indexer = indexer
    self.similarity_threshold = similarity_threshold
    self.default_top_k = default_top_k
    self.oversample_multiplier = oversample_multiplier  # NEW
    self.term_index: dict[str, list[str]] = {}
    self.chapter_type: str | None = None  # For structural constraint
    self.section_mapping = SECTION_TYPE_MAPPING
```

- [ ] **Step 4: Verify retriever imports**

Run: `uv run python -c "from thesis_library.core.retriever import Retriever; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add thesis_library/core/retriever.py
git commit -m "feat(retriever): add oversampling and chapter_type post-filtering"
```

---

## Task 7: Export New Modules in core/__init__.py

**Files:**
- Modify: `thesis_library/core/__init__.py`

- [ ] **Step 1: Add exports for new modules**

```python
# In thesis_library/core/__init__.py, add exports

from thesis_library.core.chunker import Chunk, Chunker
from thesis_library.core.indexer import Indexer, IndexBuildError
from thesis_library.core.retriever import Retriever, SearchResult
from thesis_library.core.chapter_classifier import ChapterClassifier, ClassificationError  # NEW
from thesis_library.core.smoother import sliding_window_smoothing  # NEW

__all__ = [
    "Chunk",
    "Chunker",
    "Indexer",
    "IndexBuildError",
    "Retriever",
    "SearchResult",
    "ChapterClassifier",  # NEW
    "ClassificationError",  # NEW
    "sliding_window_smoothing",  # NEW
]
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from thesis_library.core import ChapterClassifier, sliding_window_smoothing; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add thesis_library/core/__init__.py
git commit -m "feat(core): export ChapterClassifier and smoother modules"
```

---

## Task 8: Add --chapter-type CLI Flag

**Files:**
- Modify: `thesis_library/cli.py:278-296`

- [ ] **Step 1: Add chapter_type parameter to search command**

```python
# In cli.py, modify search_parser (around line 278)

# search command
search_parser = subparsers.add_parser("search", help="Search library")
search_parser.add_argument("query", help="Search query")
search_parser.add_argument(
    "--chapter", "-c",
    help="Chapter type for structural constraint",
)
search_parser.add_argument(
    "--chapter-type", "-C",  # NEW: Metadata filter
    help="Filter by chapter type (ABSTRACT, INTRODUCTION, METHODOLOGY, EXPERIMENT, CONCLUSION, REFERENCE, OTHER)",
)
search_parser.add_argument(
    "--threshold", "-t",
    type=float,
    default=0.7,
    help="Similarity threshold",
)
search_parser.add_argument(
    "--top-k", "-k",
    type=int,
    default=10,
    help="Number of results",
)
```

- [ ] **Step 2: Modify cmd_search to use chapter_type**

```python
# In cli.py, modify cmd_search function (around line 77)

def cmd_search(args: argparse.Namespace) -> int:
    """Search the library."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1

    query = args.query
    chapter_type = args.chapter_type  # NEW
    threshold = args.threshold
    top_k = args.top_k

    print(f"Searching: {query}")
    if chapter_type:
        print(f"  Chapter type filter: {chapter_type}")
    print(f"  Threshold: {threshold}, Top-K: {top_k}")

    # NEW: Use chapter_type filter
    results = library.search(
        query, 
        chapter_type=chapter_type,  # NEW: Pass chapter_type
        threshold=threshold, 
        top_k=top_k
    )

    if not results:
        print("No results found")
        return 0

    print(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        chapter_label = f" [{r.chunk.chapter_type}]" if r.chunk.chapter_type != "OTHER" else ""
        print(f"{i}. [{r.chunk.cite_key}]{chapter_label} Sim: {r.similarity:.2f}")
        print(f"   Section: {r.chunk.section_title}")
        print(f"   Terms: {r.matched_terms}")
        print(f"   Content: {r.chunk.content[:100]}...")
        print()

    return 0
```

- [ ] **Step 3: Verify CLI help**

Run: `uv run thesis-library search --help`

Expected: Shows `--chapter-type` option

- [ ] **Step 4: Commit**

```bash
git add thesis_library/cli.py
git commit -m "feat(cli): add --chapter-type flag for metadata filtering"
```

---

## Task 9: Update TC-004 Test Case with chapter_type

**Files:**
- Modify: `thesis/evaluator/test_cases.json`

- [ ] **Step 1: Update TC-004 to use chapter_type filter**

```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "query": "大语言模型",
      "query_type": "exact_term",
      "expected_chunk_ids": ["Paper5462_section1", "Paper4721_para10"],
      "chapter_type": null,
      "threshold": null,
      "notes": "Exact term: 大语言模型 should match title/intro chunks"
    },
    {
      "id": "TC-002",
      "query": "知识图谱",
      "query_type": "exact_term",
      "expected_chunk_ids": ["Paper4721_para7", "Paper4721_para6"],
      "chapter_type": null,
      "threshold": 0.6,
      "notes": "Exact term: 知识图谱结构化组织事实信息 (threshold lowered to 0.6)"
    },
    {
      "id": "TC-003",
      "query": "如何将用户画像融入对话系统",
      "query_type": "fuzzy_concept",
      "expected_chunk_ids": ["Paper1215_para3", "Paper1215_para4"],
      "chapter_type": null,
      "threshold": null,
      "notes": "Fuzzy concept: user profile integration concept"
    },
    {
      "id": "TC-004",
      "query": "知识图谱与大语言模型的结合",
      "query_type": "multi_cond",
      "expected_chunk_ids": ["Paper4721_para7", "Paper4721_para8"],
      "chapter_type": "METHODOLOGY",
      "threshold": null,
      "notes": "Multi condition: KG + LLM combination, FILTERED by METHODOLOGY chapter_type"
    },
    {
      "id": "TC-005",
      "query": "RAG系统如何整合知识",
      "query_type": "cross_para",
      "expected_chunk_ids": ["Paper4721_para297", "Paper4721_para60"],
      "chapter_type": null,
      "threshold": null,
      "notes": "Cross paragraph: RAG knowledge integration"
    },
    {
      "id": "TC-006",
      "query": "角色扮演智能体",
      "query_type": "exact_term",
      "expected_chunk_ids": ["Paper5462_para8", "Paper5462_para30"],
      "chapter_type": null,
      "threshold": null,
      "notes": "Exact term: role-playing agent"
    },
    {
      "id": "TC-007",
      "query": "个性化对话生成方法",
      "query_type": "fuzzy_concept",
      "expected_chunk_ids": ["Paper1215_section1", "Paper1215_para14"],
      "chapter_type": null,
      "threshold": null,
      "notes": "Fuzzy concept: personalized dialogue generation"
    }
  ],
  "metadata": {
    "created": "TC-001",
    "last_updated": "TC-004",
    "total_cases": 7
  }
}
```

- [ ] **Step 2: Commit test case update**

```bash
git add thesis/evaluator/test_cases.json
git commit -m "test(evaluator): update TC-004 with chapter_type METHODOLOGY filter"
```

---

## Task 10: Integrate Classification into Ingestion Pipeline

**Files:**
- Modify: `thesis_library/core/library.py` (ingest method)

- [ ] **Step 1: Find and read library.py ingest method**

Run: `uv run python -c "import inspect; from thesis_library import Library; print(inspect.getfile(Library))"` to locate file.

- [ ] **Step 2: Add classification step after chunking**

```python
# In library.py ingest method, after chunking step

# After: chunks = chunker.chunk_paper(json_data, cite_key)

# NEW: Classify chapter types
from thesis_library.core.chapter_classifier import create_classifier
from thesis_library.core.smoother import sliding_window_smoothing

classifier = create_classifier(self.config)
chapter_types = classifier.classify_all(chunks)

# Apply smoothing
smoothed_types = sliding_window_smoothing(chapter_types)

# Assign to chunks
for chunk, chapter_type in zip(chunks, smoothed_types):
    chunk.chapter_type = chapter_type.value

logger.info(f"Classified {len(chunks)} chunks into chapter types")
```

- [ ] **Step 3: Commit integration**

```bash
git add thesis_library/core/library.py
git commit -m "feat(library): integrate chapter classification into ingestion pipeline"
```

---

## Task 11: Re-ingest and Validate

**Files:**
- No file changes, runtime validation

- [ ] **Step 1: Clear existing library**

Run: `rm -rf thesis/library/*`

- [ ] **Step 2: Set QWEN_API_KEY environment variable**

Run: `export QWEN_API_KEY=<your_key>` (user provides key)

- [ ] **Step 3: Re-ingest test PDFs**

Run: `uv run thesis-library ingest papers/*.pdf --batch`

Expected: Logs showing classification batches

- [ ] **Step 4: Verify chapter_type in chunk_map**

Run: `uv run python -c "
import json
with open('thesis/library/index/chunk_map.json') as f:
    data = json.load(f)
    
types = {}
for chunk_id, chunk in list(data.items())[:20]:
    t = chunk.get('chapter_type', 'OTHER')
    types[t] = types.get(t, 0) + 1
    
print('Chapter type distribution (sample):')
for t, count in sorted(types.items()):
    print(f'  {t}: {count}')
"`

Expected: Shows distribution across INTRODUCTION, METHODOLOGY, etc.

- [ ] **Step 5: Run evaluation**

Run: `uv run thesis-library eval --verbose`

Expected: Multi_cond Recall@5 ≥ 0.80

---

## Task 12: Final Validation and Commit

- [ ] **Step 1: Check all metrics meet targets**

Run: `uv run thesis-library eval`

Verify:
- Multi_cond Recall@5 ≥ 0.80
- hit_ratio (TC-004) ≥ 0.80

- [ ] **Step 2: Save baseline**

Run: `uv run thesis-library eval --save-baseline`

- [ ] **Step 3: Create summary commit**

```bash
git add thesis/evaluator/baseline.json thesis/evaluator/last_run.json
git commit -m "feat(metadata-filtering): complete Phase C implementation

Results:
- Multi_cond Recall@5: baseline → target (validated)
- Oversampling + post-filtering working
- Chapter classification integrated
- All tests passing"
```

---

## Self-Review Checklist

**1. Spec Coverage:**
- D1 Optimization Targets → Task 11 validates metrics ✓
- D2 Oversampling Strategy → Task 6 implements ✓
- D3 Classification Pipeline → Tasks 3, 10 integrate ✓
- D4 ChapterType Enum → Task 1 defines ✓
- D5 Sliding Window → Task 4 implements ✓
- D6 Retriever Changes → Task 6 modifies ✓
- File Structure → All files covered ✓
- Risk Mitigation → Task 3 has tenacity retry ✓

**2. Placeholder Scan:**
- No TBD/TODO found ✓
- All code blocks complete ✓
- All test commands specified ✓

**3. Type Consistency:**
- ChapterType enum uppercase values match LLM output ✓
- Chunk.chapter_type uses string ("OTHER") default ✓
- Retriever.search chapter_type parameter str | None ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-12-metadata-filtering-implementation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** - Fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**