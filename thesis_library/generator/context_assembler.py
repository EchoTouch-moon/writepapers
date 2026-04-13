"""Context assembler - retrieval results to structured prompt.

CRITICAL: Uses chunk.id (e.g., Paper7744_s2_para1) for citation tracking,
NOT cite_key (e.g., Paper7744), to ensure fine-grained溯源.
"""

import logging
from dataclasses import dataclass

from thesis_library.core.chunker import Chunk
from thesis_library.core.retriever import SearchResult

logger = logging.getLogger(__name__)

# Token estimation (rough: 1 token ≈ 4 chars for Chinese, 0.75 words for English)
CHARS_PER_TOKEN = 4


@dataclass
class AssembledContext:
    """Assembled context for LLM generation.

    Attributes:
        query: Writing requirement
        section: Target section name
        chunks: List of search results
        prompt: Final assembled prompt
        token_estimate: Estimated token count
    """
    query: str
    section: str
    chunks: list[SearchResult]
    prompt: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    return len(text) // CHARS_PER_TOKEN


def assemble_prompt(
    query: str,
    results: list[SearchResult],
    section: str,
    max_tokens: int = 4000,
    style: str = "academic",
) -> AssembledContext:
    """Assemble retrieval results into structured prompt for LLM.

    CRITICAL FIX: Uses chunk.id for citation placeholders,
    ensuring fine-grained溯源 (Paper7744_s2_para1 vs Paper7744).

    Args:
        query: Writing requirement for current section
        results: Search results from library
        section: Target section name (e.g., "引言")
        max_tokens: Maximum tokens for context
        style: Writing style ("academic", "technical")

    Returns:
        AssembledContext with final prompt
    """
    context_blocks = []
    total_tokens = 0
    included_chunks = []

    for r in results:
        # CRITICAL: Use chunk.id for fine-grained citation tracking
        # Format: [引用ID: Paper7744_s2_para1] (来源: Paper7744, p.23)
        block = format_reference_block(r.chunk)

        block_tokens = estimate_tokens(block)
        if total_tokens + block_tokens > max_tokens:
            logger.info(f"Token limit reached at {len(included_chunks)} chunks")
            break

        context_blocks.append(block)
        included_chunks.append(r)
        total_tokens += block_tokens

    # Build final prompt
    prompt = build_prompt_template(query, section, context_blocks, style)

    return AssembledContext(
        query=query,
        section=section,
        chunks=included_chunks,
        prompt=prompt,
        token_estimate=total_tokens + estimate_tokens(prompt),
    )


def format_reference_block(chunk: Chunk) -> str:
    """Format a single reference block for LLM.

    CRITICAL: Uses chunk.id for citation tracking.

    Args:
        chunk: Chunk object with fine-grained ID

    Returns:
        Formatted reference block string
    """
    return f"""
[引用ID: {chunk.id}] (来源: {chunk.cite_key}, p.{chunk.page_number})
章节: {chunk.section_title}
类型: {chunk.chunk_type}
内容: {chunk.content}
"""


def build_prompt_template(
    query: str,
    section: str,
    context_blocks: list[str],
    style: str,
) -> str:
    """Build the final prompt template.

    Args:
        query: Writing requirement
        section: Target section
        context_blocks: Formatted reference blocks
        style: Writing style

    Returns:
        Complete prompt string
    """
    context_text = "".join(context_blocks)

    if style == "academic":
        system_instruction = """
## 写作要求
1. 使用严格的学术写作风格
2. 每个论点必须引用参考文献支撑
3. 引用格式: [引用ID] (如 [Paper7744_s2_para1])
4. 无一字无出处，避免无根据的断言
5. 保持逻辑连贯，段落衔接自然
"""
    else:
        system_instruction = """
## 写作要求
1. 使用技术文档风格
2. 引用格式: [引用ID]
3. 简洁明了，重点突出
"""

    return f"""
{system_instruction}

## 当前任务
目标章节: {section}
写作需求: {query}

## 参考文献（请严格使用引用ID标注来源）
{context_text}

## 输出内容
请根据以上参考文献，撰写{section}的内容。
记住：每个事实性陈述都要标注引用来源，格式为 [PaperXXX_sN_paraN]。
"""


def assemble_for_section(
    library,
    section: str,
    query: str,
    top_k: int = 10,
    chapter_type: str | None = None,
) -> AssembledContext:
    """Convenience function: search + assemble in one call.

    Args:
        library: Library instance
        section: Target section
        query: Writing requirement
        top_k: Number of results
        chapter_type: Optional chapter type filter

    Returns:
        AssembledContext ready for LLM
    """
    results = library.search(query, chapter_type=chapter_type, top_k=top_k)
    return assemble_prompt(query, results, section)