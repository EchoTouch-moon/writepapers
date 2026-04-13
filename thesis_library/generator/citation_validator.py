"""Citation validator - strict reference tracking for thesis generation.

Enforces fine-grained citation tracking using chunk.id format.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Citation pattern: [PaperXXX_sN_paraN] or [PaperXXX_sN_mergedN]
CITATION_PATTERN = r"\[([A-Za-z0-9_]+_s[0-9]+_[a-z]+[0-9]*)\]"
# Also accept section chunks: [PaperXXX_sectionN]
SECTION_PATTERN = r"\[([A-Za-z0-9_]+_section[0-9]*)\]"


@dataclass
class CitationReport:
    """Citation validation report.

    Attributes:
        citations: List of extracted citation IDs
        valid: List of valid citations (in available_chunks)
        invalid: List of invalid citations (not found)
        coverage: Percentage of content with citations
    """
    citations: list[str]
    valid: list[str]
    invalid: list[str]
    coverage: float


def extract_citations(text: str) -> list[str]:
    """Extract all citation IDs from generated text.

    Args:
        text: Generated text with citation markers

    Returns:
        List of citation IDs (e.g., ["Paper7744_s2_para1", "Paper4639_section1"])
    """
    # Match paragraph citations
    para_citations = re.findall(CITATION_PATTERN, text)
    # Match section citations
    section_citations = re.findall(SECTION_PATTERN, text)

    all_citations = para_citations + section_citations
    logger.info(f"Extracted {len(all_citations)} citations")
    return all_citations


def verify_citations(
    text: str,
    available_chunks: dict[str, object],
) -> CitationReport:
    """Verify citations against available chunks.

    Args:
        text: Generated text with citation markers
        available_chunks: Dict mapping chunk_id → Chunk object

    Returns:
        CitationReport with validation results
    """
    citations = extract_citations(text)
    available_ids = set(available_chunks.keys())

    valid = [c for c in citations if c in available_ids]
    invalid = [c for c in citations if c not in available_ids]

    # Calculate citation coverage
    # Estimate: count citation markers vs total sentences
    sentences = len(re.findall(r"[。.!?]", text))
    coverage = len(citations) / max(sentences, 1) if sentences > 0 else 0.0

    logger.info(f"Citation verification: {len(valid)} valid, {len(invalid)} invalid")

    return CitationReport(
        citations=citations,
        valid=valid,
        invalid=invalid,
        coverage=min(coverage, 1.0),
    )


def format_error_report(report: CitationReport) -> str:
    """Format validation errors for LLM correction prompt.

    Args:
        report: Citation validation report

    Returns:
        Formatted error message for LLM
    """
    if not report.invalid:
        return "✅ 所有引用验证通过"

    lines = ["❌ 发现无效引用，请修正："]
    for cite in report.invalid:
        lines.append(f"  - [{cite}] 未在参考文献中找到")

    lines.append("\n可用引用ID:")
    available_ids = list(report.valid)[:20]  # Show first 20
    for id in available_ids:
        lines.append(f"  - [{id}]")

    return "\n".join(lines)


def build_correction_prompt(
    original_text: str,
    report: CitationReport,
    available_chunks: dict[str, object],
) -> str:
    """Build prompt for LLM to correct invalid citations.

    Args:
        original_text: Original generated text
        report: Citation validation report
        available_chunks: Available chunks for replacement

    Returns:
        Correction prompt
    """
    error_msg = format_error_report(report)

    # Build replacement suggestions
    suggestions = []
    for invalid_cite in report.invalid[:5]:
        # Try to find similar citations by cite_key
        cite_key = invalid_cite.split("_")[0]
        similar = [id for id in available_chunks if id.startswith(cite_key)]
        if similar:
            suggestions.append(f"[{invalid_cite}] → 可替换为 [{similar[0]}]")

    return f"""
## 需要修正的文本
{original_text}

## 验证报告
{error_msg}

## 替换建议
{"".join(suggestions)}

## 修正要求
1. 修正所有无效引用
2. 保持原文结构和逻辑
3. 不要删除必要的引用标注
4. 输出修正后的完整文本
"""