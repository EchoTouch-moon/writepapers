"""Thesis Generator Module - LLM-assisted academic writing with citation tracking.

This module implements:
- LLM API client (Qwen/Doubao compatible)
- Context assembler (retrieval → prompt)
- Citation validator (strict reference tracking)
- Section planner (outline-driven workflow)
"""

from thesis_library.generator.llm_client import LLMClient, LLMConfig
from thesis_library.generator.context_assembler import (
    assemble_prompt,
    AssembledContext,
)
from thesis_library.generator.citation_validator import (
    verify_citations,
    extract_citations,
    CITATION_PATTERN,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "assemble_prompt",
    "AssembledContext",
    "verify_citations",
    "extract_citations",
    "CITATION_PATTERN",
]