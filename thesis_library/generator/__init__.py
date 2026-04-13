"""Thesis Generator Module - LLM-assisted academic writing with citation tracking.

This module implements:
- LLM API client (Qwen/DashScope compatible)
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
from thesis_library.generator.section_planner import (
    RetrievalPlan,
    SectionPlanner,
    plan_and_assemble,
    create_planner,
)

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMConfig",
    # Context Assembler
    "assemble_prompt",
    "AssembledContext",
    # Citation Validator
    "verify_citations",
    "extract_citations",
    "CITATION_PATTERN",
    # Section Planner
    "RetrievalPlan",
    "SectionPlanner",
    "plan_and_assemble",
    "create_planner",
]