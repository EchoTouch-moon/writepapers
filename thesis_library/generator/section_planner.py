"""Section Planner - LLM-assisted retrieval strategy planning.

This module implements:
- RetrievalPlan dataclass (LLM-generated retrieval strategy)
- SectionPlanner class (plan → execute → merge workflow)
- Integration with Library.search() and Library.search_by_terms()
"""

import json
import logging
from dataclasses import dataclass

from thesis_library import Library
from thesis_library.core.chunker import Chunk
from thesis_library.core.retriever import SearchResult
from thesis_library.generator.llm_client import LLMClient, LLMConfig
from thesis_library.generator.context_assembler import AssembledContext, assemble_prompt

logger = logging.getLogger(__name__)


# --- Constants ---

PLANNER_SYSTEM_PROMPT = """你是学术论文检索规划专家。分析用户的写作需求，生成多路径检索计划。

输出格式（严格JSON，不要包含其他文字）：
{
  "rationale": "检索策略的理由说明",
  "sub_queries": ["分解后的子查询1", "分解后的子查询2"],
  "key_terms": ["关键术语1", "关键术语2"],
  "chapter_suggestions": ["METHODOLOGY", null, "EXPERIMENT"]
}

规划原则：
1. sub_queries: 将复杂需求分解为多个具体查询（2-4个）
2. key_terms: 提取可能出现在文献中的技术术语（2-5个）
3. chapter_suggestions: 为每个子查询建议合适的章节类型过滤（可选，用null表示不限制）
4. 确保覆盖写作需求的不同角度"""


# --- Data Structures ---


@dataclass
class RetrievalPlan:
    """LLM-generated retrieval plan for a section.

    Attributes:
        rationale: Why these queries/terms are relevant (Chain of Thought)
        sub_queries: Decomposed queries for multi-path retrieval (2-4)
        key_terms: Technical terms to anchor term-based search (2-5)
        chapter_suggestions: Suggested chapter types per sub_query (optional)
    """

    rationale: str
    sub_queries: list[str]
    key_terms: list[str]
    chapter_suggestions: list[str | None]


# --- SectionPlanner Class ---


class SectionPlanner:
    """Plan retrieval strategy for thesis section writing.

    Workflow:
    1. Analyze section requirements via LLM → RetrievalPlan
    2. Execute multi-path retrieval (semantic + term)
    3. Deduplicate and merge results
    """

    def __init__(self, llm_client: LLMClient, library: Library) -> None:
        """Initialize planner with LLM client and library.

        Args:
            llm_client: LLM client for planning
            library: Thesis library for retrieval
        """
        self.llm_client = llm_client
        self.library = library

    def plan(self, section: str, query: str) -> RetrievalPlan:
        """Generate retrieval plan via LLM.

        Args:
            section: Target section name
            query: Writing requirement

        Returns:
            RetrievalPlan with sub_queries and key_terms
        """
        user_prompt = f"""
## 目标章节
{section}

## 写作需求
{query}

请生成检索计划（JSON格式）。
"""
        logger.info(f"Planning retrieval for section: {section}")

        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for structured output
        )

        return self._parse_plan_response(response)

    def execute_plan(
        self,
        plan: RetrievalPlan,
        top_k_per_query: int = 3,
        threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Execute multi-path retrieval and merge.

        Args:
            plan: RetrievalPlan from LLM
            top_k_per_query: Results per sub_query
            threshold: Minimum similarity threshold

        Returns:
            Deduplicated merged results
        """
        all_results: list[SearchResult] = []

        # Path A: Semantic search for each sub_query
        for i, sub_query in enumerate(plan.sub_queries):
            chapter_type = None
            if plan.chapter_suggestions and i < len(plan.chapter_suggestions):
                chapter_type = plan.chapter_suggestions[i]

            logger.info(f"Semantic search: '{sub_query}' (chapter={chapter_type})")
            results = self.library.search(
                sub_query,
                chapter_type=chapter_type,
                top_k=top_k_per_query,
                threshold=threshold,
            )
            all_results.extend(results)

        # Path B: Term search for key_terms
        if plan.key_terms:
            logger.info(f"Term search: {plan.key_terms}")
            term_chunks = self.library.search_by_terms(plan.key_terms)

            # Convert Chunk to SearchResult with placeholder score
            for chunk in term_chunks[:top_k_per_query * 2]:
                all_results.append(
                    SearchResult(
                        chunk=chunk,
                        similarity=0.8,  # Placeholder for term match
                        matched_terms=plan.key_terms,
                        section_match=True,
                    )
                )

        # Deduplicate and sort
        return self._merge_results(all_results, top_k_per_query * len(plan.sub_queries) + 3)

    def plan_and_retrieve(
        self,
        section: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> tuple[RetrievalPlan, list[SearchResult]]:
        """Main entry: plan → execute → merge.

        Args:
            section: Target section name
            query: Writing requirement
            top_k: Final result count
            threshold: Minimum similarity

        Returns:
            (RetrievalPlan, merged SearchResult list)
        """
        plan = self.plan(section, query)
        logger.info(f"🤔 规划思路: {plan.rationale}")
        logger.info(f"🔍 子查询数: {len(plan.sub_queries)}, 术语数: {len(plan.key_terms)}")

        results = self.execute_plan(plan, top_k_per_query=3, threshold=threshold)

        # Limit to top_k
        results = results[:top_k]
        logger.info(f"✅ 检索完成，合并后 {len(results)} 个结果")

        return plan, results

    def _parse_plan_response(self, response: str) -> RetrievalPlan:
        """Parse LLM JSON response with graceful fallback.

        Args:
            response: Raw LLM response

        Returns:
            RetrievalPlan (parsed or fallback)
        """
        try:
            # Try direct JSON parse
            data = json.loads(response.strip())
            return RetrievalPlan(
                rationale=data.get("rationale", ""),
                sub_queries=data.get("sub_queries", []),
                key_terms=data.get("key_terms", []),
                chapter_suggestions=data.get("chapter_suggestions", []),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, using fallback plan")

            # Fallback: extract content from response
            # Try to find JSON block in response
            import re
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return RetrievalPlan(
                        rationale=data.get("rationale", ""),
                        sub_queries=data.get("sub_queries", []),
                        key_terms=data.get("key_terms", []),
                        chapter_suggestions=data.get("chapter_suggestions", []),
                    )
                except json.JSONDecodeError:
                    pass

            # Ultimate fallback: use original query
            return RetrievalPlan(
                rationale="Fallback: JSON parsing failed, using original query",
                sub_queries=[response[:100] if len(response) > 100 else response],
                key_terms=[],
                chapter_suggestions=[None],
            )

    def _merge_results(
        self,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Deduplicate and merge results.

        Args:
            results: All results from multi-path retrieval
            top_k: Maximum results to return

        Returns:
            Deduplicated, sorted results
        """
        merged: dict[str, SearchResult] = {}

        for r in results:
            chunk_id = r.chunk.id
            if chunk_id not in merged or r.similarity > merged[chunk_id].similarity:
                merged[chunk_id] = r

        # Sort by similarity
        sorted_results = list(merged.values())
        sorted_results.sort(key=lambda x: x.similarity, reverse=True)

        return sorted_results[:top_k]


# --- Factory and Convenience Functions ---


def create_planner(
    library: Library,
    llm_config: LLMConfig | None = None,
) -> SectionPlanner:
    """Factory function to create SectionPlanner.

    Args:
        library: Thesis library instance
        llm_config: Optional LLM configuration

    Returns:
        SectionPlanner instance
    """
    config = llm_config or LLMConfig()
    client = LLMClient(config)
    return SectionPlanner(client, library)


def plan_and_assemble(
    library: Library,
    section: str,
    query: str,
    top_k: int = 10,
    threshold: float = 0.7,
    llm_config: LLMConfig | None = None,
) -> tuple[RetrievalPlan, AssembledContext]:
    """Plan retrieval, execute, and assemble context.

    Convenience function for end-to-end flow.

    Args:
        library: Thesis library
        section: Target section
        query: Writing requirement
        top_k: Result count
        threshold: Similarity threshold
        llm_config: Optional LLM config for planner

    Returns:
        (RetrievalPlan, AssembledContext)
    """
    planner = create_planner(library, llm_config)
    plan, results = planner.plan_and_retrieve(section, query, top_k, threshold)
    context = assemble_prompt(query, results, section)
    return plan, context