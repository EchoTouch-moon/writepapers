# Context Management Architecture

Three-layer context system prevents overflow while maintaining coherence.

## Layer 1: Core Context (Always Loaded)

~2000 tokens, from `global_state.json`:

```json
{
  "core_context": {
    "title": "论文题目",
    "objectives": ["目标1", "目标2"],
    "tech_route_summary": "技术路线概述",
    "chapter_outline": [
      {"number": 1, "title": "绪论", "description": "背景与目标"},
      {"number": 2, "title": "相关技术", "description": "技术基础"}
    ]
  },
  "concept_registry": {},
  "citation_registry": {}
}
```

### When to Update

- After proposal parsing: title, objectives, chapter_outline
- After each chapter: concept_registry, citation_registry

## Layer 2: Previous Chapter Summaries (On-demand)

~500 tokens per chapter, from `chapter_summaries/`:

```json
{
  "number": 2,
  "title": "相关技术",
  "key_concepts": [
    {"term": "知识图谱", "definition": "结构化语义知识库", "first_defined": true}
  ],
  "key_citations": ["Wang2023", "Li2022"],
  "contribution": "介绍了知识图谱和深度学习技术基础",
  "link_points": ["将在第四章详细介绍系统实现"]
}
```

### Loading Strategy

When writing chapter N:
- Load summary of chapter N-1 (if exists)
- Load summary of chapter N-2 (if exists)
- Maximum: 2 previous summaries (~1000 tokens)

### Link Point Resolution

Previous chapter's `link_points` → current chapter's context injection:
```
"第二章提到: '将在第四章详细介绍系统实现'"
→ Inject into Chapter 4 writing context as "需要呼应的衔接点"
```

## Layer 3: Current Chapter Sources (~5000-8000 tokens)

### Proposal Matching

Match chapter title to proposal sections:
- 绪论 → 研究背景, 研究目标
- 相关技术 → 技术路线
- 系统设计 → 系统架构描述

Extract relevant paragraphs, limit to ~1000 tokens.

### Code Matching

Match chapter topics to code analysis:
- 系统设计 → architecture, modules
- 系统实现 → key_implementations
- 系统测试 → test coverage

Extract relevant sections, limit to ~2000 tokens.

### Literature Retrieval

Via rag-citation-mcp:
```
build_context(query=<chapter_title + topics>, section_type=<type>, top_k=10)
```

Returns ~10 chunks, ~3000 tokens total.

## Context Pruning Rules

| Source | Pruning Method | Limit |
|--------|----------------|-------|
| Proposal | Match by section title | ~1000 tokens |
| Code | Match by module/topic | ~2000 tokens |
| Literature | RAG top_k=10 | ~3000 tokens |
| Previous summaries | Only N-1, N-2 | ~1000 tokens |

## Total Token Budget

| Layer | Typical | Maximum |
|-------|---------|---------|
| Core | 2000 | 2500 |
| Previous | 1000 | 1500 |
| Current | 6000 | 8000 |
| **Total** | **9000** | **12000** |

Safe for quality generation.

## Concept Registry

Track defined terms across chapters:

```json
{
  "知识图谱": {
    "definition": "结构化语义知识库...",
    "first_defined_in": 2,
    "citations": ["Wang2023"],
    "used_in_chapters": [2, 4, 5]
  }
}
```

Purpose: Avoid redefining same terms, maintain consistency.

## Citation Registry

Track citation usage:

```json
{
  "Wang2023": {
    "used_in_chapters": [2, 4, 5],
    "context": ["知识图谱构建", "系统设计"],
    "full_citation": "王某某 (2023). 知识图谱构建方法研究. 计算机学报."
  }
}
```

Purpose: Prevent duplicate citations, track context overlap.