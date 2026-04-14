---
name: thesis-writing
description: This skill should be used when the user asks to "write thesis", "generate thesis chapter", "start thesis writing", "continue thesis", or needs to write an undergraduate thesis based on project code, proposal, and literature. Provides interactive chapter-by-chapter generation with three-source integration, bounded context management, citation verification, Chinese undergraduate thesis constraints, and writing anti-patterns.
tags: [Writing, Thesis, Academic, Workflow]
version: 2.2.0
---

# Thesis-Writing Skill

Interactive thesis generation workflow with three-source integration.

**Target: 国内计算机本科毕业设计论文**

## Core Positioning

**工程实践能力为核心：** 完整技术闭环 + 规范学术格式 + 严谨逻辑链条 + 真实研究内容

**LLM角色边界：** 辅助写作工具，禁止全文代写，所有内容须经用户审核

## Core Philosophy

**Three sources. Bounded context. Chapter-by-chapter. Verified citations.**

Generate undergraduate thesis content by synthesizing:
1. **Proposal** - Research goals, technical route, chapter outline
2. **Project Code** - Architecture, modules, implementations (via CLI tools)
3. **Literature** - Relevant papers via rag-citation-mcp

Each chapter uses controlled context (~8000-11000 tokens) to maintain quality.

## Invocation

```
/thesis-writing --proposal <path> --papers <path> --code <path> [--output <path>]
/thesis-writing continue
/thesis-writing status
```

## Execution Flow

### Phase 0: Initialization

1. **Parse proposal document** (Markdown)
   - Extract: title, objectives, technical_route, chapter_outline
   - Save to `thesis/proposal_info.json`

2. **Import literature** (rag-citation-mcp)
   ```
   ingest_papers(pdf_paths=<paper_paths>, library_name="thesis")
   ```

3. **Analyze project code** (CLI tools)
   - Glob/Grep → file structure, tech stack
   - Agent → architecture, key modules
   - LSP → function definitions, references
   - Save to `thesis/code_analysis_report.json`

4. **Initialize state files**
   - `thesis/global_state.json` - core context, registries
   - `thesis/state.json` - progress tracking

5. **Confirm chapter outline** with user

### Phase 1: Chapter Writing Loop

For each chapter in outline, execute 5-step cycle:

#### Step A: Prepare Chapter Context

Load bounded context:
- Layer 1: Core context from `global_state.json` (~2000 tokens)
- Layer 2: Previous 2 chapter summaries from `chapter_summaries/` (~1000 tokens)
- Layer 3: Chapter-specific sources (~5000-8000 tokens):
  - Proposal sections matching chapter title
  - Code modules matching chapter topics
  - Literature via `build_context(query=<chapter_topics>)`

#### Step B: Write Chapter Content

Generate Markdown with:
- Content based on three sources
- Citations in `[AuthorYear]` format
- Figure suggestions as HTML comments
- Clean academic writing style

#### Step C: Verify Citations

Call citation-verification logic:
1. Format check: `\[\w+\d{4}\]` pattern
2. Existence check: citation in literature library
3. Output verification report

#### Step D: Review with User

Display chapter + citations + figure suggestions + verification report.
User options: [A] Accept, [M] Modify, [R] Rewrite, [S] Pause

#### Step E: Save Chapter

- Write to `thesis/chapters/<num>-<title>.md`
- Extract summary → `thesis/chapter_summaries/<num>-summary.json`
- Update `global_state.json` registries
- Update `state.json` progress

### Phase 2: Finalize

1. Merge chapter references → `thesis/references/all-refs.md`
2. Format conversion (optional): GB/T 7714, APA, IEEE
3. Word export (optional): docx-protocol-mcp

## Context Management

Three-layer architecture prevents overflow:

| Layer | Content | Tokens |
|-------|---------|--------|
| Core | Title, objectives, outline, registries | ~2000 |
| Previous | 2 chapter summaries | ~1000 |
| Current | Proposal+Code+Literature for this chapter | ~5000-8000 |

Total: ~8000-11000 tokens (safe range).

See `references/context-management.md` for details.

## Figure Suggestions

Automatically suggest figures based on chapter type:

| Chapter | Required Figures |
|---------|------------------|
| 需求分析 | 需求分类表, Use Case 图 |
| 系统设计 | 系统架构图, ER图 ⭐ |
| 系统测试 | 测试用例表, 测试结果表 ⭐ |

Suggestions inserted as HTML comments, summarized at chapter end.

See `references/figure-system.md` for mapping rules.

## Citation Rules

Format: `[AuthorYear]` inline, full reference at paragraph end.

Example:
```markdown
知识图谱是一种结构化的语义知识库...[Wang2023]
> 引用: 王某某 (2023). 知识图谱构建方法研究. 计算机学报.
```

Verification prevents hallucination. See `references/citation-rules.md`.

## Writing Constraints

### Code Display Rules

| Pattern | Status |
|---------|--------|
| 正文代码占比超10% | ❌ 禁止 |
| 单段代码超过30行 | ❌ 禁止 |
| 大段粘贴完整源代码 | ❌ 禁止 |
| 关键代码片段(≤30行，带注释) | ✅ 允许 |
| 文字描述算法思路 | ✅ 推荐 |

### Citation Placement Rules

| Pattern | Status |
|---------|--------|
| 段末集中罗列多个引用 | ❌ 禁止 |
| 一处引用3篇以上 | ❌ 禁止 |
| 提及技术时立即引用 | ✅ 正确 |
| 引用紧跟相关论述 | ✅ 正确 |

### Red Lines (绝对禁止)

- ❌ 编造参考文献/数据/研究成果
- ❌ "填补空白""首创"等夸大表述
- ❌ 口语化/网络化表述
- ❌ 前后矛盾/逻辑脱节
- ❌ 无编号/无引用图表

详细规范见 `references/chinese-thesis-constraints.md`。

## State Recovery

Resume interrupted sessions:
1. Check `thesis/state.json` for `current_chapter`
2. Load `global_state.json` for context
3. Continue from last pending chapter

## Additional Resources

### Reference Files

- **`references/chinese-thesis-constraints.md`** — 国内本科毕设论文完整规范（章节框架、红线禁令、测试要求）
- **`references/context-management.md`** — Three-layer architecture details
- **`references/chapter-workflow.md`** — 5-step cycle implementation (含系统提示模板)
- **`references/figure-system.md`** — Figure suggestion rules
- **`references/citation-rules.md`** — Citation format and verification
- **`references/proposal-parsing.md`** — Proposal parsing patterns

### Templates

- **`templates/global-state.json`** - Global state schema
- **`templates/chapter-summary.json`** - Chapter summary schema
- **`templates/state.json`** - Progress state schema

### Examples

- **`examples/sample-session.md`** - Example interactive session