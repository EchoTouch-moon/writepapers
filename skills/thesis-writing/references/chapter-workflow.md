# Chapter Writing Workflow

Detailed 5-step cycle for each chapter.

## Step A: Prepare Chapter Context

### Input
- `chapter`: {number, title, topics}
- `proposal_info.json`
- `code_analysis_report.json`
- `global_state.json`
- rag-citation-mcp library

### Procedure

1. **Load core context** from `global_state.json`
   - title, objectives, tech_route_summary
   - chapter_outline list

2. **Load previous summaries**
   - Read `chapter_summaries/(N-1)-summary.json` if exists
   - Read `chapter_summaries/(N-2)-summary.json` if exists
   - Extract link_points for context injection

3. **Match proposal content**
   - Find proposal sections matching chapter.title
   - Extract relevant paragraphs
   - Limit to ~1000 tokens

4. **Match code analysis**
   - Find modules matching chapter.topics
   - For 系统设计: architecture, design_patterns
   - For 系统实现: key_implementations
   - For 系统测试: test coverage
   - Limit to ~2000 tokens

5. **Retrieve literature**
   ```
   build_context(
     query=f"{chapter.title} {chapter.topics}",
     section_type=_get_section_type(chapter.title),
     top_k=10
   )
   ```

6. **Build chapter outline**
   - Generate detailed section outline
   - Mark suggested citation points

7. **Pre-judge figures**
   - Load figure mapping from `templates/figure-mapping.yaml`
   - Add suggested figures to context

### Output
```json
{
  "chapter": {"number": 4, "title": "系统设计", "topics": ["架构", "数据库"]},
  "core_context": {...},
  "previous_summaries": [...],
  "proposal_relevant": "...",
  "code_relevant": {"modules": [...], "implementations": [...]},
  "literature_relevant": [
    {"cite_key": "Wang2023", "title": "...", "relevance": "架构设计方法"}
  ],
  "outline": [
    {"section": "4.1 总体设计", "citations": ["Wang2023"]}
  ],
  "suggested_figures": [
    {"type": "系统架构图", "priority": "required", "position": "4.1开头"}
  ]
}
```

## Step B: Write Chapter Content

### System Prompt Template

```
你是毕业论文写作助手，撰写 {chapter.title} 章节。

【核心定位】
- 工程实践能力为核心，完整技术闭环
- LLM仅辅助写作，禁止全文代写
- 所有内容须经用户审核

【写作要求】
1. 基于提供的三源信息写作，不编造内容
2. 技术描述结合项目代码实际实现
3. 重要观点必须有引用支持
4. 引用格式: [AuthorYear]，正文上角标
5. 输出干净 Markdown 格式，学术化正式语体
6. 需要图表支撑时插入建议标记:
   <!-- FIGURE_SUGGESTION: 类型 | 位置 | 内容描述 | 参考代码 -->
7. 图表必须有编号、标题，正文必须引用（如"如图4-1所示"）

【代码展示约束】
- ❌ 正文代码总占比≤10%
- ❌ 单段代码不超过30行
- ❌ 禁止大段粘贴完整源代码
- ✅ 仅展示关键代码片段，带规范注释
- ✅ 用文字描述算法思路、设计决策
- ✅ 完整代码放入附录，正文仅展示核心逻辑

【引用位置约束】
- ❌ 段末集中罗列多个引用
- ❌ 一处引用3篇以上文献
- ❌ "填补空白""首创"等夸大表述
- ✅ 提及技术/方法时立即引用
- ✅ 引用紧跟相关论述，放在句子末尾
- ✅ 首次出现时引用，后续不重复

【表达规范】
- 正式学术语体，禁止口语化/网络化
- 客观被动语态，避免第一人称过度使用
- 核心信息（课题/技术栈/版本/模块）全文统一
- 专业术语、缩写、大小写全文一致

【红线禁令】
- 绝对禁止编造参考文献/数据/研究成果
- 绝对禁止前后矛盾、逻辑脱节
- 绝对禁止空泛无量化表述
- 绝对禁止无关冗余内容、通用模板化内容
```

### User Prompt Template

```
## 开题报告相关内容
{proposal_relevant}

## 项目代码分析
{code_relevant}

## 相关文献
{literature_relevant}

## 前序章节衔接点
{link_points_from_previous}

## 写作提纲
{outline}

## 图表建议
{suggested_figures}

请撰写 {chapter.title} 章节，约 2000-3000 字。
```

### Generation Process

1. Send prompt to LLM
2. Receive Markdown content
3. Parse for:
   - Citations `[AuthorYear]`
   - Figure suggestions `<!-- FIGURE_SUGGESTION: ... -->`
4. Format output

### Output Format

```markdown
---
chapter: 04-系统设计
status: draft
---

## 4 系统设计

### 4.1 总体设计

系统采用三层架构设计...[Wang2023]
> 引用: 王某某 (2023). 系统架构设计方法. 计算机学报.

<!-- FIGURE_SUGGESTION: 系统架构图 | 4.1开头 | 展示三层结构 | src/api/ -->

...

## 📊 本章图表建议汇总

| 序号 | 类型 | 建议位置 | 内容摘要 |
|------|------|----------|----------|
| 1 | 系统架构图 | 4.1 | 三层架构结构 |
```

## Step C: Verify Citations

### Verification Levels

**Level 1: Format Check**
```python
import re
pattern = r'\[\w+\d{4}\]'
citations = re.findall(pattern, content)
# Check: no malformed patterns like [wang] or [2023]
```

**Level 2: Existence Check**
```
For each citation [AuthorYear]:
  - Query rag-citation-mcp: search_citations(keyword=Author)
  - Verify year matches
  - If not found: mark as warning
```

**Level 3: Content Check (Optional)**
```
For key citations:
  - Retrieve original chunk
  - Compare with generated content
  - Flag potential misinterpretation
```

### Verification Report Format

```json
{
  "chapter": "04-系统设计",
  "total_citations": 5,
  "valid": [
    {"cite_key": "Wang2023", "status": "verified"}
  ],
  "warnings": [
    {"cite_key": "Unknown2024", "status": "not_found", "suggestion": "请确认文献来源"}
  ],
  "format_errors": []
}
```

## Step D: Review with User

### Display

1. Show chapter content (Markdown, foldable)
2. Show citation annotations
3. Show verification report
4. Show figure suggestion table

### User Options

| Option | Action |
|--------|--------|
| [A] Accept | Save chapter, continue to next |
| [M] Modify | Apply user edits, then save |
| [R] Rewrite | Adjust context, regenerate |
| [S] Pause | Save progress, exit loop |
| [G] Generate Figure | Call figure generation skill |
| [I] Insert Figure | User provides figure file |

### Handling Options

- **Accept**: Proceed to Step E
- **Modify**: Apply diff, verify citations again, proceed to Step E
- **Rewrite**: Collect user feedback, adjust prompt, return to Step B
- **Pause**: Update `state.json`, save session, exit
- **Generate Figure**: Use frontend-design skill, insert path, continue
- **Insert Figure**: Copy to `thesis/figures/`, update chapter, continue

## Step E: Save Chapter

### File Operations

1. **Write chapter file**
   ```
   thesis/chapters/{num}-{title}.md
   ```

2. **Extract and save summary**
   ```
   thesis/chapter_summaries/{num}-summary.json
   ```

3. **Update global_state.json**
   - Add concepts to concept_registry
   - Add citations to citation_registry
   - Update figure counters

4. **Update state.json**
   - Increment current_chapter
   - Add to completed_chapters
   - Update last_updated timestamp

### Summary Extraction

Use LLM to extract:
```json
{
  "number": 4,
  "title": "系统设计",
  "key_concepts": [
    {"term": "三层架构", "definition": "...", "first_defined": true}
  ],
  "key_citations": ["Wang2023"],
  "contribution": "设计了系统的总体架构和数据模型",
  "link_points": ["将在第五章介绍系统实现细节"]
}
```

### Figure Counter Update

```json
{
  "figure_counters": {
    "chapter_04": 3,
    "chapter_05": 0
  },
  "table_counters": {
    "chapter_04": 2,
    "chapter_05": 0
  }
}
```