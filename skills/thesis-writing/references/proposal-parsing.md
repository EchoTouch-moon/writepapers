# Proposal Document Parsing

Extract key information from proposal Markdown document.

## Expected Proposal Structure

Typical undergraduate thesis proposal structure:

```markdown
# 论文题目

## 研究背景与意义
...

## 国内外研究现状
...

## 研究目标与内容
...

## 技术路线/研究方法
...

## 预期成果
...

## 论文大纲/章节安排
...

## 进度计划
...
```

## Parsing Procedure

### Step 1: Identify Sections

Match headings by level and keyword:

```python
section_keywords = {
    "title": ["论文题目", "题目"],
    "background": ["研究背景", "背景", "意义"],
    "status": ["研究现状", "国内外研究", "文献综述"],
    "objectives": ["研究目标", "目标", "研究内容", "内容"],
    "technical_route": ["技术路线", "研究方法", "方法"],
    "expected_results": ["预期成果", "成果"],
    "chapter_outline": ["论文大纲", "章节安排", "大纲"],
    "schedule": ["进度计划", "计划"]
}
```

### Step 2: Extract Content

For each section:
- Extract text content
- Summarize if too long (>500 chars)
- Preserve structure (lists, paragraphs)

### Step 3: Parse Chapter Outline

From "论文大纲" section, extract chapter structure:

```markdown
论文大纲：
第一章 绪论
  - 研究背景
  - 研究目标
第二章 相关技术
  - 知识图谱技术
  - 深度学习技术
...
```

Parse into structured format:

```json
{
  "chapter_outline": [
    {
      "number": 1,
      "title": "绪论",
      "topics": ["研究背景", "研究目标"]
    },
    {
      "number": 2,
      "title": "相关技术",
      "topics": ["知识图谱技术", "深度学习技术"]
    }
  ]
}
```

## Output Schema

```json
{
  "title": "基于知识图谱的智能问答系统设计与实现",
  "background": "随着人工智能技术的发展...",
  "status": "国内外学者在知识图谱领域...",
  "objectives": [
    "构建领域知识图谱",
    "实现智能问答系统"
  ],
  "technical_route": "采用 Neo4j 作为图数据库...",
  "expected_results": [
    "完整的知识图谱构建方案",
    "可运行的问答系统"
  ],
  "chapter_outline": [
    {"number": 1, "title": "绪论", "topics": [...]},
    {"number": 2, "title": "相关技术", "topics": [...]},
    {"number": 3, "title": "需求分析", "topics": [...]},
    {"number": 4, "title": "系统设计", "topics": [...]},
    {"number": 5, "title": "系统实现", "topics": [...]},
    {"number": 6, "title": "系统测试", "topics": [...]},
    {"number": 7, "title": "结论与展望", "topics": [...]}
  ]
}
```

## Edge Cases

### Missing Sections

If section not found:
- Set field to empty or null
- Log warning
- Proceed with available content

### Non-standard Structure

If proposal has different structure:
- Use keyword matching as fallback
- Ask user to confirm extracted outline
- Allow manual outline adjustment

### Long Content

If section >1000 chars:
- Summarize to ~500 chars
- Keep key points
- Store full content separately for reference

## Storage

Save to `thesis/proposal_info.json`.

Also update `global_state.json`:
- title → core_context.title
- objectives → core_context.objectives
- technical_route → core_context.tech_route_summary
- chapter_outline → core_context.chapter_outline