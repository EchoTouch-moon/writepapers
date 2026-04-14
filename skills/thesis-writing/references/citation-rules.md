# Citation Rules and Verification

Citation format, embedding rules, and verification procedures.

## Citation Format

### Inline Format

```markdown
知识图谱是一种结构化的语义知识库...[Wang2023]
```

Pattern: `[AuthorYear]` where:
- Author: First author's last name (capitalized)
- Year: 4-digit year

### Paragraph Annotation

After paragraph with citations:

```markdown
> 引用: 王某某 (2023). 知识图谱构建方法研究. 计算机学报.
```

### Full Reference Format (GB/T 7714)

```
[1] 作者.文献题目[文献类型].期刊名,年份,卷(期):页码.DOI.
[2] Author.Title[J].Journal,Year,Vol(Issue):Pages.
```

Type markers:
- [J] - Journal article
- [D] - Dissertation
- [M] - Monograph
- [C] - Conference paper

## Verification Levels

### Level 1: Format Verification

Check citation pattern with regex:

```python
import re

def verify_format(content):
    pattern = r'\[\w+\d{4}\]'
    citations = re.findall(pattern, content)
    
    errors = []
    for cite in citations:
        # Check: first letter capitalized
        if cite[1].islower():
            errors.append({
                "original": cite,
                "corrected": f"[{cite[1].upper()}{cite[2:]}]",
                "error": "Author name should be capitalized"
            })
        # Check: 4-digit year
        year_match = re.search(r'\d{4}', cite)
        if not year_match:
            errors.append({
                "original": cite,
                "error": "Missing 4-digit year"
            })
    
    return citations, errors
```

### Level 2: Existence Verification

Query rag-citation-mcp for each citation:

```python
def verify_existence(citations, library_name="thesis"):
    results = []
    for cite in citations:
        # Parse AuthorYear
        author = cite[1:-5]  # Extract author
        year = cite[-5:-1]   # Extract year
        
        # Query MCP
        matches = search_citations(keyword=author, paper_type=None)
        
        # Find match with same year
        found = False
        for m in matches:
            if str(m["year"]) == year:
                found = True
                results.append({
                    "cite_key": cite,
                    "status": "verified",
                    "details": m
                })
                break
        
        if not found:
            results.append({
                "cite_key": cite,
                "status": "not_found",
                "suggestion": f"请确认 '{cite}' 是否存在于文献库"
            })
    
    return results
```

### Level 3: Content Verification (Optional)

For high-risk citations, verify content alignment:

```python
def verify_content(cite_key, generated_text, library_name="thesis"):
    # Retrieve original chunk
    chunks = build_context(query=cite_key, top_k=3)
    
    # Compare semantic alignment
    # (This is expensive, only for key citations)
    
    return {
        "cite_key": cite_key,
        "alignment": "high/medium/low",
        "warning": "..." if alignment == "low" else None
    }
```

## Verification Report Format

```json
{
  "chapter": "02-相关技术",
  "verification_time": "2026-04-11T15:30:00",
  "total_citations": 5,
  "valid": [
    {
      "cite_key": "Wang2023",
      "status": "verified",
      "details": {
        "authors": ["王某某"],
        "year": 2023,
        "title": "知识图谱构建方法研究",
        "venue": "计算机学报"
      }
    }
  ],
  "warnings": [
    {
      "cite_key": "Unknown2024",
      "status": "not_found",
      "suggestion": "请确认文献来源，可能未导入文献库"
    }
  ],
  "format_errors": [
    {
      "original": "[wang2023]",
      "corrected": "[Wang2023]",
      "location": "第3段第2行",
      "error": "Author name should be capitalized"
    }
  ]
}
```

## Hallucination Prevention

### Red Flags

1. Citation not in library → potential hallucination
2. Citation year doesn't match → verify manually
3. Multiple citations for same concept → check consistency
4. Citation in unsupported context → verify alignment

### Auto-Correction

When format error detected:
1. Auto-correct capitalization
2. Prompt user for missing citations
3. Offer to remove unverifiable citations

### User Confirmation

For warnings, ask user:
- "Citation [Unknown2024] not found. Skip or provide source?"
- "Citation [wang2023] has format error. Auto-correct to [Wang2023]?"

## Citation Placement Rules

### ❌ Prohibited Patterns (Anti-Concentration)

**Problem: Citation Clustering**
```markdown
# ❌ WRONG - Citations concentrated at paragraph end
知识图谱技术近年来发展迅速，许多学者进行了相关研究[Wang2023][Li2022][Zhang2021]。
系统架构设计采用了分层思想[Chen2020][Liu2019]。
```

**Problems with this pattern:**
- Readers cannot identify which citation supports which claim
- Appears as lazy "citation dumping"
- Violates academic writing conventions

### ✅ Correct Patterns (Contextual Placement)

**Pattern 1: Immediate Citation**
```markdown
# ✅ CORRECT - Citation immediately follows relevant claim
知识图谱是一种结构化的语义知识库，用于存储实体及其关系[Wang2023]。
基于图神经网络的知识图谱推理方法能够有效处理复杂关系[Liu2022]。
```

**Pattern 2: First-Mention Rule**
```markdown
# ✅ CORRECT - Cite when concept first appears
本系统采用知识图谱技术构建数据模型[Wang2023]。知识图谱的构建过程包括...
（后续复述知识图谱时无需重复引用[Wang2023]）
```

**Pattern 3: Comparative Citation**
```markdown
# ✅ CORRECT - Different viewpoints cite different sources
传统关系数据库在处理复杂关联时效率较低[Zhang2020]，
而图数据库通过原生图结构提供了更高效的查询能力[Chen2021]。
```

### Placement Rules Summary

| Rule | Description |
|------|-------------|
| **就近原则** | 引用紧跟被支持的论述，放在句子末尾 |
| **首次引用** | 技术概念首次出现时引用，后续不重复 |
| **一观点一引用** | 一个观点最多引用1-2篇最相关文献 |
| **对比分引** | 不同观点分别引用各自来源 |
| **禁堆砌** | 禁止一个位置同时引用3篇以上 |

### Verification Check for Placement

```python
def check_citation_placement(content):
    """Check for citation clustering patterns."""
    # Find paragraphs with multiple consecutive citations
    pattern = r'\[\w+\d{4}\]\s*\[\w+\d{4}\]'
    clusters = re.findall(pattern, content)
    
    warnings = []
    if clusters:
        warnings.append({
            "type": "citation_cluster",
            "message": f"发现 {len(clusters)} 处引用堆砌，请分散引用位置",
            "suggestion": "每个观点单独引用，避免集中罗列"
        })
    
    # Check citations at paragraph ends without prior claims
    # ... additional checks
    
    return warnings
```

## Citation Registry

Track all used citations in `global_state.json`:

```json
{
  "citation_registry": {
    "Wang2023": {
      "authors": ["王某某"],
      "year": 2023,
      "title": "知识图谱构建方法研究",
      "venue": "计算机学报",
      "used_in_chapters": [2, 4, 5],
      "contexts": ["知识图谱概述", "系统架构", "实现细节"]
    }
  }
}
```

Purpose:
- Prevent duplicate citations in same context
- Track citation reuse across chapters
- Generate final reference list