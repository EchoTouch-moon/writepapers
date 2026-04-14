# Figure Suggestion System

Automatic figure suggestions based on chapter type.

## Trigger Points

1. **Pre-judgment** (prepare_chapter): Based on chapter type
2. **Identification** (write_chapter): LLM inserts markers
3. **Review** (review_chapter): User processes suggestions

## Figure Types

| Type | Description | Typical Chapters |
|------|-------------|------------------|
| 流程图 | Algorithm/business flow | 系统实现, 系统设计 |
| 架构图 | System architecture | 系统设计 ⭐ |
| ER图 | Data model diagram | 系统设计 ⭐ |
| 时序图 | Interaction sequence | 系统设计 |
| 数据表 | Test results, comparison | 系统测试 ⭐ |
| 统计图 | Performance curves | 系统测试 |
| 截图 | Interface screenshots | 系统实现 |

⭐ = Required for typical undergraduate thesis.

## Suggestion Format

### HTML Comment Marker

```html
<!-- FIGURE_SUGGESTION_START -->
**图表建议 #N**
- **类型**: 系统架构图
- **建议位置**: 4.1 总体设计 开头
- **内容描述**: 展示系统三层架构：
  - 前端层: Vue.js + Element UI
  - 后端层: FastAPI + 知识图谱服务
  - 数据层: Neo4j + PostgreSQL
- **参考实现**: src/api/, src/core/
- **生成方式**: draw.io / Mermaid
<!-- FIGURE_SUGGESTION_END -->
```

### End-of-Chapter Summary Table

```markdown
## 📊 本章图表建议汇总

| 序号 | 类型 | 建议位置 | 内容摘要 | 生成方式 |
|------|------|----------|----------|----------|
| 1 | 系统架构图 | 4.1 | 三层架构 | draw.io |
| 2 | ER图 | 4.2 | 数据模型 | Mermaid |
| 3 | 流程图 | 4.3 | 核心算法 | draw.io |
```

## Numbering Convention

- 图: `图X.Y` (e.g., 图4.1 = Chapter 4, Figure 1)
- 表: `表X.Y` (e.g., 表5.2 = Chapter 5, Table 2)

### Caption Placement

- 图题注: Below figure, centered
- 表题注: Above table, centered

### Reference in Text

- ✅ 正确: "如图4.1所示"
- ❌ 错误: "如下图所示"

## User Processing Options

| Action | Result |
|--------|--------|
| [G] Generate | Call frontend-design skill, create figure |
| [I] Insert | User provides figure file, copy to thesis/figures/ |
| [S] Skip | Mark as "待补充", record in state.json |
| [D] Delete | Remove suggestion |

## Storage

```
thesis/
├── figures/
│   ├── arch-diagram.drawio
│   ├── arch-diagram.png
│   └── er-diagram.png
└── tables/
    └── test-results.md    # Markdown table
```

## Counter Management

In `global_state.json`:

```json
{
  "figure_counters": {"chapter_04": 3, "chapter_05": 0},
  "table_counters": {"chapter_04": 2, "chapter_05": 0}
}
```

Update after each chapter save.