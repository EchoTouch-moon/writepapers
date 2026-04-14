# Example Thesis-Writing Session

Sample interactive session demonstrating the workflow.

## Phase 0: Initialization

```
User: /thesis-writing --proposal ./proposal.md --papers ./papers/ --code ./my-project/

Claude: 开始毕业论文写作流程...

### Step 1: 解析开题报告
已提取:
- 题目: 基于知识图谱的智能问答系统设计与实现
- 目标: [构建领域知识图谱, 实现智能问答系统]
- 技术路线: Neo4j + FastAPI + Vue.js
- 章节大纲: 7章

### Step 2: 导入文献
已导入 15 篇 PDF 文献到文献库

### Step 3: 分析项目代码
项目结构:
- src/api/ (FastAPI 后端)
- src/kg/ (知识图谱构建)
- src/web/ (Vue.js 前端)
技术栈: Python, FastAPI, Neo4j, Vue.js
核心模块: kg_builder, qa_engine, web_app

### Step 4: 论文大纲确认

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 1 | 绪论 | 背景、现状、目标 |
| 2 | 相关技术 | 知识图谱、深度学习 |
| 3 | 需求分析 | 功能需求、性能需求 |
| 4 | 系统设计 | 架构、数据模型 |
| 5 | 系统实现 | 核心模块实现 |
| 6 | 系统测试 | 测试用例、结果 |
| 7 | 结论与展望 | 总结、改进方向 |

请确认大纲是否正确？可调整章节顺序或内容。

User: 确认，开始写作

Claude: 开始第一章写作...
```

## Phase 1: Chapter Writing (示例: 第四章)

```
Claude: 正在准备第四章"系统设计"的上下文...

### 章节上下文
- 开题报告相关: 技术路线部分
- 代码相关: architecture, kg_builder模块
- 文献检索: 检索到5篇相关文献
- 前章衔接: 第三章提到"将在第四章详细介绍设计方案"
- 图表预判: 系统架构图(必需), ER图(必需)

### 写作提纲
4.1 总体设计 - 系统三层架构 [Wang2023]
4.2 数据库设计 - Neo4j图数据库设计 [Li2022]
4.3 接口设计 - RESTful API设计

生成内容中...

---

## 4 系统设计

### 4.1 总体设计

本系统采用三层架构设计...[Wang2023]
> 引用: 王某某 (2023). 系统架构设计方法. 计算机学报.

<!-- FIGURE_SUGGESTION: 系统架构图 | 4.1开头 | 展示三层结构 -->

...

### 📊 本章图表建议汇总

| 序号 | 类型 | 建议位置 | 内容摘要 |
|------|------|----------|----------|
| 1 | 系统架构图 | 4.1 | 三层架构结构 |
| 2 | ER图 | 4.2 | 知识图谱实体关系 |

### 🔍 引用验证报告

总引用数: 5
有效: [Wang2023, Li2022, Zhang2021]
警告: 无

---

请选择操作:
[A] 接受  [M] 修改  [R] 重写  [S] 暂停  [G] 生成图表

User: A

Claude: 已保存第四章到 thesis/chapters/04-系统设计.md

提取章节摘要:
- 核心概念: 三层架构, RESTful API
- 关键引用: Wang2023, Li2022
- 贡献: 设计了系统总体架构和数据库结构
- 衔接点: "将在第五章介绍系统实现细节"

继续第五章写作...
```

## Phase 2: Finalize

```
Claude: 所有章节已完成！

### 统计
- 总章节: 7章
- 总引用: 23条
- 待补充图表: 5个

### 输出文件
thesis/
├── chapters/
│   ├── 01-绪论.md
│   ├── 02-相关技术.md
│   ├── ...
│   └── 07-结论与展望.md
├── references/
│   └── all-refs.md
├── figures/
│   └── (待用户补充)
├── global_state.json
└── state.json

是否需要:
1. 导出参考文献 (GB/T 7714格式)?
2. 合并导出 Word 文档?

User: 导出参考文献

Claude: 已生成 thesis/references/all-refs.md (GB/T 7714格式)

论文写作完成！
```