# 进度文件 Schema

## 文件位置

与 `model.docx` 同目录下的 `thesis-progress.json`。

## Schema

```json
{
  "version": "1.0",
  "template_path": "model.docx",
  "document_id": null,
  "output_path": "output/thesis-draft.docx",
  "research_topic": "",
  "current_phase": 0,
  "sections": {
    "ch1_intro": {
      "status": "pending",
      "phase": 2,
      "section_title": "1 绪论",
      "subsections": ["1.1 研究背景", "1.2 研究目的与意义", "1.3 国内外研究现状", "1.4 研究内容", "1.5 论文组织结构"],
      "word_count_target": 3000,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "ch2_tech": {
      "status": "pending",
      "phase": 3,
      "section_title": "2 相关技术",
      "subsections": ["2.1 相关理论", "2.2 相关技术", "2.3 本章小结"],
      "word_count_target": 3000,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "ch3_requirements": {
      "status": "pending",
      "phase": 4,
      "section_title": "3 需求分析",
      "subsections": ["3.1 系统需求概述", "3.2 功能需求分析", "3.3 非功能需求分析", "3.4 本章小结"],
      "word_count_target": 2500,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "ch4_design": {
      "status": "pending",
      "phase": 5,
      "section_title": "4 系统设计",
      "subsections": ["4.1 系统总体设计", "4.2 系统架构设计", "4.3 数据库设计", "4.4 详细设计", "4.5 本章小结"],
      "word_count_target": 3000,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "ch5_implementation": {
      "status": "pending",
      "phase": 6,
      "section_title": "5 系统实现与测试",
      "subsections": ["5.1 开发环境", "5.2 系统实现", "5.3 系统测试", "5.4 本章小结"],
      "word_count_target": 4000,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "ch6_conclusion": {
      "status": "pending",
      "phase": 7,
      "section_title": "6 结论与展望",
      "subsections": ["6.1 工作总结", "6.2 不足与展望"],
      "word_count_target": 1500,
      "completed_word_count": 0,
      "summary": null,
      "iterations": 0
    },
    "references": {
      "status": "pending",
      "phase": 8,
      "section_title": "参考文献"
    },
    "acknowledgement": {
      "status": "pending",
      "phase": 8,
      "section_title": "致谢"
    },
    "abstract_cn": {
      "status": "pending",
      "phase": 9,
      "section_title": "摘要"
    },
    "abstract_en": {
      "status": "pending",
      "phase": 9,
      "section_title": "Abstract"
    }
  },
  "literature_pool": [],
  "style_applied": false,
  "created_at": "",
  "last_updated": ""
}
```

## 状态流转

```
pending → writing → completed
                   ↘ revising → completed
```

## 跨会话恢复流程

1. 检查 `thesis-progress.json` 是否存在
2. 如果存在且 `current_phase > 0`：
   - 读取进度文件
   - 找到所有 `status === "writing"` 的章节（未完成的中断任务）
   - 将其重置为 `pending`
   - 重新 `docx_parse(template_path)` 加载模板
   - 如有增量保存文件（`thesis-draft-chN.docx`），优先加载该文件
3. 报告当前进度给用户
4. 从 `current_phase` 继续

## 增量保存

每完成一个 phase 后：
```json
{
  "output_path": "output/thesis-draft-ch1.docx"  // phase 2
  "output_path": "output/thesis-draft-ch2.docx"  // phase 3
  ...
}
```

恢复时加载最新的增量保存文件而非原始模板。
