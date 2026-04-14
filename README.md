# WritePapers

基于本地文献库的 LLM 辅助论文写作工具。将 PDF 文献导入本地向量检索库，利用大语言模型生成带引用追踪的论文章节。

## 功能特点

- **PDF 文献入库** — 自动解析 PDF、提取元数据、智能分块、章节分类
- **向量语义检索** — 基于 FAISS + sentence-transformers 的混合检索（术语锚定 + 结构约束 + 语义匹配）
- **LLM 论文生成** — 调用 Qwen 等大模型，基于检索到的文献内容生成论文章节，自动追踪引用
- **检索质量评估** — 内置 RAG 评估框架，支持 Recall@K、MRR@K 指标与回归对比

## 系统架构

```
PDF 文献 → 解析分块 → 向量索引 + 术语索引
                            ↓
                     检索策略（三步法）
                  术语锚定 → 结构约束 → 语义匹配
                            ↓
                   LLM 生成（带引用追踪）
```

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/EchoTouch-moon/writepapers.git
cd writepapers

# 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install -e .
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# DASHSCOPE_API_KEY=your_key_here
```

API Key 获取方式：[阿里云百炼平台](https://bailian.console.alibabacloud.com/)

### 使用方式

#### 命令行工具

```bash
# 查看帮助
thesis-library --help

# 导入 PDF 文献
thesis-library ingest paper1.pdf paper2.pdf paper3.pdf

# 重建索引
thesis-library index

# 语义检索
thesis-library search "基于角色扮演的对话系统"

# 按章节类型过滤检索
thesis-library search "记忆机制优化" --chapter-type METHODOLOGY

# 查看文献库状态
thesis-library status

# 列出所有文献
thesis-library list

# 查看术语表
thesis-library terms
```

#### Python API

```python
from thesis_library import Library, LibraryConfig

# 初始化文献库
config = LibraryConfig(library_dir="thesis/library")
library = Library(config)

# 导入 PDF
metadata_list = library.ingest(["paper1.pdf", "paper2.pdf"])

# 检索
results = library.search("数字孪生对话系统设计", top_k=5)

for r in results:
    print(f"[{r.chunk.cite_key}] {r.similarity:.2f} - {r.chunk.section_title}")
    print(f"  {r.chunk.content[:100]}...")
```

#### 生成论文章节

```python
from thesis_library.generator import LLMClient, LLMConfig
from thesis_library.generator.context_assembler import ContextAssembler
from thesis_library.generator.citation_validator import CitationValidator

# 初始化组件
library = Library()
assembler = ContextAssembler(library)
validator = CitationValidator()

# 组装上下文
context = assembler.assemble(
    query="基于大语言模型的对话系统",
    chapter_type="INTRODUCTION",
)

# 生成内容
client = LLMClient(LLMConfig())
content = client.generate_with_retry(
    prompt=context.prompt,
    system_prompt=context.system_prompt,
)

# 验证引用
citations = validator.extract_and_validate(content, context.used_chunks)
```

## 检索策略

检索采用三步级联策略：

| 步骤 | 方法 | 说明 |
|------|------|------|
| 1. 术语锚定 | 倒排索引 | 从查询中提取中英文术语，通过术语索引缩小候选范围 |
| 2. 结构约束 | 章节类型映射 | 根据当前撰写的论文章节类型，过滤不相关的文本块 |
| 3. 语义匹配 | FAISS 向量检索 | 对候选集进行语义相似度排序，返回 Top-K 结果 |

## 评估框架

内置 RAG 检索质量评估工具：

```bash
# 运行评估
thesis-library eval --k 5

# 添加测试用例（交互式）
thesis-library eval-add

# 保存为基线
thesis-library eval --save-baseline

# 详细输出
thesis-library eval -v
```

评估指标：
- **Recall@K** — 前 K 个结果中包含目标文献的比例
- **MRR@K** — 目标文献在检索结果中的排名倒数均值
- **Noise Rate** — 无关结果占比

## 项目结构

```
writepapers/
├── thesis_library/           # 核心库
│   ├── __init__.py           # Library 主接口
│   ├── cli.py                # CLI 入口
│   ├── config.py             # 配置（LibraryConfig, ChapterType）
│   ├── core/                 # 文献处理核心
│   │   ├── chunker.py        # PDF 内容分块
│   │   ├── indexer.py        # FAISS 向量索引 + 术语索引
│   │   ├── retriever.py      # 混合检索策略
│   │   ├── pdf_processor.py  # PDF 解析
│   │   ├── metadata_extractor.py  # 元数据提取
│   │   ├── chapter_classifier.py  # LLM 章节分类
│   │   └── smoother.py       # 滑动窗口平滑
│   ├── generator/            # 论文生成
│   │   ├── llm_client.py     # LLM API 客户端
│   │   ├── context_assembler.py    # 上下文组装
│   │   ├── citation_validator.py   # 引用验证
│   │   └── section_planner.py      # 检索策略规划
│   └── evaluator/            # RAG 评估
│       ├── evaluator.py      # 评估引擎
│       ├── metrics.py        # 评估指标
│       └── report.py         # 报告生成
├── thesis/                   # 论文数据（gitignore 大文件）
├── pyproject.toml
└── README.md
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `faiss-cpu` | 向量相似度检索 |
| `sentence-transformers` | 文本嵌入模型（默认 MiniLM） |
| `opendataloader-pdf` | PDF 解析 |
| `tenacity` | API 调用重试 |

嵌入模型默认使用 `paraphrase-multilingual-MiniLM-L12-v2`（384 维，支持中英文，Mac 兼容）。

## 许可证

[MIT License](LICENSE)
