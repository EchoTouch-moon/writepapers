# LLM 辅助毕业论文写作系统设计文档

> 设计日期: 2026-04-12
> 项目路径: /Users/v/new-idea/writepapers

---

## 1. 项目概述

### 1.1 目标

构建一个基于 LLM 的毕业论文辅助写作系统，核心特性：

1. **本地文献库**：使用 opendataloader-pdf 解析参考文献，无需外部 MCP 服务
2. **严格检索**：组合策略（术语锚定 → 结构约束 → 高阈值语义匹配）
3. **智能循环**：自动检测技术术语，提示用户检索并重写相关段落
4. **三源集成**：Proposal + Project Code + Literature（替换原有 rag-citation-mcp）

### 1.2 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 环境设置 | uv隔离环境 | 已完成，避免污染全局环境 |
| 文献检索策略 | 语义向量检索 | 能发现概念关联（如「知识图谱」→「本体」） |
| 检索严格性 | 组合策略（D） | 多层过滤确保结果紧密相关 |
| 系统整合方式 | 扩展 thesis-writing skill | 现有设计成熟，只替换 Literature 模块 |
| 循环触发模式 | 自动检测并提示（B） | 平衡自动化与用户控制权 |

---

## 2. 系统架构

```
writepapers/                              # 项目根目录
├── .venv/                                # uv 环境（已完成）
├── pyproject.toml                        # 项目配置（已完成）
│
├── thesis_library/                       # 本地文献库模块
│   ├── __init__.py                       # 模块入口 + Library 类
│   ├── core/
│   │   ├── pdf_processor.py              # PDF 解析（opendataloader-pdf）
│   │   ├── chunker.py                    # 智能分块（章节+段落）
│   │   ├── indexer.py                    # 向量索引（Faiss）
│   │   ├── retriever.py                  # 组合检索策略
│   │   └── metadata_extractor.py         # 元数据提取（作者/年份/标题）
│   ├── cli.py                            # CLI 入口
│   └── config.py                         # 配置管理
│
├── thesis/                                # thesis-writing skill 工作目录
│   ├── library/                           # 文献库数据
│   │   ├── papers/                        # 解析后的论文
│   │   │   ├── {cite_key}.md              # Markdown 内容
│   │   │   ├── {cite_key}.json            # 结构化数据 + bounding boxes
│   │   │   └── {cite_key}_chunks.json     # 分块数据
│   │   ├── index/                         # 向量索引
│   │   │   ├── embeddings.faiss           # Faiss 索引
│   │   │   ├── chunk_map.json             # chunk_id → source 映射
│   │   │   └── term_index.json            # 术语倒排索引
│   │   └── metadata.json                  # 文献元数据注册表
│   ├── chapters/                          # 生成的章节
│   ├── chapter_summaries/                 # 章节摘要
│   ├── global_state.json                  # 全局状态（扩展）
│   └── state.json                         # 进度状态
│
└── docs/
    └── superpowers/specs/
        └── 2026-04-12-thesis-library-design.md  # 本设计文档
```

---

## 3. 核心模块设计

### 3.1 PDF 解析模块 (`pdf_processor.py`)

**职责**：批量解析 PDF → Markdown + JSON

**关键技术点**：
- 使用 `opendataloader_pdf.convert()` 批量处理
- Hybrid 模式处理复杂表格/公式/图表（可选）
- 输出包含 bounding boxes 用于引用定位

```python
# 核心接口
class PDFProcessor:
    def process_batch(pdf_paths: list[str], use_hybrid: bool = False) -> list[dict]:
        """批量处理，返回解析结果列表"""
    
    def process_single(pdf_path: str) -> dict:
        """单文件处理"""
```

**JSON 输出结构**（来自 opendataloader-pdf）：
```json
{
  "type": "heading|paragraph|table|list|image|formula",
  "id": 42,
  "page number": 1,
  "bounding box": [left, bottom, right, top],
  "heading level": 1,  // 仅 heading
  "content": "文本内容",
  "rows": [...],        // 仅 table
  "list items": [...]   // 仅 list
}
```

### 3.2 智能分块模块 (`chunker.py`)

**职责**：将论文内容分块，保留结构信息

**分块策略**：
1. **章节级分块**：按 heading 切分，每章节作为一个大块
2. **段落级分块**：章节内按 paragraph/list/table 切分
3. **表格特殊处理**：表格作为一个完整块，保留 row/column 结构

```python
@dataclass
class Chunk:
    id: str                          # chunk_id
    cite_key: str                    # 来源论文引用键
    content: str                     # 文本内容
    chunk_type: str                  # section|paragraph|table|list
    section_title: str               # 所属章节标题
    page_number: int                 # 页码
    bounding_box: list[float]        # 坐标
    parent_id: str | None            # 所属章节 chunk_id

class Chunker:
    def chunk_paper(json_data: list[dict], cite_key: str) -> list[Chunk]:
        """将 JSON 结构数据分块"""
    
    def save_chunks(chunks: list[Chunk], output_path: str):
        """保存分块数据"""
```

**分块输出**：`{cite_key}_chunks.json`

### 3.3 向量索引模块 (`indexer.py`)

**职责**：构建向量索引 + 术语倒排索引

**技术选型**：
- Embedding 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - 支持中英文
  - 模型较小 (~118MB)，本地运行
- 索引：Faiss IndexFlatIP（内积索引，适合小规模）

```python
class Indexer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_map = {}  # chunk_id → Chunk
    
    def build_index(chunks: list[Chunk]) -> None:
        """构建向量索引"""
    
    def add_chunks(chunks: list[Chunk]) -> None:
        """增量添加 chunks"""
    
    def build_term_index(chunks: list[Chunk]) -> dict:
        """构建术语倒排索引（用于术语锚定）"""
```

**术语提取策略**：
- 从 chunk content 中提取候选术语（N-gram + 词频）
- 使用技术词典辅助（可选）
- 术语 → chunk_ids 映射

### 3.4 组合检索模块 (`retriever.py`)

**职责**：执行组合检索策略

**组合策略流程**：
```
输入：查询文本 + 当前章节类型

Step 1: 术语锚定
  → 从查询中提取技术术语
  → 术语倒排索引查找相关 chunks
  → 结果：候选集 A

Step 2: 结构约束
  → 根据章节类型限定搜索范围
  → 章节类型 → 可引用章节类型映射
  → 结果：候选集 B（A 的子集）

Step 3: 语义匹配
  → 向量检索，相似度阈值 ≥ 0.7
  → 返回 top_k 结果

输出：检索结果列表（带相似度分数和来源信息）
```

**章节类型映射表**：
```python
SECTION_TYPE_MAPPING = {
    "绪论": ["Introduction", "Background", "Related Work"],
    "相关技术": ["Methodology", "Approach", "Technical Background"],
    "需求分析": ["Requirements", "Use Case", "Functional Analysis"],
    "系统设计": ["Architecture", "Design", "Database Design", "System Design"],
    "系统实现": ["Implementation", "Implementation Details", "Technical Implementation"],
    "系统测试": ["Testing", "Experiment", "Evaluation", "Results"],
    "总结与展望": ["Conclusion", "Future Work", "Discussion"],
}
```

```python
@dataclass
class SearchResult:
    chunk: Chunk
    similarity: float                 # 相似度分数
    matched_terms: list[str]          # 匹配的术语
    section_match: bool               # 是否符合结构约束

class Retriever:
    def search(
        query: str,
        chapter_type: str,
        threshold: float = 0.7,
        top_k: int = 10
    ) -> list[SearchResult]:
        """组合检索"""
    
    def search_by_terms(terms: list[str]) -> list[Chunk]:
        """纯术语检索（Step1）"""
    
    def semantic_search(query: str, threshold: float, top_k: int) -> list[Chunk]:
        """纯语义检索（Step3）"""
```

### 3.5 元数据提取模块 (`metadata_extractor.py`)

**职责**：从解析结果中提取论文元数据

**提取策略**：
- **标题**：JSON 中第一个 `heading level: 1`
- **作者/年份/期刊**：需要用户补充或从 PDF 首页推断
- **引用键生成**：`{FirstAuthorLastName}{Year}`

```python
@dataclass
class PaperMetadata:
    cite_key: str                    # 如 Wang2023
    title: str
    authors: list[str]
    year: int
    venue: str                       # 期刊/会议名
    pdf_path: str
    md_path: str
    json_path: str

class MetadataExtractor:
    def extract(json_data: list[dict], pdf_path: str) -> PaperMetadata:
        """提取元数据"""
    
    def generate_cite_key(metadata: PaperMetadata) -> str:
        """生成引用键"""
```

**用户补充接口**：CLI 提供 `thesis-library edit-meta {cite_key}` 命令

### 3.6 CLI 模块 (`cli.py`)

**职责**：提供命令行交互接口

```bash
# 命令列表
thesis-library ingest <pdf_paths...> [--hybrid]   # 导入论文
thesis-library index                              # 重建索引
thesis-library search <query> [--chapter <type>]  # 检索
thesis-library list                               # 列出所有论文
thesis-library edit-meta <cite_key>               # 编辑元数据
thesis-library status                             # 显示状态
```

---

## 4. 与 thesis-writing Skill 集成

### 4.1 Skill 文件修改

**修改文件**：
1. `SKILL.md` - 添加 LiteratureLibrary 模块引用
2. `references/chapter-workflow.md` - 修改 Step A/C 的检索逻辑
3. `references/literature-library.md` - 新增：本地文献库使用指南
4. `templates/global-state.json` - 扩展字段

**新增字段** (`global-state.json`)：
```json
{
  "core_context": {...},
  "concept_registry": {},
  "citation_registry": {},
  "link_chain": [],
  "figure_counters": {},
  "table_counters": {},
  
  // 新增
  "library_config": {
    "library_dir": "thesis/library",
    "index_status": "built|pending",
    "papers_count": 0,
    "last_indexed": ""
  },
  "term_registry": {
    // 已发现的技术术语 → 使用次数
    "知识图谱": {"count": 3, "citations": ["Wang2023", "Li2022"]}
  }
}
```

### 4.2 Step A 修改：Prepare Chapter Context

**原逻辑**：
```python
# 使用 rag-citation-mcp
build_context(query=chapter_topics, section_type=..., top_k=10)
```

**新逻辑**：
```python
# 使用本地 LiteratureLibrary
from thesis_library import Library

library = Library(library_dir="thesis/library")
literature_relevant = library.search(
    query=f"{chapter.title} {chapter.topics}",
    chapter_type=chapter.title,  # 结构约束
    threshold=0.7,
    top_k=10
)
```

### 4.3 Step B 增加自动检测

**新增：技术术语检测步骤**

在生成章节内容后，自动检测段落中的技术术语：

```python
def detect_terms(content: str, library: Library) -> list[dict]:
    """检测内容中的技术术语"""
    # 从已建立的 term_registry 中匹配
    known_terms = library.get_known_terms()
    
    detected = []
    for term in known_terms:
        if term in content:
            # 查询是否有相关引用
            related_chunks = library.search_by_terms([term])
            if related_chunks:
                detected.append({
                    "term": term,
                    "position": content.find(term),
                    "potential_citations": [c.cite_key for c in related_chunks[:3]]
                })
    return detected
```

### 4.4 新增 Step D'：检索并重写循环

**在 Step D（Review with User）之后增加**：

```
Step D': 检索优化循环

1. 显示检测到的技术术语列表
   "检测到以下技术术语可能有引用机会：
   - 知识图谱（第2段）→ 可引用 Wang2023, Li2022
   - 三层架构（第4段）→ 可引用 Chen2020"

2. 用户选择
   [S] Skip - 跳过，继续下一章
   [R] Retrieve - 选择术语，执行检索
   [A] Auto - 自动检索所有检测到的术语

3. 执行检索
   - 调用 library.search(term, chapter_type, threshold=0.7)
   - 返回相关 chunks

4. 重写段落
   - 提供原段落 + 检索结果
   - LLM 重写，添加引用
   - 显示新旧对比

5. 用户确认
   [Y] Accept - 接受重写
   [N] Reject - 保持原内容
   [M] Modify - 手动修改

6. 循环直到用户选择 [S]
```

### 4.5 Step C 修改：引用验证

**原逻辑**：使用 rag-citation-mcp 验证

**新逻辑**：使用本地 `metadata.json` 验证

```python
def verify_citation(cite_key: str, metadata_path: str) -> dict:
    """验证引用是否存在于本地库"""
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    if cite_key in metadata:
        return {
            "status": "verified",
            "details": metadata[cite_key]
        }
    else:
        return {
            "status": "not_found",
            "suggestion": f"请确认 '{cite_key}' 是否已导入文献库"
        }
```

---

## 5. 数据流设计

### 5.1 文献导入流程

```
用户输入：pdf_paths (PDF 文件列表)

┌─────────────────┐
│  PDFProcessor   │  opendataloader_pdf.convert()
│  process_batch  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ MetadataExtractor│  提取标题，生成 cite_key
│ extract         │  用户补充作者/年份
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Chunker       │  JSON → chunks
│   chunk_paper   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Indexer       │  构建向量索引 + 术语索引
│   build_index   │
└─────────────────┘
        │
        ▼
thesis/library/
├── papers/{cite_key}.md
├── papers/{cite_key}.json
├── papers/{cite_key}_chunks.json
├── index/embeddings.faiss
├── index/chunk_map.json
├── index/term_index.json
└── metadata.json
```

### 5.2 章节写作流程（含检索循环）

```
Step A: Prepare Context
├── 加载 global_state.json
├── 加载前序章节摘要
├── 加载 proposal 相关内容
├── 加载 code 分析报告
└── 调用 Library.search() 加载文献上下文

Step B: Write Chapter
└── LLM 生成章节内容（带引用占位符）

Step B': Detect Terms (新增)
├── 检测内容中的技术术语
├── 查询术语索引，找潜在引用
└── 输出检测报告

Step C: Verify Citations
├── 格式验证 [\w+\d{4}]
├── 存在性验证（本地 metadata.json）
└── 输出验证报告

Step D: Review with User
├── 显示章节 + 引用 + 检测报告
└── 用户选择 [A] [M] [R] [S]

Step D': Retrieve & Rewrite (新增)
├── 用户选择术语执行检索
├── 显示检索结果
├── LLM 重写段落
└── 用户确认 [Y] [N] [M]

Step E: Save Chapter
├── 保存章节文件
├── 提取摘要
├── 更新 global_state.json
│   ├── concept_registry
│   ├── citation_registry
│   └── term_registry (新增)
└── 更新 state.json
```

---

## 6. 配置设计

### 6.1 项目配置 (`pyproject.toml`)

```toml
[project]
name = "writepapers"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opendataloader-pdf[hybrid]>=2.2.1",
    "faiss-cpu>=1.7.4",
    "sentence-transformers>=2.2.0",
]

[project.scripts]
thesis-library = "thesis_library.cli:main"
```

### 6.2 文献库配置 (`thesis_library/config.py`)

```python
from dataclasses import dataclass

@dataclass
class LibraryConfig:
    library_dir: str = "thesis/library"
    papers_dir: str = "thesis/library/papers"
    index_dir: str = "thesis/library/index"
    metadata_file: str = "thesis/library/metadata.json"
    
    # 分块配置
    max_chunk_size: int = 500      # 最大 chunk 字数
    min_chunk_size: int = 100      # 最小 chunk 字数
    
    # 索引配置
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    index_type: str = "FlatIP"     # Faiss 索引类型
    
    # 检索配置
    similarity_threshold: float = 0.7
    default_top_k: int = 10
    
    # 术语配置
    min_term_freq: int = 2         # 最小词频才算术语
    max_term_length: int = 10      # 最大术语长度
```

---

## 7. 错误处理设计

### 7.1 PDF 解析错误

```python
class PDFParseError(Exception):
    """PDF 解析失败"""
    def __init__(self, pdf_path: str, reason: str):
        self.pdf_path = pdf_path
        self.reason = reason

# 处理策略
try:
    result = pdf_processor.process_single(pdf_path)
except PDFParseError as e:
    logger.error(f"解析失败: {e.pdf_path}, 原因: {e.reason}")
    # 记录到失败列表，继续处理其他文件
```

### 7.2 索引构建错误

```python
class IndexBuildError(Exception):
    """索引构建失败"""
    def __init__(self, reason: str):
        self.reason = reason

# 处理策略
try:
    indexer.build_index(chunks)
except IndexBuildError as e:
    logger.error(f"索引构建失败: {e.reason}")
    # 回退到纯关键词检索
```

### 7.3 检索无结果

```python
# 当检索返回空结果时的处理
results = retriever.search(query, chapter_type)
if not results:
    logger.info(f"未找到相关文献: {query}")
    # 提示用户可能需要导入更多相关文献
    # 或降低阈值重新检索
```

---

## 8. 测试设计

### 8.1 单元测试

```python
# tests/test_chunker.py
def test_chunk_paper_basic():
    """测试基本分块"""
    
def test_chunk_table_preservation():
    """测试表格结构保留"""

# tests/test_indexer.py
def test_build_index():
    """测试索引构建"""

def test_add_chunks_incremental():
    """测试增量添加"""

# tests/test_retriever.py
def test_search_combined_strategy():
    """测试组合检索策略"""

def test_search_with_threshold():
    """测试阈值过滤"""
```

### 8.2 集成测试

```python
# tests/test_library.py
def test_full_pipeline():
    """测试完整流程：导入 → 分块 → 索引 → 检索"""
    
def test_search_for_chapter():
    """测试为特定章节检索文献"""
```

---

## 9. 依赖清单

### 9.1 Python 依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| opendataloader-pdf[hybrid] | ≥2.2.1 | PDF 解析 |
| faiss-cpu | ≥1.7.4 | 向量索引 |
| sentence-transformers | ≥2.2.0 | Embedding 模型 |
| torch | 已安装 | sentence-transformers 依赖 |
| numpy | 已安装 | Faiss 依赖 |
| pydantic | 已安装 | 数据验证 |

### 9.2 系统依赖

| 依赖 | 版本 | 状态 |
|------|------|------|
| Java | ≥11 | ✅ 已安装 (Java 17) |
| Python | ≥3.11 | ✅ 已安装 (3.11.15) |

---

## 10. 实施计划概要

### Phase 1: 核心模块实现（3-4 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| PDF 解析模块 | `pdf_processor.py` | P0 |
| 智能分块模块 | `chunker.py` | P0 |
| 元数据提取模块 | `metadata_extractor.py` | P0 |

### Phase 2: 索引与检索（2-3 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 向量索引模块 | `indexer.py` | P0 |
| 组合检索模块 | `retriever.py` | P0 |

### Phase 3: CLI 与集成（1-2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| CLI 模块 | `cli.py` | P1 |
| thesis-writing skill 扩展 | SKILL.md 等 | P1 |

### Phase 4: 测试与调试（1天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 单元测试 | `tests/` | P1 |
| 集成测试 | `tests/test_library.py` | P1 |

---

## 11. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| Embedding 模型加载慢 | 首次启动延迟 | 模型预加载 + 缓存 |
| 中文论文术语提取不准确 | 检索质量下降 | 提供用户自定义术语词典 |
| Faiss 索引内存占用大 | 大规模论文库性能问题 | 使用 IndexIVFFlat 分段索引 |
| PDF 结构识别错误 | 分块混乱 | Hybrid 模式 + 用户手动修正 |

---

## 附录 A：术语词典示例

```json
{
  "知识图谱": {
    "aliases": ["Knowledge Graph", "KG", "语义网络"],
    "related_terms": ["本体", "实体", "关系", "RDF", "SPARQL"]
  },
  "三层架构": {
    "aliases": ["Three-tier Architecture", "MVC"],
    "related_terms": ["表示层", "业务层", "数据层", "分层设计"]
  }
}
```

---

## 附录 B：检索结果输出格式

```json
{
  "query": "知识图谱构建方法",
  "chapter_type": "系统设计",
  "results": [
    {
      "chunk_id": "Wang2023_section3_para2",
      "cite_key": "Wang2023",
      "similarity": 0.85,
      "matched_terms": ["知识图谱", "构建方法"],
      "section_match": true,
      "content": "知识图谱的构建过程包括实体识别、关系抽取和知识融合三个主要步骤...",
      "section_title": "Methodology",
      "page_number": 5,
      "bounding_box": [72.0, 400.0, 540.0, 450.0]
    }
  ],
  "total": 5,
  "threshold_used": 0.7
}
```