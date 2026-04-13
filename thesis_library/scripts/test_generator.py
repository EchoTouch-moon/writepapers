#!/usr/bin/env python3
"""Manual test script for Thesis Generator.

Run this script after setting DASHSCOPE_API_KEY:
    export DASHSCOPE_API_KEY="your-key-here"
    uv run thesis_library/scripts/test_generator.py
"""

import os
import sys

from thesis_library import Library
from thesis_library.config import LibraryConfig
from thesis_library.generator import (
    assemble_prompt,
    LLMClient,
    LLMConfig,
    verify_citations,
    extract_citations,
)


def test_generator():
    """Test complete generation flow."""
    # Check API key (from config or environment)
    config = LibraryConfig()
    api_key = config.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='your-key-here'")
        sys.exit(1)

    print("=== Step 1: Search ===")
    library = Library()
    results = library.search("角色扮演智能体的研究方法", top_k=3, threshold=0.7)
    print(f"找到 {len(results)} 个结果:")
    for r in results:
        print(f"  [{r.chunk.id}] Sim: {r.similarity:.2f}")

    print("\n=== Step 2: Assemble Prompt ===")
    context = assemble_prompt(
        query="撰写关于角色扮演智能体研究方法的段落，约200字",
        results=results,
        section="研究方法",
        max_tokens=2000,
    )
    print(f"组装完成，预估 {context.token_estimate} tokens")
    print("\n--- Prompt Preview ---")
    print(context.prompt[:800])
    print("...")

    print("\n=== Step 3: LLM Generation ===")
    llm_config = LLMConfig(api_key=api_key, model="qwen-plus")
    client = LLMClient(llm_config)
    try:
        generated = client.generate(
            prompt=context.prompt,
            temperature=0.7,
            max_tokens=512,
        )
        print(f"生成完成，{len(generated)} 字")
        print("\n--- Generated Text ---")
        print(generated)
    except Exception as e:
        print(f"❌ LLM API 调用失败: {e}")
        sys.exit(1)

    print("\n=== Step 4: Citation Verification ===")
    # Build available chunks dict
    available_chunks = {r.chunk.id: r.chunk for r in results}
    report = verify_citations(generated, available_chunks)

    print(f"提取引用: {len(report.citations)} 个")
    print(f"有效引用: {len(report.valid)} 个")
    print(f"无效引用: {len(report.invalid)} 个")
    print(f"覆盖率: {report.coverage:.1%}")

    if report.citations:
        print("\n引用列表:")
        for cite in report.citations:
            status = "✅" if cite in report.valid else "❌"
            print(f"  {status} [{cite}]")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_generator()