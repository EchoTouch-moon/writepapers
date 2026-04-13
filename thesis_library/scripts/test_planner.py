#!/usr/bin/env python3
"""Manual test script for Thesis Section Planner.

Run this script after setting DASHSCOPE_API_KEY:
    export DASHSCOPE_API_KEY="your-key-here"
    uv run thesis_library/scripts/test_planner.py
"""

import os
import sys

from thesis_library import Library
from thesis_library.config import LibraryConfig
from thesis_library.generator import (
    SectionPlanner,
    plan_and_assemble,
    create_planner,
    LLMConfig,
    assemble_prompt,
    verify_citations,
    LLMClient,
)


def test_planner():
    """Test complete planner + generation flow."""
    # Check API key
    config = LibraryConfig()
    api_key = config.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='your-key-here'")
        sys.exit(1)

    print("=== Step 1: Initialize Library ===")
    library = Library()
    print(f"Library loaded, {len(library.indexer.chunk_map)} chunks available")

    print("\n=== Step 2: Section Planning ===")
    planner = create_planner(library)

    section = "系统设计"
    query = "介绍角色扮演智能体的记忆机制，并对比长上下文与外挂知识库的优劣"

    plan, results = planner.plan_and_retrieve(section, query, top_k=8)

    print(f"\n🤔 规划思路: {plan.rationale}")
    print(f"🔍 子查询: {plan.sub_queries}")
    print(f"📝 关键术语: {plan.key_terms}")
    print(f"📚 章节建议: {plan.chapter_suggestions}")

    print(f"\n=== Step 3: Retrieved Results ({len(results)} chunks) ===")
    for r in results[:5]:
        print(f"  [{r.chunk.id}] Sim: {r.similarity:.2f}")

    print("\n=== Step 4: Assemble Context ===")
    context = assemble_prompt(query, results, section)
    print(f"组装完成，预估 {context.token_estimate} tokens")

    print("\n=== Step 5: LLM Generation ===")
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

    print("\n=== Step 6: Citation Verification ===")
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

    # Summary
    print("\n📊 测试总结:")
    print(f"  - RetrievalPlan JSON 解析: ✅")
    print(f"  - 多路径检索: {len(plan.sub_queries)} sub_queries + {len(plan.key_terms)} terms")
    print(f"  - 结果合并去重: {len(results)} unique chunks")
    print(f"  - 引用验证: {len(report.valid)}/{len(report.citations)} valid")


def test_plan_and_assemble():
    """Test convenience function."""
    print("\n=== Test plan_and_assemble() ===")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("跳过（需要 API key）")
        return

    library = Library()
    llm_config = LLMConfig(api_key=api_key)

    plan, context = plan_and_assemble(
        library,
        section="研究背景",
        query="大语言模型的发展历程与应用场景",
        top_k=5,
        llm_config=llm_config,
    )

    print(f"Plan rationale: {plan.rationale[:50]}...")
    print(f"Context tokens: {context.token_estimate}")
    print("✅ plan_and_assemble() works")


if __name__ == "__main__":
    test_planner()
    # test_plan_and_assemble()  # Optional second test