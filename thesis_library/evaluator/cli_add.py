"""Interactive CLI for adding test cases."""

import json
import logging
from pathlib import Path

from thesis_library import Library
from thesis_library.evaluator.test_cases import QueryType, TestCase

logger = logging.getLogger(__name__)


def run_eval_add(
    library: Library,
    test_cases_path: str,
) -> int:
    """Interactive test case creation.

    Flow:
    1. User enters query
    2. System searches and shows top results
    3. User selects expected chunks (or types 'manual' for fallback)
    4. User specifies query_type, chapter_type, notes
    5. Save to test_cases.json

    Args:
        library: Library instance
        test_cases_path: Path to test_cases.json

    Returns:
        0 on success, 1 on failure
    """
    # Load existing test cases to determine next ID
    path = Path(test_cases_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_cases: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        existing_cases = data.get("test_cases", [])

    next_id = f"TC-{len(existing_cases) + 1:03d}"

    print("\n=== Add Test Case ===")
    print(f"ID: {next_id}")

    # Step 1: Enter query
    query = input("Query: ").strip()
    if not query:
        print("Query required. Aborted.")
        return 1

    # Step 2: Search with default params
    print("\nSearching...")
    results = library.search(query, top_k=5)

    if results:
        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            content_preview = r.chunk.content[:80].replace("\n", " ")
            print(f"  {i}. [{r.chunk.id}] \"{content_preview}...\"")
    else:
        print("No results found.")

    # Step 3: Select expected chunks
    print("\nSelect expected chunks:")
    print("  - Enter numbers (comma-separated): 1,2,3")
    print("  - Or type 'manual' to enter chunk IDs directly")
    print("  - Or type 'search' to search with larger top_k")

    selection = input("Selection: ").strip().lower()

    expected_chunk_ids: list[str] = []

    if selection == "manual":
        # Fallback: manual chunk ID entry
        manual_ids = input("Enter chunk IDs (comma-separated): ").strip()
        expected_chunk_ids = [id.strip() for id in manual_ids.split(",") if id.strip()]

    elif selection == "search":
        # Expand search
        top_k = input("Search with larger top_k (default 20): ").strip()
        top_k = int(top_k) if top_k else 20
        results = library.search(query, top_k=top_k)

        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            content_preview = r.chunk.content[:60].replace("\n", " ")
            print(f"  {i}. [{r.chunk.id}] \"{content_preview}...\"")

        selection = input("Selection (numbers): ").strip()
        indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
        expected_chunk_ids = [results[i - 1].chunk.id for i in indices if 0 < i <= len(results)]

    elif selection:
        # Parse selection as numbers
        indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
        expected_chunk_ids = [results[i - 1].chunk.id for i in indices if 0 < i <= len(results)]

    if not expected_chunk_ids:
        print("No expected chunks selected. Aborted.")
        return 1

    print(f"\nExpected chunks: {expected_chunk_ids}")

    # Step 4: Query type
    print("\nQuery type:")
    print("  1. exact_term")
    print("  2. fuzzy_concept")
    print("  3. multi_cond")
    print("  4. cross_para")

    type_selection = input("Select (1-4): ").strip()
    query_type_map = {
        "1": QueryType.EXACT_TERM,
        "2": QueryType.FUZZY_CONCEPT,
        "3": QueryType.MULTI_CONDITION,
        "4": QueryType.CROSS_PARAGRAPH,
    }
    query_type = query_type_map.get(type_selection, QueryType.EXACT_TERM)

    # Step 5: Chapter type (optional)
    chapter_type = input("Chapter type constraint (optional, press Enter to skip): ").strip()
    chapter_type = chapter_type if chapter_type else None

    # Step 6: Notes (optional)
    notes = input("Notes (optional): ").strip()
    notes = notes if notes else None

    # Step 7: Threshold override (optional)
    threshold_input = input(
        f"Threshold override (current: {library.config.similarity_threshold}, press Enter to keep): "
    ).strip()
    threshold = float(threshold_input) if threshold_input else None

    # Create test case
    test_case = TestCase(
        id=next_id,
        query=query,
        query_type=query_type,
        expected_chunk_ids=expected_chunk_ids,
        chapter_type=chapter_type,
        threshold=threshold,
        notes=notes,
    )

    # Save to file
    existing_cases.append(test_case.to_dict())

    data = {
        "test_cases": existing_cases,
        "metadata": {
            "created": existing_cases[0].get("id", "unknown") if existing_cases else "unknown",
            "last_updated": test_case.id,
            "total_cases": len(existing_cases),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved as {next_id}")
    return 0