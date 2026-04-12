"""CLI interface for thesis-library."""

import argparse
import json
import logging
import sys
from pathlib import Path

from thesis_library import Library, LibraryConfig
from thesis_library.evaluator import Evaluator, generate_report, load_baseline, save_baseline, save_last_run
from thesis_library.evaluator.cli_add import run_eval_add

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest PDF files into library."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    pdf_paths = args.pdf_paths
    if not pdf_paths:
        print("No PDF files provided")
        return 1

    # Validate paths
    valid_paths = [p for p in pdf_paths if Path(p).exists()]
    if not valid_paths:
        print("No valid PDF files found")
        return 1

    print(f"Ingesting {len(valid_paths)} PDF files...")

    try:
        metadata_list = library.ingest(
            valid_paths,
            use_hybrid=args.hybrid,
            interactive=not args.batch,
        )

        for m in metadata_list:
            print(f"  ✓ {m.cite_key}: {m.title}")

        print(f"\nSuccessfully ingested {len(metadata_list)} papers")
        return 0

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        print(f"Error: {e}")
        return 1


def cmd_index(args: argparse.Namespace) -> int:
    """Rebuild the index."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    print("Rebuilding index...")
    try:
        library.rebuild_index()
        status = library.status()
        print(f"Index rebuilt: {status['chunks_count']} chunks, {status['terms_count']} terms")
        return 0
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        print(f"Error: {e}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search the library."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1

    query = args.query
    chapter_type = args.chapter_type.upper() if args.chapter_type else None
    threshold = args.threshold
    top_k = args.top_k

    print(f"Searching: {query}")
    if chapter_type:
        print(f"  Chapter type: {chapter_type}")
    print(f"  Threshold: {threshold}, Top-K: {top_k}")

    results = library.search(query, chapter_type, threshold, top_k)

    if not results:
        print("No results found")
        return 0

    print(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.chunk.cite_key}] Sim: {r.similarity:.2f}")
        print(f"   Section: {r.chunk.section_title}")
        print(f"   Terms: {r.matched_terms}")
        print(f"   Content: {r.chunk.content[:100]}...")
        print()

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all papers in library."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    papers = library.list_papers()

    if not papers:
        print("No papers in library")
        return 0

    print(f"\n{len(papers)} papers in library:\n")
    for p in papers:
        print(f"  [{p.cite_key}] {p.title}")
        print(f"    Authors: {', '.join(p.authors[:3])}")
        print(f"    Year: {p.year}, Venue: {p.venue}")
        print()

    return 0


def cmd_edit_meta(args: argparse.Namespace) -> int:
    """Edit paper metadata."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    cite_key = args.cite_key
    metadata = library.get_paper(cite_key)

    if not metadata:
        print(f"Paper not found: {cite_key}")
        return 1

    print(f"Current metadata for {cite_key}:")
    print(f"  Title: {metadata.title}")
    print(f"  Authors: {', '.join(metadata.authors)}")
    print(f"  Year: {metadata.year}")
    print(f"  Venue: {metadata.venue}")

    # Interactive editing
    print("\nEdit fields (press Enter to keep current value):")

    updates: dict = {}

    # Title
    new_title = input(f"Title [{metadata.title}]: ").strip()
    if new_title:
        updates["title"] = new_title

    # Authors
    new_authors = input(f"Authors [{', '.join(metadata.authors)}]: ").strip()
    if new_authors:
        updates["authors"] = [a.strip() for a in new_authors.split(",")]

    # Year
    new_year = input(f"Year [{metadata.year}]: ").strip()
    if new_year:
        try:
            updates["year"] = int(new_year)
        except ValueError:
            print("Invalid year")

    # Venue
    new_venue = input(f"Venue [{metadata.venue}]: ").strip()
    if new_venue:
        updates["venue"] = new_venue

    # Cite key
    new_key = input(f"Cite key [{cite_key}]: ").strip()
    if new_key:
        updates["cite_key"] = new_key

    if not updates:
        print("No changes made")
        return 0

    updated = library.edit_metadata(cite_key, updates)
    if updated:
        print(f"\nUpdated: {updated.cite_key}")
        return 0
    else:
        print("Update failed")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show library status."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    status = library.status()

    print("\nLibrary Status:")
    print(f"  Directory: {status['library_dir']}")
    print(f"  Papers: {status['papers_count']}")
    print(f"  Index: {status['index_status']}")
    print(f"  Chunks: {status['chunks_count']}")
    print(f"  Terms: {status['terms_count']}")

    return 0


def cmd_terms(args: argparse.Namespace) -> int:
    """List known terms."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    terms = library.get_known_terms()

    if not terms:
        print("No terms indexed")
        return 0

    print(f"\n{len(terms)} known terms:\n")
    for term in sorted(terms)[:50]:
        chunks = library.search_by_terms([term])
        print(f"  {term}: {len(chunks)} chunks")

    if len(terms) > 50:
        print(f"\n  ... and {len(terms) - 50} more")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run RAG evaluation."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1

    evaluator = Evaluator(library, args.test_cases)

    if not evaluator.test_cases:
        print(f"No test cases found in {args.test_cases}")
        print("Use 'thesis-library eval-add' to add test cases.")
        return 1

    print(f"Running evaluation with {len(evaluator.test_cases)} test cases...\n")

    result = evaluator.run(k=args.k)

    # Load baseline if exists
    baseline = load_baseline(args.baseline) if Path(args.baseline).exists() else None

    # Generate and print report
    report = generate_report(result, baseline, verbose=args.verbose)
    print(report)

    # Save last run
    save_last_run(result, args.last_run)

    # Save as baseline if requested
    if args.save_baseline:
        library_config = {
            "similarity_threshold": library.config.similarity_threshold,
            "max_chunk_size": library.config.max_chunk_size,
            "embedding_model": library.config.embedding_model,
        }
        save_baseline(result, args.baseline, library_config)
        print(f"\nSaved baseline to {args.baseline}")

    return 0


def cmd_eval_add(args: argparse.Namespace) -> int:
    """Add test case interactively."""
    config = LibraryConfig(library_dir=args.library_dir)
    library = Library(config)

    if not library.indexer.index:
        print("Index not built. Run 'thesis-library index' first.")
        return 1

    return run_eval_add(library, args.test_cases)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="thesis-library",
        description="Thesis literature library management",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--library-dir",
        default="thesis/library",
        help="Library directory path",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF files")
    ingest_parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="PDF file paths to ingest",
    )
    ingest_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid mode for complex content",
    )
    ingest_parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode (no interactive prompts)",
    )

    # index command
    index_parser = subparsers.add_parser("index", help="Rebuild index")

    # search command
    search_parser = subparsers.add_parser("search", help="Search library")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--chapter-type",
        help="Chapter type for structural constraint",
    )
    search_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Similarity threshold",
    )
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List papers")

    # edit-meta command
    edit_parser = subparsers.add_parser("edit-meta", help="Edit paper metadata")
    edit_parser.add_argument("cite_key", help="Citation key")

    # status command
    status_parser = subparsers.add_parser("status", help="Show status")

    # terms command
    terms_parser = subparsers.add_parser("terms", help="List known terms")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Run RAG evaluation")
    eval_parser.add_argument(
        "--test-cases",
        default="thesis/evaluator/test_cases.json",
        help="Path to test cases JSON file",
    )
    eval_parser.add_argument(
        "--baseline",
        default="thesis/evaluator/baseline.json",
        help="Path to baseline JSON file",
    )
    eval_parser.add_argument(
        "--last-run",
        default="thesis/evaluator/last_run.json",
        help="Path to last run JSON file",
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value for Recall@K and MRR@K",
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose noise metrics",
    )
    eval_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current result as new baseline",
    )

    # eval-add command
    eval_add_parser = subparsers.add_parser("eval-add", help="Add test case interactively")
    eval_add_parser.add_argument(
        "--test-cases",
        default="thesis/evaluator/test_cases.json",
        help="Path to test cases JSON file",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "edit-meta":
        return cmd_edit_meta(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "terms":
        return cmd_terms(args)
    elif args.command == "eval":
        return cmd_eval(args)
    elif args.command == "eval-add":
        return cmd_eval_add(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())