"""PDF processing module using opendataloader-pdf."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """PDF parsing failed."""

    def __init__(self, pdf_path: str, reason: str) -> None:
        self.pdf_path = pdf_path
        self.reason = reason
        super().__init__(f"PDF parse error: {pdf_path} - {reason}")


class PDFProcessor:
    """Process PDF files using opendataloader-pdf.

    The opendataloader-pdf API writes output to a directory, so this processor
    creates a temporary output directory and reads the generated files.

    Attributes:
        use_hybrid: Whether to use hybrid mode for complex content
    """

    def __init__(self, use_hybrid: bool = False) -> None:
        self.use_hybrid = use_hybrid

    def process_single(self, pdf_path: str) -> dict[str, Any]:
        """Process a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - md_content: Markdown text
                - json_content: Structured JSON data with bounding boxes
                - pdf_path: Original PDF path

        Raises:
            PDFParseError: If parsing fails
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise PDFParseError(pdf_path, "File not found")

        try:
            import opendataloader_pdf

            logger.info(f"Processing PDF: {pdf_path}")

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)

                # Call opendataloader_pdf.convert
                # It writes files to output_dir
                opendataloader_pdf.convert(
                    input_path=[str(pdf_file)],
                    output_dir=str(output_dir),
                    format="markdown,json",
                    hybrid="docling-fast" if self.use_hybrid else None,
                )

                # Find the generated files
                # Output files are named based on input PDF name
                base_name = pdf_file.stem

                md_file = output_dir / f"{base_name}.md"
                json_file = output_dir / f"{base_name}.json"

                # Some versions might use different naming
                if not md_file.exists():
                    md_files = list(output_dir.glob("*.md"))
                    if md_files:
                        md_file = md_files[0]

                if not json_file.exists():
                    json_files = list(output_dir.glob("*.json"))
                    if json_files:
                        json_file = json_files[0]

                # Read the contents
                md_content = ""
                json_content: list[dict] = []

                if md_file.exists():
                    md_content = md_file.read_text(encoding="utf-8")
                else:
                    logger.warning(f"No markdown output for {pdf_path}")

                if json_file.exists():
                    with open(json_file, encoding="utf-8") as f:
                        json_data = json.load(f)
                    # opendataloader-pdf outputs a dict with 'kids' containing elements
                    if isinstance(json_data, dict) and "kids" in json_data:
                        json_content = json_data["kids"]
                    elif isinstance(json_data, list):
                        json_content = json_data
                    else:
                        json_content = []
                else:
                    logger.warning(f"No JSON output for {pdf_path}")

                if not md_content and not json_content:
                    raise PDFParseError(pdf_path, "No content extracted")

                return {
                    "md_content": md_content,
                    "json_content": json_content,
                    "pdf_path": pdf_path,
                }

        except ImportError as e:
            raise ImportError(
                "opendataloader-pdf not installed. "
                "Install with: uv pip install opendataloader-pdf[hybrid]"
            ) from e
        except Exception as e:
            if isinstance(e, PDFParseError):
                raise
            raise PDFParseError(pdf_path, str(e)) from e

    def process_batch(
        self, pdf_paths: list[str], use_hybrid: bool | None = None
    ) -> list[dict[str, Any]]:
        """Process multiple PDF files in batch.

        Args:
            pdf_paths: List of PDF file paths
            use_hybrid: Override hybrid mode setting

        Returns:
            List of parsing results, same format as process_single

        Note:
            opendataloader-pdf recommends batching all files in one call
            because each convert() spawns a JVM process.
        """
        hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        # Filter valid paths
        valid_paths = []
        for path in pdf_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                logger.warning(f"Skipping non-existent file: {path}")

        if not valid_paths:
            logger.warning("No valid PDF files to process")
            return []

        try:
            import opendataloader_pdf

            logger.info(f"Batch processing {len(valid_paths)} PDF files (hybrid={hybrid})")

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)

                # Batch all files in ONE call (JVM spawn optimization)
                opendataloader_pdf.convert(
                    input_path=valid_paths,
                    output_dir=str(output_dir),
                    format="markdown,json",
                    hybrid="docling-fast" if hybrid else None,
                )

                # Read all generated files
                results: list[dict[str, Any]] = []

                for pdf_path in valid_paths:
                    pdf_file = Path(pdf_path)
                    base_name = pdf_file.stem

                    md_file = output_dir / f"{base_name}.md"
                    json_file = output_dir / f"{base_name}.json"

                    # Handle potential naming variations
                    if not md_file.exists():
                        md_files = list(output_dir.glob("*.md"))
                        if md_files:
                            md_file = md_files[0]

                    if not json_file.exists():
                        json_files = list(output_dir.glob("*.json"))
                        if json_files:
                            json_file = json_files[0]

                    md_content = ""
                    json_content: list[dict] = []

                    if md_file.exists():
                        md_content = md_file.read_text(encoding="utf-8")

                    if json_file.exists():
                        with open(json_file, encoding="utf-8") as f:
                            json_data = json.load(f)
                        # opendataloader-pdf outputs a dict with 'kids' containing elements
                        if isinstance(json_data, dict) and "kids" in json_data:
                            json_content = json_data["kids"]
                        elif isinstance(json_data, list):
                            json_content = json_data
                        else:
                            json_content = []

                    if md_content or json_content:
                        results.append({
                            "md_content": md_content,
                            "json_content": json_content,
                            "pdf_path": pdf_path,
                        })
                    else:
                        logger.warning(f"No content for {pdf_path}")

                return results

        except ImportError as e:
            raise ImportError(
                "opendataloader-pdf not installed. "
                "Install with: uv pip install opendataloader-pdf[hybrid]"
            ) from e
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fall back to single-file processing
            results: list[dict[str, Any]] = []
            for path in valid_paths:
                try:
                    result = self.process_single(path)
                    results.append(result)
                except PDFParseError as e:
                    logger.error(f"Failed: {e.pdf_path} - {e.reason}")
            return results