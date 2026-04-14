"""Metadata extraction from parsed paper content."""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata for a parsed paper.

    Attributes:
        cite_key: Citation key (e.g., Wang2023)
        title: Paper title
        authors: List of author names
        year: Publication year
        venue: Journal or conference name
        pdf_path: Original PDF path
        md_path: Path to parsed Markdown file
        json_path: Path to parsed JSON file
        chunks_path: Path to chunks file
    """

    cite_key: str
    title: str
    authors: list[str]
    year: int
    venue: str
    pdf_path: str
    md_path: str
    json_path: str
    chunks_path: str


class MetadataExtractor:
    """Extract metadata from parsed paper JSON.

    Strategy:
        - Title: First heading level 1 in JSON
        - Authors/Year/Venue: Inferred from PDF first page or user input
        - cite_key: {FirstAuthorLastName}{Year}
    """

    def __init__(self) -> None:
        pass

    def extract(
        self,
        json_data: list[dict[str, Any]],
        pdf_path: str,
        papers_dir: str = "thesis/library/papers",
    ) -> PaperMetadata:
        """Extract metadata from parsed JSON.

        Args:
            json_data: Parsed JSON content from opendataloader-pdf
            pdf_path: Original PDF path
            papers_dir: Directory for parsed paper files

        Returns:
            PaperMetadata object (may require user to fill missing fields)
        """
        # Extract title from first heading
        title = self._extract_title(json_data)

        # Generate placeholder cite_key (user should edit)
        cite_key = self._generate_placeholder_cite_key(title, pdf_path)

        # Try to extract authors/year from first page
        first_page = [e for e in json_data if e.get("page number", 1) == 1]
        authors = self._extract_authors(first_page)
        year = self._extract_year(first_page)
        venue = self._extract_venue(first_page)

        # Build file paths
        pdf_name = Path(pdf_path).stem
        md_path = str(Path(papers_dir) / f"{cite_key}.md")
        json_path = str(Path(papers_dir) / f"{cite_key}.json")
        chunks_path = str(Path(papers_dir) / f"{cite_key}_chunks.json")

        return PaperMetadata(
            cite_key=cite_key,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            pdf_path=pdf_path,
            md_path=md_path,
            json_path=json_path,
            chunks_path=chunks_path,
        )

    def _extract_title(self, json_data: list[dict[str, Any]]) -> str:
        """Extract title from first heading level 1."""
        for element in json_data:
            if element.get("type") == "heading" and element.get("heading level", 1) == 1:
                return element.get("content", "").strip()
        # Fallback: first content element
        if json_data:
            return json_data[0].get("content", "Unknown Title").strip()
        return "Unknown Title"

    def _extract_authors(self, first_page: list[dict[str, Any]]) -> list[str]:
        """Try to extract author names from first page.

        Heuristic: Look for patterns like "Author Name, Author Name"
        """
        for element in first_page:
            content = element.get("content", "")
            # Skip headings
            if element.get("type") == "heading":
                continue

            # Look for email patterns which often accompany author names
            if "@" in content:
                # Extract names before email
                names = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)", content)
                if names:
                    return names[:5]  # Limit to 5 authors

            # Look for comma-separated names pattern
            if "," in content and len(content) < 200:
                parts = content.split(",")
                names = [p.strip() for p in parts if len(p.strip()) > 2]
                if len(names) >= 2:
                    return names[:5]

        return ["Unknown Author"]

    def _extract_year(self, first_page: list[dict[str, Any]]) -> int:
        """Try to extract publication year.

        Heuristic: Look for 4-digit years in range 1990-2030
        """
        for element in first_page:
            content = element.get("content", "")
            years = re.findall(r"\b(19[9][0-9]|20[0-2][0-9])\b", content)
            if years:
                return int(years[0])

        return 2024  # Default placeholder

    def _extract_venue(self, first_page: list[dict[str, Any]]) -> str:
        """Try to extract venue (journal/conference name).

        Heuristic: Look for capitalized phrases after year
        """
        for element in first_page:
            content = element.get("content", "")
            # Look for patterns like "Published in: Venue Name"
            match = re.search(r"(?:published in|journal|conference)[:\s]+([A-Z][^,]+)", content)
            if match:
                return match.group(1).strip()

        return "Unknown Venue"

    def _generate_placeholder_cite_key(self, title: str, pdf_path: str) -> str:
        """Generate a placeholder cite key.

        Uses PDF filename if title extraction is unreliable.
        """
        # Use PDF filename stem
        pdf_name = Path(pdf_path).stem
        # Clean up for cite key format
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", pdf_name)
        if clean_name:
            return clean_name[:20]
        return f"Paper{hash(title) % 10000}"

    def generate_cite_key(self, first_author_last_name: str, year: int) -> str:
        """Generate proper cite key from author name and year.

        Args:
            first_author_last_name: Last name of first author
            year: Publication year

        Returns:
            Citation key like "Wang2023"
        """
        # Clean name for cite key
        clean_name = re.sub(r"[^a-zA-Z]", "", first_author_last_name)
        return f"{clean_name}{year}"

    def update_cite_key(self, metadata: PaperMetadata, new_cite_key: str) -> PaperMetadata:
        """Update cite key and regenerate file paths.

        Args:
            metadata: Existing metadata
            new_cite_key: New citation key

        Returns:
            Updated metadata with new paths
        """
        papers_dir = Path(metadata.md_path).parent

        return PaperMetadata(
            cite_key=new_cite_key,
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            venue=metadata.venue,
            pdf_path=metadata.pdf_path,
            md_path=str(papers_dir / f"{new_cite_key}.md"),
            json_path=str(papers_dir / f"{new_cite_key}.json"),
            chunks_path=str(papers_dir / f"{new_cite_key}_chunks.json"),
        )

    def save_metadata(self, metadata: PaperMetadata, metadata_registry_path: str) -> None:
        """Add paper metadata to the registry file.

        Args:
            metadata: Paper metadata to save
            metadata_registry_path: Path to metadata.json registry
        """
        path = Path(metadata_registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry or create new
        registry: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                registry = json.load(f)

        # Add/update entry
        registry[metadata.cite_key] = asdict(metadata)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved metadata for {metadata.cite_key}")

    def load_metadata_registry(self, metadata_registry_path: str) -> dict[str, PaperMetadata]:
        """Load all metadata from registry file.

        Args:
            metadata_registry_path: Path to metadata.json

        Returns:
            Dictionary mapping cite_key to PaperMetadata
        """
        path = Path(metadata_registry_path)
        if not path.exists():
            return {}

        with open(path, encoding="utf-8") as f:
            registry = json.load(f)

        return {key: PaperMetadata(**value) for key, value in registry.items()}