"""Smart chunking module for parsed paper content."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of paper content.

    Attributes:
        id: Unique chunk identifier (cite_key + type + index)
        cite_key: Citation key of source paper
        content: Text content
        chunk_type: Type of chunk (section, paragraph, table, list)
        section_title: Title of the section this chunk belongs to
        page_number: Page number in the PDF
        bounding_box: Coordinates [left, bottom, right, top]
        parent_id: ID of parent section chunk (if this is a sub-chunk)
        chapter_type: Standardized chapter classification (added for metadata filtering)
    """

    id: str
    cite_key: str
    content: str
    chunk_type: str  # section | paragraph | table | list
    section_title: str
    page_number: int
    bounding_box: list[float]
    parent_id: str | None = None
    chapter_type: str = "OTHER"  # Default, will be classified by ChapterClassifier


class Chunker:
    """Chunk parsed paper JSON into manageable pieces.

    Attributes:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
    """

    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._current_section = ""
        self._section_counter = 0
        self._merged_para_counter = 0

    def chunk_paper(
        self, json_data: list[dict[str, Any]], cite_key: str
    ) -> list[Chunk]:
        """Convert parsed JSON data into chunks.

        Args:
            json_data: List of content elements from opendataloader-pdf
            cite_key: Citation key for the paper

        Returns:
            List of Chunk objects preserving structure and bounding boxes
        """
        chunks: list[Chunk] = []
        section_chunks: dict[str, Chunk] = {}  # Track section chunks for parent_id

        # First pass: identify section structure
        for element in json_data:
            if element.get("type") == "heading":
                level = element.get("heading level", 1)
                if level <= 2:  # Only track main sections
                    self._section_counter += 1
                    self._current_section = element.get("content", "").strip()

        # Reset counters for actual chunking
        self._current_section = ""
        self._section_counter = 0
        self._merged_para_counter = 0  # Reset merged paragraph counter
        para_counter = 0
        table_counter = 0
        list_counter = 0

        # Buffer for merging short paragraphs
        paragraph_buffer: list[dict] = []
        buffer_content = ""

        for element in json_data:
            elem_type = element.get("type", "")
            content = element.get("content", "")
            page = element.get("page number", 1)
            bbox = element.get("bounding box", [0.0, 0.0, 0.0, 0.0])

            if elem_type == "heading":
                # Flush paragraph buffer before new section
                if buffer_content:
                    chunks.extend(self._flush_paragraph_buffer(
                        buffer_content, paragraph_buffer, cite_key, section_chunks
                    ))
                    buffer_content = ""
                    paragraph_buffer = []

                level = element.get("heading level", 1)
                if level <= 2:
                    self._section_counter += 1
                    self._current_section = content.strip()
                    para_counter = 0
                    self._merged_para_counter = 0  # Reset merged counter for new section

                    # Create section chunk
                    chunk_id = f"{cite_key}_section{self._section_counter}"
                    section_chunk = Chunk(
                        id=chunk_id,
                        cite_key=cite_key,
                        content=content.strip(),
                        chunk_type="section",
                        section_title=self._current_section,
                        page_number=page,
                        bounding_box=bbox,
                    )
                    chunks.append(section_chunk)
                    section_chunks[self._current_section] = section_chunk

            elif elem_type == "paragraph":
                # Filter out extremely short fragments (< 10 chars)
                # These are likely parsing artifacts from special PDF fonts
                if len(content.strip()) < 10:
                    continue

                # Buffer short paragraphs for merging
                if len(content) < self.min_chunk_size:
                    buffer_content += content.strip() + "\n"
                    paragraph_buffer.append(element)
                else:
                    # Flush buffer first
                    if buffer_content:
                        chunks.extend(self._flush_paragraph_buffer(
                            buffer_content, paragraph_buffer, cite_key, section_chunks
                        ))
                        buffer_content = ""
                        paragraph_buffer = []

                    para_counter += 1
                    # Split long paragraphs
                    if len(content) > self.max_chunk_size:
                        sub_chunks = self._split_long_content(
                            content, cite_key, "paragraph", page, bbox
                        )
                        for i, sub in enumerate(sub_chunks):
                            parent = section_chunks.get(self._current_section)
                            chunks.append(Chunk(
                                id=f"{cite_key}_s{self._section_counter}_para{para_counter}_sub{i}",
                                cite_key=cite_key,
                                content=sub,
                                chunk_type="paragraph",
                                section_title=self._current_section,
                                page_number=page,
                                bounding_box=bbox,
                                parent_id=parent.id if parent else None,
                            ))
                    else:
                        parent = section_chunks.get(self._current_section)
                        chunks.append(Chunk(
                            id=f"{cite_key}_s{self._section_counter}_para{para_counter}",
                            cite_key=cite_key,
                            content=content.strip(),
                            chunk_type="paragraph",
                            section_title=self._current_section,
                            page_number=page,
                            bounding_box=bbox,
                            parent_id=parent.id if parent else None,
                        ))

            elif elem_type == "table":
                # Flush paragraph buffer before table
                if buffer_content:
                    chunks.extend(self._flush_paragraph_buffer(
                        buffer_content, paragraph_buffer, cite_key, section_chunks
                    ))
                    buffer_content = ""
                    paragraph_buffer = []

                table_counter += 1
                # Keep tables as single chunks, preserve row structure
                rows = element.get("rows", [])
                table_content = self._format_table(rows)

                parent = section_chunks.get(self._current_section)
                chunks.append(Chunk(
                    id=f"{cite_key}_s{self._section_counter}_table{table_counter}",
                    cite_key=cite_key,
                    content=table_content,
                    chunk_type="table",
                    section_title=self._current_section,
                    page_number=page,
                    bounding_box=bbox,
                    parent_id=parent.id if parent else None,
                ))

            elif elem_type == "list":
                # Flush paragraph buffer before list
                if buffer_content:
                    chunks.extend(self._flush_paragraph_buffer(
                        buffer_content, paragraph_buffer, cite_key, section_chunks
                    ))
                    buffer_content = ""
                    paragraph_buffer = []

                list_counter += 1
                items = element.get("list items", [])
                list_content = "\n".join(f"- {item}" for item in items)

                parent = section_chunks.get(self._current_section)
                chunks.append(Chunk(
                    id=f"{cite_key}_s{self._section_counter}_list{list_counter}",
                    cite_key=cite_key,
                    content=list_content,
                    chunk_type="list",
                    section_title=self._current_section,
                    page_number=page,
                    bounding_box=bbox,
                    parent_id=parent.id if parent else None,
                ))

        # Flush remaining buffer at end
        if buffer_content:
            chunks.extend(self._flush_paragraph_buffer(
                buffer_content, paragraph_buffer, cite_key, section_chunks
            ))

        logger.info(f"Created {len(chunks)} chunks for {cite_key}")
        return chunks

    def _flush_paragraph_buffer(
        self,
        buffer_content: str,
        paragraph_buffer: list[dict],
        cite_key: str,
        section_chunks: dict[str, "Chunk"],
    ) -> list[Chunk]:
        """Flush accumulated short paragraphs as merged chunks."""
        chunks: list[Chunk] = []

        if not buffer_content.strip():
            return chunks

        # Split buffer into max_chunk_size pieces
        if len(buffer_content) > self.max_chunk_size:
            pieces = self._split_long_content(
                buffer_content, cite_key, "paragraph",
                paragraph_buffer[0].get("page number", 1),
                paragraph_buffer[0].get("bounding box", [0.0, 0.0, 0.0, 0.0])
            )
            for i, piece in enumerate(pieces):
                parent = section_chunks.get(self._current_section)
                chunks.append(Chunk(
                    id=f"{cite_key}_s{self._section_counter}_merged{self._merged_para_counter + i}",
                    cite_key=cite_key,
                    content=piece,
                    chunk_type="paragraph",
                    section_title=self._current_section,
                    page_number=paragraph_buffer[0].get("page number", 1),
                    bounding_box=paragraph_buffer[0].get("bounding box", [0.0, 0.0, 0.0, 0.0]),
                    parent_id=parent.id if parent else None,
                ))
            self._merged_para_counter += len(pieces)
        else:
            parent = section_chunks.get(self._current_section)
            chunks.append(Chunk(
                id=f"{cite_key}_s{self._section_counter}_merged{self._merged_para_counter}",
                cite_key=cite_key,
                content=buffer_content.strip(),
                chunk_type="paragraph",
                section_title=self._current_section,
                page_number=paragraph_buffer[0].get("page number", 1) if paragraph_buffer else 1,
                bounding_box=paragraph_buffer[0].get("bounding box", [0.0, 0.0, 0.0, 0.0]) if paragraph_buffer else [0.0, 0.0, 0.0, 0.0],
                parent_id=parent.id if parent else None,
            ))
            self._merged_para_counter += 1

        return chunks

    def _split_long_content(
        self,
        content: str,
        cite_key: str,
        chunk_type: str,
        page: int,
        bbox: list[float],
    ) -> list[str]:
        """Split content that exceeds max_chunk_size.

        Strategy: Split by sentences, then combine until reaching limit.
        """
        # Simple sentence split (could be improved for Chinese)
        sentences = []
        for sent in content.replace("。", "。\n").split("\n"):
            sent = sent.strip()
            if sent:
                sentences.append(sent)

        chunks: list[str] = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) > self.max_chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = current + " " + sent if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _format_table(self, rows: list[list[str]]) -> str:
        """Format table rows as markdown."""
        if not rows:
            return ""

        lines = []
        for i, row in enumerate(rows):
            cells = " | ".join(str(cell) for cell in row)
            lines.append(f"| {cells} |")
            if i == 0:  # Header separator
                sep = " | ".join("---" for _ in row)
                lines.append(f"| {sep} |")

        return "\n".join(lines)

    def save_chunks(self, chunks: list[Chunk], output_path: str) -> None:
        """Save chunks to JSON file.

        Args:
            chunks: List of Chunk objects
            output_path: Path to output JSON file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(chunk) for chunk in chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str) -> list[Chunk]:
        """Load chunks from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of Chunk objects
        """
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        return [Chunk(**item) for item in data]