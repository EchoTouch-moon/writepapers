"""Chapter classification using Qwen API with batch processing."""

import json
import logging
import os
import urllib.request
import urllib.error
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from thesis_library.config import ChapterType, LibraryConfig
from thesis_library.core.chunker import Chunk

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Chapter classification failed."""
    pass


# Anthropic-compatible endpoint for Qwen Coding Plan
QWEN_CODING_ENDPOINT = "https://coding.dashscope.aliyuncs.com/apps/anthropic/v1/messages"


@dataclass
class ClassifierConfig:
    """Configuration for chapter classifier."""
    api_key: str
    model: str = "qwen3.5-plus"  # Qwen Coding Plan model
    batch_size: int = 5
    endpoint: str = QWEN_CODING_ENDPOINT


class ChapterClassifier:
    """Batch classify chunks using Qwen API (Anthropic-compatible endpoint).

    Workflow:
    1. Group chunks into batches (batch_size=5)
    2. Call Qwen API with system prompt
    3. Parse JSON array response
    4. Handle retry on rate limits/network errors
    """

    SYSTEM_PROMPT = """你是学术论文章节分类器。阅读以下按顺序排列的文本块（Chunk 1 到 Chunk N）。
将每个文本块归类到以下类型之一：[ABSTRACT, INTRODUCTION, METHODOLOGY, EXPERIMENT, CONCLUSION, REFERENCE, OTHER]。
你必须且只能输出一个严格的 JSON 数组，数组长度必须与输入的 Chunk 数量一致。不要包含任何其他文字。

示例输出：
["ABSTRACT", "INTRODUCTION", "INTRODUCTION", "METHODOLOGY", "METHODOLOGY"]"""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, urllib.error.URLError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _call_api(self, user_prompt: str) -> str:
        """Call Qwen API (Anthropic-compatible) with retry logic."""
        logger.info(f"Calling Qwen API with {len(user_prompt)} chars prompt...")

        # Build request body for Anthropic-compatible endpoint
        request_body = {
            "model": self.config.model,
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"}
            ]
        }

        request_data = json.dumps(request_body).encode("utf-8")

        # Build request
        req = urllib.request.Request(
            self.config.endpoint,
            data=request_data,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))

                # Extract text content from response
                # Response format: {"content": [{"type": "text", "text": "..."}, ...]}
                text_content = ""
                for item in response_data.get("content", []):
                    if item.get("type") == "text":
                        text_content += item.get("text", "")

                if not text_content:
                    raise ClassificationError("Empty response from API")

                return text_content.strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"API HTTP error {e.code}: {error_body}")
            raise ClassificationError(f"API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            logger.error(f"API URL error: {e.reason}")
            raise

    def classify_batch(self, chunks: list[Chunk]) -> list[ChapterType]:
        """Classify a batch of chunks.

        Args:
            chunks: List of chunks to classify (max batch_size)

        Returns:
            List of ChapterType enums
        """
        if not chunks:
            return []

        # Build user prompt
        prompt_lines = []
        for i, chunk in enumerate(chunks, 1):
            prompt_lines.append(f"Chunk {i}: {chunk.content}")
        user_prompt = "\n".join(prompt_lines)

        # Call API
        try:
            response_text = self._call_api(user_prompt)

            # Parse JSON array
            labels = json.loads(response_text.strip())

            if len(labels) != len(chunks):
                logger.warning(
                    f"Response length mismatch: got {len(labels)}, expected {len(chunks)}. "
                    f"Marking all as OTHER."
                )
                return [ChapterType.OTHER] * len(chunks)

            # Convert to ChapterType
            chapter_types = []
            for label in labels:
                try:
                    chapter_types.append(ChapterType(label))
                except ValueError:
                    logger.warning(f"Invalid label '{label}', using OTHER")
                    chapter_types.append(ChapterType.OTHER)

            return chapter_types

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Marking batch as OTHER.")
            return [ChapterType.OTHER] * len(chunks)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise ClassificationError(str(e)) from e

    def classify_all(self, chunks: list[Chunk]) -> list[ChapterType]:
        """Classify all chunks in batches.

        Args:
            chunks: All chunks from a paper

        Returns:
            List of ChapterType enums for each chunk
        """
        all_types: list[ChapterType] = []

        # Process in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_types = self.classify_batch(batch)
            all_types.extend(batch_types)

            logger.info(f"Classified batch {i//self.config.batch_size + 1}: {len(batch)} chunks")

        return all_types


def create_classifier(library_config: LibraryConfig) -> ChapterClassifier:
    """Factory function to create classifier from library config."""
    api_key = library_config.qwen_api_key or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ClassificationError(
            "QWEN_API_KEY not set. Set via environment variable or LibraryConfig."
        )

    return ChapterClassifier(ClassifierConfig(
        api_key=api_key,
        model=library_config.classifier_model,
        batch_size=library_config.classifier_batch_size,
    ))