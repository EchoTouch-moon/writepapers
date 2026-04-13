"""LLM API client for thesis generation.

Supports:
- Alibaba Cloud DashScope (Qwen models)
- OpenAI-compatible endpoints (Doubao, etc.)
"""

import json
import logging
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# OpenAI-compatible endpoint for Qwen
DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-plus"


@dataclass
class LLMConfig:
    """LLM API configuration.

    Attributes:
        api_url: API endpoint URL
        model: Model name
        api_key: API key (from environment)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    """
    api_url: str = DEFAULT_API_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")


class LLMClient:
    """LLM API client for thesis generation.

    Minimal wrapper around OpenAI-compatible endpoints.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise ValueError("API key required. Set DASHSCOPE_API_KEY environment variable.")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Generated text

        Raises:
            urllib.error.URLError: API request failed
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Calling LLM API: {self.config.model}")

        try:
            req = urllib.request.Request(
                self.config.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                logger.info(f"LLM generated {len(content)} chars")
                return content

        except urllib.error.URLError as e:
            logger.error(f"LLM API failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise ValueError(f"Invalid API response: {result}")

    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_retries: int = 3,
    ) -> str:
        """Generate with automatic retry on failure."""
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, min=4, max=10))
        def _generate():
            return self.generate(prompt, system_prompt)

        return _generate()