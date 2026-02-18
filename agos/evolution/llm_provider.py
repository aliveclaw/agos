"""Thin LLM provider for ALMA evolution features.

Wraps the Anthropic SDK for use by MetaEvolver (ideation),
CodeEvolver (self-reflection, iterate-on-strategy).
Keeps token budgets tight — each call is 400-2000 tokens.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LLMProvider:
    """Minimal LLM interface for evolution features."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Single-turn completion. Returns text content."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
