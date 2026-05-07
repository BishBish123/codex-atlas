"""Groq-backed LLM synthesizer for codex-atlas.

When ``GROQ_API_KEY`` is set, calls Groq's chat-completions endpoint to produce
a cited, concise answer from retrieved code chunks. When unset, falls back to
``StitchSynthesizer`` so the agent runs end-to-end without any API key.

Endpoint:  POST https://api.groq.com/openai/v1/chat/completions
Auth:      Authorization: Bearer <GROQ_API_KEY>
Model:     llama-3.3-70b-versatile  (verified May 2026)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from codex_atlas.agent import StitchSynthesizer, _display_path
from codex_atlas.store import StoredChunk

try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False

_log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_TIMEOUT_S = 30.0
_SYSTEM_PROMPT = (
    "You are a code-Q&A assistant. Answer questions about the provided code "
    "concisely. Include inline `[file:line]` citations for every claim. "
    "Do not invent symbols that are not present in the code chunks."
)


def _build_user_message(query: str, chunks: list[StoredChunk]) -> str:
    """Format the query + chunks into a single user turn."""
    parts: list[str] = [f"Question: {query}", "", "Code chunks (with file:line citations):"]
    for c in chunks:
        shown = _display_path(c.file_path)
        parts.append(
            f"\n### `{c.qualified_name}` ({shown}:{c.lineno_start}-{c.lineno_end})\n"
            f"```python\n{c.text.rstrip()}\n```"
        )
    parts.append(
        "\nAnswer concisely with inline `[file:line]` citations for every claim. "
        "Don't invent symbols."
    )
    return "\n".join(parts)


@dataclass
class GroqSynthesizer:
    """LLM-backed synthesizer using Groq's chat-completions API.

    When ``GROQ_API_KEY`` is not set (or ``httpx`` is unavailable), the
    instance operates in mock mode and delegates every call to an inner
    ``StitchSynthesizer``. This makes the agent fully functional offline
    and in CI without any environment configuration.

    Attributes:
        model: Groq model ID to use.
        timeout: HTTP request timeout in seconds.
    """

    model: str = _GROQ_MODEL
    timeout: float = _TIMEOUT_S
    _mock: bool = field(init=False)
    _api_key: str = field(init=False, default="")
    _fallback: StitchSynthesizer = field(init=False)

    def __post_init__(self) -> None:
        key = os.environ.get("GROQ_API_KEY", "")
        self._api_key = key
        self._mock = not key or not _HTTPX_AVAILABLE
        self._fallback = StitchSynthesizer()

    async def synthesize(self, query: str, chunks: list[StoredChunk]) -> str:
        """Synthesize an answer from retrieved chunks.

        Delegates to ``StitchSynthesizer`` when operating in mock mode.
        On API error (including auth failure and timeout), logs a warning
        and falls back to mock output.
        """
        if self._mock:
            return await self._fallback.synthesize(query, chunks)
        return await self._call_groq(query, chunks)

    async def _call_groq(self, query: str, chunks: list[StoredChunk]) -> str:
        """Call the Groq endpoint; retry once on 5xx; fallback on errors."""
        user_message = _build_user_message(query, chunks)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                async with _httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(_GROQ_URL, json=payload, headers=headers)
                if resp.status_code == 200:
                    return self._parse_response(resp.json())
                if resp.status_code >= 500 and attempt == 0:
                    # One retry on 5xx
                    _log.warning(
                        "groq_synthesizer.5xx_retry",
                        extra={"status": resp.status_code, "attempt": attempt},
                    )
                    continue
                # 4xx or second 5xx → fall back
                _log.warning(
                    "groq_synthesizer.api_error",
                    extra={"status": resp.status_code, "body": resp.text[:200]},
                )
                return await self._fallback.synthesize(query, chunks)
            except _httpx.TimeoutException as exc:
                last_exc = exc
                _log.warning("groq_synthesizer.timeout", extra={"attempt": attempt})
                break
            except Exception as exc:  # broad catch intentional — fallback path
                last_exc = exc
                _log.warning(
                    "groq_synthesizer.exception",
                    extra={"exc": str(exc)},
                )
                break

        if last_exc is not None:
            _log.warning(
                "groq_synthesizer.fallback",
                extra={"reason": str(last_exc)},
            )
        return await self._fallback.synthesize(query, chunks)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> str:
        """Extract the assistant message text from the API response."""
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected Groq response shape: {data!r}") from exc
