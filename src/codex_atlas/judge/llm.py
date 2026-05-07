"""LLM-as-judge answer-faithfulness scorer.

Uses Anthropic Claude (preferred) or OpenAI as a backend, selected by
whichever env var is set.  Requires ``httpx`` (already in the project's
core dependencies).

Key design choices
------------------
- **httpx** is a soft import: if the package is not installed the module
  still loads and ``LLMJudge.__init__`` raises ``ImportError`` with a
  helpful message.
- **30-second timeout** on every request; one automatic retry on 5xx or
  429.
- **JSON response parsing**: the prompt instructs the LLM to return
  ``{"score": <float>, "rationale": <str>}``.  If parsing fails the
  judge logs a warning and falls back to :class:`HeuristicJudge`.
- **No LLM state** is kept between ``score`` calls; each call is fully
  self-contained.
"""

from __future__ import annotations

import json
import logging
import os
import re

from codex_atlas.judge.protocol import JudgeScore

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0
_MAX_RETRIES = 1

_SYSTEM_PROMPT = """\
You are a faithfulness judge for a code-question-answering agent.
Given a user question, a list of expected qualified symbol names, the
agent's answer, and the symbols the agent cited, score whether the
answer faithfully covers the expected content.

Return ONLY valid JSON on a single line with exactly these two keys:
{"score": <float between 0.0 and 1.0>, "rationale": "<one sentence explanation>"}

Scoring rubric:
- 1.0: The answer covers all (or essentially all) expected qualified names.
- 0.5: The answer partially covers the expected qualified names.
- 0.0: The answer does not cover any expected qualified name or is off-topic.
"""


def _build_user_message(
    question: str,
    expected_qualified_names: list[str],
    answer: str,
    citations: list[str],
) -> str:
    names_block = "\n".join(f"- {n}" for n in expected_qualified_names)
    cite_block = "\n".join(f"- {c}" for c in citations) if citations else "(none)"
    return (
        f"Question:\n{question}\n\n"
        f"Expected qualified names:\n{names_block}\n\n"
        f"Agent citations:\n{cite_block}\n\n"
        f"Agent answer:\n{answer}"
    )


def _extract_json(text: str) -> dict[str, object]:
    """Extract the first JSON object from *text*, even if wrapped in markdown."""
    stripped = text.strip()
    try:
        return json.loads(stripped)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if m:
        return json.loads(m.group(1))  # type: ignore[no-any-return]
    m2 = re.search(r"\{.*?\}", stripped, re.DOTALL)
    if m2:
        return json.loads(m2.group(0))  # type: ignore[no-any-return]
    raise ValueError(f"No JSON object found in LLM response: {text!r}")


class LLMJudge:
    """Answer-faithfulness judge backed by an LLM API.

    Reads ``ANTHROPIC_API_KEY`` first, then ``OPENAI_API_KEY``.  Raises
    ``OSError`` if neither is set.  Use :func:`make_judge` for graceful
    fallback to :class:`~codex_atlas.judge.heuristic.HeuristicJudge`.

    Args:
        anthropic_key: Override the env-var lookup (testing convenience).
        openai_key: Override the env-var lookup (testing convenience).

    Raises:
        OSError: Neither ``ANTHROPIC_API_KEY`` nor ``OPENAI_API_KEY`` is
            set and no key was supplied directly.
        ImportError: ``httpx`` is not installed.
    """

    def __init__(
        self,
        *,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
    ) -> None:
        try:
            import httpx as _httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required for LLMJudge. "
                "Install it with: pip install httpx"
            ) from exc

        self._anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key = openai_key or os.environ.get("OPENAI_API_KEY", "")

        if not self._anthropic_key and not self._openai_key:
            raise OSError(
                "LLMJudge requires an API key. Set ANTHROPIC_API_KEY (preferred) "
                "or OPENAI_API_KEY in the environment, or pass the key directly."
            )

        self._use_anthropic = bool(self._anthropic_key)

        # Lazy import to avoid circular; fallback instantiated once.
        from codex_atlas.judge.heuristic import HeuristicJudge

        self._fallback = HeuristicJudge()

    async def score(
        self,
        question: str,
        expected_qualified_names: list[str],
        answer: str,
        citations: list[str],
    ) -> JudgeScore:
        """Score *answer* against *expected_qualified_names* using an LLM.

        On network failure or malformed JSON the judge logs a warning and
        falls back to :class:`~codex_atlas.judge.heuristic.HeuristicJudge`.

        Args:
            question: The user question that prompted the answer.
            expected_qualified_names: Gold symbol names the answer should cover.
            answer: The agent reply to evaluate.
            citations: Qualified names the agent explicitly cited.

        Returns:
            :class:`~codex_atlas.judge.protocol.JudgeScore` with
            ``mode="llm"`` on success or ``mode="heuristic"`` on fallback.
        """
        import httpx

        user_message = _build_user_message(
            question, expected_qualified_names, answer, citations
        )
        raw: str | None = None
        last_exc: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                if self._use_anthropic:
                    raw = await self._call_anthropic(httpx, user_message)
                else:
                    raw = await self._call_openai(httpx, user_message)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "LLMJudge: %s %s, retrying (attempt %d/%d)",
                            exc.response.status_code,
                            exc.response.reason_phrase,
                            attempt + 1,
                            _MAX_RETRIES + 1,
                        )
                        continue
                logger.warning("LLMJudge: HTTP error %s, falling back", exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("LLMJudge: unexpected error %r, falling back", exc)
                last_exc = exc
                break

        if raw is None:
            logger.warning(
                "LLMJudge: falling back to heuristic judge after error: %r", last_exc
            )
            return await self._fallback.score(
                question, expected_qualified_names, answer, citations
            )

        try:
            data = _extract_json(raw)
            score_val = float(data["score"])  # type: ignore[arg-type]
            rationale = str(data.get("rationale", ""))
            score_val = max(0.0, min(1.0, score_val))
        except Exception as exc:
            logger.warning(
                "LLMJudge: could not parse LLM response %r: %r — falling back", raw, exc
            )
            return await self._fallback.score(
                question, expected_qualified_names, answer, citations
            )

        return JudgeScore(score=score_val, rationale=rationale, mode="llm")

    async def _call_anthropic(self, httpx: object, user_message: str) -> str:
        import httpx as _httpx

        payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 256,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
        async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
            return str(body["content"][0]["text"])

    async def _call_openai(self, httpx: object, user_message: str) -> str:
        import httpx as _httpx

        payload = {
            "model": "gpt-4o-mini",
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openai_key}",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
            return str(body["choices"][0]["message"]["content"])


__all__ = ["LLMJudge"]
