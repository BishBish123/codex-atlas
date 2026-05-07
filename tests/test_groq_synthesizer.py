"""Tests for GroqSynthesizer and the make_synthesizer factory.

All tests operate offline -- no real Groq API calls are made.
httpx is patched via unittest.mock wherever the live path would execute.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from codex_atlas.agent import Agent, StitchSynthesizer
from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.retriever import RetrievalResult, Route
from codex_atlas.store import StoredChunk
from codex_atlas.synthesis import GroqSynthesizer, make_synthesizer
from codex_atlas.synthesis.groq import _GROQ_URL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(qname: str, text: str = "def foo(): pass\n") -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=text,
        score=0.9,
    )


def _groq_response(content: str) -> dict[str, Any]:
    """Minimal Groq chat-completions response shape."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def _make_mock_httpx_response(status_code: int, body: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body or {}
    mock_resp.text = json.dumps(body or {})
    return mock_resp


def _make_async_client_cm(response: MagicMock) -> tuple[MagicMock, AsyncMock]:
    """Build a context-manager mock for httpx.AsyncClient."""
    mock_client: AsyncMock = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, mock_client


def _stub_retrieval(chunks: list[StoredChunk] | None = None) -> RetrievalResult:
    return RetrievalResult(
        route=Route.LOOKUP,
        confidence=0.9,
        signals=["stub"],
        chunks=chunks or [],
        extra_qualified_names=[],
    )


# ---------------------------------------------------------------------------
# Mock-mode (no GROQ_API_KEY)
# ---------------------------------------------------------------------------


class TestGroqSynthesizerMockMode:
    """When GROQ_API_KEY is absent the synthesizer must be in mock mode."""

    def test_no_key_sets_mock_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        synth = GroqSynthesizer()
        assert synth._mock is True

    async def test_mock_mode_delegates_to_stitch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        synth = GroqSynthesizer()
        chunks = [_chunk("m.foo")]
        result = await synth.synthesize("what does foo do", chunks)

        # Should match StitchSynthesizer output exactly
        stitch = StitchSynthesizer()
        expected = await stitch.synthesize("what does foo do", chunks)
        assert result == expected

    async def test_mock_mode_empty_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        synth = GroqSynthesizer()
        result = await synth.synthesize("what does foo do", [])
        assert "could not find" in result.lower()


# ---------------------------------------------------------------------------
# Real-key path (mocked httpx)
# ---------------------------------------------------------------------------


class TestGroqSynthesizerRealKey:
    """When GROQ_API_KEY is set, the synthesizer calls the Groq endpoint."""

    async def test_calls_correct_url_and_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        synth = GroqSynthesizer()
        assert synth._mock is False

        resp_mock = _make_mock_httpx_response(200, _groq_response("The answer."))
        cm, mock_client = _make_async_client_cm(resp_mock)

        with patch("codex_atlas.synthesis.groq._httpx.AsyncClient", return_value=cm):
            result = await synth.synthesize("what does foo do", [_chunk("m.foo")])

        assert result == "The answer."
        # Verify request shape
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert _GROQ_URL in str(call_kwargs)
        headers = call_kwargs.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer test_key"
        payload = call_kwargs.kwargs.get("json", {})
        assert payload["model"] == "llama-3.3-70b-versatile"
        assert any(m["role"] == "system" for m in payload["messages"])
        assert any(m["role"] == "user" for m in payload["messages"])

    async def test_request_body_contains_query_and_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        synth = GroqSynthesizer()
        chunks = [_chunk("m.bar", "def bar(): return 42\n")]
        resp_mock = _make_mock_httpx_response(200, _groq_response("Bar returns 42."))
        cm, mock_client = _make_async_client_cm(resp_mock)

        with patch("codex_atlas.synthesis.groq._httpx.AsyncClient", return_value=cm):
            await synth.synthesize("what does bar return", chunks)

        payload = mock_client.post.call_args.kwargs["json"]
        user_content = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        assert "what does bar return" in user_content
        assert "m.bar" in user_content

    async def test_response_parsed_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        synth = GroqSynthesizer()
        resp_mock = _make_mock_httpx_response(
            200, _groq_response("Parsed answer with citation [x.py:1].")
        )
        cm, _mock_client = _make_async_client_cm(resp_mock)

        with patch("codex_atlas.synthesis.groq._httpx.AsyncClient", return_value=cm):
            result = await synth.synthesize("q", [_chunk("m.baz")])

        assert result == "Parsed answer with citation [x.py:1]."


# ---------------------------------------------------------------------------
# Auth error fallback
# ---------------------------------------------------------------------------


class TestGroqSynthesizerAuthError:
    async def test_401_falls_back_to_stitch(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "bad_key")
        synth = GroqSynthesizer()
        chunks = [_chunk("m.foo")]

        resp_mock = _make_mock_httpx_response(401, {"error": "invalid_api_key"})
        cm, _client = _make_async_client_cm(resp_mock)

        with (
            caplog.at_level(logging.WARNING, logger="codex_atlas.synthesis.groq"),
            patch("codex_atlas.synthesis.groq._httpx.AsyncClient", return_value=cm),
        ):
            result = await synth.synthesize("q", chunks)

        # Should fall back to stitch output
        stitch = StitchSynthesizer()
        expected = await stitch.synthesize("q", chunks)
        assert result == expected
        # Warning must have been logged
        assert any("groq_synthesizer.api_error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Timeout fallback
# ---------------------------------------------------------------------------


class TestGroqSynthesizerTimeout:
    async def test_timeout_falls_back_to_stitch(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        synth = GroqSynthesizer()
        chunks = [_chunk("m.foo")]

        mock_client: AsyncMock = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_client)
        cm.__aexit__ = AsyncMock(return_value=None)

        with (
            caplog.at_level(logging.WARNING, logger="codex_atlas.synthesis.groq"),
            patch("codex_atlas.synthesis.groq._httpx.AsyncClient", return_value=cm),
        ):
            result = await synth.synthesize("q", chunks)

        stitch = StitchSynthesizer()
        expected = await stitch.synthesize("q", chunks)
        assert result == expected
        assert any("groq_synthesizer.timeout" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestMakeSynthesizer:
    def test_returns_stitch_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        synth = make_synthesizer()
        assert isinstance(synth, StitchSynthesizer)

    def test_returns_groq_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        synth = make_synthesizer()
        assert isinstance(synth, GroqSynthesizer)
        assert synth._mock is False

    def test_groq_instance_mock_false_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "real_key")
        synth = make_synthesizer()
        assert isinstance(synth, GroqSynthesizer)
        assert not synth._mock

    def test_groq_instance_mock_true_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        synth = GroqSynthesizer()
        assert synth._mock is True


# ---------------------------------------------------------------------------
# Agent integration: factory wires through
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Minimal stub retriever for Agent wiring tests."""

    def __init__(self, chunks: list[StoredChunk] | None = None) -> None:
        self._chunks = chunks or []

    async def retrieve(
        self, query: str, *, route_override: Route | None = None
    ) -> RetrievalResult:
        return _stub_retrieval(self._chunks)


class TestAgentUsesFactory:
    """Confirm Agent defaults to GroqSynthesizer when key is set."""

    async def test_agent_uses_groq_synth_when_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        agent = Agent(_StubRetriever())  # type: ignore[arg-type]
        assert isinstance(agent._synth, GroqSynthesizer)

    async def test_agent_uses_stitch_when_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        agent = Agent(_StubRetriever())  # type: ignore[arg-type]
        assert isinstance(agent._synth, StitchSynthesizer)
