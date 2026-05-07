"""Tests for the codex_atlas observability (Langfuse tracing) module.

Coverage:
- No env vars → make_tracer() returns NullTracer.
- NullTracer no-ops: start_run / record_event / record_validation / finish_run
  all return without raising and produce the expected sentinel values.
- With env vars but no SDK installed → LangfuseTracer falls back to no-op;
  agent still runs cleanly.
- With a mocked Langfuse SDK → LangfuseTracer in live mode; assert that
  the right SDK calls are made with the right call counts.
- Method-call counting: 1 trace per run, N events per trace
  (matching len(state.trace) + len(state.tool_calls)), 1 validation if
  any, 1 finish.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from codex_atlas.agent import (
    Agent,
    Node,
    TraceEvent,
    ValidationReport,
)
from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.observability import LangfuseTracer, NullTracer, make_tracer
from codex_atlas.observability.langfuse import make_tracer as _make_tracer_direct
from codex_atlas.retriever import RetrievalResult, Route
from codex_atlas.store import StoredChunk

# ---------------------------------------------------------------------------
# Helpers shared with test_agent.py
# ---------------------------------------------------------------------------


def _stored(qname: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=f"def {qname.rsplit('.', maxsplit=1)[-1]}(): pass\n",
        score=0.9,
    )


@dataclass
class StubRetriever:
    responses: list[RetrievalResult]
    calls: int = 0

    async def retrieve(
        self, query: str, *, route_override: Route | None = None
    ) -> RetrievalResult:
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _result(
    chunks: list[StoredChunk],
    confidence: float = 0.9,
    route: Route = Route.LOOKUP,
) -> RetrievalResult:
    return RetrievalResult(
        route=route,
        confidence=confidence,
        signals=["stub"],
        chunks=chunks,
        extra_qualified_names=[],
    )


# ---------------------------------------------------------------------------
# 1. make_tracer() with no env vars → NullTracer
# ---------------------------------------------------------------------------


def test_make_tracer_no_env_returns_null_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = make_tracer()
    assert isinstance(tracer, NullTracer)


def test_make_tracer_only_public_key_returns_null_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = make_tracer()
    assert isinstance(tracer, NullTracer)


def test_make_tracer_only_secret_key_returns_null_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    tracer = make_tracer()
    assert isinstance(tracer, NullTracer)


# ---------------------------------------------------------------------------
# 2. NullTracer interface contract
# ---------------------------------------------------------------------------


def test_null_tracer_start_run_returns_sentinel() -> None:
    t = NullTracer()
    tid = t.start_run("what does Encoder do")
    assert isinstance(tid, str)
    assert tid == "null"


def test_null_tracer_record_event_does_not_raise() -> None:
    t = NullTracer()
    tid = t.start_run("q")
    ev = TraceEvent(node=Node.RETRIEVE, started_at=0.0, elapsed_ms=10.0, detail="")
    t.record_event(tid, ev)  # must not raise


def test_null_tracer_record_validation_does_not_raise() -> None:
    t = NullTracer()
    tid = t.start_run("q")
    report = ValidationReport(
        n_claims=1,
        n_grounded=1,
        ungrounded_claims=[],
        grounded_qualified_names=["m.fn"],
        is_acceptable=True,
    )
    t.record_validation(tid, report)  # must not raise


@pytest.mark.asyncio
async def test_null_tracer_finish_run_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """finish_run with NullTracer must not raise even with a real AgentResult."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = NullTracer()
    tid = t.start_run("q")

    # Build a minimal AgentResult via Agent.run so we have the right type.
    retriever = StubRetriever(responses=[_result([_stored("m.fn")])])
    agent = Agent(retriever, tracer=t)

    result = await agent.run("q")
    t.finish_run(tid, result)  # must not raise


# ---------------------------------------------------------------------------
# 3. Agent.run works unchanged with NullTracer (no env vars)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_run_with_null_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    chunks = [_stored("m.fn")]
    retriever = StubRetriever(responses=[_result(chunks)])
    agent = Agent(retriever)

    result = await agent.run("what does m.fn do")
    assert result.answer != ""
    assert result.attempts == 1
    assert isinstance(agent._tracer, NullTracer)


# ---------------------------------------------------------------------------
# 4. With env vars but no SDK → LangfuseTracer falls back to no-op
# ---------------------------------------------------------------------------


def test_langfuse_tracer_falls_back_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LangfuseTracer.__init__ ImportError path → _client stays None."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    # Temporarily hide the langfuse package even if it's installed.
    original = sys.modules.get("langfuse")
    sys.modules["langfuse"] = None  # type: ignore[assignment]
    try:
        tracer = LangfuseTracer()
    finally:
        if original is None:
            sys.modules.pop("langfuse", None)
        else:
            sys.modules["langfuse"] = original

    assert tracer._client is None
    # All methods must still be no-ops.
    tid = tracer.start_run("q")
    assert tid == "null"
    ev = TraceEvent(node=Node.GRADE, started_at=0.0, elapsed_ms=5.0, detail="grade=1.0")
    tracer.record_event(tid, ev)
    tracer.record_validation(
        tid,
        ValidationReport(
            n_claims=0,
            n_grounded=0,
            ungrounded_claims=[],
            grounded_qualified_names=[],
            is_acceptable=True,
        ),
    )


@pytest.mark.asyncio
async def test_agent_run_clean_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent.run completes without raising when langfuse SDK is absent."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    original = sys.modules.get("langfuse")
    sys.modules["langfuse"] = None  # type: ignore[assignment]
    try:
        tracer = LangfuseTracer()
        chunks = [_stored("m.fn")]
        retriever = StubRetriever(responses=[_result(chunks)])
        agent = Agent(retriever, tracer=tracer)
        result = await agent.run("what does m.fn do")
    finally:
        if original is None:
            sys.modules.pop("langfuse", None)
        else:
            sys.modules["langfuse"] = original

    assert result.answer != ""


# ---------------------------------------------------------------------------
# 5. Mocked Langfuse SDK — live mode call counts
# ---------------------------------------------------------------------------


def _build_mock_langfuse() -> tuple[MagicMock, MagicMock]:
    """Return (mock_module, mock_client) with v2/v3-style API."""
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_trace.id = "trace-123"
    mock_span = MagicMock()
    mock_trace.span.return_value = mock_span
    mock_client.trace.return_value = mock_trace

    mock_module = MagicMock()
    mock_module.Langfuse.return_value = mock_client
    # v2/v3: no get_client attribute → uses Langfuse() constructor.
    del mock_module.get_client
    return mock_module, mock_client


@pytest.mark.asyncio
async def test_langfuse_live_mode_call_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """With mocked SDK: 1 trace, N events, 1 validation span, 1 finish."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    mock_module, mock_client = _build_mock_langfuse()
    mock_trace = mock_client.trace.return_value

    with patch.dict(sys.modules, {"langfuse": mock_module}):
        tracer = LangfuseTracer()
        assert tracer._client is not None

        chunks = [_stored("m.fn")]
        retriever = StubRetriever(responses=[_result(chunks)])
        agent = Agent(retriever, tracer=tracer)
        result = await agent.run("what does m.fn do")

    # 1 trace was created.
    mock_client.trace.assert_called_once()

    # flush was called once at finish_run.
    mock_client.flush.assert_called_once()

    # span() was called for events + validation. Count them:
    # _retrieve appends CLASSIFY + RETRIEVE trace events → 2 spans
    # _grade appends GRADE event → 1 span
    # _answer appends ANSWER event → 1 span
    # _validate appends VALIDATE event → 1 span
    # record_event for tool_call → 1 span
    # record_validation → 1 span
    # Total: at least 7 span calls.
    span_call_count = mock_trace.span.call_count
    assert span_call_count >= 7, f"expected >= 7 span calls, got {span_call_count}"

    # trace.update was called at finish_run.
    mock_trace.update.assert_called()

    # Result is valid.
    assert result.answer != ""
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_langfuse_live_mode_tool_call_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool calls produce spans with route in the name."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    mock_module, mock_client = _build_mock_langfuse()
    mock_trace = mock_client.trace.return_value

    with patch.dict(sys.modules, {"langfuse": mock_module}):
        tracer = LangfuseTracer()
        chunks = [_stored("m.fn")]
        retriever = StubRetriever(responses=[_result(chunks, route=Route.LOOKUP)])
        agent = Agent(retriever, tracer=tracer)
        await agent.run("what does m.fn do")

    # At least one span should have "tool_call:" in its name.
    span_names = [
        call.args[0] if call.args else call.kwargs.get("name", "")
        for call in mock_trace.span.call_args_list
    ]
    assert any("tool_call:" in n for n in span_names), f"span names: {span_names}"


# ---------------------------------------------------------------------------
# 6. make_tracer() with both env vars → returns LangfuseTracer (with mock SDK)
# ---------------------------------------------------------------------------


def test_make_tracer_with_env_vars_returns_langfuse_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_module, _ = _build_mock_langfuse()
    with patch.dict(sys.modules, {"langfuse": mock_module}):
        tracer = _make_tracer_direct()

    assert isinstance(tracer, LangfuseTracer)


# ---------------------------------------------------------------------------
# 7. tracer constructor arg on Agent overrides default make_tracer()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_accepts_explicit_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit NullTracer passed to Agent.__init__ is used verbatim."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    null_tracer = NullTracer()
    chunks = [_stored("m.fn")]
    retriever = StubRetriever(responses=[_result(chunks)])
    agent = Agent(retriever, tracer=null_tracer)

    assert agent._tracer is null_tracer
    result = await agent.run("q")
    assert result.answer != ""


# ---------------------------------------------------------------------------
# 8. Validation report is emitted once per run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_report_emitted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    mock_module, mock_client = _build_mock_langfuse()
    mock_trace = mock_client.trace.return_value

    with patch.dict(sys.modules, {"langfuse": mock_module}):
        tracer = LangfuseTracer()
        chunks = [_stored("m.fn")]
        retriever = StubRetriever(responses=[_result(chunks)])
        agent = Agent(retriever, tracer=tracer)
        await agent.run("q")

    # Exactly one "validation" span should have been created.
    validation_spans = [
        call
        for call in mock_trace.span.call_args_list
        if (call.args and "validation" in str(call.args[0]))
        or (call.kwargs and "validation" in str(call.kwargs.get("name", "")))
    ]
    assert len(validation_spans) == 1, f"expected 1 validation span, got {len(validation_spans)}"
