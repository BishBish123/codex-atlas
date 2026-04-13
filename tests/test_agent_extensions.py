"""Tests for the agent's validate step + cancel/timeout + tool_calls log."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from codex_atlas.agent import (
    Agent,
    AgentConfig,
    CancelReason,
    CitationValidator,
    Node,
    ValidationReport,
)
from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.retriever import RetrievalResult, Route
from codex_atlas.store import StoredChunk


def _stored(qname: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=f"def {qname.rsplit('.', 1)[-1]}(): pass\n",
        score=0.9,
    )


def _result(
    chunks: list[StoredChunk], confidence: float = 0.9, route: Route = Route.LOOKUP
) -> RetrievalResult:
    return RetrievalResult(
        route=route,
        confidence=confidence,
        signals=["stub"],
        chunks=chunks,
        extra_qualified_names=[],
    )


@dataclass
class StubRetriever:
    responses: list[RetrievalResult]
    calls: int = 0

    async def retrieve(self, query: str) -> RetrievalResult:
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


@dataclass
class SlowRetriever:
    """Retriever that sleeps before returning, used to test timeouts."""

    delay_s: float

    async def retrieve(self, query: str) -> RetrievalResult:
        await asyncio.sleep(self.delay_s)
        return _result([_stored("m.foo")])


class TestCitationValidator:
    async def test_grounded_claim_accepted(self) -> None:
        validator = CitationValidator()
        chunks = [_stored("m.foo")]
        answer = "see `m.foo` for details"
        report = await validator.validate("q", answer, chunks)
        assert report.is_acceptable
        assert "m.foo" in report.grounded_qualified_names

    async def test_ungrounded_claim_rejected(self) -> None:
        validator = CitationValidator()
        chunks = [_stored("m.foo")]
        # answer cites a name that's NOT in the chunks.
        answer = "the function `m.bar` does the work"
        report = await validator.validate("q", answer, chunks)
        assert not report.is_acceptable
        assert "m.bar" in report.ungrounded_claims

    async def test_no_claims_treated_as_acceptable(self) -> None:
        # Hedging answer with no backticked qnames should pass — punishing
        # it would push the agent toward fabrication.
        validator = CitationValidator()
        report = await validator.validate("q", "I could not find this in the corpus.", [])
        assert report.is_acceptable
        assert report.n_claims == 0

    async def test_partial_grounding_meets_threshold(self) -> None:
        # 2/3 grounded with default threshold 0.5 -> acceptable.
        validator = CitationValidator(accept_threshold=0.5)
        chunks = [_stored("m.foo"), _stored("m.bar")]
        answer = "uses `m.foo`, `m.bar`, and `m.ghost`"
        report = await validator.validate("q", answer, chunks)
        assert report.is_acceptable
        assert "m.ghost" in report.ungrounded_claims

    async def test_threshold_can_reject_majority_grounded(self) -> None:
        # threshold 0.9 rejects anything below 90% grounded.
        validator = CitationValidator(accept_threshold=0.9)
        chunks = [_stored("m.foo")]
        answer = "uses `m.foo` and `m.ghost`"
        report = await validator.validate("q", answer, chunks)
        assert not report.is_acceptable

    async def test_suffix_match_accepts_class_method(self) -> None:
        validator = CitationValidator()
        chunks = [_stored("pkg.module.Cls.method")]
        # Answer cites just the rightmost chunk — should still match.
        answer = "see `Cls.method`"
        report = await validator.validate("q", answer, chunks)
        assert report.is_acceptable

    async def test_validator_rejects_wrong_module_suffix_match(self) -> None:
        # Retrieved chunk is in pkg_a; the answer claims pkg_b. Rejecting
        # this is the whole point of v0.2 grounding: the trailing path
        # ``Cls.method`` is the same, but the leading module prefix of the
        # claim (``pkg_b``) is not in the retrieval set's known prefixes
        # (``{pkg_a, pkg_a.Cls}``), so the claim is ungrounded.
        validator = CitationValidator()
        chunks = [_stored("pkg_a.Cls.method")]
        answer = "see `pkg_b.Cls.method` for the implementation"
        report = await validator.validate("q", answer, chunks)
        assert "pkg_b.Cls.method" in report.ungrounded_claims
        assert not report.is_acceptable

    async def test_validator_accepts_exact_qualified_match(self) -> None:
        # Retrieved + claim are byte-identical: the rule-1 exact match.
        validator = CitationValidator()
        chunks = [_stored("pkg_a.Cls.method")]
        answer = "see `pkg_a.Cls.method`"
        report = await validator.validate("q", answer, chunks)
        assert report.is_acceptable
        assert "pkg_a.Cls.method" in report.grounded_qualified_names

    async def test_validator_handles_short_unqualified_reference(self) -> None:
        # A claim shaped like ``Cls.method`` (no module) is accepted when
        # any retrieved qname ends in it. This is the suffix-only rule —
        # documented as a pass-through: the claim is too short to verify
        # the implied module, but rejecting it would punish well-formed
        # answers that use class-level shorthand.
        validator = CitationValidator()
        chunks = [_stored("pkg_a.Cls.method")]
        answer = "use `Cls.method` here"
        report = await validator.validate("q", answer, chunks)
        # Deterministic: pass-through.
        assert report.is_acceptable
        assert "Cls.method" in report.grounded_qualified_names

    async def test_zero_claims_logs_structured_event(self, caplog) -> None:  # type: ignore[no-untyped-def]
        # v0.2 policy: zero specific claims still passes, but we log it
        # so production traces can audit how often it happens.
        validator = CitationValidator()
        with caplog.at_level(logging.INFO, logger="codex_atlas.agent"):
            report = await validator.validate("q", "no specifics here", [])
        assert report.n_claims == 0
        assert report.is_acceptable
        # A structured info-log was emitted under the agent logger.
        assert any("zero_claims" in rec.message for rec in caplog.records)


class TestAgentValidationIntegration:
    async def test_run_returns_validation_report(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = Agent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        assert isinstance(result.validation, ValidationReport)

    async def test_validate_node_in_trace(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = Agent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        nodes = [e.node for e in result.trace]
        assert Node.VALIDATE in nodes


class TestToolCallsLog:
    async def test_tool_calls_recorded_per_retrieve(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = Agent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q")
        assert len(result.tool_calls) >= 1
        tc = result.tool_calls[0]
        assert tc.route is Route.LOOKUP
        assert tc.n_chunks == 1
        assert tc.confidence == 0.9

    async def test_tool_calls_count_matches_attempts(self) -> None:
        # Two empty retrievals trigger two retries -> three tool_calls total.
        retriever = StubRetriever(
            responses=[
                _result([], confidence=0.9),
                _result([], confidence=0.9),
                _result([_stored("m.foo")], confidence=0.9),
            ]
        )
        agent = Agent(retriever, config=AgentConfig(max_attempts=3))  # type: ignore[arg-type]
        result = await agent.run("q")
        # The agent retrieves once per attempt up to max_attempts.
        assert len(result.tool_calls) == result.attempts


class TestCancelOnTimeout:
    async def test_run_timeout_marks_cancelled(self) -> None:
        retriever = SlowRetriever(delay_s=0.5)
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(run_timeout_s=0.05),
        )
        result = await agent.run("q")
        assert result.cancelled is CancelReason.TIMEOUT

    async def test_step_timeout_marks_cancelled(self) -> None:
        retriever = SlowRetriever(delay_s=0.5)
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(step_timeout_s=0.05),
        )
        result = await agent.run("q")
        assert result.cancelled is CancelReason.TIMEOUT

    async def test_cancel_event_in_trace(self) -> None:
        retriever = SlowRetriever(delay_s=0.5)
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(run_timeout_s=0.05),
        )
        result = await agent.run("q")
        assert any(e.node is Node.CANCEL for e in result.trace)

    async def test_cancelled_run_returns_empty_answer(self) -> None:
        retriever = SlowRetriever(delay_s=0.5)
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(run_timeout_s=0.05),
        )
        result = await agent.run("q")
        assert result.answer == ""
        assert result.citations == []

    async def test_no_timeout_does_not_cancel(self) -> None:
        retriever = SlowRetriever(delay_s=0.01)
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(run_timeout_s=5.0),
        )
        result = await agent.run("q")
        assert result.cancelled is None
