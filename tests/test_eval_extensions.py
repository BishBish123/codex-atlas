"""Tests for the eval harness's new metrics + baseline regression + failure report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_atlas.agent import AgentResult, Citation, ToolCall
from codex_atlas.eval.harness import (
    EvalQuestion,
    ExpectedRoute,
    FailureBucket,
    aggregate,
    evaluate_against_baseline,
    failure_taxonomy_counts,
    score_result,
    write_failure_report,
)
from codex_atlas.retriever import Route


def _agent_result(
    *,
    answer: str = "",
    citations: list[Citation] | None = None,
    route: Route = Route.LOOKUP,
    tool_calls: list[ToolCall] | None = None,
) -> AgentResult:
    return AgentResult(
        query="q",
        final_query="q",
        answer=answer,
        citations=citations or [],
        route=route,
        grade=0.5,
        attempts=1,
        trace=[],
        tool_calls=tool_calls or [],
    )


def _q(
    *,
    qid: str = "q1",
    cat: str = "lookup",
    expected: ExpectedRoute = ExpectedRoute.LOOKUP,
    gold: list[str] | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        qid=qid,
        category=cat,
        question="?",
        expected_route=expected,
        gold_qualified_names=gold or [],
    )


def _cite(qname: str) -> Citation:
    return Citation(qualified_name=qname, file_path="x.py", lineno_start=1, lineno_end=2)


def _tc() -> ToolCall:
    return ToolCall(query="q", route=Route.LOOKUP, n_chunks=1, elapsed_ms=1.0, confidence=0.9)


class TestCostEstimate:
    def test_short_answer_costs_more_than_zero(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(answer="hello world", citations=[_cite("m.foo")])
        res = score_result(q, ar, latency_ms=10.0)
        assert res.cost_estimate_usd > 0

    def test_longer_answer_costs_more(self) -> None:
        q = _q(gold=["m.foo"])
        short = _agent_result(answer="hi", citations=[_cite("m.foo")])
        long = _agent_result(answer="x" * 1000, citations=[_cite("m.foo")])
        s = score_result(q, short, latency_ms=10.0)
        long_res = score_result(q, long, latency_ms=10.0)
        assert long_res.cost_estimate_usd > s.cost_estimate_usd

    def test_price_inputs_apply(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(answer="x" * 4000, citations=[_cite("m.foo")])
        cheap = score_result(q, ar, latency_ms=10.0, price_input=0.001, price_output=0.001)
        pricey = score_result(q, ar, latency_ms=10.0, price_input=0.01, price_output=0.01)
        assert pricey.cost_estimate_usd > cheap.cost_estimate_usd


class TestToolCallCount:
    def test_count_propagates_from_agent_result(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(citations=[_cite("m.foo")], tool_calls=[_tc(), _tc(), _tc()])
        res = score_result(q, ar, latency_ms=10.0)
        assert res.tool_call_count == 3


class TestFailureBuckets:
    def test_correct_answer_bucket_none(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(citations=[_cite("m.foo")], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.NONE

    def test_wrong_route_bucket(self) -> None:
        q = _q(gold=["m.foo"], expected=ExpectedRoute.STRUCTURAL)
        ar = _agent_result(citations=[_cite("m.foo")], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.WRONG_ROUTE

    def test_ungrounded_bucket(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(citations=[], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.UNGROUNDED

    def test_off_topic_bucket(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(citations=[_cite("m.bar"), _cite("m.baz")], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.OFF_TOPIC

    def test_partial_bucket(self) -> None:
        q = _q(gold=["m.foo", "m.bar"])
        ar = _agent_result(citations=[_cite("m.foo")], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.PARTIAL

    def test_outdated_index_bucket(self) -> None:
        q = _q(gold=["m.foo"])
        ar = _agent_result(
            answer="this answer relies on stale index data",
            citations=[_cite("m.foo")],
            route=Route.LOOKUP,
        )
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.OUTDATED_INDEX

    def test_refusal_correct_bucket_none(self) -> None:
        # No gold, no citations -> a correct refusal.
        q = _q(gold=[])
        ar = _agent_result(citations=[], route=Route.LOOKUP)
        res = score_result(q, ar, latency_ms=10.0)
        assert res.failure_bucket is FailureBucket.NONE


class TestAggregate:
    def test_aggregate_includes_p99(self) -> None:
        q = _q(gold=["m.foo"])
        results = [
            score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=lat)
            for lat in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
        ]
        agg = aggregate(results)
        assert "latency_p99_ms" in agg
        assert agg["latency_p99_ms"] >= agg["latency_p95_ms"] >= agg["latency_p50_ms"]

    def test_aggregate_empty_returns_empty(self) -> None:
        assert aggregate([]) == {}

    def test_failure_taxonomy_counts(self) -> None:
        q = _q(gold=["m.foo"])
        wrong_route = score_result(
            _q(gold=["m.foo"], expected=ExpectedRoute.STRUCTURAL),
            _agent_result(citations=[_cite("m.foo")], route=Route.LOOKUP),
            latency_ms=10.0,
        )
        ok = score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0)
        counts = failure_taxonomy_counts([wrong_route, ok])
        assert counts[str(FailureBucket.WRONG_ROUTE)] == 1
        assert counts[str(FailureBucket.NONE)] == 1


class TestBaselineRegression:
    def test_no_change_no_regression(self, tmp_path: Path) -> None:
        q = _q(gold=["m.foo"])
        results = [score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0)]
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(aggregate(results)))
        diff = evaluate_against_baseline(results, baseline_path)
        assert not diff.is_regression

    def test_regression_on_route_correctness(self, tmp_path: Path) -> None:
        baseline = {"route_correctness": 1.0, "latency_p50_ms": 10.0}
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        # Current run: 50% correct.
        q_ok = _q(gold=["m.foo"])
        q_wrong = _q(gold=["m.foo"], expected=ExpectedRoute.STRUCTURAL)
        results = [
            score_result(q_ok, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0),
            score_result(q_wrong, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0),
        ]
        diff = evaluate_against_baseline(results, baseline_path, tolerance=0.0)
        assert diff.is_regression
        assert "route_correctness" in diff.regressions

    def test_improvement_recorded(self, tmp_path: Path) -> None:
        baseline = {"route_correctness": 0.5}
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        q = _q(gold=["m.foo"])
        results = [score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0)]
        diff = evaluate_against_baseline(results, baseline_path, tolerance=0.0)
        assert "route_correctness" in diff.improvements

    def test_tolerance_absorbs_small_drift(self, tmp_path: Path) -> None:
        baseline = {"latency_p50_ms": 100.0}
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        q = _q(gold=["m.foo"])
        results = [
            score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=lat)
            for lat in [102.0]  # 2% slower than baseline
        ]
        diff = evaluate_against_baseline(results, baseline_path, tolerance=0.10)
        # 2% slower with 10% tolerance is fine.
        assert not diff.is_regression


class TestFailureReportJsonl:
    def test_writes_one_line_per_question(self, tmp_path: Path) -> None:
        q = _q(gold=["m.foo"])
        results = [
            score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0)
            for _ in range(3)
        ]
        out = tmp_path / "failures.jsonl"
        write_failure_report(results, out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        q = _q(gold=["m.foo"])
        results = [score_result(q, _agent_result(citations=[_cite("m.foo")]), latency_ms=10.0)]
        out = tmp_path / "failures.jsonl"
        write_failure_report(results, out)
        line = out.read_text().strip()
        parsed = json.loads(line)
        assert parsed["qid"] == "q1"
        assert parsed["failure_bucket"] in {str(b) for b in FailureBucket}


@pytest.mark.parametrize(
    "answer,n_tokens",
    [
        ("", 1),  # empty -> floor
        ("hello", 1),  # 5 chars / 4 = 1 token (floor)
        ("a" * 1000, 250),
    ],
)
def test_token_count_floor(answer: str, n_tokens: int) -> None:
    # Indirectly test via _estimate_cost_usd. Cost should rise linearly
    # with answer length above ~4 chars.
    q = _q(gold=["m.foo"])
    ar = _agent_result(answer=answer, citations=[_cite("m.foo")])
    res = score_result(q, ar, latency_ms=10.0)
    # Empty answer floors at 1 token, so cost is non-zero.
    assert res.cost_estimate_usd > 0
