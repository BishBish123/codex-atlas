"""Tests for the RAGAS metric layer.

Coverage:
- Each heuristic with hand-built fixtures (perfect / no-overlap / partial).
- ``make_ragas_metrics()`` factory: with-library (mocked) returns library
  wrappers; without returns heuristics.
- Library wrapper smoke: ragas mocked → assert it's called with the right args.
- Aggregate end-to-end: ``atlas eval --metrics ragas`` produces a REPORT.md
  with all 4 RAGAS rows.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codex_atlas.cli import app
from codex_atlas.eval.ragas.factory import make_ragas_metrics
from codex_atlas.eval.ragas.heuristic import (
    AnswerRelevancyHeuristic,
    ContextPrecisionHeuristic,
    ContextRecallHeuristic,
    FaithfulnessHeuristic,
)
from codex_atlas.eval.ragas.library import RagasLibraryWrapper
from codex_atlas.eval.ragas.protocol import RagasMetric

# ---------------------------------------------------------------------------
# FaithfulnessHeuristic
# ---------------------------------------------------------------------------


class TestFaithfulnessHeuristic:
    @pytest.fixture()
    def metric(self) -> FaithfulnessHeuristic:
        return FaithfulnessHeuristic()

    def test_name(self, metric: FaithfulnessHeuristic) -> None:
        assert metric.name == "faithfulness"

    @pytest.mark.asyncio()
    async def test_empty_ground_truth_vacuous(self, metric: FaithfulnessHeuristic) -> None:
        score = await metric.compute(
            question="q", contexts=[], answer="any text", ground_truth=[]
        )
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_perfect_match_all_in_answer(self, metric: FaithfulnessHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="pkg.Foo and pkg.Bar are the core classes.",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_no_overlap_score_zero(self, metric: FaithfulnessHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="completely unrelated answer about weather",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio()
    async def test_partial_overlap(self, metric: FaithfulnessHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="pkg.Foo is here",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio()
    async def test_case_insensitive(self, metric: FaithfulnessHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="PKG.FOO handles everything.",
            ground_truth=["pkg.Foo"],
        )
        assert score == pytest.approx(1.0)

    def test_satisfies_protocol(self, metric: FaithfulnessHeuristic) -> None:
        assert isinstance(metric, RagasMetric)


# ---------------------------------------------------------------------------
# AnswerRelevancyHeuristic
# ---------------------------------------------------------------------------


class TestAnswerRelevancyHeuristic:
    @pytest.fixture()
    def metric(self) -> AnswerRelevancyHeuristic:
        return AnswerRelevancyHeuristic()

    def test_name(self, metric: AnswerRelevancyHeuristic) -> None:
        assert metric.name == "answer_relevancy"

    @pytest.mark.asyncio()
    async def test_identical_content_max_score(self, metric: AnswerRelevancyHeuristic) -> None:
        score = await metric.compute(
            question="what does retriever do",
            contexts=[],
            answer="retriever does retrieval work",
            ground_truth=[],
        )
        # "retriever" is in both → high cosine sim
        assert score > 0.5

    @pytest.mark.asyncio()
    async def test_no_keyword_overlap_low_score(
        self, metric: AnswerRelevancyHeuristic
    ) -> None:
        score = await metric.compute(
            question="what does retriever do",
            contexts=[],
            answer="the sky is blue and cats are nice",
            ground_truth=[],
        )
        assert score < 0.5

    @pytest.mark.asyncio()
    async def test_empty_question_vacuous(self, metric: AnswerRelevancyHeuristic) -> None:
        score = await metric.compute(
            question="",
            contexts=[],
            answer="some answer text about code",
            ground_truth=[],
        )
        # Empty keyword set → vacuously 1.0
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_empty_answer_vacuous(self, metric: AnswerRelevancyHeuristic) -> None:
        score = await metric.compute(
            question="what does retriever do",
            contexts=[],
            answer="",
            ground_truth=[],
        )
        assert score == 1.0

    def test_satisfies_protocol(self, metric: AnswerRelevancyHeuristic) -> None:
        assert isinstance(metric, RagasMetric)


# ---------------------------------------------------------------------------
# ContextPrecisionHeuristic
# ---------------------------------------------------------------------------


class TestContextPrecisionHeuristic:
    @pytest.fixture()
    def metric(self) -> ContextPrecisionHeuristic:
        return ContextPrecisionHeuristic()

    def test_name(self, metric: ContextPrecisionHeuristic) -> None:
        assert metric.name == "context_precision"

    @pytest.mark.asyncio()
    async def test_no_contexts_vacuous(self, metric: ContextPrecisionHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="a",
            ground_truth=["pkg.Foo"],
        )
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_no_ground_truth_vacuous(self, metric: ContextPrecisionHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["some chunk about stuff"],
            answer="a",
            ground_truth=[],
        )
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_all_chunks_relevant(self, metric: ContextPrecisionHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["pkg.Foo is a class", "pkg.Bar is a method"],
            answer="a",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_no_chunks_relevant(self, metric: ContextPrecisionHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["unrelated chunk about weather", "another irrelevant chunk"],
            answer="a",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio()
    async def test_partial_precision(self, metric: ContextPrecisionHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["pkg.Foo is here", "nothing relevant here"],
            answer="a",
            ground_truth=["pkg.Foo"],
        )
        assert score == pytest.approx(0.5)

    def test_satisfies_protocol(self, metric: ContextPrecisionHeuristic) -> None:
        assert isinstance(metric, RagasMetric)


# ---------------------------------------------------------------------------
# ContextRecallHeuristic
# ---------------------------------------------------------------------------


class TestContextRecallHeuristic:
    @pytest.fixture()
    def metric(self) -> ContextRecallHeuristic:
        return ContextRecallHeuristic()

    def test_name(self, metric: ContextRecallHeuristic) -> None:
        assert metric.name == "context_recall"

    @pytest.mark.asyncio()
    async def test_empty_ground_truth_vacuous(self, metric: ContextRecallHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["some chunk"],
            answer="a",
            ground_truth=[],
        )
        assert score == 1.0

    @pytest.mark.asyncio()
    async def test_no_contexts_zero(self, metric: ContextRecallHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=[],
            answer="a",
            ground_truth=["pkg.Foo"],
        )
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio()
    async def test_all_gold_recalled(self, metric: ContextRecallHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["chunk mentioning pkg.Foo and pkg.Bar"],
            answer="a",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_no_gold_recalled(self, metric: ContextRecallHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["chunk about weather and sunshine"],
            answer="a",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio()
    async def test_partial_recall(self, metric: ContextRecallHeuristic) -> None:
        score = await metric.compute(
            question="q",
            contexts=["chunk mentioning pkg.Foo"],
            answer="a",
            ground_truth=["pkg.Foo", "pkg.Bar"],
        )
        assert score == pytest.approx(0.5)

    def test_satisfies_protocol(self, metric: ContextRecallHeuristic) -> None:
        assert isinstance(metric, RagasMetric)


# ---------------------------------------------------------------------------
# make_ragas_metrics factory
# ---------------------------------------------------------------------------


class TestMakeRagasMetrics:
    def test_returns_four_metrics(self) -> None:
        with patch.dict("sys.modules", {"ragas": None}):
            # ragas not installed → heuristics
            metrics = make_ragas_metrics()
        assert len(metrics) == 4

    def test_metric_names(self) -> None:
        with patch.dict("sys.modules", {"ragas": None}):
            metrics = make_ragas_metrics()
        names = [m.name for m in metrics]
        assert names == [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    def test_without_library_returns_heuristics(self) -> None:
        with patch.dict("sys.modules", {"ragas": None}):
            metrics = make_ragas_metrics()
        for m in metrics:
            assert not isinstance(m, RagasLibraryWrapper)

    def test_with_library_returns_wrappers(self) -> None:
        fake_ragas = MagicMock()
        with patch.dict("sys.modules", {"ragas": fake_ragas, "ragas.metrics": MagicMock()}):
            metrics = make_ragas_metrics()
        assert all(isinstance(m, RagasLibraryWrapper) for m in metrics)

    def test_all_satisfy_protocol(self) -> None:
        with patch.dict("sys.modules", {"ragas": None}):
            metrics = make_ragas_metrics()
        for m in metrics:
            assert isinstance(m, RagasMetric)


# ---------------------------------------------------------------------------
# RagasLibraryWrapper smoke tests
# ---------------------------------------------------------------------------


class TestRagasLibraryWrapper:
    def _make_wrapper(self, ragas_name: str) -> RagasLibraryWrapper:
        return RagasLibraryWrapper(ragas_name, FaithfulnessHeuristic())

    @pytest.mark.asyncio()
    async def test_falls_back_to_heuristic_when_no_library(self) -> None:
        wrapper = self._make_wrapper("faithfulness")
        with patch.dict("sys.modules", {"ragas": None, "ragas.metrics": None}):
            # Reset resolution cache
            wrapper._resolved = False
            wrapper._ragas_metric = None
            score = await wrapper.compute(
                question="q",
                contexts=[],
                answer="pkg.Foo is here",
                ground_truth=["pkg.Foo"],
            )
        # Heuristic: pkg.Foo appears verbatim in answer → 1.0
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_falls_back_without_llm_key(self) -> None:
        """Library installed but no LLM key → heuristic fallback."""
        wrapper = self._make_wrapper("faithfulness")
        fake_ragas = MagicMock()
        env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
        wrapper._resolved = False
        wrapper._ragas_metric = None
        with (
            patch.dict("sys.modules", {"ragas": fake_ragas, "ragas.metrics": MagicMock()}),
            patch.dict("os.environ", env, clear=True),
        ):
            score = await wrapper.compute(
                question="q",
                contexts=[],
                answer="pkg.Foo is here",
                ground_truth=["pkg.Foo"],
            )
        # Heuristic fallback: 1.0
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_calls_ragas_when_resolved(self) -> None:
        """When _resolve() returns a metric, compute delegates to it."""
        mock_metric = MagicMock()
        mock_metric.score.return_value = {"faithfulness": 0.75}

        mock_dataset_cls = MagicMock()
        mock_dataset_cls.from_dict.return_value = MagicMock()

        wrapper = self._make_wrapper("faithfulness")

        import codex_atlas.eval.ragas.library as lib_mod

        def fake_resolve(self: RagasLibraryWrapper) -> object:
            return mock_metric

        original_resolve = lib_mod.RagasLibraryWrapper._resolve
        lib_mod.RagasLibraryWrapper._resolve = fake_resolve  # type: ignore[method-assign]
        try:
            with patch.dict("sys.modules", {"datasets": MagicMock(Dataset=mock_dataset_cls)}):
                score = await wrapper.compute(
                    question="q",
                    contexts=["context chunk"],
                    answer="some answer",
                    ground_truth=["pkg.Foo"],
                )
        finally:
            lib_mod.RagasLibraryWrapper._resolve = original_resolve  # type: ignore[method-assign]

        # mock_metric.score was called, returned {"faithfulness": 0.75}
        assert score == pytest.approx(0.75)

    @pytest.mark.asyncio()
    async def test_falls_back_on_ragas_exception(self) -> None:
        """If ragas raises during compute, heuristic fallback is used."""
        mock_metric = MagicMock()
        mock_metric.score.side_effect = RuntimeError("ragas internal error")

        wrapper = self._make_wrapper("faithfulness")

        import codex_atlas.eval.ragas.library as lib_mod

        def fake_resolve(self: RagasLibraryWrapper) -> object:
            return mock_metric

        original_resolve = lib_mod.RagasLibraryWrapper._resolve
        lib_mod.RagasLibraryWrapper._resolve = fake_resolve  # type: ignore[method-assign]
        try:
            mock_dataset_cls = MagicMock()
            mock_dataset_cls.from_dict.return_value = MagicMock()
            with patch.dict("sys.modules", {"datasets": MagicMock(Dataset=mock_dataset_cls)}):
                score = await wrapper.compute(
                    question="q",
                    contexts=[],
                    answer="pkg.Foo is here",
                    ground_truth=["pkg.Foo"],
                )
        finally:
            lib_mod.RagasLibraryWrapper._resolve = original_resolve  # type: ignore[method-assign]

        # Heuristic fallback: pkg.Foo in answer → 1.0
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end: atlas eval --metrics ragas produces REPORT.md with RAGAS rows
# ---------------------------------------------------------------------------


class TestEvalMetricsFlag:
    def test_metrics_ragas_produces_report_with_all_four_rows(self) -> None:
        """atlas eval --metrics ragas writes REPORT.md with all 4 RAGAS rows."""
        runner = CliRunner()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
        }
        result = runner.invoke(
            app,
            ["eval", "--judge", "heuristic", "--metrics", "ragas"],
            env=env,
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        report_path = Path("evals/REPORT.md")
        assert report_path.exists(), "REPORT.md was not written"
        report = report_path.read_text()

        assert re.search(r"Faithfulness \(RAGAS\)", report), "Missing RAGAS faithfulness row"
        assert re.search(r"Answer relevancy \(RAGAS\)", report), "Missing answer relevancy row"
        assert re.search(r"Context precision \(RAGAS\)", report), "Missing context precision row"
        assert re.search(r"Context recall \(RAGAS\)", report), "Missing context recall row"

    def test_metrics_custom_only_skips_ragas_rows(self) -> None:
        """atlas eval --metrics custom skips RAGAS scoring (shows n/a)."""
        runner = CliRunner()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
        }
        result = runner.invoke(
            app,
            ["eval", "--judge", "heuristic", "--metrics", "custom"],
            env=env,
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        report_path = Path("evals/REPORT.md")
        assert report_path.exists()
        report = report_path.read_text()

        # RAGAS rows still render (always in the template) but values are n/a
        assert "| Faithfulness (RAGAS) | n/a |" in report
        assert "| Answer relevancy (RAGAS) | n/a |" in report
        assert "| Context precision (RAGAS) | n/a |" in report
        assert "| Context recall (RAGAS) | n/a |" in report

    def test_default_metrics_includes_ragas(self) -> None:
        """Default (--metrics custom,ragas) produces real RAGAS values."""
        runner = CliRunner()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
        }
        result = runner.invoke(
            app,
            ["eval", "--judge", "heuristic"],
            env=env,
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        report = Path("evals/REPORT.md").read_text()
        # All 4 rows should have numeric values (not n/a) under default mode
        for label in [
            "Faithfulness (RAGAS)",
            "Answer relevancy (RAGAS)",
            "Context precision (RAGAS)",
            "Context recall (RAGAS)",
        ]:
            match = re.search(rf"\| {re.escape(label)} \| ([0-9.]+|n/a) \|", report)
            assert match is not None, f"Row '{label}' not found in report"
            # Default mode computes RAGAS → should be numeric
            assert match.group(1) != "n/a", f"Expected numeric value for '{label}'"
