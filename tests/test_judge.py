"""Tests for the codex_atlas.judge package.

Tests are grouped into:
- HeuristicJudge correctness
- LLMJudge with mocked httpx (no real LLM calls)
- make_judge factory
- Cohen's kappa formula
- CLI: --judge flag integration
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from codex_atlas.cli import app
from codex_atlas.judge.calibration import JudgeAgreement, cohens_kappa
from codex_atlas.judge.factory import make_judge
from codex_atlas.judge.heuristic import HeuristicJudge
from codex_atlas.judge.llm import LLMJudge
from codex_atlas.judge.protocol import JudgeProtocol

# ---------------------------------------------------------------------------
# HeuristicJudge
# ---------------------------------------------------------------------------


class TestHeuristicJudge:
    """Correctness tests for the fraction-cited / text-substring heuristic."""

    @pytest.fixture()
    def judge(self) -> HeuristicJudge:
        return HeuristicJudge()

    @pytest.mark.asyncio()
    async def test_empty_expected_names_vacuously_faithful(
        self, judge: HeuristicJudge
    ) -> None:
        result = await judge.score(
            question="anything",
            expected_qualified_names=[],
            answer="some answer",
            citations=[],
        )
        assert result.score == 1.0
        assert result.mode == "heuristic"

    @pytest.mark.asyncio()
    async def test_perfect_citation_match(self, judge: HeuristicJudge) -> None:
        """All expected names in citations → score 1.0."""
        result = await judge.score(
            question="q",
            expected_qualified_names=["pkg.Foo", "pkg.Bar"],
            answer="some text",
            citations=["pkg.Foo", "pkg.Bar"],
        )
        assert result.score == 1.0
        assert result.mode == "heuristic"

    @pytest.mark.asyncio()
    async def test_no_overlap_score_zero(self, judge: HeuristicJudge) -> None:
        """No citations, no text match → score 0.0."""
        result = await judge.score(
            question="q",
            expected_qualified_names=["pkg.Foo", "pkg.Bar"],
            answer="completely unrelated text about weather",
            citations=[],
        )
        assert result.score == 0.0
        assert result.mode == "heuristic"

    @pytest.mark.asyncio()
    async def test_partial_citation_fractional_score(
        self, judge: HeuristicJudge
    ) -> None:
        """Half the gold names cited → score 0.5."""
        result = await judge.score(
            question="q",
            expected_qualified_names=["pkg.Foo", "pkg.Bar"],
            answer="irrelevant",
            citations=["pkg.Foo"],
        )
        assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio()
    async def test_text_fallback_when_no_citations(
        self, judge: HeuristicJudge
    ) -> None:
        """Name appears verbatim in answer text but not in citations."""
        result = await judge.score(
            question="q",
            expected_qualified_names=["pkg.Foo"],
            answer="The pkg.Foo class does the heavy lifting.",
            citations=[],
        )
        assert result.score == 1.0

    @pytest.mark.asyncio()
    async def test_suffix_match_in_citations(self, judge: HeuristicJudge) -> None:
        """Suffix match (short name vs fully-qualified) counts as a hit."""
        result = await judge.score(
            question="q",
            expected_qualified_names=["mymod.MyClass.my_method"],
            answer="some text",
            citations=["MyClass.my_method"],
        )
        assert result.score == 1.0

    @pytest.mark.asyncio()
    async def test_score_is_max_of_citation_and_text(
        self, judge: HeuristicJudge
    ) -> None:
        """The final score is max(citation_frac, text_frac), not their average."""
        # citation_fraction = 0.5, text_fraction = 1.0 → score = 1.0
        result = await judge.score(
            question="q",
            expected_qualified_names=["pkg.Foo", "pkg.Bar"],
            answer="pkg.Foo is here and pkg.Bar is also here",
            citations=["pkg.Foo"],
        )
        assert result.score == 1.0

    def test_protocol_satisfaction(self) -> None:
        """HeuristicJudge satisfies JudgeProtocol at runtime."""
        assert isinstance(HeuristicJudge(), JudgeProtocol)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """LLMJudge tests with mocked httpx — no real API calls."""

    def _make_anthropic_response(self, score: float, rationale: str) -> Any:
        """Build a fake Anthropic response object."""
        body = {
            "content": [
                {"text": json.dumps({"score": score, "rationale": rationale})}
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _make_openai_response(self, score: float, rationale: str) -> Any:
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"score": score, "rationale": rationale})
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio()
    async def test_parses_anthropic_json_response(self) -> None:
        """LLMJudge correctly parses a well-formed Anthropic response."""

        mock_resp = self._make_anthropic_response(0.8, "Covers most key points.")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            judge = LLMJudge(anthropic_key="test-key-anthropic")
            result = await judge.score(
                question="what does Foo do",
                expected_qualified_names=["pkg.Foo"],
                answer="Foo does X.",
                citations=["pkg.Foo"],
            )

        assert result.score == pytest.approx(0.8)
        assert result.rationale == "Covers most key points."
        assert result.mode == "llm"

    @pytest.mark.asyncio()
    async def test_parses_openai_json_response(self) -> None:
        """LLMJudge correctly parses a well-formed OpenAI response."""

        mock_resp = self._make_openai_response(0.5, "Partial coverage.")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            judge = LLMJudge(openai_key="test-key-openai")
            result = await judge.score(
                question="explain Foo",
                expected_qualified_names=["pkg.Foo", "pkg.Bar"],
                answer="Foo does Y.",
                citations=["pkg.Foo"],
            )

        assert result.score == pytest.approx(0.5)
        assert result.mode == "llm"

    @pytest.mark.asyncio()
    async def test_malformed_response_falls_back_to_heuristic(self) -> None:
        """Malformed LLM JSON → fail-soft to HeuristicJudge."""

        bad_resp = MagicMock()
        bad_resp.json.return_value = {"content": [{"text": "not valid json {{{"}]}
        bad_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=bad_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="what does Foo do",
                expected_qualified_names=["pkg.Foo"],
                answer="pkg.Foo is great",
                citations=["pkg.Foo"],
            )

        # Falls back to heuristic; answer text contains pkg.Foo → score = 1.0
        assert result.mode == "heuristic"
        assert result.score == 1.0

    def test_no_key_raises_os_error(self) -> None:
        """LLMJudge raises OSError when no API key is available."""

        env_backup = {
            k: os.environ.pop(k)
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if k in os.environ
        }
        try:
            with pytest.raises(OSError, match="API key"):
                LLMJudge()
        finally:
            os.environ.update(env_backup)

    @pytest.mark.asyncio()
    async def test_score_clamped_to_unit_interval(self) -> None:
        """LLMJudge clamps scores outside [0, 1] to the boundary."""

        # LLM returns 1.5 → should be clamped to 1.0
        mock_resp = self._make_anthropic_response(1.5, "Over-confident.")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="q",
                expected_qualified_names=["pkg.Foo"],
                answer="Foo is everything.",
                citations=["pkg.Foo"],
            )

        assert result.score == pytest.approx(1.0)
        assert result.mode == "llm"


# ---------------------------------------------------------------------------
# make_judge factory
# ---------------------------------------------------------------------------


class TestMakeJudge:
    def test_heuristic_mode_always_returns_heuristic(self) -> None:
        judge = make_judge(mode="heuristic")
        assert isinstance(judge, HeuristicJudge)

    def test_auto_without_keys_returns_heuristic(self) -> None:
        env_backup = {
            k: os.environ.pop(k)
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if k in os.environ
        }
        try:
            judge = make_judge(mode="auto")
            assert isinstance(judge, HeuristicJudge)
        finally:
            os.environ.update(env_backup)

    def test_auto_with_anthropic_key_returns_llm_judge(self) -> None:

        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "fake-key-for-test"
        try:
            judge = make_judge(mode="auto")
            assert isinstance(judge, LLMJudge)
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_auto_with_openai_key_returns_llm_judge(self) -> None:

        env_backup_a = os.environ.pop("ANTHROPIC_API_KEY", None)
        env_backup_o = os.environ.pop("OPENAI_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "fake-openai-key"
        try:
            judge = make_judge(mode="auto")
            assert isinstance(judge, LLMJudge)
        finally:
            if env_backup_a is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup_a
            if env_backup_o is not None:
                os.environ["OPENAI_API_KEY"] = env_backup_o
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_llm_mode_without_keys_raises_os_error(self) -> None:
        env_backup = {
            k: os.environ.pop(k)
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if k in os.environ
        }
        try:
            with pytest.raises(OSError, match="API key"):
                make_judge(mode="llm")
        finally:
            os.environ.update(env_backup)

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown judge mode"):
            make_judge(mode="bogus")


# ---------------------------------------------------------------------------
# Cohen's kappa formula
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement_balanced(self) -> None:
        """Perfect agreement on balanced labels → kappa = 1.0."""
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=1),
            JudgeAgreement("q2", human_score=0, judge_score=0),
        ]
        assert cohens_kappa(pairs) == pytest.approx(1.0)

    def test_perfect_disagreement_balanced(self) -> None:
        """Perfect disagreement on balanced labels → kappa = -1.0."""
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=0),
            JudgeAgreement("q2", human_score=0, judge_score=1),
        ]
        assert cohens_kappa(pairs) == pytest.approx(-1.0)

    def test_empty_list_returns_zero(self) -> None:
        assert cohens_kappa([]) == 0.0

    def test_chance_level_agreement(self) -> None:
        """Two raters each labelling 50% positive independently at chance → kappa ~ 0."""
        # Human: [1, 1, 0, 0], Judge: [1, 0, 1, 0] — independent of each other
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=1),
            JudgeAgreement("q2", human_score=1, judge_score=0),
            JudgeAgreement("q3", human_score=0, judge_score=1),
            JudgeAgreement("q4", human_score=0, judge_score=0),
        ]
        # p_o = 2/4 = 0.5, p_e = 0.5*0.5 + 0.5*0.5 = 0.5 → kappa = 0.0
        assert cohens_kappa(pairs) == pytest.approx(0.0)

    def test_degenerate_all_positive_returns_zero(self) -> None:
        """All labels positive → p_e=1, undefined kappa → return 0."""
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=1),
            JudgeAgreement("q2", human_score=1, judge_score=1),
        ]
        # p_e = 1.0, so we return 0.0 (no division by zero)
        assert cohens_kappa(pairs) == 0.0

    def test_partial_agreement(self) -> None:
        """3/4 agree on balanced labels → kappa = 0.5."""
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=1),
            JudgeAgreement("q2", human_score=1, judge_score=0),  # disagree
            JudgeAgreement("q3", human_score=0, judge_score=0),
            JudgeAgreement("q4", human_score=0, judge_score=0),
        ]
        # p_o = 3/4, p_human_1=2/4=0.5, p_judge_1=1/4=0.25
        # p_e = 0.5*0.25 + 0.5*0.75 = 0.125 + 0.375 = 0.5
        # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.25/0.5 = 0.5
        assert cohens_kappa(pairs) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CLI integration: --judge flag
# ---------------------------------------------------------------------------


class TestCLIJudgeFlag:
    def test_heuristic_judge_no_keys_needed(self) -> None:
        """atlas eval --judge heuristic works without any API keys."""

        runner = CliRunner()
        # Strip any API keys from environment for this test.
        env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
        result = runner.invoke(
            app,
            ["eval", "--judge", "heuristic"],
            env=env,
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_llm_judge_without_keys_exits_with_error(self) -> None:
        """atlas eval --judge llm without keys exits non-zero with a friendly message."""

        runner = CliRunner()
        env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
        result = runner.invoke(
            app,
            ["eval", "--judge", "llm"],
            env=env,
        )
        assert result.exit_code != 0
        # Should mention the API key requirement
        assert "API key" in result.output or "key" in result.output.lower()
