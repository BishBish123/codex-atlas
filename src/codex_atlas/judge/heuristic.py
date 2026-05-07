"""Heuristic answer-faithfulness judge — zero LLM dependencies.

Scoring formula
---------------
Given ``expected_qualified_names`` (E) and the agent's ``citations`` (C)
plus the ``answer`` text:

1. **Citation fraction** — fraction of names in E that appear in C
   (exact match or suffix match, same logic as the retrieval harness).
2. **Text fraction** — fraction of names in E that appear verbatim as
   a substring in the answer text (case-insensitive, NFKC-normalised).
3. **Final score** = max(citation_fraction, text_fraction)

Rationale: the agent may format citations differently from what the gold
expects, so we give credit for verbatim presence in the answer text as a
secondary signal.  Taking the maximum prevents the two signals from
averaging down a correctly-cited answer that has no text mention.

Special cases
-------------
- Empty ``expected_qualified_names`` → score = 1.0 (vacuously faithful).
- Empty ``citations`` AND empty ``answer`` → score = 0.0.
"""

from __future__ import annotations

import unicodedata

from codex_atlas.judge.protocol import JudgeProtocol, JudgeScore


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _suffix_match(name: str, pool: list[str]) -> bool:
    """True if *name* equals or suffix-matches any element of *pool*."""
    for p in pool:
        if p == name:
            return True
        if p.endswith(f".{name}") or name.endswith(f".{p}"):
            return True
    return False


class HeuristicJudge:
    """Deterministic answer-faithfulness judge.

    No API keys required.  Suitable for CI, local development, and as a
    fallback when no LLM key is configured.

    Implements :class:`~codex_atlas.judge.protocol.JudgeProtocol`.
    """

    async def score(
        self,
        question: str,
        expected_qualified_names: list[str],
        answer: str,
        citations: list[str],
    ) -> JudgeScore:
        """Score the answer using citation-overlap and text-substring heuristics.

        Args:
            question: Not used by the heuristic judge (included for
                protocol compatibility).
            expected_qualified_names: Gold qualified names to look for.
            answer: The agent's reply to evaluate.
            citations: Qualified names the agent explicitly cited.

        Returns:
            :class:`~codex_atlas.judge.protocol.JudgeScore` with
            ``mode="heuristic"``.
        """
        del question  # not used by this judge

        if not expected_qualified_names:
            return JudgeScore(
                score=1.0,
                rationale="No expected qualified names; vacuously faithful.",
                mode="heuristic",
            )

        # 1. Citation fraction — how many gold names appear in citations.
        cited_hits = sum(
            1 for g in expected_qualified_names if _suffix_match(g, citations)
        )
        citation_fraction = cited_hits / len(expected_qualified_names)

        # 2. Text fraction — how many gold names appear verbatim in the answer.
        norm_answer = _normalize(answer)
        text_hits = sum(
            1 for g in expected_qualified_names if _normalize(g) in norm_answer
        )
        text_fraction = text_hits / len(expected_qualified_names)

        score = max(citation_fraction, text_fraction)

        if citation_fraction >= 1.0:
            rationale = (
                f"All {len(expected_qualified_names)} expected name(s) "
                "found in citations."
            )
        elif citation_fraction > 0.0:
            rationale = (
                f"{cited_hits}/{len(expected_qualified_names)} expected name(s) "
                "found in citations"
                + (
                    f"; {text_hits}/{len(expected_qualified_names)} in answer text."
                    if text_fraction > citation_fraction
                    else "."
                )
            )
        elif text_fraction > 0.0:
            rationale = (
                f"No citation matches but {text_hits}/{len(expected_qualified_names)} "
                "expected name(s) found verbatim in answer text."
            )
        else:
            rationale = (
                f"None of the {len(expected_qualified_names)} expected name(s) "
                "found in citations or answer text."
            )

        return JudgeScore(score=score, rationale=rationale, mode="heuristic")


# Runtime protocol check — evaluated once at import time.
assert isinstance(HeuristicJudge(), JudgeProtocol), (
    "HeuristicJudge does not satisfy JudgeProtocol"
)

__all__ = ["HeuristicJudge"]
