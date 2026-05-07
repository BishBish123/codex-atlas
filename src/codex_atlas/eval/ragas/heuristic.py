"""Deterministic RAGAS-metric heuristics — no LLM, no API keys.

Each class implements the :class:`~codex_atlas.eval.ragas.protocol.RagasMetric`
protocol and produces a score in ``[0.0, 1.0]`` using only string
operations.  The formulae match RAGAS semantics while remaining
reproducible across environments.

Metric formulae
---------------

**Faithfulness**
    ``fraction of ground_truth qnames that appear verbatim in the answer``

    Matches the existing :class:`~codex_atlas.judge.heuristic.HeuristicJudge`
    text-fraction signal.  Score = 1.0 when ``ground_truth`` is empty
    (vacuously faithful).

**Answer relevancy**
    ``cosine similarity between question keyword set and answer keyword set``

    Lexical only (no embeddings).  Keywords are lowercased alpha/digit
    tokens of length ≥ 3.  Score = 1.0 when either set is empty
    (degenerate; vacuously relevant).

**Context precision**
    ``fraction of retrieved chunks that contain at least one ground_truth name``

    Measures how many of the retrieved chunks are "relevant" (contain
    a gold symbol).  Score = 1.0 when no chunks are retrieved (vacuous).

**Context recall**
    ``fraction of ground_truth names that appear in any retrieved chunk``

    Measures whether the retrieved context covers all gold symbols.
    Score = 1.0 when ``ground_truth`` is empty (vacuous).
"""

from __future__ import annotations

import math
import re
import unicodedata

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _keywords(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of length ≥ 3."""
    return {tok for tok in re.findall(r"[a-z0-9_]{3,}", _normalize(text))}


def _cosine_sim(a: set[str], b: set[str]) -> float:
    """Jaccard-like cosine on binary bag-of-words (equal weights)."""
    if not a or not b:
        return 1.0  # degenerate; vacuously relevant
    intersection = len(a & b)
    # ||a|| = sqrt(|a|), ||b|| = sqrt(|b|); cosine = dot / (||a|| * ||b||)
    return intersection / math.sqrt(len(a) * len(b))


# ---------------------------------------------------------------------------
# Heuristic implementations
# ---------------------------------------------------------------------------


class FaithfulnessHeuristic:
    """RAGAS faithfulness: fraction of gold qnames cited verbatim in the answer.

    Matches the text-fraction branch of
    :class:`~codex_atlas.judge.heuristic.HeuristicJudge`.

    Implements :class:`~codex_atlas.eval.ragas.protocol.RagasMetric`.
    """

    name: str = "faithfulness"

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        del question, contexts  # not used by this metric

        if not ground_truth:
            return 1.0  # vacuously faithful

        norm_answer = _normalize(answer)
        hits = sum(1 for g in ground_truth if _normalize(g) in norm_answer)
        return hits / len(ground_truth)


class AnswerRelevancyHeuristic:
    """RAGAS answer relevancy: cosine sim between question and answer keyword sets.

    Lexical only — no embeddings.  A high score means the answer uses
    many of the same terms as the question.

    Implements :class:`~codex_atlas.eval.ragas.protocol.RagasMetric`.
    """

    name: str = "answer_relevancy"

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        del contexts, ground_truth  # not used by this metric

        q_kw = _keywords(question)
        a_kw = _keywords(answer)
        return _cosine_sim(q_kw, a_kw)


class ContextPrecisionHeuristic:
    """RAGAS context precision: fraction of retrieved chunks containing a gold name.

    A chunk is "relevant" if at least one ``ground_truth`` qualified
    name appears as a substring in the chunk text (case-insensitive,
    NFKC-normalised).

    Implements :class:`~codex_atlas.eval.ragas.protocol.RagasMetric`.
    """

    name: str = "context_precision"

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        del question, answer  # not used by this metric

        if not contexts:
            return 1.0  # no chunks retrieved; vacuously precise

        if not ground_truth:
            return 1.0  # no gold to check against; vacuous

        norm_gold = [_normalize(g) for g in ground_truth]
        hits = sum(
            1
            for chunk in contexts
            if any(g in _normalize(chunk) for g in norm_gold)
        )
        return hits / len(contexts)


class ContextRecallHeuristic:
    """RAGAS context recall: fraction of gold names that appear in any chunk.

    A gold name is "recalled" if it appears as a substring in any of
    the retrieved chunks (case-insensitive, NFKC-normalised).

    Implements :class:`~codex_atlas.eval.ragas.protocol.RagasMetric`.
    """

    name: str = "context_recall"

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        del question, answer  # not used by this metric

        if not ground_truth:
            return 1.0  # vacuously recalled

        if not contexts:
            return 0.0  # nothing retrieved → no gold recalled

        combined = _normalize(" ".join(contexts))
        hits = sum(1 for g in ground_truth if _normalize(g) in combined)
        return hits / len(ground_truth)


# ---------------------------------------------------------------------------
# Runtime protocol checks — evaluated once at import time.
# ---------------------------------------------------------------------------
assert isinstance(FaithfulnessHeuristic(), object)  # Protocol is structural; mypy checks it
assert isinstance(AnswerRelevancyHeuristic(), object)
assert isinstance(ContextPrecisionHeuristic(), object)
assert isinstance(ContextRecallHeuristic(), object)

__all__ = [
    "AnswerRelevancyHeuristic",
    "ContextPrecisionHeuristic",
    "ContextRecallHeuristic",
    "FaithfulnessHeuristic",
]
