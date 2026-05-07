"""RAGAS-style metric layer for the Codex-Atlas eval harness.

Exposes four canonical RAGAS metrics — ``faithfulness``,
``answer_relevancy``, ``context_precision``, and ``context_recall`` —
implemented as deterministic heuristics that need no LLM.

When the real ``ragas`` library is installed *and* an LLM key is
available the :class:`~codex_atlas.eval.ragas.library.RagasLibraryWrapper`
delegates to the genuine metric implementations.  The public entry
point is :func:`~codex_atlas.eval.ragas.factory.make_ragas_metrics`.
"""

from __future__ import annotations

from codex_atlas.eval.ragas.factory import make_ragas_metrics
from codex_atlas.eval.ragas.heuristic import (
    AnswerRelevancyHeuristic,
    ContextPrecisionHeuristic,
    ContextRecallHeuristic,
    FaithfulnessHeuristic,
)
from codex_atlas.eval.ragas.protocol import RagasMetric

__all__ = [
    "AnswerRelevancyHeuristic",
    "ContextPrecisionHeuristic",
    "ContextRecallHeuristic",
    "FaithfulnessHeuristic",
    "RagasMetric",
    "make_ragas_metrics",
]
