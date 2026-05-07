"""RAGAS metric factory.

:func:`make_ragas_metrics` is the primary entry point.  It returns four
:class:`~codex_atlas.eval.ragas.protocol.RagasMetric` objects — one per
canonical RAGAS metric — using library wrappers when ``ragas`` is
installed and an LLM key is available, otherwise deterministic
heuristics.
"""

from __future__ import annotations

import logging

from codex_atlas.eval.ragas.protocol import RagasMetric

logger = logging.getLogger(__name__)


def make_ragas_metrics() -> list[RagasMetric]:
    """Return the four RAGAS metrics, best implementation available.

    Resolution order per metric:

    1. :class:`~codex_atlas.eval.ragas.library.RagasLibraryWrapper` —
       delegates to the real ``ragas`` library when installed *and* an
       LLM key (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``) is present.
    2. Deterministic heuristic — used as the wrapper's fallback, and
       directly when the library is absent or no key is set.

    Returns:
        Four-element list in the canonical order:
        ``[faithfulness, answer_relevancy, context_precision, context_recall]``.
    """
    try:
        import ragas  # noqa: F401

        from codex_atlas.eval.ragas.library import make_library_wrappers

        logger.debug("ragas library found; returning library wrappers.")
        wrappers: list[RagasMetric] = make_library_wrappers()  # type: ignore[assignment]
        return wrappers
    except ImportError:
        pass

    logger.debug("ragas library not installed; returning heuristic metrics.")
    from codex_atlas.eval.ragas.heuristic import (
        AnswerRelevancyHeuristic,
        ContextPrecisionHeuristic,
        ContextRecallHeuristic,
        FaithfulnessHeuristic,
    )

    return [
        FaithfulnessHeuristic(),
        AnswerRelevancyHeuristic(),
        ContextPrecisionHeuristic(),
        ContextRecallHeuristic(),
    ]


__all__ = ["make_ragas_metrics"]
