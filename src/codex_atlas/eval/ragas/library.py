"""RagasLibraryWrapper — soft-imports the real ``ragas`` library.

When ``ragas>=0.2`` is installed *and* an LLM key
(``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``) is present, this module
delegates to ragas's own metric implementations.  In every other
situation it transparently falls back to the deterministic
:mod:`~codex_atlas.eval.ragas.heuristic` implementations.

The wrapper is a thin adapter so the eval harness code is identical
whether the library is present or not — it just calls
``await metric.compute(...)``.

LLM gating
----------
Even when ``ragas`` is installed, we do *not* call the library metrics
unless an LLM key is available: ragas's faithfulness and
answer_relevancy metrics use an LLM as judge.  Running them without a
key would raise at eval time.  The env-var check mirrors the one in
:mod:`~codex_atlas.judge.factory`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from codex_atlas.eval.ragas.heuristic import (
    AnswerRelevancyHeuristic,
    ContextPrecisionHeuristic,
    ContextRecallHeuristic,
    FaithfulnessHeuristic,
)
from codex_atlas.eval.ragas.protocol import RagasMetric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM availability check
# ---------------------------------------------------------------------------


def _llm_available() -> bool:
    """Return True when at least one LLM key is set in the environment."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    )


# ---------------------------------------------------------------------------
# Library wrapper
# ---------------------------------------------------------------------------


class RagasLibraryWrapper:
    """Wraps one ragas library metric and falls back to a heuristic.

    Parameters
    ----------
    ragas_metric_name:
        The attribute name on the ``ragas.metrics`` module, e.g.
        ``"faithfulness"`` or ``"answer_relevancy"``.
    fallback:
        A heuristic implementation to use when the library or an LLM
        key is unavailable.
    """

    def __init__(
        self,
        ragas_metric_name: str,
        fallback: RagasMetric,
    ) -> None:
        self._ragas_name = ragas_metric_name
        self._fallback = fallback
        self._ragas_metric: Any = None
        self._resolved = False
        self.name: str = fallback.name

    def _resolve(self) -> Any:
        """Try to import the ragas metric once; cache the result."""
        if self._resolved:
            return self._ragas_metric
        self._resolved = True
        if not _llm_available():
            logger.debug(
                "No LLM key set; RagasLibraryWrapper(%s) will use heuristic fallback.",
                self._ragas_name,
            )
            return None
        try:
            import ragas.metrics as _rm

            metric = getattr(_rm, self._ragas_name, None)
            if metric is None:
                logger.warning(
                    "ragas.metrics.%s not found; using heuristic fallback.",
                    self._ragas_name,
                )
                return None
            self._ragas_metric = metric
            logger.info("RagasLibraryWrapper: using real ragas.metrics.%s", self._ragas_name)
            return self._ragas_metric
        except ImportError:
            logger.debug(
                "ragas library not installed; RagasLibraryWrapper(%s) using heuristic.",
                self._ragas_name,
            )
            return None

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        """Compute the metric, delegating to ragas when available.

        The real ragas library expects a ``Dataset``; we build a minimal
        one-row HuggingFace-compatible dict and call
        ``metric.score(dataset)``.  If anything goes wrong (import
        failure, missing key, unexpected API shape) we fall back to the
        heuristic.
        """
        metric = self._resolve()
        if metric is None:
            return await self._fallback.compute(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
            )

        try:
            # ragas>=0.2 Dataset API:
            # from datasets import Dataset
            # ds = Dataset.from_dict({...})
            # score = metric.score(ds)
            from datasets import Dataset

            ds = Dataset.from_dict(
                {
                    "question": [question],
                    "contexts": [contexts],
                    "answer": [answer],
                    "ground_truths": [ground_truth],
                }
            )
            result = metric.score(ds)
            # ragas returns a dict keyed by metric name; grab the first value.
            if isinstance(result, dict):
                val = next(iter(result.values()), None)
                if val is not None:
                    return float(val)
            return await self._fallback.compute(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
            )
        except Exception:
            logger.debug(
                "RagasLibraryWrapper(%s): ragas call failed; using heuristic.",
                self._ragas_name,
                exc_info=True,
            )
            return await self._fallback.compute(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
            )


# ---------------------------------------------------------------------------
# Convenience factory: return 4 wrappers
# ---------------------------------------------------------------------------


def make_library_wrappers() -> list[RagasLibraryWrapper]:
    """Return one RagasLibraryWrapper per canonical RAGAS metric."""
    return [
        RagasLibraryWrapper("faithfulness", FaithfulnessHeuristic()),
        RagasLibraryWrapper("answer_relevancy", AnswerRelevancyHeuristic()),
        RagasLibraryWrapper("context_precision", ContextPrecisionHeuristic()),
        RagasLibraryWrapper("context_recall", ContextRecallHeuristic()),
    ]


__all__ = ["RagasLibraryWrapper", "make_library_wrappers"]
