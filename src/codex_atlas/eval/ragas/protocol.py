"""RagasMetric Protocol — the interface every RAGAS metric must satisfy.

Any class with a ``name`` string attribute and an ``async compute``
method matching the signature below satisfies the protocol and can be
used interchangeably by the eval harness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RagasMetric(Protocol):
    """Structural protocol for RAGAS-style eval metrics.

    Attributes:
        name: Short snake-case identifier, e.g. ``"faithfulness"``.

    Methods:
        compute: Score a single question/context/answer triple.
    """

    name: str

    async def compute(
        self,
        *,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: list[str],
    ) -> float:
        """Compute a score in [0.0, 1.0] for one evaluation sample.

        Args:
            question: The user question that generated the answer.
            contexts: Retrieved text chunks (used as context by the agent).
            answer: The agent's generated answer.
            ground_truth: Gold qualified names (``expected_qualified_names``
                from :class:`~codex_atlas.eval.harness.EvalQuestion`).

        Returns:
            Float in ``[0.0, 1.0]``.
        """
        ...


__all__ = ["RagasMetric"]
