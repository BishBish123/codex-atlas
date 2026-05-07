"""Judge protocol and JudgeScore dataclass.

Every judge implementation — heuristic or LLM — must return a
``JudgeScore`` from its ``async score`` method.  The protocol is kept in a
separate module so other packages can depend on just the interface without
pulling in httpx or any LLM-specific imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeScore:
    """Result of a single answer-level faithfulness judgement.

    Attributes:
        score: Float in [0.0, 1.0].  1.0 means the answer fully covers
            all expected qualified names; 0.0 means none are covered.
        rationale: Human-readable explanation of the score.
        mode: Whether the score came from a real LLM call or the
            heuristic fallback.
    """

    score: float
    rationale: str
    mode: Literal["llm", "heuristic"]


@runtime_checkable
class JudgeProtocol(Protocol):
    """Structural protocol for answer-faithfulness judges.

    Any class with an ``async score`` method matching this signature
    satisfies the protocol and can be used wherever a judge is expected.
    """

    async def score(
        self,
        question: str,
        expected_qualified_names: list[str],
        answer: str,
        citations: list[str],
    ) -> JudgeScore:
        """Score *answer* against *expected_qualified_names*.

        Args:
            question: The user question that prompted the answer.
            expected_qualified_names: Gold qualified names the answer should
                reference or cover.
            answer: The agent's reply to evaluate.
            citations: Qualified names the agent explicitly cited.

        Returns:
            A :class:`JudgeScore` with ``score``, ``rationale``, and ``mode``.
        """
        ...


__all__ = ["JudgeProtocol", "JudgeScore"]
