"""LLM-as-judge answer-faithfulness scoring framework.

Public API
----------
- :class:`~codex_atlas.judge.protocol.JudgeProtocol` — structural protocol
  every judge must satisfy.
- :class:`~codex_atlas.judge.protocol.JudgeScore` — score dataclass returned
  by every judge.
- :class:`~codex_atlas.judge.llm.LLMJudge` — real LLM judge (requires
  ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``).
- :class:`~codex_atlas.judge.heuristic.HeuristicJudge` — deterministic
  fallback, zero dependencies.
- :func:`~codex_atlas.judge.factory.make_judge` — factory that returns the
  best available judge for the current environment.

Env-key gating
--------------
When ``ANTHROPIC_API_KEY`` (or ``OPENAI_API_KEY``) is set,
:func:`make_judge` returns an :class:`LLMJudge`.  Otherwise it silently
returns :class:`HeuristicJudge`, keeping the default eval path (no env
keys, no network) fully functional.
"""

from codex_atlas.judge.factory import make_judge
from codex_atlas.judge.heuristic import HeuristicJudge
from codex_atlas.judge.llm import LLMJudge
from codex_atlas.judge.protocol import JudgeProtocol, JudgeScore

__all__ = [
    "HeuristicJudge",
    "JudgeProtocol",
    "JudgeScore",
    "LLMJudge",
    "make_judge",
]
