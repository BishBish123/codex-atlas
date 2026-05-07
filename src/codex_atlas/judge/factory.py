"""Judge factory — returns the best available judge given the environment.

:func:`make_judge` is the primary entry point for production code.  It
checks for API keys and returns an :class:`~codex_atlas.judge.llm.LLMJudge`
when one is found, otherwise falls back silently to
:class:`~codex_atlas.judge.heuristic.HeuristicJudge`.

This means the default eval path (no env keys) always works without any
configuration, keeping CI hermetic.
"""

from __future__ import annotations

import logging
import os

from codex_atlas.judge.protocol import JudgeProtocol

logger = logging.getLogger(__name__)


def make_judge(
    *,
    mode: str = "auto",
) -> JudgeProtocol:
    """Return the appropriate judge for the current environment.

    Args:
        mode: One of ``"auto"``, ``"llm"``, or ``"heuristic"``.

            - ``"auto"`` (default): return :class:`~codex_atlas.judge.llm.LLMJudge`
              if ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` is set,
              else :class:`~codex_atlas.judge.heuristic.HeuristicJudge`.
            - ``"llm"``: always return :class:`~codex_atlas.judge.llm.LLMJudge`;
              raises ``OSError`` if no key is configured.
            - ``"heuristic"``: always return
              :class:`~codex_atlas.judge.heuristic.HeuristicJudge`.

    Returns:
        An object satisfying :class:`~codex_atlas.judge.protocol.JudgeProtocol`.

    Raises:
        ValueError: If *mode* is not one of the accepted values.
        OSError: If ``mode="llm"`` and no API key is set.
        ImportError: If ``mode="llm"`` and httpx is not installed.
    """
    from codex_atlas.judge.heuristic import HeuristicJudge

    if mode == "heuristic":
        return HeuristicJudge()

    if mode == "llm":
        from codex_atlas.judge.llm import LLMJudge

        return LLMJudge()  # raises OSError / ImportError if unconfigured

    if mode == "auto":
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        has_openai = bool(os.environ.get("OPENAI_API_KEY", ""))
        if has_anthropic or has_openai:
            try:
                from codex_atlas.judge.llm import LLMJudge

                judge = LLMJudge()
                logger.info(
                    "LLMJudge active (%s key found)",
                    "ANTHROPIC_API_KEY" if has_anthropic else "OPENAI_API_KEY",
                )
                return judge
            except ImportError:
                logger.warning(
                    "API key found but httpx is not installed; "
                    "falling back to heuristic judge. "
                    "Install httpx with: pip install httpx"
                )
        return HeuristicJudge()

    raise ValueError(
        f"Unknown judge mode: {mode!r}. Valid values: 'auto', 'llm', 'heuristic'."
    )


__all__ = ["make_judge"]
