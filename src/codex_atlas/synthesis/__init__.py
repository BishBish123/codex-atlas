"""Synthesis layer for codex-atlas.

Exports ``GroqSynthesizer`` and the ``make_synthesizer`` factory.

``make_synthesizer()`` returns a ``GroqSynthesizer`` when ``GROQ_API_KEY``
is present in the environment, otherwise a ``StitchSynthesizer``.  The
``Agent`` calls this factory at construction time so the right backend
is selected automatically without any caller changes.
"""

from __future__ import annotations

import os

from codex_atlas.agent import StitchSynthesizer, Synthesizer
from codex_atlas.synthesis.groq import GroqSynthesizer

__all__ = ["GroqSynthesizer", "make_synthesizer"]


def make_synthesizer() -> Synthesizer:
    """Return the best available synthesizer for the current environment.

    * ``GROQ_API_KEY`` set → ``GroqSynthesizer`` (LLM-backed).
    * ``GROQ_API_KEY`` unset → ``StitchSynthesizer`` (deterministic).
    """
    if os.environ.get("GROQ_API_KEY"):
        return GroqSynthesizer()
    return StitchSynthesizer()
