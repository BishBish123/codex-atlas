"""Eval harness: golden set + per-question scoring + report writer."""

from codex_atlas.eval.golden import load_golden_set
from codex_atlas.eval.harness import EvalQuestion, EvalResult, run_eval, score_result

__all__ = [
    "EvalQuestion",
    "EvalResult",
    "load_golden_set",
    "run_eval",
    "score_result",
]
