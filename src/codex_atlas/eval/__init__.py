"""Eval harness: golden set + per-question scoring + report writer."""

from codex_atlas.eval.golden import load_golden_set
from codex_atlas.eval.harness import (
    BaselineDiff,
    EvalQuestion,
    EvalResult,
    FailureBucket,
    aggregate,
    evaluate_against_baseline,
    failure_taxonomy_counts,
    run_eval,
    score_result,
    write_failure_report,
)

__all__ = [
    "BaselineDiff",
    "EvalQuestion",
    "EvalResult",
    "FailureBucket",
    "aggregate",
    "evaluate_against_baseline",
    "failure_taxonomy_counts",
    "load_golden_set",
    "run_eval",
    "score_result",
    "write_failure_report",
]
