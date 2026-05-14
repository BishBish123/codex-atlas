"""Eval harness: per-question scoring + aggregate report.

Metrics (no LLM required so the harness runs in CI):

- `route_correctness` — did the classifier pick the expected route?
- `citation_recall` — fraction of gold qualified names present in the
  agent's citations
- `citation_precision` — fraction of cited qualified names that match a
  gold (or partial-suffix-match) gold name
- `latency_ms` — wall-clock per question
- `attempts` — how many re-query loops the agent took
- `tool_calls` — how many retriever invocations per question
- `cost_estimate_usd` — token count x model price (approximation)

Aggregate-level the harness also reports `latency_p99` and a
``failure_taxonomy`` bucket-count so a senior reviewer can see *why*
the harness is missing answers, not just *that* it is.

Anything LLM-dependent (faithfulness, answer-relevancy) lives behind a
`Grader` plug point and is *not* required to make the harness green —
that keeps it portable while leaving room for the production LLM judge
to drop in.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from codex_atlas.agent import Agent, AgentResult
from codex_atlas.retriever import Route


class ExpectedRoute(StrEnum):
    LOOKUP = "lookup"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    SUMMARIZATION = "summarization"
    NEIGHBORHOOD = "neighborhood"
    IMPORT_CHAIN = "import_chain"


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    category: str
    question: str
    expected_route: ExpectedRoute
    gold_qualified_names: list[str] = field(default_factory=list)


class FailureBucket(StrEnum):
    """Why an answer was wrong / unsatisfying. One per question, max."""

    NONE = "none"
    MISSING_NODE = "missing_node"  # gold qname not in graph at all
    WRONG_ROUTE = "wrong_route"  # classifier picked the wrong route
    HALLUCINATED = "hallucinated"  # cited qname not in retrieved chunks
    PARTIAL = "partial"  # some gold cited, some missed
    UNGROUNDED = "ungrounded"  # answer cites nothing
    OFF_TOPIC = "off_topic"  # citations match no gold name
    # ``OUTDATED_INDEX`` is reserved for future structured detection
    # (e.g. the retriever surfacing an explicit ``freshness_warning``
    # signal when the index is older than the corpus). It is currently
    # unreachable: the previous answer-text heuristic ("answer contains
    # the word 'stale' or 'outdated'") was non-deterministic — paraphrase
    # changed the bucket. The bucket name is kept so downstream consumers
    # of the JSON dump don't break when the structured signal lands.
    OUTDATED_INDEX = "outdated_index"  # reserved; see harness docstring


# Token / pricing approximation for `cost_estimate_usd`. The numbers are
# illustrative — tweak them per provider — and live here rather than at
# call sites so a single edit covers every metric report.
DEFAULT_PRICE_USD_PER_1K_INPUT = 0.0030
DEFAULT_PRICE_USD_PER_1K_OUTPUT = 0.0150


@dataclass(frozen=True)
class EvalResult:
    qid: str
    category: str
    question: str
    expected_route: ExpectedRoute
    actual_route: Route
    route_correct: bool
    citation_recall: float
    citation_precision: float
    latency_ms: float
    attempts: int
    answer_preview: str
    cited_qualified_names: list[str]
    tool_call_count: int = 0
    cost_estimate_usd: float = 0.0
    failure_bucket: FailureBucket = FailureBucket.NONE
    answer_full: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_result(
    question: EvalQuestion,
    agent_result: AgentResult,
    latency_ms: float,
    *,
    price_input: float = DEFAULT_PRICE_USD_PER_1K_INPUT,
    price_output: float = DEFAULT_PRICE_USD_PER_1K_OUTPUT,
    indexed_qnames: frozenset[str] | None = None,
) -> EvalResult:
    cited = [c.qualified_name for c in agent_result.citations]

    route_correct = str(agent_result.route) == str(question.expected_route)

    if question.gold_qualified_names:
        hits = sum(1 for g in question.gold_qualified_names if _matches(g, cited))
        recall = hits / len(question.gold_qualified_names)
    else:
        # Failure-likely / out-of-scope questions: 1.0 when the agent
        # returned no citations (correct refusal), 0.0 otherwise.
        recall = 1.0 if not cited else 0.0

    if cited:
        if question.gold_qualified_names:
            cite_hits = sum(
                1 for c in cited if any(_matches(g, [c]) for g in question.gold_qualified_names)
            )
            precision = cite_hits / len(cited)
        else:
            precision = 0.0
    else:
        precision = 1.0 if not question.gold_qualified_names else 0.0

    bucket = _classify_failure(
        question=question,
        cited=cited,
        recall=recall,
        precision=precision,
        route_correct=route_correct,
        answer=agent_result.answer,
        indexed_qnames=indexed_qnames,
    )
    cost = _estimate_cost_usd(
        question=question.question,
        answer=agent_result.answer,
        price_input=price_input,
        price_output=price_output,
    )

    return EvalResult(
        qid=question.qid,
        category=question.category,
        question=question.question,
        expected_route=question.expected_route,
        actual_route=agent_result.route,
        route_correct=route_correct,
        citation_recall=recall,
        citation_precision=precision,
        latency_ms=latency_ms,
        attempts=agent_result.attempts,
        answer_preview=agent_result.answer[:160].replace("\n", " "),
        cited_qualified_names=cited,
        tool_call_count=len(agent_result.tool_calls),
        cost_estimate_usd=cost,
        failure_bucket=bucket,
        answer_full=agent_result.answer,
    )


def _classify_failure(  # noqa: PLR0911
    *,
    question: EvalQuestion,
    cited: list[str],
    recall: float,
    precision: float,
    route_correct: bool,
    answer: str,
    indexed_qnames: frozenset[str] | None = None,
) -> FailureBucket:
    """Bucket a question into the 7-mode failure taxonomy.

    Order matters: we report the first bucket that matches so a single
    failure is never double-counted.

    ``OUTDATED_INDEX`` is intentionally unreachable here: the previous
    "answer contains 'stale' or 'outdated'" heuristic was paraphrase-
    sensitive (an LLM rewording the same finding bucketed differently),
    so it has been removed. The bucket value remains in the enum and
    will be repopulated once the retriever exposes a structured
    ``freshness_warning`` signal — at which point this classifier will
    consult that flag rather than the answer text.

    ``indexed_qnames`` is the set of qualified names actually present in
    the index. When provided, gold qnames that aren't in the index are
    classified as ``MISSING_NODE`` — distinguishing "harness expected a
    symbol that doesn't exist in this corpus" from "agent missed it".
    """
    # ``answer`` is unused now that the OUTDATED_INDEX heuristic is gone,
    # but the parameter is preserved for the future structured signal
    # (the harness will pass the retriever's freshness flag here).
    del answer

    # Refusal questions: gold is empty, so recall=1.0 means correct refusal.
    if not question.gold_qualified_names:
        if cited:
            return FailureBucket.OFF_TOPIC
        return FailureBucket.NONE

    # MISSING_NODE — every gold qname is absent from the index. This is a
    # corpus-mismatch signal for the harness, not an agent failure. We
    # only flag when ALL gold names are missing AND the agent surfaced
    # nothing matching, otherwise it's covered by the recall/precision
    # buckets below.
    if (
        indexed_qnames is not None
        and not cited
        and all(g not in indexed_qnames for g in question.gold_qualified_names)
    ):
        return FailureBucket.MISSING_NODE

    if not route_correct:
        return FailureBucket.WRONG_ROUTE
    if not cited:
        return FailureBucket.UNGROUNDED
    if recall == 0.0 and precision == 0.0:
        return FailureBucket.OFF_TOPIC
    if 0.0 < recall < 1.0:
        return FailureBucket.PARTIAL
    if precision < 0.5 and recall >= 1.0:
        # Got everything we wanted but also produced unsupported names.
        return FailureBucket.HALLUCINATED
    return FailureBucket.NONE


def _estimate_cost_usd(
    *,
    question: str,
    answer: str,
    price_input: float,
    price_output: float,
) -> float:
    """Token-count approximation: ~4 chars / token, then multiply by price."""
    in_tokens = max(1, len(question) // 4)
    out_tokens = max(1, len(answer) // 4)
    return (in_tokens / 1000.0) * price_input + (out_tokens / 1000.0) * price_output


def _matches(gold: str, cited: list[str]) -> bool:
    """Match if any cited qname equals or end-suffix-matches the gold name.

    Suffix match is necessary because the agent's structural route may
    return `pkg.Class.method` when the question's gold expectation was
    just `Class.method` (or vice-versa).
    """
    for c in cited:
        if c == gold:
            return True
        if c.endswith(f".{gold}") or gold.endswith(f".{c}"):
            return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_eval(
    agent: Agent,
    questions: list[EvalQuestion],
    *,
    indexed_qnames: frozenset[str] | None = None,
) -> list[EvalResult]:
    """Run every question through the agent and score the results.

    ``indexed_qnames`` is the optional set of qnames in the corpus
    index. When provided, scoring distinguishes ``MISSING_NODE``
    (gold qname doesn't exist in the index) from ``UNGROUNDED`` (agent
    failed to surface an existing qname).
    """
    out: list[EvalResult] = []
    for q in questions:
        t0 = time.perf_counter()
        result = await agent.run(q.question)
        latency = (time.perf_counter() - t0) * 1000.0
        out.append(score_result(q, result, latency, indexed_qnames=indexed_qnames))
    return out


def aggregate(results: list[EvalResult]) -> dict[str, float | int]:
    """Compute every aggregate metric without rendering markdown.

    Convenient for ``evaluate_against_baseline`` and for tests that
    don't want to parse the report. Numbers track ``render_report``
    output exactly.

    Percentile method: ``numpy.percentile`` with the default
    ``linear`` interpolation. For very small samples (n < 10) the
    naive index-based picks the original code used produced misleading
    "p99 = max" or "p95 = p50" depending on rounding; we explicitly
    fall back to the sample max for p95/p99 in that regime so the
    metric is stable + interpretable. See ``evals/INTERPRETATION.md``.
    """
    if not results:
        return {}
    n = len(results)
    latencies_arr = np.asarray([r.latency_ms for r in results], dtype=float)
    if n < 10:
        # Tiny samples: percentile interpolation is misleading. Report
        # the worst observed value as a conservative upper bound.
        p95 = float(latencies_arr.max())
        p99 = float(latencies_arr.max())
    else:
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
    p50 = float(np.percentile(latencies_arr, 50))
    return {
        "n": n,
        "route_correctness": sum(1 for r in results if r.route_correct) / n,
        "citation_recall_mean": sum(r.citation_recall for r in results) / n,
        "citation_precision_mean": sum(r.citation_precision for r in results) / n,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "tool_call_count_mean": sum(r.tool_call_count for r in results) / n,
        "cost_estimate_usd_total": sum(r.cost_estimate_usd for r in results),
    }


def failure_taxonomy_counts(results: list[EvalResult]) -> dict[str, int]:
    """Count how many questions landed in each failure bucket."""
    counts: Counter[str] = Counter()
    for bucket in FailureBucket:
        counts[str(bucket)] = 0
    for r in results:
        counts[str(r.failure_bucket)] += 1
    return dict(counts)


def render_report(results: list[EvalResult]) -> str:
    """Markdown report summarising the run."""
    if not results:
        return "# Eval report\n\nNo results to render.\n"

    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    agg = aggregate(results)
    n = len(results)
    lines = ["# Codex-Atlas eval report", ""]
    lines += [
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {n} |",
        f"| Route correctness | {agg['route_correctness']:.1%} |",
        f"| Citation recall (mean) | {agg['citation_recall_mean']:.2f} |",
        f"| Citation precision (mean) | {agg['citation_precision_mean']:.2f} |",
        f"| p50 latency (ms) | {agg['latency_p50_ms']:.1f} |",
        f"| p95 latency (ms) | {agg['latency_p95_ms']:.1f} |",
        f"| p99 latency (ms) | {agg['latency_p99_ms']:.1f} |",
        f"| Tool calls / q (mean) | {agg['tool_call_count_mean']:.2f} |",
        f"| Cost estimate (USD, total) | {agg['cost_estimate_usd_total']:.4f} |",
        "",
    ]

    lines += ["## By category", ""]
    lines += [
        "| Category | n | Route acc. | Recall | Precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cat, rs in sorted(by_category.items()):
        cat_acc = sum(1 for r in rs if r.route_correct) / len(rs)
        cat_recall = sum(r.citation_recall for r in rs) / len(rs)
        cat_prec = sum(r.citation_precision for r in rs) / len(rs)
        lines.append(f"| {cat} | {len(rs)} | {cat_acc:.0%} | {cat_recall:.2f} | {cat_prec:.2f} |")
    lines += [""]

    counts = failure_taxonomy_counts(results)
    lines += ["## Failure taxonomy", ""]
    lines += [
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for bucket, count in counts.items():
        # OUTDATED_INDEX is currently unreachable in ``_classify_failure``
        # (the answer-text heuristic was paraphrase-sensitive and got
        # removed). The bucket is kept in the enum for future structured
        # detection — until then it'd render as a permanent zero-count
        # row that misleads readers about the taxonomy's coverage. Skip
        # it from the rendered table; consumers of the JSON dump still
        # see it in ``failure_taxonomy_counts``.
        if bucket == str(FailureBucket.OUTDATED_INDEX):
            continue
        lines.append(f"| {bucket} | {count} |")
    lines += [""]

    lines += ["## Per question", ""]
    lines += [
        "| qid | route ok | recall | prec | ms | tools | bucket | preview |",
        "| --- | :---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in results:
        check = "yes" if r.route_correct else "no"
        lines.append(
            f"| {r.qid} | {check} | {r.citation_recall:.2f} | "
            f"{r.citation_precision:.2f} | {r.latency_ms:.1f} | "
            f"{r.tool_call_count} | {r.failure_bucket} | {r.answer_preview[:80]} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Baseline regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineDiff:
    """A baseline-vs-current diff suitable for a CI gate.

    `regressions` lists metrics that got worse beyond the tolerance.
    `improvements` lists metrics that got better. `unchanged` is silent
    by design (CI doesn't need to celebrate flat numbers).
    """

    regressions: dict[str, tuple[float, float]]  # metric -> (baseline, current)
    improvements: dict[str, tuple[float, float]]
    is_regression: bool


# Metrics that should be *higher* in the new run than in the baseline.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "route_correctness",
        "citation_recall_mean",
        "citation_precision_mean",
    }
)
# Metrics that should be *lower*.
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "tool_call_count_mean",
        "cost_estimate_usd_total",
    }
)


_LATENCY_METRICS: frozenset[str] = frozenset(
    {"latency_p50_ms", "latency_p95_ms", "latency_p99_ms"}
)


def evaluate_against_baseline(
    current: list[EvalResult],
    baseline_path: str | Path,
    *,
    tolerance: float = 0.05,
) -> BaselineDiff:
    """Diff `current` against a saved baseline JSON; flag regressions.

    Tolerance is a fraction of the baseline value: a 5% tolerance means
    a metric that drops by less than 5% of its baseline doesn't count as
    a regression. This keeps the CI gate from blocking on noise while
    still surfacing real drift.

    Latency tolerance has an extra absolute floor: when the baseline
    JSON includes a ``latency_tolerance_ms`` field, every
    ``latency_p*_ms`` metric is allowed to drift by up to that many
    milliseconds in addition to the proportional ``tolerance`` band.
    Without it, a baseline pinned at single-digit milliseconds would
    flag a 1ms wall-clock jitter (~25% of 4.5ms) as a regression on
    every busy CI runner.
    """
    baseline = json.loads(Path(baseline_path).read_text())
    cur = aggregate(current)
    regressions: dict[str, tuple[float, float]] = {}
    improvements: dict[str, tuple[float, float]] = {}
    # Optional absolute-ms slack for latency metrics. Anything else
    # ignores it — only latency is jittery enough to need an absolute
    # floor on top of the proportional band.
    latency_floor_ms = float(baseline.get("latency_tolerance_ms", 0.0) or 0.0)

    def _slack(metric: str, baseline_value: float) -> float:
        proportional = tolerance * max(abs(baseline_value), 1e-9)
        if metric in _LATENCY_METRICS and latency_floor_ms > 0.0:
            return max(proportional, latency_floor_ms)
        return proportional

    # Only diff metrics the baseline actually recorded — you can't regress
    # against a number you didn't measure.
    for metric in _HIGHER_IS_BETTER:
        if metric not in baseline:
            continue
        b = float(baseline[metric])
        c = float(cur.get(metric, 0.0))
        s = _slack(metric, b)
        if c < b - s:
            regressions[metric] = (b, c)
        elif c > b + s:
            improvements[metric] = (b, c)
    for metric in _LOWER_IS_BETTER:
        if metric not in baseline:
            continue
        b = float(baseline[metric])
        c = float(cur.get(metric, 0.0))
        s = _slack(metric, b)
        if c > b + s:
            regressions[metric] = (b, c)
        elif c < b - s:
            improvements[metric] = (b, c)
    return BaselineDiff(
        regressions=regressions,
        improvements=improvements,
        is_regression=bool(regressions),
    )


def write_failure_report(results: list[EvalResult], path: str | Path) -> Path:
    """Write a JSONL per-question dump for manual debugging.

    One line per question, full answer + every cited qname + the
    failure bucket. Designed for `cat path | jq` workflows.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "qid": r.qid,
                        "category": r.category,
                        "question": r.question,
                        "expected_route": str(r.expected_route),
                        "actual_route": str(r.actual_route),
                        "route_correct": r.route_correct,
                        "citation_recall": r.citation_recall,
                        "citation_precision": r.citation_precision,
                        "latency_ms": r.latency_ms,
                        "attempts": r.attempts,
                        "tool_call_count": r.tool_call_count,
                        "cost_estimate_usd": r.cost_estimate_usd,
                        "failure_bucket": str(r.failure_bucket),
                        "cited_qualified_names": r.cited_qualified_names,
                        "answer": r.answer_full,
                    }
                )
                + "\n"
            )
    return out
