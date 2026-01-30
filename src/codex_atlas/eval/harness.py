"""Eval harness: per-question scoring + aggregate report.

Metrics (no LLM required so the harness runs in CI):

- `route_correctness` — did the classifier pick the expected route?
- `citation_recall` — fraction of gold qualified names present in the
  agent's citations
- `citation_precision` — fraction of cited qualified names that match a
  gold (or partial-suffix-match) gold name
- `latency_ms` — wall-clock per question
- `attempts` — how many re-query loops the agent took

Anything LLM-dependent (faithfulness, answer-relevancy) lives behind a
`Grader` plug point and is *not* required to make the harness green —
that keeps it portable while leaving room for the production LLM judge
to drop in. Note: this module is named `harness` in the `eval` package,
not the Python builtin `eval()` (which is not used anywhere here).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from codex_atlas.agent import Agent, AgentResult
from codex_atlas.retriever import Route


class ExpectedRoute(StrEnum):
    LOOKUP = "lookup"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    SUMMARIZATION = "summarization"


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    category: str
    question: str
    expected_route: ExpectedRoute
    gold_qualified_names: list[str] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_result(
    question: EvalQuestion, agent_result: AgentResult, latency_ms: float
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
    )


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


async def run_eval(agent: Agent, questions: list[EvalQuestion]) -> list[EvalResult]:
    out: list[EvalResult] = []
    for q in questions:
        t0 = time.perf_counter()
        result = await agent.run(q.question)
        latency = (time.perf_counter() - t0) * 1000.0
        out.append(score_result(q, result, latency))
    return out


def render_report(results: list[EvalResult]) -> str:
    """Markdown report summarising the run."""
    if not results:
        return "# Eval report\n\nNo results to render.\n"

    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    lines = ["# Codex-Atlas eval report", ""]
    n = len(results)
    route_acc = sum(1 for r in results if r.route_correct) / n
    avg_recall = sum(r.citation_recall for r in results) / n
    avg_precision = sum(r.citation_precision for r in results) / n
    p50_latency = sorted(r.latency_ms for r in results)[n // 2]
    lines += [
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {n} |",
        f"| Route correctness | {route_acc:.1%} |",
        f"| Citation recall (mean) | {avg_recall:.2f} |",
        f"| Citation precision (mean) | {avg_precision:.2f} |",
        f"| p50 latency (ms) | {p50_latency:.1f} |",
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

    lines += ["## Per question", ""]
    lines += [
        "| qid | route ok | recall | prec | ms | attempts | preview |",
        "| --- | :---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        check = "yes" if r.route_correct else "no"
        lines.append(
            f"| {r.qid} | {check} | {r.citation_recall:.2f} | "
            f"{r.citation_precision:.2f} | {r.latency_ms:.1f} | "
            f"{r.attempts} | {r.answer_preview[:80]} |"
        )

    return "\n".join(lines) + "\n"
