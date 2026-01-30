"""The 12-question golden eval set.

We deliberately use *this* repo (codex-atlas) as the eval corpus so the
suite is hermetic: no external clone required, every gold citation is
verifiable in the same git tree, and reviewers can re-run `make eval`
without standing up a separate dataset.

Categories cover the brief's 5 question types: lookup, structural,
multi-hop, failure-likely (non-existent symbol), out-of-scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codex_atlas.eval.harness import EvalQuestion, ExpectedRoute

_DEFAULT_GOLDEN = [
    # ------------------------------ lookup ------------------------------
    EvalQuestion(
        qid="lookup-1",
        category="lookup",
        question="what does Encoder do",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.embed.Encoder"],
    ),
    EvalQuestion(
        qid="lookup-2",
        category="lookup",
        question="explain how chunk_id is constructed",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.indexer.ast_parser.Chunk.chunk_id"],
    ),
    EvalQuestion(
        qid="lookup-3",
        category="lookup",
        question="how does the heuristic grader score retrieval",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.agent.HeuristicGrader.grade"],
    ),
    # ------------------------------ structural ------------------------------
    EvalQuestion(
        qid="structural-1",
        category="structural",
        question="who calls find_callers",
        expected_route=ExpectedRoute.STRUCTURAL,
        gold_qualified_names=[
            "codex_atlas.indexer.graph.CallGraph.find_callers",
            "codex_atlas.indexer.graph.CallGraph.neighbors",
        ],
    ),
    EvalQuestion(
        qid="structural-2",
        category="structural",
        question="who calls _extract_qualified_name",
        expected_route=ExpectedRoute.STRUCTURAL,
        gold_qualified_names=[
            "codex_atlas.retriever._extract_qualified_name",
            "codex_atlas.retriever.Retriever._structural",
        ],
    ),
    EvalQuestion(
        qid="structural-3",
        category="structural",
        question="callers of register_async",
        expected_route=ExpectedRoute.STRUCTURAL,
        gold_qualified_names=[
            "codex_atlas.store.ChunkStore.upsert_chunks",
            "codex_atlas.store.ChunkStore.search",
        ],
    ),
    # ------------------------------ multi-hop ------------------------------
    EvalQuestion(
        qid="hybrid-1",
        category="multi-hop",
        question="show me all retriever-related code",
        expected_route=ExpectedRoute.HYBRID,
        gold_qualified_names=[
            "codex_atlas.retriever.Retriever",
            "codex_atlas.retriever.classify",
            "codex_atlas.retriever.RetrieverConfig",
        ],
    ),
    EvalQuestion(
        qid="hybrid-2",
        category="multi-hop",
        question="end-to-end indexing pipeline",
        expected_route=ExpectedRoute.HYBRID,
        gold_qualified_names=[
            "codex_atlas.indexer.walker.parse_corpus",
            "codex_atlas.indexer.ast_parser.parse_python_file",
            "codex_atlas.indexer.graph.CallGraph.ingest",
        ],
    ),
    # ------------------------------ summarization ------------------------------
    EvalQuestion(
        qid="summary-1",
        category="summarization",
        question="walk me through the agent loop",
        expected_route=ExpectedRoute.SUMMARIZATION,
        gold_qualified_names=[
            "codex_atlas.agent.Agent",
            "codex_atlas.agent.Agent.run",
        ],
    ),
    EvalQuestion(
        qid="summary-2",
        category="summarization",
        question="overview of the indexer module",
        expected_route=ExpectedRoute.SUMMARIZATION,
        gold_qualified_names=[
            "codex_atlas.indexer",
            "codex_atlas.indexer.ast_parser",
            "codex_atlas.indexer.graph",
        ],
    ),
    # ------------------------------ failure-likely ------------------------------
    EvalQuestion(
        qid="refusal-1",
        category="failure-likely",
        question="what does frobnicate_widget do",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=[],  # no answer expected; agent should say so
    ),
    # ------------------------------ out-of-scope ------------------------------
    EvalQuestion(
        qid="oos-1",
        category="out-of-scope",
        question="how do I deploy this to AWS Lambda",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=[],  # not in the corpus; agent should hedge
    ),
]


def load_golden_set(path: str | Path | None = None) -> list[EvalQuestion]:
    """Load the golden set. With `path=None`, returns the in-source default.

    Persisting + reloading from JSON makes this commit-friendly even if
    the eval set evolves outside Python.
    """
    if path is None:
        return list(_DEFAULT_GOLDEN)
    raw = json.loads(Path(path).read_text())
    return [EvalQuestion(**row) for row in raw]


def save_golden_set(questions: list[EvalQuestion], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    @dataclass
    class _Row:
        qid: str
        category: str
        question: str
        expected_route: str
        gold_qualified_names: list[str]

    rows = [
        {
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "expected_route": str(q.expected_route),
            "gold_qualified_names": list(q.gold_qualified_names),
        }
        for q in questions
    ]
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return out
