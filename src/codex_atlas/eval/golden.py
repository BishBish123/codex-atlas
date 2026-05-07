"""The 30-question golden eval set.

We deliberately use *this* repo (codex-atlas) as the eval corpus so the
suite is hermetic: no external clone required, every gold citation is
verifiable in the same git tree, and reviewers can re-run `make eval`
without standing up a separate dataset.

Categories use the 8-category taxonomy: lookup, structural, multi-hop,
summarization, neighborhood, import-chain, failure-likely, out-of-scope.
The brief's "5 question types" map onto this richer taxonomy as follows:
  lookup       -> lookup (3 q)
  structural   -> structural (3 q)
  hybrid       -> multi-hop (3 q)
  summarization -> summarization (3 q) + neighborhood (3 q)
  failure      -> failure-likely (2 q) + out-of-scope (2 q)
  import-chain -> import-chain (3 q)
Total: 6+5+4+4+4+3+2+2 = 30 questions (8-category taxonomy, expanded from the original 16-question core).
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
    # Note: `category` is a free-form human label (e.g. "multi-hop") and is
    # independent of `expected_route` (e.g. ExpectedRoute.HYBRID). The two
    # dimensions are intentionally separate: category groups questions for the
    # "By category" report table; expected_route is the classifier target.
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
    # ------------------------------ neighborhood ------------------------------
    EvalQuestion(
        qid="neighborhood-1",
        category="neighborhood",
        question="neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers",
        expected_route=ExpectedRoute.NEIGHBORHOOD,
        gold_qualified_names=[
            "codex_atlas.indexer.graph.CallGraph.find_callers",
            "codex_atlas.indexer.graph.CallGraph.neighbors",
        ],
    ),
    EvalQuestion(
        qid="neighborhood-2",
        category="neighborhood",
        question="everything around codex_atlas.retriever.Retriever.retrieve",
        expected_route=ExpectedRoute.NEIGHBORHOOD,
        gold_qualified_names=[
            "codex_atlas.retriever.Retriever.retrieve",
        ],
    ),
    # ------------------------------ import_chain ------------------------------
    EvalQuestion(
        qid="import-chain-1",
        category="import-chain",
        question="which modules import codex_atlas.store",
        expected_route=ExpectedRoute.IMPORT_CHAIN,
        gold_qualified_names=[
            "codex_atlas.agent",
            "codex_atlas.cli",
            "codex_atlas.mcp_server",
            "codex_atlas.retriever",
        ],
    ),
    EvalQuestion(
        qid="import-chain-2",
        category="import-chain",
        question="import chain for codex_atlas.indexer.graph",
        expected_route=ExpectedRoute.IMPORT_CHAIN,
        gold_qualified_names=[
            "codex_atlas.retriever",
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
    # ========================= NEW QUESTIONS (14) ============================
    # ------------------------------ lookup (3 more) --------------------------
    EvalQuestion(
        qid="lookup-4",
        category="lookup",
        question="what does FakeEncoder do",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.embed.FakeEncoder"],
    ),
    EvalQuestion(
        qid="lookup-5",
        category="lookup",
        question="how does InMemoryChunkStore store chunks",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.store.InMemoryChunkStore"],
    ),
    EvalQuestion(
        qid="lookup-6",
        category="lookup",
        question="what does the StitchSynthesizer produce",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=["codex_atlas.agent.StitchSynthesizer"],
    ),
    # ------------------------------ structural (2 more) ----------------------
    EvalQuestion(
        qid="structural-4",
        category="structural",
        question="who calls hybrid_score",
        expected_route=ExpectedRoute.STRUCTURAL,
        gold_qualified_names=[
            "codex_atlas.retriever.hybrid_score",
            "codex_atlas.retriever.Retriever._hybrid",
        ],
    ),
    EvalQuestion(
        qid="structural-5",
        category="structural",
        question="callees of parse_corpus",
        expected_route=ExpectedRoute.STRUCTURAL,
        gold_qualified_names=[
            "codex_atlas.indexer.walker.parse_corpus",
            "codex_atlas.indexer.ast_parser.parse_python_file",
        ],
    ),
    # ------------------------------ multi-hop (2 more) -----------------------
    EvalQuestion(
        qid="hybrid-3",
        category="multi-hop",
        question="show me all embed-related code",
        expected_route=ExpectedRoute.HYBRID,
        gold_qualified_names=[
            "codex_atlas.embed.Encoder",
            "codex_atlas.embed.FakeEncoder",
        ],
    ),
    EvalQuestion(
        qid="hybrid-4",
        category="multi-hop",
        question="find all code that does cache-related work",
        expected_route=ExpectedRoute.HYBRID,
        gold_qualified_names=[
            "codex_atlas.store.InMemoryChunkStore",
            "codex_atlas.store.ChunkStore",
        ],
    ),
    # ------------------------------ summarization (2 more) -------------------
    EvalQuestion(
        qid="summary-3",
        category="summarization",
        question="walk me through how the retriever classifies a query",
        expected_route=ExpectedRoute.SUMMARIZATION,
        gold_qualified_names=[
            "codex_atlas.retriever.classify",
            "codex_atlas.retriever.Retriever.retrieve",
        ],
    ),
    EvalQuestion(
        qid="summary-4",
        category="summarization",
        question="overview of the store module",
        expected_route=ExpectedRoute.SUMMARIZATION,
        gold_qualified_names=[
            "codex_atlas.store",
            "codex_atlas.store.InMemoryChunkStore",
            "codex_atlas.store.ChunkStore",
        ],
    ),
    # ------------------------------ neighborhood (2 more) --------------------
    EvalQuestion(
        qid="neighborhood-3",
        category="neighborhood",
        question="neighborhood of codex_atlas.indexer.graph.CallGraph.ingest",
        expected_route=ExpectedRoute.NEIGHBORHOOD,
        gold_qualified_names=[
            "codex_atlas.indexer.graph.CallGraph.ingest",
            "codex_atlas.indexer.graph.CallGraph.add_symbol",
        ],
    ),
    EvalQuestion(
        qid="neighborhood-4",
        category="neighborhood",
        question="everything around codex_atlas.agent.Agent.run",
        expected_route=ExpectedRoute.NEIGHBORHOOD,
        gold_qualified_names=[
            "codex_atlas.agent.Agent.run",
            "codex_atlas.agent.Agent",
        ],
    ),
    # ------------------------------ import_chain (1 more) -------------------
    EvalQuestion(
        qid="import-chain-3",
        category="import-chain",
        question="which modules import codex_atlas.embed",
        expected_route=ExpectedRoute.IMPORT_CHAIN,
        gold_qualified_names=[
            "codex_atlas.cli",
            "codex_atlas.mcp_server",
            "codex_atlas.retriever",
        ],
    ),
    # ------------------------------ failure-likely (1 more) -----------------
    EvalQuestion(
        qid="refusal-2",
        category="failure-likely",
        question="what does authenticate_user do",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=[],  # symbol does not exist in corpus; agent should say so
    ),
    # ------------------------------ out-of-scope (1 more) -------------------
    EvalQuestion(
        qid="oos-2",
        category="out-of-scope",
        question="what is the weather in San Francisco today",
        expected_route=ExpectedRoute.LOOKUP,
        gold_qualified_names=[],  # entirely off_topic; agent should refuse / reroute
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
