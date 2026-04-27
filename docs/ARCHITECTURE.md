# Architecture

Codex-Atlas is split into five layers, each with a tight contract.
This doc reads bottom-up because that mirrors how data flows through
the system.

## 1. Indexer

`src/codex_atlas/indexer/`

- `walker.py` walks a directory tree, skipping `.venv`, `__pycache__`,
  `.git`, `node_modules`, build/dist artefacts, and language caches.
  Each `.py` file is parsed exactly once.
- `ast_parser.py` is a `NodeVisitor` over the standard library `ast`
  module. For every file it produces:
  - one `Symbol` per module / class / function / method, tagged with
    `is_async`, `decorators`, and `docstring`;
  - one `Chunk` per function-or-method (we chunk at function granularity
    because that's the unit that fits in a typical LLM prompt with the
    surrounding signature);
  - `imports` (display strings) **and** `import_refs`
    (`local-name -> dotted-target` bindings) — the latter feeds the
    graph's imports-aware call resolution;
  - `calls`: `(caller_qualified_name, callee_unqualified_name)` pairs,
    attributed only when there is an enclosing function;
  - `type_aliases`: PEP 695 `type X = Y` plus PEP 613
    `X: TypeAlias = Y` plus best-effort plain-assignment aliases.
- Lambdas / arbitrary expressions in call position are dropped on
  purpose — they have no resolvable callee name and would create noise
  nodes in the graph.

## 2. Call graph

`src/codex_atlas/indexer/graph.py`

A NetworkX `MultiDiGraph` with three edge kinds:

| edge | meaning |
| --- | --- |
| `defines` | module → class / function / method directly inside it |
| `imports` | module → imported target (best-effort, may dangle) |
| `calls` | caller → callee |

Call resolution is two-stage:

1. **Imports-map first.** Per-module `local-name -> dotted-target` from
   `ImportRef`s. If the callee's short name resolves to a target that
   exists in the graph, record an edge to that exact target only.
2. **Short-name fallback.** Otherwise, record an edge per qualified-name
   match in the corpus (preserves v1 behaviour for unresolved calls).

Why NetworkX over Neo4j: a 50K-node single-codebase graph fits in
memory comfortably, the API surface the retriever needs is three
functions, and persistence is a single diffable JSON file. ADR-001
covers the alternative (FAISS for vectors) and ADR-002 covers why
graph-walk is the first route the agent considers.

## 3. Vector store

`src/codex_atlas/store.py`

pgvector on Postgres 17 via `asyncpg`. One row per indexed chunk
(`id`, `qualified_name`, `file_path`, `lineno_start`, `lineno_end`,
`kind`, `text`, `vec`). HNSW index on `vec` with `vector_cosine_ops`,
parameters tuned for the 10K–50K chunk range a single-codebase ingest
hits. `upsert_chunks` is idempotent on `(file_path, qualified_name,
lineno_start)`.

The encoder is a `Protocol` so we can swap blake2b (`FakeEncoder`,
deterministic + offline tests) for `sentence-transformers` (production)
without code changes.

## 4. Retriever

`src/codex_atlas/retriever.py`

Six adaptive routes, picked by a regex-based classifier:

| route | trigger phrasing | strategy |
| --- | --- | --- |
| lookup | "what does X do" | vector top-k |
| structural | "who calls X" | graph callers/callees |
| hybrid | "show me all auth-related X" | top-k + 1-hop graph expansion |
| summarization | "walk me through X" | wider top-k + 2-hop neighbours |
| neighborhood | "neighborhood of X" | both-direction BFS to depth 2 |
| import_chain | "which modules import X" | reverse imports walk |

Confidences are calibrated against a 30-question hand-graded
development set so the agent's grader can compare them on a meaningful
scale (see `evals/INTERPRETATION.md`).

The `hybrid_score` helper combines cosine similarity, graph distance
(decaying as `1 / (1 + d)`), and a coarse term-overlap full-text score
into a single combined score. Weights are tunable; the scorer
normalises them so callers can pass un-normalised vectors.

## 5. Agent state machine

`src/codex_atlas/agent.py`

Seven nodes, all timestamped:

```
classify -> retrieve -> grade ─┐
                ▲              │
                │   grade < threshold AND attempts < max
                └── rewrite_query ──┘
                            │
                            ▼
                         answer
                            │
                            ▼
                        validate
```

- **classify** is implicit inside the retriever — we emit a separate
  trace event so the LangGraph vocabulary stays intact.
- **rewrite_query** appends a clarifier when retrieval comes back
  weak and re-runs.
- **validate** (default `CitationValidator`) extracts every backticked
  qualified name from the answer and flags any that aren't in the
  retrieved chunks. The threshold (default 0.5) is conservative on
  purpose; tighten it for production.
- **cancel** fires when a per-step or whole-run timeout trips. Result
  carries `cancelled=CancelReason.TIMEOUT` and an empty answer/citations.

`tool_calls` is a structured log of every retriever invocation
(query, route, n_chunks, elapsed_ms, confidence). External
observability planes (Langfuse, OpenTelemetry) subscribe to it without
re-implementing the state machine. ADR-003 covers why we picked a
hand-rolled state machine over LangGraph or a ReAct loop.

## 6. MCP server

`src/codex_atlas/mcp_server.py`

FastMCP wrapping seven tools and one resource:

- `search_code(query, top_k)` — adaptive-route search
- `explain_function(qualified_name)` — chunk + immediate neighbours
- `find_callers(qualified_name, depth)` — graph-only callers traversal
- `summarize_module(module_path)` — wider retrieval + 2-hop expansion
- `search_codebase(query, top_k, route?)` — explicit route override
- `get_graph_neighborhood(symbol, depth)` — both-direction BFS
- `explain(symbol)` — agent end-to-end on a symbol
- `codebase://stats` — node/edge counts + language breakdown

ADR-004 documents the tool-shape decisions.

## 7. Eval harness

`src/codex_atlas/eval/`

Eight metrics, no LLM required:

- `route_correctness`
- `citation_recall`, `citation_precision`
- `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`
- `tool_call_count`
- `cost_estimate_usd` (tokens × per-1K price; pluggable)

A 7-bucket failure taxonomy (`missing_node`, `wrong_route`,
`hallucinated`, `partial`, `ungrounded`, `off_topic`, `outdated_index`)
classifies each wrong answer so reviewers can see *why* the harness is
losing accuracy, not just *that* it is.

`evaluate_against_baseline` diffs the current run against a saved
JSON baseline; the CLI exits non-zero on regression beyond the
configured tolerance. `write_failure_report` dumps a per-question
JSONL so debugging is `cat | jq`.

## CI

`.github/workflows/ci.yml` runs lint + format-check + mypy + tests
across Python 3.11 and 3.12 on every PR and main push. Concurrency
groups cancel in-progress runs on the same ref. Permissions are
limited to `contents: read`.

## Limitations

A short list of known gaps the indexer + retriever do *not* try to
solve. Each is in scope for a later release; recording them here keeps
the gap visible without polluting the test suite with x-fail markers.

- **Star imports.** `from foo import *` is recorded as an `ImportRef`
  with `local="*"` but never resolved. We do not enumerate the names
  re-exported by `foo` (this would require importing the module at
  index time, which the walker deliberately avoids — the AST parser is
  pure-syntactic so it works on partially-broken corpora). Calls
  resolved via a star-bound name fall back to the short-name index, so
  the call edge is still recorded; only the *imports* edge is silently
  unresolved. Guidance: `__all__`-driven re-exports are the supported
  pattern.
- **Dynamic dispatch.** Calls through `getattr(obj, name)(...)` or
  registry-driven lookups (e.g. plugin systems) are invisible to the
  AST parser. The graph captures only what the source spells out.
- **Cross-language calls.** Python -> C extension transitions stop at
  the FFI boundary; the graph is Python-only.
