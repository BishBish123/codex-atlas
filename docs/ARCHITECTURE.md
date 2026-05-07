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
hits. `upsert_chunks` is idempotent on stable `(file_path,
qualified_name)` IDs plus per-file tombstoning, so re-indexing the
same file replaces stale chunks even when their line ranges shift.

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

## 7. Observability

`src/codex_atlas/observability/`

An env-key-gated Langfuse tracing adapter that subscribes to the agent's
structured `trace` and `tool_calls` lists and emits them as Langfuse spans.

**Components:**

- `LangfuseTracer` — live tracer; imports the `langfuse` SDK lazily so the
  module loads without it installed. Supports both v2/v3 (`trace()` /
  `span()` API) and v4 (`start_observation()` API).
- `NullTracer` — no-op fallback with the same interface. Used when
  `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` are absent.
- `make_tracer()` — factory; returns `LangfuseTracer` when both keys are
  set, `NullTracer` otherwise.

**Env vars:**

| Variable | Effect |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Required to enable live tracing. |
| `LANGFUSE_SECRET_KEY` | Required to enable live tracing. |
| `LANGFUSE_HOST` | Optional. Defaults to `https://cloud.langfuse.com`. |

**Instrumentation points in `Agent.run`:**

1. `start_run(query)` — opens a top-level Langfuse trace.
2. `record_event(trace_id, event)` — called after each `TraceEvent` and
   `ToolCall` is appended to the internal state (classify, retrieve, grade,
   rewrite, answer, validate, cancel).
3. `record_validation(trace_id, report)` — emits the `CitationValidator`
   result as a separate span.
4. `finish_run(trace_id, result)` — attaches the final answer summary to
   the trace and calls `flush()` so buffered events survive short-lived
   CLI invocations.

The `tracer` is a constructor arg on `Agent` (default: `make_tracer()`),
so tests can inject `NullTracer` explicitly without touching env vars.

## 8. Eval harness

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

## ADR-05: Store protocol + env-var-driven backend selection

### Context

The original `store.py` module contained a single concrete pgvector adapter
(`ChunkStore`) and a test-only `InMemoryChunkStore`. There was no first-class
way for operators to switch backends without modifying source: the `cli.py`
chose between them via `--store=memory` / `--store=postgres` flag logic, and
the `mcp_server.py` read `ATLAS_STORE` for the same choice. Adding a second
pgvector adapter (with a different DDL schema and simpler method surface) while
keeping existing code working required a stable public interface and a clean
dispatch point.

### Decision

1. **Convert `store.py` to a package** (`store/`). All existing names (`ChunkStore`,
   `InMemoryChunkStore`, `StoredChunk`, `ChunkStoreProtocol`, …) are re-exported
   from `store/__init__.py` so every existing `from codex_atlas.store import X`
   import continues to work unchanged.

2. **Add `PgVectorChunkStore`** in `store/pgvector.py`. This backend uses a
   simplified schema (named `chunks`, embedding column named `embedding` instead
   of `vec`) and ivfflat indexing instead of HNSW, and exposes a lighter method
   surface (`upsert_chunks(list[StoredChunk])`, `upsert_with_embeddings`, `search`,
   `fetch_by_qualified_name`, `get`, `clear`). It is gated behind `ATLAS_PG_DSN`
   and soft-fails the `asyncpg`/`pgvector` imports so the package stays importable
   on memory-only installs.

3. **Add `make_chunk_store()`** in `store/__init__.py`. A single factory that reads
   `ATLAS_CHUNK_STORE` (`memory` | `pgvector`) and dispatches:
   - `memory`   → `InMemoryChunkStore()` (default)
   - `pgvector` → `PgVectorChunkStore(dsn=os.environ["ATLAS_PG_DSN"])`
   Unknown backends raise `RuntimeError` with an actionable message.

4. **No changes to `cli.py` or `mcp_server.py`**. The existing `--store` flag
   logic and `ATLAS_STORE` env var remain in place and continue to operate the
   existing `ChunkStore` / `InMemoryChunkStore` pair. `make_chunk_store()` is
   an additive hook for new tooling that wants env-var-driven dispatch.

### Rationale

- **Backward compatibility**: converting to a package instead of adding a second
  flat module preserves every existing import without a grep-and-replace.
- **Single dispatch point**: `make_chunk_store()` is the only place that reads
  `ATLAS_CHUNK_STORE`, so adding a third backend (e.g. SQLite, Qdrant) is a
  one-function change with no ripple to callers.
- **Soft-fail imports**: `asyncpg` and `pgvector` are already hard dependencies
  in `pyproject.toml`, but the `try/except ImportError` guards in `pgvector.py`
  make the error message actionable on mis-configured installs rather than
  producing an opaque `ModuleNotFoundError` at import time.
- **ivfflat over HNSW for `PgVectorChunkStore`**: ivfflat appends cheaply on
  INSERT (no graph rebuild), which suits the incremental-reindex pattern the
  CLI uses. HNSW is retained in the original `ChunkStore` where its higher
  recall-at-low-probes matters for the production query path.

## ADR-06: Dual call-graph backend — NetworkX (default) + Neo4j (optional)

### Context

The original call graph was implemented entirely in-memory with NetworkX
(`src/codex_atlas/indexer/graph.py`). For single-user portfolio deployments
(the primary target) this is ideal — a 50K-node graph fits in RAM, persistence
is a single diffable JSON file, and no external service is required.

As codex-atlas is deployed against larger mono-repos (500K+ symbols) or in
multi-user team settings, the in-memory graph becomes a bottleneck: the JSON
snapshot can exceed several hundred MB and every reader must load the full
graph before answering even a single query. Neo4j AuraDB solves both problems —
it persists the graph natively and evaluates Cypher queries server-side — but
requiring it for every user would break the "zero-infra quick-start" promise.

### Decision

1. **Add `Neo4jCallGraph`** in `src/codex_atlas/indexer/neo4j_graph.py`.
   Uses the official `neo4j` async Python driver. Reads `ATLAS_NEO4J_URI`,
   `ATLAS_NEO4J_USERNAME` (default `neo4j`), and `ATLAS_NEO4J_PASSWORD` from
   environment variables. The driver import is a soft-fail — the module loads
   cleanly when `neo4j` is absent; `Neo4jCallGraph()` raises `ImportError` at
   instantiation time with an actionable message.

2. **Add `make_call_graph()` factory** in `src/codex_atlas/indexer/__init__.py`.
   Reads `ATLAS_GRAPH_BACKEND` (`networkx` | `neo4j`). Default is `networkx`
   so no existing deployment breaks. Unknown values raise `RuntimeError`.

3. **Add `--graph-backend` option to `atlas index`**. Passes through to the
   factory via env-var so operators can switch backends per-run without
   modifying their environment permanently.

4. **Ship `docker-compose.neo4j.yml`** with `make neo4j-up` / `make neo4j-down`
   targets for local development, mirroring the existing pgvector compose file.

5. **Add `neo4j>=5.20` to the `[real]` extras** in `pyproject.toml`.
   Not a hard dependency — memory-only installs stay lean.

### Rationale

- **Default unchanged**: `ATLAS_GRAPH_BACKEND` defaults to `networkx`. Every
  existing test, CLI invocation, and MCP server configuration continues to
  work with zero changes.
- **Soft-fail imports**: identical pattern to `PgVectorChunkStore`. Importable
  without the driver; actionable error at runtime.
- **Cypher schema is minimal and idempotent**: one uniqueness constraint
  (`symbol_qname`) + one index (`symbol_module`). Both use `IF NOT EXISTS`.
- **Relationship types are allow-listed**: Cypher does not support parameterised
  relationship types, so `add_edge()` validates `kind` against a fixed
  frozenset (`CALLS`, `IMPORTS`) and formats it into the template string.
  This prevents injection while keeping the API clean.

### When to pick each backend

| Criteria | Pick NetworkX | Pick Neo4j |
| --- | --- | --- |
| Quick-start / CI | Yes | No |
| Offline / no Docker | Yes | No |
| > 500K symbols | No | Yes |
| Multi-user sharing | No | Yes |
| Need graph analytics (PageRank, shortest path) | No | Yes |

## ADR-07: Dual agent backend — hand-rolled (default) + LangGraph (optional)

### Context

The project brief promises LangGraph as the orchestration framework.
`agent.py` implements a "LangGraph-style" hand-rolled async state machine
(classify → retrieve → grade → rewrite_query → answer → validate → cancel)
that deliberately avoids the real LangGraph package to keep cold-start fast
and the dependency list short.

As codex-atlas is positioned as a serious RAG reference, reviewers correctly
ask why it uses a bespoke state machine rather than the industry-standard
framework. Adding the real LangGraph backend answers that question without
breaking any existing deployment.

### Decision

1. **Add `LangGraphAgent`** in `src/codex_atlas/agent_langgraph.py`.
   Uses a real `StateGraph` with seven nodes (classify, retrieve, grade,
   rewrite_query, answer, validate, cancel) and conditional edges:
   - `grade` → `rewrite_query` (when grade < threshold AND attempts < max)
   - `grade` → `answer` (otherwise)
   - `rewrite_query` → `retrieve` (loop-back edge)
   - `answer` → `validate` → `END`
   The `langgraph` import is soft-fail: the module loads cleanly without
   the package installed; `LangGraphAgent.__init__` raises a `RuntimeError`
   with an install hint at construction time.

2. **Add `make_agent()` factory** in `src/codex_atlas/__init__.py`.
   Reads `ATLAS_AGENT_BACKEND` (env var) or a `backend=` kwarg:
   - `"hand-rolled"` (default) → existing `Agent`
   - `"langgraph"` → `LangGraphAgent`
   Unknown values raise `RuntimeError` with valid choices listed.

3. **Add `--backend hand-rolled|langgraph` to `atlas ask`** CLI.
   Default unchanged (`hand-rolled`). The flag sets the backend for a
   single invocation without touching env vars.

4. **Add `langgraph>=0.2` to `[real]` extras** in `pyproject.toml`.
   Not a hard dependency — memory-only installs stay lean.

### Rationale

- **Default unchanged**: `ATLAS_AGENT_BACKEND` defaults to `"hand-rolled"`.
  Every existing test, CLI invocation, MCP server, and eval run continues
  to work with zero changes. No new transitive imports at startup.
- **Same Protocol implementations**: `LangGraphAgent` reuses the same
  `Retriever`, `Grader`, `Synthesizer`, `Validator` Protocol objects as
  `Agent`. The swap is purely orchestration — behavior is identical.
- **Soft-fail import**: identical pattern to `PgVectorChunkStore` and
  `Neo4jCallGraph`. The package stays importable without langgraph; the
  error is actionable at construction time rather than at `import` time.
- **Why hand-rolled stays default**:
  - No extra dependency (`langgraph>=0.2` pulls `pydantic`, `langchain-core`,
    `httpx`, etc. — adds ~15 transitive packages).
  - Faster cold-start: the hand-rolled machine starts in microseconds; the
    LangGraph graph compilation + `StateGraph.compile()` adds overhead.
  - Simpler debugging: the hand-rolled machine is ~400 lines of vanilla
    asyncio with no framework magic between nodes.
  - CI stays hermetic: the langgraph tests use a minimal in-process stub
    and never require the real package.

### State graph shape

```
retrieve → grade ─┐
    ▲              │ grade < threshold AND attempts < max
    │              ▼
    └── rewrite_query
                   │
                   │ grade >= threshold OR attempts exhausted
                   ▼
               answer → validate → END
```

Seven nodes total; the `cancel` node is registered but only reachable
if a caller injects a cancellation signal into the state before graph
invocation (the current implementation does not exercise it, but it
mirrors the hand-rolled machine's node vocabulary for parity).

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
