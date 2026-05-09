# Codex-Atlas

> Agentic GraphRAG over a Python codebase. Adaptive vector + graph hybrid retrieval with reflection + bounded re-query, exposed as a **Model Context Protocol** server.

[![ci](https://github.com/BishBish123/codex-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/codex-atlas/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![mcp](https://img.shields.io/badge/mcp-spec--conformant-success)](https://modelcontextprotocol.io/)
[![eval](https://img.shields.io/badge/route%20accuracy-93.8%25-brightgreen)](evals/REPORT.md)

**New here?** See [DEMO.md](DEMO.md) for a 5-minute walkthrough — indexed commands, expected output, and what to evaluate.

---

## What it does

You point Codex-Atlas at a Python repo. It extracts function-grain chunks via the stdlib `ast` module (capturing async-def, decorators, docstrings, and type aliases), embeds them into pgvector, and builds an in-memory call graph (NetworkX) with imports-aware call resolution. When you ask a question:

1. **Classify** the query — is it `lookup`, `structural`, `hybrid`, `summarization`, `neighborhood`, or `import_chain`?
2. **Retrieve** with the route that matches:
   * lookup → vector top-k
   * structural → graph callers / callees
   * hybrid → vector top-k → 1-hop graph expansion + tunable hybrid scorer (cosine + graph-distance + tfidf-ish overlap)
   * summarization → wider top-k + 2-hop graph neighbours
   * neighborhood → both-direction BFS to depth 2
   * import_chain → reverse imports walk
3. **Grade** the retrieval. If it's weak, **rewrite** the query and retry (capped at `max_attempts=3`, with optional per-step / whole-run timeouts).
4. **Answer** with citations into the source.
5. **Validate** — `CitationValidator` flags any backticked qname in the answer that isn't in the retrieved chunks. Accept threshold is configurable.

Every step is recorded in a structured trace + a `tool_calls` log so adding Langfuse / OpenTelemetry later is a 30-line adapter.

The whole thing is wrapped as a FastMCP server with 7 tools + 1 resource:

| Tool | Purpose |
| --- | --- |
| `search_code(query, top_k)` | Adaptive-route retrieval + synthesis |
| `explain_function(qualified_name)` | Pull a chunk + its immediate neighbours |
| `find_callers(qualified_name, depth)` | Graph-only callers traversal |
| `summarize_module(module_path)` | Wider retrieval + 2-hop expansion |
| `search_codebase(query, top_k, route?)` | Adaptive search with explicit route override |
| `get_graph_neighborhood(symbol, depth)` | Both-direction BFS on the call graph |
| `explain(symbol)` | Agent end-to-end on a symbol |
| `codebase://stats` (resource) | Node/edge counts + language breakdown |

## Architecture

```
                   ┌─ MCP client (Claude Code / Cursor / Inspector)
                   ▼
             ┌────────────┐
             │ FastMCP    │  7 tools + 1 resource, Pydantic-typed
             └──────┬─────┘
                    ▼
         ┌──────────────────────────┐
         │   Agent (state machine)  │  classify → retrieve → grade
         │   ┌─────────┐            │       ▲           │
         │   │ classify│            │       │   if grade < threshold
         │   ├─────────┤            │       │   AND attempts < max
         │   │retrieve │── extras ──┘       │           │
         │   ├─────────┤                    │           ▼
         │   │  grade  │              rewrite_query ────┘
         │   ├─────────┤
         │   │ answer  │ → markdown + Citations[]
         │   └─────────┘
         └──────┬─────────┬──────────┐
                ▼         ▼          ▼
        ┌────────────┐ ┌──────────┐ ┌────────────┐
        │ pgvector   │ │ Call graph│ │ Synthesizer │
        │ (Postgres) │ │ (NetworkX)│ │ (stitch    │
        │            │ │ persisted │ │  or LLM)   │
        │            │ │ as JSON   │ │            │
        └────────────┘ └──────────┘ └────────────┘
```

## Quick start

```bash
git clone https://github.com/BishBish123/codex-atlas.git
cd codex-atlas
make install          # uv sync (+ embed extra on supported platforms)
# Intel macOS: `make install` and `make install-min` are equivalent — `pyproject.toml`
# already gates `torch` / `sentence-transformers` behind `platform_machine == 'arm64'`,
# so the heavy ML extras are skipped automatically.

# No Docker, no DB — index into a JSON snapshot at data/chunks.json.
uv run atlas index src/ --store=memory

# Ask a question (reads data/chunks.json + data/graph.json).
uv run atlas ask "who calls find_callers" --store=memory

# Other subcommands worth knowing:
uv run atlas search "who calls _traverse" --store=memory          # raw retrieval, no synthesis
uv run atlas explain "codex_atlas.retriever.Retriever.retrieve"   # symbol + immediate neighbours

# Point it at any other Python repo (absolute path):
uv run atlas index /Users/me/my-project --store=memory
# The MCP config below picks up data/{graph,chunks}.json from wherever you ran `atlas index`.

# Run the golden test set.
uv run atlas eval --json-out evals/scores.json
cat evals/REPORT.md

# Run the unit tests.
uv run pytest -q                          # or: make test
```

Want pgvector instead? Boot Postgres + pgvector and switch the flag:

```bash
docker run -d --name codex-atlas-pg \
    -e POSTGRES_PASSWORD=bench -e POSTGRES_USER=bench -e POSTGRES_DB=bench \
    -p 5433:5432 pgvector/pgvector:pg17
export POSTGRES_DSN=postgresql://bench:bench@localhost:5433/bench
uv run atlas index src/ --store=postgres --encoder fake --drop-existing
uv run atlas ask "who calls find_callers" --store=postgres
```

## Demo cast

A pre-recorded terminal session is committed at `assets/demo.cast`. It captures
real stdout from six CLI commands against the bundled corpus.

**Play locally** (requires [asciinema](https://asciinema.org/)):

```bash
bash scripts/play_demo.sh
# or directly:
asciinema play assets/demo.cast
```

**Watch on asciinema.org:**

[![asciicast](https://asciinema.org/a/LhtcqI2ZjiJY2j4W.svg)](https://asciinema.org/a/LhtcqI2ZjiJY2j4W)

**Refresh the cast** after code changes:

```bash
make demo-cast        # regenerates assets/demo.cast from real CLI output
```

## pgvector chunk store

Codex-Atlas ships two chunk-store backends. The default is `memory` (a JSON snapshot at `data/chunks.json` — no DB required). The `pgvector` backend persists chunks in Postgres with a pgvector extension and supports ANN search via cosine distance.

### Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT  PRIMARY KEY,
    qualified_name TEXT  NOT NULL,
    file_path      TEXT  NOT NULL,
    lineno_start   INT   NOT NULL,
    lineno_end     INT   NOT NULL,
    text           TEXT  NOT NULL,
    embedding      vector(384)   -- bge-small-en-v1.5 dim
);

CREATE INDEX IF NOT EXISTS chunks_qname_idx     ON chunks (qualified_name);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**Why ivfflat with lists=100?** For the typical 10k-100k-chunk corpus a single codebase produces, ivfflat partitions vectors into 100 Voronoi cells. At query time only the nearest cell(s) are probed, giving O(n/100) scan with >95% recall at `probes=10`. HNSW gives marginally higher recall but its build cost is O(n × m × ef_construction) and its on-disk size is 2-5× larger — a poor trade for an incremental-upsert workload.

### env var contract

| Variable | Default | Purpose |
| --- | --- | --- |
| `ATLAS_CHUNK_STORE` | `memory` | Backend selector: `memory` or `pgvector`. |
| `ATLAS_PG_DSN` | _(required for pgvector)_ | asyncpg DSN for the pgvector backend. |

`make_chunk_store()` (in `codex_atlas.store`) reads these at import time and returns the right backend. Existing `--store=memory` / `--store=postgres` CLI flags are unchanged.

### Local development with Docker

```bash
# Start Postgres + pgvector (port 5433, no clash with system Postgres)
make pg-up

# Index into pgvector
export ATLAS_PG_DSN=postgresql://atlas:atlas@localhost:5433/atlas
ATLAS_CHUNK_STORE=pgvector uv run atlas index src/ --store=pgvector

# Ask a question
ATLAS_CHUNK_STORE=pgvector uv run atlas ask "who calls find_callers" --store=pgvector

# Tear down
make pg-down
```

## Neo4j graph backend

Codex-Atlas ships a **Neo4j AuraDB / Community** graph backend alongside the
default in-memory NetworkX backend. Both implement the same family of
call/import graph operations; the right choice depends on your deployment:

| Dimension | NetworkX (default) | Neo4j |
| --- | --- | --- |
| Setup | Zero — no container | Needs AuraDB or local container |
| Persistence | JSON snapshot (`data/graph.json`) | Neo4j database |
| Scale | Up to ~50K nodes comfortably in RAM | Millions of nodes |
| Queries | Synchronous Python BFS | Cypher via async driver |
| When to pick | Single-user, CI, offline | Multi-user, large mono-repo, BI |

### Env-var contract

| Variable | Default | Purpose |
| --- | --- | --- |
| `ATLAS_GRAPH_BACKEND` | `networkx` | Backend selector: `networkx` or `neo4j`. |
| `ATLAS_NEO4J_URI` | _(required for neo4j)_ | Bolt or `neo4j+s` URI, e.g. `bolt://localhost:7687`. |
| `ATLAS_NEO4J_USERNAME` | `neo4j` | Database username. |
| `ATLAS_NEO4J_PASSWORD` | — | Database password. |

`make_call_graph()` (in `codex_atlas.indexer`) reads `ATLAS_GRAPH_BACKEND` and
returns the right backend. The `--graph-backend` CLI flag on `atlas index`
sets the backend for a single run without touching env vars.

### Cypher schema

Applied by `Neo4jCallGraph.setup()` (idempotent):

```cypher
CREATE CONSTRAINT symbol_qname IF NOT EXISTS FOR (s:Symbol) REQUIRE s.qualified_name IS UNIQUE;
CREATE INDEX symbol_module IF NOT EXISTS FOR (s:Symbol) ON (s.module);
```

### Local development with Docker

```bash
# Start Neo4j Community (bolt 7687, browser 7474, password neo4j-dev)
make neo4j-up

# Index into Neo4j
export ATLAS_NEO4J_URI=bolt://localhost:7687
export ATLAS_NEO4J_USERNAME=neo4j
export ATLAS_NEO4J_PASSWORD=neo4j-dev
uv run atlas index src/ --graph-backend neo4j

# Tear down
make neo4j-down
```

### Install the driver

The `neo4j>=5.20` driver is part of the `[real]` extras group:

```bash
uv sync --extra real
```

## Run as an MCP server

> **Prerequisite:** Run `uv run atlas index src/ --store=memory` first — the MCP server reads `data/graph.json` and crashes if it doesn't exist.

### Add to Claude Code in one line

```bash
# Register codex-atlas as an MCP server in Claude Code.
# Replace <absolute-path-to-codex-atlas> with the actual absolute path to
# your checkout on disk (e.g. /Users/you/projects/codex-atlas).
claude mcp add codex-atlas \
  -e ATLAS_STORE=memory \
  -e ATLAS_GRAPH_PATH=<absolute-path-to-codex-atlas>/data/graph.json \
  -e ATLAS_CHUNKS_PATH=<absolute-path-to-codex-atlas>/data/chunks.json \
  -- uv run atlas-mcp
```

After registering, restart Claude Code and the seven tools (`search_code`, `explain_function`, `find_callers`, `summarize_module`, `search_codebase`, `get_graph_neighborhood`, `explain`) are available in every conversation.

The MCP server defaults to `ATLAS_STORE=memory` (reads `data/chunks.json`), so no
Postgres container is needed.  Agent timeouts default to `ATLAS_STEP_TIMEOUT_S=10` (per
retrieval/synthesis step) and `ATLAS_RUN_TIMEOUT_S=30` (whole run); set either to `0` to
disable.  On timeout, tools return `{"error": "agent_timeout", "phase": "<last_node>"}`.

```bash
# stdio for Claude Desktop / Claude Code (memory store, no DSN needed).
# Use absolute paths so MCP clients with unpredictable cwds resolve them
# correctly — `$(pwd)/...` expands at shell-eval time so the snippet stays
# copy-pastable from inside the cloned repo.
ATLAS_STORE=memory ATLAS_GRAPH_PATH=$(pwd)/data/graph.json \
    uv run atlas-mcp

# HTTP for the MCP Inspector / remote clients
ATLAS_STORE=memory ATLAS_GRAPH_PATH=$(pwd)/data/graph.json \
    uv run atlas-mcp --transport http --port 8090

# Postgres backend (only when ATLAS_STORE=postgres)
POSTGRES_DSN=$POSTGRES_DSN ATLAS_STORE=postgres ATLAS_GRAPH_PATH=$(pwd)/data/graph.json \
    uv run atlas-mcp
```

Then in your MCP client config. **Use absolute paths** — Claude Desktop / Code launches MCP
servers with an unpredictable cwd, so relative paths like `data/graph.json` break every tool
call with `FileNotFoundError`:

```json
{
  "mcpServers": {
    "codex-atlas": {
      "command": "uv",
      "args": ["run", "atlas-mcp"],
      "env": {
        "ATLAS_STORE": "memory",
        "ATLAS_GRAPH_PATH": "<absolute-path-to-codex-atlas>/data/graph.json",
        "ATLAS_CHUNKS_PATH": "<absolute-path-to-codex-atlas>/data/chunks.json"
      }
    }
  }
}
```

For the Postgres backend, swap the env block:

```json
"env": {
  "ATLAS_STORE": "postgres",
  "POSTGRES_DSN": "postgresql://bench:bench@localhost:5433/bench",
  "ATLAS_GRAPH_PATH": "<absolute-path-to-codex-atlas>/data/graph.json"
}
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ATLAS_STORE` | `memory` | Chunk-store backend: `memory` (reads `ATLAS_CHUNKS_PATH`) or `postgres` (uses `POSTGRES_DSN`). |
| `ATLAS_GRAPH_PATH` | `data/graph.json` | Persisted call graph the server loads at startup. Use an absolute path under MCP. |
| `ATLAS_CHUNKS_PATH` | `data/chunks.json` | Chunk snapshot read when `ATLAS_STORE=memory`. Use an absolute path under MCP. |
| `ATLAS_ENCODER` | `fake` | Encoder name. `fake` = deterministic blake2b (no model download). Any other value loads a `sentence-transformers` model (e.g. `BAAI/bge-small-en-v1.5`). |
| `ATLAS_STEP_TIMEOUT_S` | `10` | Per-step (retriever / synthesiser) timeout in seconds. `0` disables. |
| `ATLAS_RUN_TIMEOUT_S` | `30` | Whole-run timeout in seconds. `0` disables. |
| `POSTGRES_DSN` | _required for postgres_ | DSN for the pgvector backend. |

## Tracing

Codex-Atlas ships a Langfuse tracing adapter that is **env-key-gated**: when
both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set it attempts to
import the `langfuse` SDK and emit spans for every agent step. When either
key is missing, or when the SDK is not installed, every call is a silent
no-op — the rest of the application continues unchanged.

### Enable tracing

```bash
# Install the optional langfuse dependency.
uv sync --extra real

# Set env vars and run any atlas command — spans are emitted automatically.
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
# Optional: self-hosted Langfuse
# export LANGFUSE_HOST=https://my-langfuse.example.com

uv run atlas ask "what does Retriever do" --store=memory
```

### What is traced

| Span | Emitted for |
| --- | --- |
| `codex-atlas-run` | One agent run (top-level trace) |
| `node:classify` / `node:retrieve` / `node:grade` / … | Each state machine step |
| `tool_call:<route>` | Each retriever invocation (query, n_chunks, elapsed_ms) |
| `validation` | `CitationValidator` result (n_claims, n_grounded, is_acceptable) |

### No-op fallback

If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are absent, or if
`langfuse` is not installed, a `NullTracer` is used that matches the same
interface and is guaranteed never to raise. Existing tests (no env vars set)
are unaffected.

### Langfuse account setup

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (free tier available).
2. Create a project and copy the **Public Key** and **Secret Key** from Settings → API Keys.
3. Export the keys before running any `atlas` command.

Self-hosted deployments: set `LANGFUSE_HOST` to your instance URL.

## LLM-as-judge

The eval harness includes an answer-level faithfulness judge (RAGAS-shape) that runs
after every eval question.  The judge is env-key-gated:

| Env var | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | Uses `claude-3-5-haiku-20241022` (preferred) |
| `OPENAI_API_KEY` | Uses `gpt-4o-mini` (fallback) |
| _(neither)_ | `HeuristicJudge` — deterministic, no LLM, CI-safe |

The judge scores each answer on a [0, 1] scale and reports `faithfulness_mean` in REPORT.md.

```bash
# Default (auto): HeuristicJudge when no key set, LLMJudge when key present
uv run atlas eval

# Force heuristic (deterministic, no API call)
uv run atlas eval --judge heuristic

# Force LLM (raises if no key)
ANTHROPIC_API_KEY=sk-... uv run atlas eval --judge llm
```

### Cohen's kappa calibration

Compare the judge against human labels to measure calibration:

```bash
# Edit evals/calibration.csv with real human_score labels, then:
make calibrate
# → writes evals/CALIBRATION.md with kappa + agreement matrix
```

The bundled `evals/calibration.csv` has placeholder labels.  Replace the `human_score`
column with real annotations before trusting the kappa value.

See `evals/INTERPRETATION.md` for the scoring formula, expected ranges, and kappa
interpretation guide.

## Eval results

The 30-question golden set runs `atlas eval` against this repo's own `src/` (a hermetic corpus — no external clone required). Real numbers in [evals/REPORT.md](evals/REPORT.md).

See `evals/REPORT.md` for current metrics — regenerated by `make eval`.

The lookup / hybrid / summarization recall is intentionally measured with `FakeEncoder` (deterministic blake2b — no model download required). With a real `BAAI/bge-small-en-v1.5` encoder, those numbers jump significantly. The harness is wired to take any `Encoder` Protocol implementation, so swapping is one line.

`atlas eval` defaults to `--store=memory` — the harness rebuilds the index from `--corpus` (default `src/`) on the fly, so no Postgres + pgvector container is required to reproduce the baseline. `evals/baseline.json` is regenerated from a `--store=memory` run, which keeps the CI gate hermetic. Pass `--store=postgres` if you want to evaluate against an existing pgvector index instead.

The eval report includes a hand-written **7-failure-mode taxonomy** with mitigation status — that's the section a senior reviewer should read first.

## Real-encoder eval

`evals/REPORT.bge.md` is the companion to `REPORT.md` — the same 30-question golden
set run with `BAAI/bge-small-en-v1.5` instead of `FakeEncoder`. It lives next to
`REPORT.md` and is kept up to date by the `real_encoder_eval` CI workflow.

| File | Encoder | Generated by |
| --- | --- | --- |
| `evals/REPORT.md` | `FakeEncoder` (blake2b) | `make eval` / CI (`ci.yml`) |
| `evals/REPORT.bge.md` | `BAAI/bge-small-en-v1.5` | `real_encoder_eval` workflow |

The `real_encoder_eval` workflow runs on push to `main` and weekly (Monday 04:00 UTC)
on `ubuntu-latest`, where `torch` / `sentence-transformers` Linux wheels are available.
It uploads `evals/REPORT.bge.md` + `evals/scores-bge.json` as a downloadable artifact
(`bge-eval-report`); download and commit to update the committed copy.

**Local refresh** (Linux or Apple-Silicon macOS only — torch not available on Intel macOS):

```bash
uv sync --extra dev --extra embed
uv run atlas index src/ --store=memory --encoder BAAI/bge-small-en-v1.5
ATLAS_ENCODER=BAAI/bge-small-en-v1.5 uv run atlas eval \
    --encoder BAAI/bge-small-en-v1.5 \
    --json-out evals/scores-bge.json \
    --out evals/REPORT.bge.md
```

**Expected lift over FakeEncoder** (lookup recall: ~0 → ~0.65; see `evals/REPORT.bge.md`
and `evals/INTERPRETATION.md` for per-category breakdown).

## RAGAS metrics

The eval harness now reports four [RAGAS](https://docs.ragas.io/)-style metrics alongside the existing custom metrics.  They appear in the `## Headline` table of every `REPORT.md`.

| Metric | Formula (heuristic mode) |
| --- | --- |
| **Faithfulness** | Fraction of gold qnames that appear verbatim in the answer text |
| **Answer relevancy** | Cosine sim between question keyword set and answer keyword set (lexical) |
| **Context precision** | Fraction of retrieved chunks containing at least one gold qname |
| **Context recall** | Fraction of gold qnames appearing in any retrieved chunk |

**Flags:**

```bash
# Default: compute all four RAGAS metrics (heuristic, no LLM needed)
uv run atlas eval

# Skip RAGAS for speed
uv run atlas eval --metrics custom

# RAGAS only (skip existing harness metrics in the CLI output)
uv run atlas eval --metrics ragas
```

**Library mode:** If `ragas>=0.2` is installed (`uv sync --extra real`) *and* an LLM key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) is set, the harness delegates to the real ragas library implementations for faithfulness and answer relevancy.  Without a key or the library, it falls back to the deterministic heuristics above — CI always runs deterministically.

See `evals/INTERPRETATION.md` for expected ranges and what `FakeEncoder` does to each metric.

## LangGraph backend

Codex-Atlas ships two agent backends:

| Backend | Env var value | Dep required | When to use |
| --- | --- | --- | --- |
| Hand-rolled (default) | `hand-rolled` | None | CI, quick-start, offline; zero extra deps |
| LangGraph | `langgraph` | `langgraph>=0.2` | When you want a framework-backed state machine |

### Enable it

```bash
# Install the optional dependency
uv sync --extra real

# Via env var (affects atlas ask + any Python caller)
ATLAS_AGENT_BACKEND=langgraph uv run atlas ask "who calls Retriever.retrieve" --store=memory

# Via CLI flag (single invocation)
uv run atlas ask "who calls Retriever.retrieve" --backend langgraph --store=memory

# Via Python
from codex_atlas import make_agent
agent = make_agent(retriever, backend="langgraph")
result = await agent.run("who calls Retriever.retrieve")
```

### State-graph shape

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

Seven nodes (classify, retrieve, grade, rewrite_query, answer, validate, cancel);
conditional edge from `grade` drives the rewrite loop exactly as the hand-rolled machine does.

### Semantics

Both backends implement the same `async def run(query) -> AgentResult` interface and reuse
the same `Retriever`, `Grader`, `Synthesizer`, and `Validator` Protocol implementations.
Results are equivalent for the same inputs. The hand-rolled backend remains default because:
- No extra transitive dependencies
- Faster cold-start (no `StateGraph.compile()` overhead per run)
- Simpler stack traces when debugging

If `langgraph` is not installed, `LangGraphAgent` raises a `RuntimeError` with an install hint
at construction time — the rest of the application is unaffected.

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| MCP SDK | FastMCP (`>=2.0`, resolves to 3.x today) | Auto-schema from Pydantic; standard SDK |
| Type contracts | Pydantic v2 | What FastMCP introspects for tool schemas |
| Vector DB | pgvector on Postgres 17 | SQL-native; recruiters trust Postgres; HNSW built-in |
| Graph store | NetworkX (in-memory) | A 50K-node single-codebase graph fits easily; sidesteps Neo4j signup |
| Embedder | sentence-transformers (`bge-small`) | Free, fast, top-tier on MTEB; gated extra for Intel macOS |
| Agent loop | Hand-rolled (default) + LangGraph (opt-in) | Hand-rolled: zero deps, fast cold-start; LangGraph: framework-backed, same semantics |
| CLI | Typer + Rich | Standard for new Python in 2026 |
| Tests | pytest + pytest-asyncio | mocked DB / API |

## Layout

```
src/codex_atlas/
  indexer/        AST parser + walker + CallGraph (NetworkX)
  embed.py        Encoder protocol + FakeEncoder (no torch)
  store.py        Async pgvector adapter (asyncpg + pgvector-py)
  retriever.py    Heuristic classifier + 6-route retriever
  agent.py        State machine: classify → retrieve → grade → rewrite → answer
  mcp_server.py   FastMCP wrapping 7 tools + 1 resource
  cli.py          `atlas index | ask | search | explain | mcp | eval`
  eval/           Golden set + harness + render_report

tests/            unit tests (no DB) + integration markers
evals/REPORT.md       Eval report (regenerated on every CI run)
evals/INTERPRETATION.md  How to read the eval metrics + 7-bucket failure taxonomy
docs/ARCHITECTURE.md  Deep dive into indexer / graph / retriever / agent
docs/ADR-001..004     pgvector-vs-faiss / graph-walk-first / state-machine / mcp-shape
```

## Examples

`examples/mcp_client.py` — a runnable end-to-end client that spawns `atlas-mcp` over stdio
and calls three tools (`search_code`, `find_callers`, `get_graph_neighborhood`), then
pretty-prints the structured results.

```bash
# After indexing (see Quick start above):
uv run python examples/mcp_client.py           # rich text output
uv run python examples/mcp_client.py --json    # raw JSON, pipe-friendly
uv run python examples/mcp_client.py --dry-run # fixture data, no server spawn
```

See [`examples/README.md`](examples/README.md) for full details and expected output shape.

## Real LLM synthesizer

The default synthesis layer is `StitchSynthesizer` — a deterministic chunk-concatenation step that requires no API key and works offline. Set `GROQ_API_KEY` to upgrade to `GroqSynthesizer`, which calls [Groq's chat-completions API](https://console.groq.com/docs/api-reference) with `llama-3.3-70b-versatile` and returns an LLM-generated answer with inline `[file:line]` citations.

```bash
export GROQ_API_KEY=gsk_...
uv run atlas ask "who calls find_callers" --store=memory
```

**Gating behaviour:**

| `GROQ_API_KEY` | Synthesizer used |
| --- | --- |
| set | `GroqSynthesizer` — LLM-backed via Groq |
| unset | `StitchSynthesizer` — deterministic chunk-stitch, no API call |

On API errors (auth failure, timeout, 5xx after one retry) `GroqSynthesizer` logs a warning and falls back to `StitchSynthesizer` automatically, so the agent never hard-crashes due to a Groq outage.

The factory is exposed as `codex_atlas.synthesis.make_synthesizer()` and is the default synthesizer wired into `Agent` — passing a custom `synthesizer=` argument to `Agent(...)` still works and overrides the factory.

Get a free API key at [console.groq.com](https://console.groq.com).

## Honest limitations (what to read in evals/REPORT.md)

The "what I did NOT measure" section is in the eval report. The headline:

- v2 is Python-only (tree-sitter for JS/Go/Rust is the next commit).
- Call resolution is imports-aware *when* an import binding is present; falls back to short-name matching otherwise.
- The default grader is heuristic, not LLM-as-judge — promote it for production. The validator flags ungrounded claims regardless.
- No Langfuse wired in yet (the trace events + `tool_calls` log are designed to map cleanly).
- Refusal behaviour for failure-likely / out-of-scope questions is partial — `CitationValidator` rejects ungrounded answers but the agent still attempts a retrieval first.

## Deploy

Codex-Atlas ships deploy-ready artifacts for Fly.io. No account is required to build or test them — the Dockerfile, `fly.toml`, and `scripts/deploy.sh` are validated by `make test` without touching Docker or Fly.

### Quickstart

```bash
# 1. Copy and edit fly.toml — replace REPLACE-ME with your app name.
sed -i 's/codex-atlas-REPLACE-ME/codex-atlas-myname/' fly.toml

# 2. Create the Fly.io app (one-time).
fly launch --no-deploy --copy-config

# 3. Deploy.
./scripts/deploy.sh --app codex-atlas-myname
```

### Required secrets

Set these via `fly secrets set` or by exporting them before running `scripts/deploy.sh`
(the script reads a local `.env` file automatically if present):

| Secret | Required | Purpose |
| --- | --- | --- |
| `ATLAS_PG_DSN` | For pgvector store | `postgresql://user:pass@host/db` |
| `ATLAS_NEO4J_URI` | For Neo4j graph backend | `bolt://...` or `neo4j+s://...` |
| `ATLAS_NEO4J_USERNAME` | For Neo4j backend | default `neo4j` |
| `ATLAS_NEO4J_PASSWORD` | For Neo4j backend | Neo4j password |
| `GROQ_API_KEY` | Optional | Enables `GroqSynthesizer`; falls back to stitch |
| `ANTHROPIC_API_KEY` | Optional | Enables LLM judge; falls back to heuristic |
| `OPENAI_API_KEY` | Optional | Fallback LLM judge if no Anthropic key |
| `LANGFUSE_PUBLIC_KEY` | Optional | Enables Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | Optional | Enables Langfuse tracing |

The default deploy uses `ATLAS_STORE=memory` (no Postgres required). Switch to `pgvector`
by setting `ATLAS_STORE=pgvector` and `ATLAS_PG_DSN` in `fly.toml`'s `[env]` block
and via `fly secrets set`.

### Cost estimate

A `shared-cpu-1x` machine with 1024 MB RAM in `ord` costs approximately **$3–5/mo** with
`auto_stop_machines = "stop"` and `min_machines_running = 1`. The extra RAM (vs. the
512 MB default) is needed because codex-atlas loads the full call graph + chunk store
into process memory at startup; 512 MB OOMs on mid-size repos (>30K chunks).

### Makefile targets

```bash
make docker-build   # Build the image locally (no push)
make docker-run     # Run locally on port 8000
make deploy         # Run scripts/deploy.sh (requires fly CLI + auth)
```

## License

MIT. See [LICENSE](LICENSE).
