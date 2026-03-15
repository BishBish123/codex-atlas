# Codex-Atlas

> Agentic GraphRAG over a Python codebase. Adaptive vector + graph hybrid retrieval with reflection + bounded re-query, exposed as a **Model Context Protocol** server.

[![ci](https://github.com/BishBish123/codex-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/codex-atlas/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![mcp](https://img.shields.io/badge/mcp-spec--conformant-success)]()
[![eval](https://img.shields.io/badge/route%20accuracy-93.8%25-brightgreen)](evals/REPORT.md)

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

## Run as an MCP server

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

## Eval results

The 16-question golden set runs `atlas eval` against this repo's own `src/` (a hermetic corpus — no external clone required). Real numbers in [evals/REPORT.md](evals/REPORT.md).

| Metric | Value |
| --- | ---: |
| Route correctness | 93.8% |
| Citation recall (mean, overall) | 0.22 |
| Citation recall (mean, structural) | 0.67 |
| p50 latency (ms) | 7.3 |

The lookup / hybrid / summarization recall is intentionally measured with `FakeEncoder` (deterministic blake2b — no model download required). With a real `BAAI/bge-small-en-v1.5` encoder, those numbers jump significantly. The harness is wired to take any `Encoder` Protocol implementation, so swapping is one line.

`atlas eval` defaults to `--store=memory` — the harness rebuilds the index from `--corpus` (default `src/`) on the fly, so no Postgres + pgvector container is required to reproduce the baseline. `evals/baseline.json` is regenerated from a `--store=memory` run, which keeps the CI gate hermetic. Pass `--store=postgres` if you want to evaluate against an existing pgvector index instead.

The eval report includes a hand-written **7-failure-mode taxonomy** with mitigation status — that's the section a senior reviewer should read first.

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| MCP SDK | FastMCP (`>=2.0`, resolves to 3.x today) | Auto-schema from Pydantic; standard SDK |
| Type contracts | Pydantic v2 | What FastMCP introspects for tool schemas |
| Vector DB | pgvector on Postgres 17 | SQL-native; recruiters trust Postgres; HNSW built-in |
| Graph store | NetworkX (in-memory) | A 50K-node single-codebase graph fits easily; sidesteps Neo4j signup |
| Embedder | sentence-transformers (`bge-small`) | Free, fast, top-tier on MTEB; gated extra for Intel macOS |
| Agent loop | Hand-rolled async state machine | LangGraph-style nodes without the dep; trace surface is identical |
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

## Honest limitations (what to read in evals/REPORT.md)

The "what I did NOT measure" section is in the eval report. The headline:

- v2 is Python-only (tree-sitter for JS/Go/Rust is the next commit).
- Call resolution is imports-aware *when* an import binding is present; falls back to short-name matching otherwise.
- The default grader is heuristic, not LLM-as-judge — promote it for production. The validator flags ungrounded claims regardless.
- No Langfuse wired in yet (the trace events + `tool_calls` log are designed to map cleanly).
- Refusal behaviour for failure-likely / out-of-scope questions is partial — `CitationValidator` rejects ungrounded answers but the agent still attempts a retrieval first.

## License

MIT. See [LICENSE](LICENSE).
