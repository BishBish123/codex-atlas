# Troubleshooting

Real ops gotchas surfaced during development. Each entry: what you'll see, what's actually happening, how to fix it.

## MCP server startup

### Symptom: Every MCP tool call returns `FileNotFoundError` immediately after `claude mcp add`
**Cause:** `atlas-mcp` is launched by the MCP client with an unpredictable working directory; `data/graph.json` resolves relative to that cwd, not the repo root.
**Fix:** Always pass absolute paths in the `claude mcp add` command: `-e ATLAS_GRAPH_PATH=<abs-path>/data/graph.json -e ATLAS_CHUNKS_PATH=<abs-path>/data/chunks.json`.
**Source:** README "Run as an MCP server" section; ADR-004

### Symptom: `atlas-mcp` crashes on startup with `KeyError` or `json.JSONDecodeError`
**Cause:** `data/graph.json` does not exist — `atlas index` has not been run yet (or ran against a different directory).
**Fix:** `uv run atlas index src/ --store=memory` from the repo root before starting the MCP server. The README blockquote calls this out as a hard prerequisite.
**Source:** README prerequisite blockquote; loop_history.md V2

## Platform / encoder

### Symptom: `make install` skips `torch` and `sentence-transformers` silently on Intel macOS
**Cause:** `pyproject.toml` gates the ML extras behind `platform_machine == 'arm64'`; Intel macOS has no torch wheels.
**Fix:** This is expected. Run `make eval` with the default `FakeEncoder` (deterministic blake2b — no model). For real-encoder eval on Intel Mac, use the Docker path: `python:3.12-slim` Linux container where wheels exist.
**Source:** README Quick start Intel macOS note; loop_history.md V2

### Symptom: `uv run atlas eval --encoder BAAI/bge-small-en-v1.5` on Intel macOS raises `torch` import error
**Cause:** No torch wheel for macOS x86_64 as of 2026. The `[embed]` extras group only installs on arm64.
**Fix:** Run inside a Linux container: `docker run --rm -v $(pwd):/app -w /app python:3.12-slim bash -c "pip install uv && uv sync --extra embed && uv run atlas eval --encoder BAAI/bge-small-en-v1.5"`.
**Source:** loop_history.md V2 (real-encoder eval via Docker)

## Dataclass field-order bug (Python 3.12, Linux)

### Symptom: `TypeError: non-default argument 'embedding' follows default argument` inside `embed.py` when running with a real encoder on Linux/Python 3.12
**Cause:** The `_ST` inner dataclass in `embed.py` had a field-order error inherited from the `Encoder` Protocol via structural typing. The fix was to move the non-default field before default fields. This was caught during V2 Docker verification and is patched.
**Fix:** Pull the latest code — the fix is committed. If you see it on an older checkout, check `embed.py` dataclass field ordering.
**Source:** loop_history.md V2 (`_ST` inner dataclass field-order error surfaced + fixed)

## Graph store / backend selection

### Symptom: `ATLAS_GRAPH_BACKEND=neo4j` but queries fail with `ServiceUnavailable`
**Cause:** `ATLAS_NEO4J_URI` / `ATLAS_NEO4J_PASSWORD` not set, or Neo4j container is not running.
**Fix:** `make neo4j-up` first, then set `ATLAS_NEO4J_URI=bolt://localhost:7687` and `ATLAS_NEO4J_PASSWORD=neo4j-dev`.
**Source:** README Neo4j graph backend section

### Symptom: Not sure which `ATLAS_GRAPH_BACKEND` to pick
**Cause:** Two backends exist: `networkx` (default, in-memory, JSON snapshot) and `neo4j` (Bolt driver, scales to millions of nodes).
**Fix:** Use `networkx` for local dev and CI (zero external deps). Switch to `neo4j` only for multi-user deployments or repos with >50K symbols.
**Source:** README Neo4j vs NetworkX table

## Eval / scoring

### Symptom: `make eval` formerly ran `atlas eval --help` instead of the harness
**Cause:** Historical bug in the Makefile — the target called `--help` accidentally. Fixed in Round 1.
**Fix:** Pull latest; `make eval` now runs `uv run atlas eval --json-out evals/scores.json`.
**Source:** loop_history.md Round 1 ("make eval now runs harness (was --help)")

### Symptom: `evals/REPORT.md` shows `Faithfulness (mean) = 0.00` with `--judge llm` despite having an API key
**Cause:** The `--judge llm` flag raises `OSError` / `ImportError` (network unreachable or SDK missing), and the old CLI caught only `IncompleteRunError`. Fixed in Round 8.
**Fix:** Pull latest — `run` command now catches `(OSError, ImportError)` from `make_judge()` with a friendly message. Verify the key is exported: `echo $ANTHROPIC_API_KEY`.
**Source:** loop_history.md Round 7 ("`--judge llm` no-key dumps full traceback") + Round 8 fix

## pgvector chunk store

### Symptom: `uv run atlas index --store=postgres` raises `asyncpg` import error
**Cause:** `asyncpg` is not in the base deps; it's in the `[real]` extras group.
**Fix:** `uv sync --extra real`.
**Source:** README pgvector chunk store section
