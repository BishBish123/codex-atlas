# Codex-Atlas — 5-minute recruiter walkthrough

> See also: `docs/CLI.md` for the full CLI reference, `evals/REPORT.md` for
> the latest eval numbers, and `docs/ADR-*.md` for design rationale.

---

## What this is

Codex-Atlas is an agentic GraphRAG system that answers questions about a Python
codebase. You point it at a source tree; it parses every `.py` file with the
stdlib `ast` module, embeds each function-grain chunk, builds a call graph
(NetworkX or Neo4j), and exposes a six-route adaptive retriever. When a question
arrives it classifies the intent (`lookup`, `structural`, `hybrid`,
`summarization`, `neighborhood`, `import_chain`), retrieves with the matching
strategy, grades the result, rewrites the query and retries if the grade is
weak, then synthesises an answer with source citations. The whole pipeline is
wrapped as a FastMCP server with 7 tools + 1 resource so Claude Code / Cursor
can call it like any other MCP tool.

---

## 5-minute demo

All commands run without Docker or a database — they use the in-memory store
(`data/chunks.json` + `data/graph.json`).

**Prerequisites:** Python 3.11+, `uv` installed.

```bash
git clone https://github.com/BishBish123/codex-atlas.git
cd codex-atlas
uv sync --extra dev       # ~30 s first time; torch/sentence-transformers gated
```

1. **Index the bundled source tree.**

   ```bash
   uv run atlas index src/ --store=memory
   ```

   Watch the Walker stat line: file count, chunk count, call-edge count, and
   the graph serialised at `data/graph.json`. No database needed — the chunk
   snapshot goes to `data/chunks.json`.

2. **Ask a structural question (caller-graph route).**

   ```bash
   uv run atlas ask "who calls find_callers" --store=memory
   ```

   Notice the `route=structural` label in the trace. The agent walks the call
   graph directly instead of doing a vector search. The answer cites
   `codex_atlas.retriever.Retriever.retrieve` with a backtick-fenced qualified
   name — that citation is validated by `CitationValidator` against the
   retrieved chunks before the response is accepted.

3. **Raw retrieval (no synthesis) to see route selection.**

   ```bash
   uv run atlas search "embedding encode" --store=memory
   ```

   The router picks `route=lookup` (pure vector). Try swapping in
   "who imports retriever" to trigger `route=import_chain`.

4. **Explain a symbol (forced structural route).**

   ```bash
   uv run atlas explain "codex_atlas.retriever.Retriever.retrieve" --store=memory
   ```

   The `explain` subcommand pins `route_override=structural` so the classifier
   never re-routes a bare qualified name to the vector-only `lookup` path. The
   answer header reads `# codex_atlas.retriever.Retriever.retrieve` — no
   "who calls" rewrite noise.

5. **Run the 30-question golden eval.**

   ```bash
   uv run atlas eval --store=memory
   cat evals/REPORT.md
   ```

   The punchline: **93.3% route correctness** across six route categories, all
   without a live LLM (the heuristic judge runs deterministically). The report
   also shows RAGAS-style faithfulness / answer-relevancy scores and a
   per-category breakdown. Notice the `structural` category — structural routes
   hit 100% correctness because the classifier's heuristic covers the explicit
   phrase patterns reliably.

6. **Play the pre-recorded terminal cast.**

   ```bash
   asciinema play assets/demo.cast   # requires asciinema installed
   # or:
   bash scripts/play_demo.sh
   ```

   The cast captures real stdout from the same six commands above — no
   post-processing.

---

## What to evaluate

Each check takes under 30 seconds.

1. **Test count** — `uv run pytest -q -m "not integration"` should report
   550+ tests, all green, with zero DB required. The suite covers: indexer,
   retriever, agent, MCP server, CLI, eval harness, LangGraph backend (mocked),
   Neo4j backend (mocked), pgvector store (mocked), RAGAS wrapper, and
   observability adapters.

2. **Eval report** — open `evals/REPORT.md`. Verify 30 questions, route
   correctness ≥ 93%, and that the per-category table includes all six route
   types. The `evals/baseline.json` file is the regression gate — `make eval`
   exits non-zero if any aggregate metric drops more than 5%.

3. **Honest disclosure** — read `evals/INTERPRETATION.md`. Citation
   recall/precision are intentionally low because the heuristic judge only
   checks for exact qualified-name matches; the file explains what this means
   and what would improve it (LLM judge, fuzzy matching). The project does not
   paper over the limitation.

4. **ADR trail** — open `docs/ADR-001-pgvector-vs-faiss.md` through
   `ADR-004-mcp-server-shape.md`. Each records the option considered, the
   tradeoff that decided it, and the consequence. The LangGraph backend swap
   (`ADR-003`) is live code — `--backend=langgraph` is a real flag and its
   tests mock the graph without hitting a network.

5. **CLI surface** — `uv run atlas --help` lists seven subcommands including
   `calibrate` (Cohen's kappa between the judge and human labels). Run
   `uv run atlas calibrate --help` to see it is a real, documented command, not
   scaffolding.

---

## Where to look

| Artifact | Purpose |
| --- | --- |
| `docs/ADR-001` – `ADR-004` | Design decisions with explicit tradeoffs |
| `docs/ARCHITECTURE.md` | Five-layer diagram, data-flow narrative |
| `evals/REPORT.md` | Latest eval numbers (30q, six route types) |
| `evals/INTERPRETATION.md` | Honest reading guide — what each metric means |
| `evals/CALIBRATION.md` | Cohen's kappa calibration report |
| `assets/demo.cast` | Pre-recorded asciinema terminal session |
| `docs/CLI.md` | Auto-generated full CLI reference |
| `src/codex_atlas/` | Source: `agent/`, `retriever/`, `indexer/`, `eval/`, `mcp_server.py` |
| `tests/` | 550+ unit tests, zero live-service dependencies |

**Playback:** `asciinema play assets/demo.cast`
**Embed link:** upload `assets/demo.cast` to asciinema.org and use the
generated `<img>` embed.
