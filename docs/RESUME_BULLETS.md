# Codex-Atlas — resume bullets

## Defensible (use these)

- **Designed a 7-node GraphRAG agent state machine** (classify → retrieve → grade → rewrite → answer → validate → cancel) with a Protocol-driven plugin architecture supporting 4 vector store backends (in-memory, pgvector, Neo4j, JSON), verified by 575 tests collected under mypy --strict and ruff clean.

  *Evidence:* `src/codex_atlas/agent.py:53–60` defines the `Node` StrEnum with all 7 nodes; `pytest --collect-only -q` returns "575 tests collected"; `docs/ADR-003-agent-state-machine-vs-react.md` documents the deliberate choice over LangGraph.

- **Benchmarked two encoder configurations on a 30-question golden eval**: FakeEncoder (deterministic blake2b) vs. BAAI/bge-small-en-v1.5, showing citation recall 0.13 → 0.28 (+0.15) and faithfulness 0.38 → 0.64 (+0.26) when switching to real semantic vectors, with route correctness stable at 93.3% (28/30 questions).

  *Evidence:* `evals/REPORT.md` (FakeEncoder baseline) and `evals/REPORT.bge.md` (bge-small results, both headline tables and per-category breakdown, run 2026-05-06 via Docker on Intel macOS).

- **Implemented a dual-layer SQL safety classifier** for the MCP postgres-dba server: sqlparse token-walk (Layer 1) rejects multi-statement batches, CTEs with inner mutations, EXPLAIN ANALYZE, and 53 side-effecting functions; asyncpg `readonly=True` transaction (Layer 2) catches any parser miss — verified by 91 safety-classifier unit tests.

  *Evidence:* `src/codex_atlas/mcp_server.py` + `src/codex_atlas/store.py`; the 91-test classifier count is from `tests/test_store.py` suite; `docs/ADR-001-pgvector-vs-faiss.md` and `docs/ADR-004-mcp-server-shape.md` document the design rationale.

- **p50/p95/p99 latencies on the 30-question golden set with bge-small**: 184.9 ms / 563.1 ms / 798.7 ms (CPU-only Docker inference on Intel macOS); FakeEncoder baseline is 2.6 ms / 4.1 ms / 5.5 ms (pure in-process hash, no network).

  *Evidence:* `evals/REPORT.bge.md` headline table, latency rows; `evals/REPORT.md` for FakeEncoder baseline.

---

## Stretch / claim with caveat (use cautiously)

- **"MCP server exposing 5 tools over stdio/HTTP-SSE transport"** — the server exists (`src/codex_atlas/mcp_server.py`) and `tests/test_mcp_server.py` tests it. What is not defensible: claiming a public live MCP endpoint — there is no deployed URL, only a `fly.toml` config.

  *Pushback:* "Where's the live endpoint?" — honest answer: "The `fly.toml` is present; it wasn't deployed as part of this project."

- **"93.3% route correctness on 30 questions"** — true, but the golden set is 30 hand-curated questions self-written for this repo. A skeptic will note it's not an independent benchmark.

  *Pushback:* "Who wrote the golden set?" — you did, for this project. Defensible as "internal eval harness"; not defensible as "validated against an external benchmark."

- **"Langfuse observability integration"** — `src/codex_atlas/observability/langfuse.py` exists with `LangfuseTracer` and `NullTracer`; tests exercise it. Not defensible: claiming a live Langfuse dashboard or traces visible to a reviewer.

---

## DO NOT claim

- **"Sub-500ms p95 latency"** — the bge-small p95 is 563.1 ms (CPU-only Docker). The FakeEncoder p95 is 4.1 ms but measures hash retrieval, not a real encoder pipeline. Neither number supports a "sub-500ms p95" claim.

  *Alternate defensible version:* "p50 latency of 185 ms on a 30-question eval with bge-small-en-v1.5 (CPU-only Docker); p95 563 ms."

- **"RAGAS evaluation scores"** — the RAGAS metrics (faithfulness 0.38, answer relevancy 0.14, context precision 0.35, context recall 0.35) in `REPORT.md` are computed with FakeEncoder (semantically meaningless vectors) and documented as such in `evals/INTERPRETATION.md`. They are not a meaningful RAGAS result.

  *Alternate defensible version:* "Built a RAGAS-compatible harness computing 4 retrieval metrics; FakeEncoder baseline shows the metric pipeline is wired; real-encoder numbers tracked in REPORT.bge.md."

- **"LLM judge with Cohen's kappa calibration"** — `evals/CALIBRATION.md` exists but `evals/calibration.csv` has stub human labels (all 0/1 placeholders). The kappa number is not from real human annotations.

  *Alternate:* "Designed a pluggable judge interface (heuristic + LLM) with a kappa calibration harness; stub labels in place, real human annotation is the next step."

- **"Deployed to Fly.io"** — `fly.toml` exists but no live URL is verifiable.

---

## How to defend each bullet in an interview

**Bullet 1 — 7-node state machine, 575 tests:**
> "The agent loop is in `src/codex_atlas/agent.py`. The `Node` StrEnum at line 53 enumerates classify, retrieve, grade, rewrite_query, answer, validate, cancel — seven states. The loop is hand-rolled rather than LangGraph (ADR-003 explains why: the graph is static, the only conditional is `grade < threshold AND attempts < max`, and adding LangGraph would import a graph compiler to run a while loop). I can run `pytest --collect-only -q` live — you'll see 575 collected."

**Bullet 2 — encoder comparison, recall 0.13 → 0.28:**
> "Both REPORT.md (FakeEncoder) and REPORT.bge.md (bge-small) are committed. The headline table shows citation recall 0.13 vs 0.28. The per-category breakdown shows where the lift comes from: lookup went 0.00 → 0.33, structural 0.60 → 0.80, summarization 0.00 → 0.38. Multi-hop stayed 0.00 because the route classifier misfired on 2/4 questions regardless of encoder — that's a routing bug, not a retrieval bug."

**Bullet 3 — dual-layer safety, 91 tests:**
> "Layer 1 is `is_read_only_sql` in `safety.py`: sqlparse token-walk, rejects 53 side-effecting functions by name. Layer 2 is the asyncpg `readonly=True` transaction — if the parser misses something, Postgres itself rejects the write. I have 91 classifier unit tests in `test_safety.py` plus an integration test `test_classifier_miss_blocked_by_db_readonly` that explicitly exercises the Layer 2 fallback."

**Bullet 4 — bge-small latencies:**
> "The numbers are in REPORT.bge.md, run 2026-05-06 via Docker on an Intel MacBook (CPU-only inference). p50 is 185 ms, p95 is 563 ms, p99 is 799 ms. The FakeEncoder baseline in REPORT.md is 2.6/4.1/5.5 ms — those measure in-process hash retrieval with no model inference. The ~50x latency increase is bge-small doing inference on CPU."
