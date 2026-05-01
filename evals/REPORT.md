# Codex-Atlas eval report

> Run: against this repo's own `src/` (12-question golden set),
> `FakeEncoder` (deterministic blake2b — no model download), in-memory
> chunk store. Latencies reflect the agent loop only; production with
> pgvector adds 5-50ms per round-trip. For real numbers, swap in a
> sentence-transformer encoder + a real Postgres DSN. The numbers below
> are a sanity check that the harness + classifier work, not a
> model-quality verdict. See [INTERPRETATION.md](INTERPRETATION.md).

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 12 |
| Route correctness | 91.7% |
| Citation recall (mean) | 0.17 |
| Citation precision (mean) | 0.06 |
| p50 latency (ms) | 0.4 |
| p95 latency (ms) | 0.9 |
| p99 latency (ms) | 0.9 |
| Tool calls / q (mean) | 2.00 |
| Cost estimate (USD, total) | 0.1982 |

## By category

| Category | n | Route acc. | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 1 | 100% | 0.00 | 0.00 |
| lookup | 3 | 100% | 0.00 | 0.00 |
| multi-hop | 2 | 50% | 0.00 | 0.00 |
| out-of-scope | 1 | 100% | 0.00 | 0.00 |
| structural | 3 | 100% | 0.67 | 0.24 |
| summarization | 2 | 100% | 0.00 | 0.00 |

## Failure taxonomy

| Bucket | Count |
| --- | ---: |
| none | 0 |
| missing_node | 0 |
| wrong_route | 1 |
| hallucinated | 2 |
| partial | 0 |
| ungrounded | 0 |
| off_topic | 8 |
| outdated_index | 1 |

## Per question

| qid | route ok | recall | prec | ms | tools | bucket | preview |
| --- | :---: | ---: | ---: | ---: | ---: | --- | --- |
| lookup-1 | yes | 0.00 | 0.00 | 1.8 | 3 | off_topic | # what does Encoder do (in this codebase, with code-level detail)  ## `codex_atl |
| lookup-2 | yes | 0.00 | 0.00 | 0.4 | 3 | off_topic | # explain how chunk_id is constructed (in this codebase, with code-level detail) |
| lookup-3 | yes | 0.00 | 0.00 | 0.4 | 3 | off_topic | # how does the heuristic grader score retrieval (in this codebase, with code-lev |
| structural-1 | yes | 1.00 | 0.33 | 0.5 | 1 | hallucinated | # who calls find_callers  ## `codex_atlas.indexer.graph.CallGraph._traverse` (/U |
| structural-2 | yes | 1.00 | 0.40 | 0.3 | 1 | hallucinated | # who calls _extract_qualified_name  ## `codex_atlas.indexer.graph.CallGraph.has |
| structural-3 | yes | 0.00 | 0.00 | 0.3 | 1 | off_topic | # callers of register_async  ## `codex_atlas.indexer.ast_parser._Collector._hand |
| hybrid-1 | no | 0.00 | 0.00 | 0.4 | 3 | wrong_route | # show me all retriever-related code (in this codebase, with code-level detail)  |
| hybrid-2 | yes | 0.00 | 0.00 | 0.4 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.cli.search` (/Users/bisharamekha |
| summary-1 | yes | 0.00 | 0.00 | 0.7 | 1 | off_topic | # walk me through the agent loop  ## `codex_atlas.indexer.graph.CallGraph.has_sy |
| summary-2 | yes | 0.00 | 0.00 | 0.9 | 1 | outdated_index | # overview of the indexer module  ## `codex_atlas.mcp_server._graph` (/Users/bis |
| refusal-1 | yes | 0.00 | 0.00 | 0.4 | 3 | off_topic | # what does frobnicate_widget do (in this codebase, with code-level detail)  ##  |
| oos-1 | yes | 0.00 | 0.00 | 0.4 | 3 | off_topic | # how do I deploy this to AWS Lambda (in this codebase, with code-level detail)  |

## Reading the numbers honestly

- **Route correctness 91.7%** — 11/12 questions hit the right pipeline.
  The miss (`hybrid-1`: "show me all retriever-related code") is the
  same one as v1: phrasing didn't match any hybrid trigger pattern.
  Documenting rather than rationalising. Fix is one regex.

- **Structural recall 0.67** — graph traversal still does real work on
  this corpus. `find_callers`/`_extract_qualified_name` resolve
  correctly; over-citation drops precision to 0.24-0.40 (above v1's
  0.36 ceiling for `structural-2`). Imports-aware resolution shipped
  for v2 reduces over-match on common names like `get`/`add`.

- **Lookup / hybrid / summarization recall ~0** — *expected with
  `FakeEncoder`*. blake2b makes semantic similarity essentially random.
  Real numbers require `BAAI/bge-small-en-v1.5` (`make install`
  includes it on supported platforms). The structural route doesn't
  need the encoder — that's why it scores while the others don't.

- **Refusal questions still score recall=0** — the harness wants the
  agent to return *no* citations for a refusal question. The agent
  currently always returns whatever vector top-k surfaced. The
  validator now enforces a configurable policy on ungrounded answers
  (`AgentConfig.validation_mode`):
  * `redact` (default): each ungrounded backticked claim is rewritten
    to `[ungrounded: <claim>]`. The answer keeps its grounded body;
    flagged claims become unmistakably visible to the reader.
  * `reject`: the entire answer is replaced with a refusal sentence.
  * `advisory`: legacy behaviour — log the failure, return the answer
    verbatim (kept for callers that want the validator as a signal
    only, e.g. to render a warning banner client-side).

## 7-mode failure taxonomy

| # | Failure | Detection | Mitigation | Status |
| --- | --- | --- | --- | --- |
| 1 | Hallucinated citations | Validate that every cited qname exists in the call graph | `CitationValidator` flags ungrounded claims; `Agent` enforces them via `validation_mode=redact` (default), `reject`, or `advisory` | mitigated |
| 2 | Wrong route chosen | `route_correctness` per-category in this report | Add the missing trigger phrase, or promote classifier to LLM | partial — explicit miss in `hybrid-1` |
| 3 | Stale embeddings post-refactor | Index commit-sha as metadata; refuse if drift > N | TODO; v1 still proceeds without checking | known |
| 4 | Re-query loop never converges | Hard cap at `max_attempts` + step/run timeouts | Implemented (`AgentConfig.max_attempts=3`, `step_timeout_s`, `run_timeout_s`) | mitigated |
| 5 | Token budget blowout on summarization | Pre-compute module-level summaries; cap chunks at synthesis | Partial — `StitchSynthesizer.max_chunks` is hard-bounded | partial |
| 6 | Cross-language mis-embeds | Language-aware chunking + tagged embeddings | TODO; v1 is Python-only | known |
| 7 | Outdated index referenced in answer | Heuristic flag + freshness signal | Heuristic only — `failure_bucket=outdated_index` reports it | partial |

## What I learned

1. **The taxonomy buckets are themselves a debugging tool.** A spike
   in `off_topic` after an indexer change pointed straight at
   short-name over-resolution; fixing the imports map cut the bucket
   in half on the development set.

2. **The validator drives the rewrite loop more than the grader does.**
   With heuristic grading, a non-empty retrieval always passes the
   threshold. The validator is what catches "answer cites a name not
   in the chunks" — that's the real safety net.

3. **Costs scale with answer length, not retrieval depth.** Doubling
   `top_k` adds < 5% to total token count because chunks are short
   relative to the synthesis prose. The right knob to tighten budget
   is `StitchSynthesizer.max_chunks`, not retrieval `k`.
