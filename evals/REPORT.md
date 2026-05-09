# Codex-Atlas eval report

> Run: against this repo's own `src/` (16-question golden set),
> `FakeEncoder` (deterministic blake2b — no model download),
> **`InMemoryChunkStore` backend** (`atlas eval --store=memory`). The
> harness is reproducible without a Postgres DSN. Latencies reflect
> the agent loop only; the pgvector backend (`--store=postgres`) adds
> 5-50ms per round-trip in production. For real model-quality numbers
> swap in a sentence-transformer encoder. The numbers below are a
> sanity check that the harness + classifier work end-to-end, not a
> model-quality verdict. See [INTERPRETATION.md](INTERPRETATION.md).

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 16 |
| Route correctness | 93.8% |
| Citation recall (mean) | 0.19 |
| Citation precision (mean) | 0.06 |
| p50 latency (ms) | 2.0 |
| p95 latency (ms) | 3.2 |
| p99 latency (ms) | 3.2 |
| Tool calls / q (mean) | 2.00 |
| Cost estimate (USD, total) | 0.2771 |

## By category

| Category | n | Route acc. | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 1 | 100% | 0.00 | 0.00 |
| import-chain | 2 | 100% | 0.00 | 0.00 |
| lookup | 3 | 100% | 0.00 | 0.00 |
| multi-hop | 2 | 50% | 0.00 | 0.00 |
| neighborhood | 2 | 100% | 0.50 | 0.12 |
| out-of-scope | 1 | 100% | 0.00 | 0.00 |
| structural | 3 | 100% | 0.67 | 0.24 |
| summarization | 2 | 100% | 0.00 | 0.00 |

## Failure taxonomy

| Bucket | Count |
| --- | ---: |
| none | 0 |
| missing_node | 0 |
| wrong_route | 1 |
| hallucinated | 3 |
| partial | 0 |
| ungrounded | 2 |
| off_topic | 10 |

## Per question

| qid | route ok | recall | prec | ms | tools | bucket | preview |
| --- | :---: | ---: | ---: | ---: | ---: | --- | --- |
| lookup-1 | yes | 0.00 | 0.00 | 3.2 | 3 | off_topic | # what does Encoder do (in this codebase, with code-level detail)  ## `codex_atl |
| lookup-2 | yes | 0.00 | 0.00 | 3.2 | 3 | off_topic | # explain how chunk_id is constructed (in this codebase, with code-level detail) |
| lookup-3 | yes | 0.00 | 0.00 | 3.0 | 3 | off_topic | # how does the heuristic grader score retrieval (in this codebase, with code-lev |
| structural-1 | yes | 1.00 | 0.33 | 0.6 | 1 | hallucinated | # who calls find_callers  ## `codex_atlas.indexer.graph.CallGraph._traverse` (/U |
| structural-2 | yes | 1.00 | 0.40 | 0.4 | 1 | hallucinated | # who calls _extract_qualified_name  ## `codex_atlas.indexer.graph.CallGraph.has |
| structural-3 | yes | 0.00 | 0.00 | 1.2 | 1 | off_topic | # callers of register_async  ## `codex_atlas.indexer.ast_parser._Collector._hand |
| hybrid-1 | no | 0.00 | 0.00 | 2.9 | 3 | wrong_route | # show me all retriever-related code (in this codebase, with code-level detail)  |
| hybrid-2 | yes | 0.00 | 0.00 | 2.5 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.agent.Agent.run` (/Users/bishara |
| summary-1 | yes | 0.00 | 0.00 | 2.1 | 1 | off_topic | # walk me through the agent loop  ## `codex_atlas.mcp_server._agent` (/Users/bis |
| summary-2 | yes | 0.00 | 0.00 | 1.9 | 1 | off_topic | # overview of the indexer module  ## `codex_atlas.mcp_server._graph` (/Users/bis |
| neighborhood-1 | yes | 1.00 | 0.25 | 0.3 | 1 | hallucinated | # neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers  ## `codex_at |
| neighborhood-2 | yes | 0.00 | 0.00 | 0.3 | 1 | off_topic | # everything around codex_atlas.retriever.Retriever.retrieve  ## `codex_atlas.ag |
| import-chain-1 | yes | 0.00 | 0.00 | 0.2 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| import-chain-2 | yes | 0.00 | 0.00 | 0.2 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-1 | yes | 0.00 | 0.00 | 3.1 | 3 | off_topic | # what does frobnicate_widget do (in this codebase, with code-level detail)  ##  |
| oos-1 | yes | 0.00 | 0.00 | 3.2 | 3 | off_topic | # how do I deploy this to AWS Lambda (in this codebase, with code-level detail)  |

## Reading the numbers honestly

- **Route correctness 93.8%** — 15/16 questions hit the right pipeline.
  The miss is `hybrid-1` ("show me all retriever-related code") —
  phrasing didn't match any hybrid trigger pattern. The fix is one
  regex; documenting rather than rationalising.

- **Structural recall 0.67** — graph traversal still does real work on
  this corpus. `find_callers` / `_extract_qualified_name` resolve
  correctly; over-citation drops precision to 0.24-0.40.
  Imports-aware resolution (with v2 relative-import resolution) keeps
  short-name fallback noise contained for common names.

- **Lookup / hybrid / summarization recall low** — *expected with
  `FakeEncoder`*. blake2b makes semantic similarity essentially
  random. Real numbers require `BAAI/bge-small-en-v1.5`
  (`make install` includes it on supported platforms). Structural,
  neighborhood, and import-chain routes don't need the encoder —
  that's why they score while the others don't.

- **`ungrounded` x 2 on import_chain** — the import-chain golden
  questions ask for `codex_atlas.store` and `codex_atlas.indexer.graph`
  modules; the chunk store only has function/method bodies, not
  module-level entries. Expected behaviour: the agent surfaces no
  chunks and the validator (now enforcing) lets the empty-result
  message stand. Future work is to index module-level chunks too.

- **Refusal questions** — the harness wants the agent to return *no*
  citations for a refusal question. With the validator now enforcing
  `validation_mode=redact` (default), ungrounded backticked claims in
  refusal answers are rewritten to `[ungrounded: ...]` markers; the
  citation set is still over-large because the synthesiser includes
  whatever vector top-k surfaced. Production turns the validator up
  to `reject` for refusal-shaped queries.

## 7-mode failure taxonomy

| # | Failure | Detection | Mitigation | Status |
| --- | --- | --- | --- | --- |
| 1 | Hallucinated citations | Validate that every cited qname exists in the call graph | `CitationValidator` flags ungrounded claims; `Agent` enforces them via `validation_mode=redact` (default), `reject`, or `advisory` | mitigated |
| 2 | Wrong route chosen | `route_correctness` per-category in this report | Add the missing trigger phrase, or promote classifier to LLM | partial — explicit miss in `hybrid-1` |
| 3 | Stale embeddings post-refactor | Index commit-sha as metadata; refuse if drift > N | TODO; v2 still proceeds without checking | known |
| 4 | Re-query loop never converges | Hard cap at `max_attempts` + step/run timeouts | Implemented (`AgentConfig.max_attempts=3`, `step_timeout_s`, `run_timeout_s`) | mitigated |
| 5 | Token budget blowout on summarization | Pre-compute module-level summaries; cap chunks at synthesis | Partial — `StitchSynthesizer.max_chunks` is hard-bounded | partial |
| 6 | Cross-language mis-embeds | Language-aware chunking + tagged embeddings | TODO; v2 is Python-only | known |
| 7 | Gold qname missing from index | `MISSING_NODE` bucket detected from `indexed_qnames` | Implemented in `score_result` / `run_eval` | mitigated |
