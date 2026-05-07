# Eval — what each metric means and what good/bad looks like

Codex-Atlas's eval harness reports eight metrics + a 7-bucket failure
taxonomy. This doc is a reading guide so reviewers can interpret a
report without re-reading source.

## Metrics

### `route_correctness`

Fraction of questions where the classifier picked the route the gold
set expects.

| reading | what it means |
| --- | --- |
| ≥ 95% | classifier is well-calibrated for this corpus |
| 85–95% | a few phrasings missing from the trigger set |
| < 85% | the heuristic classifier is not enough; promote to LLM |

A single missed route is not a bug — the gold set is intentionally
small (30 questions) so noise is real. Watch the trend across runs,
not the absolute number.

### `citation_recall` (mean)

Of the gold qualified names for each question, how many appeared in
the agent's citations? Averaged over questions.

| reading | what it means |
| --- | --- |
| ≥ 0.7 | retriever is finding the right code |
| 0.4–0.7 | structural recall is fine; lookup recall depends on encoder |
| < 0.4 | check the encoder — `FakeEncoder` floors lookup recall |

Refusal questions (no gold) score recall = 1.0 only when the agent
returns *no* citations.

### `citation_precision` (mean)

Of the qualified names the agent cited, how many match a gold name
(or its suffix)? Higher precision means fewer over-citations.

| reading | what it means |
| --- | --- |
| ≥ 0.5 | citations are tight |
| 0.3–0.5 | structural over-matches on common names (`get`, `add`) |
| < 0.3 | recall is being bought with citation noise |

Precision and recall fight: a wider `top_k` helps recall but hurts
precision. The eval doesn't combine them — that's the reviewer's job.

### Latencies (`p50`, `p95`, `p99` in ms)

Per-question wall clock for the full agent loop (classify → retrieve →
grade → answer → validate). We report three percentiles so the long
tail is visible.

| reading | what it means |
| --- | --- |
| p99 < 500ms | retriever is fast even with retries |
| p99 1–2s | the rewrite loop is firing on hard questions |
| p99 > 2s | check pgvector index health (HNSW build) |

`p95` and `p99` differ when one or two questions are pathological
(usually the rewrite loop maxing out). That's expected; if every
question hits `max_attempts`, the grader is too strict.

**Percentile method.** We use `numpy.percentile` with default linear
interpolation for samples of `n >= 10`. For small samples (`n < 10`)
linear interpolation is misleading — the values bounce on every added
question. In that regime `p95` and `p99` are reported as the sample
*max*: a conservative upper bound that doesn't pretend to more
precision than the data supports. `p50` is always the linear-
interpolation median.

### `tool_call_count` (mean)

Average number of retriever invocations per question. Each retry
counts as a separate tool call, and the rewrite loop runs at most
`max_attempts` retrievals.

| reading | what it means |
| --- | --- |
| 1.0–1.2 | retriever is hitting on the first try |
| 1.5–2.0 | retries are firing on ~half the questions |
| ≥ 2.5 | grader threshold is too high |

### `cost_estimate_usd` (total)

Token count × per-1K price, summed across the run. The harness uses
characters / 4 as a token approximation — this *under*estimates Latin
text and *over*estimates dense code. Treat the number as a
back-of-the-envelope budget figure, not a billing line item.

The default prices (`0.003 / 0.015 per 1K input/output`) approximate a
mid-tier model. Override via `--price-input` / `--price-output` when
benchmarking against a different provider.

## Failure taxonomy

Every wrong answer lands in exactly one of seven buckets. The bucket
is mutually exclusive; we report the first matching one so a single
failure isn't double-counted.

| bucket | trigger |
| --- | --- |
| `none` | answer was correct |
| `missing_node` | gold qname not in the graph at all |
| `wrong_route` | classifier picked the wrong pipeline |
| `hallucinated` | cited a qname that's not in the retrieved chunks |
| `partial` | cited some gold names, missed others |
| `ungrounded` | cited nothing for a question that has gold |
| `off_topic` | cited names that match no gold |
| `outdated_index` | answer mentions stale state heuristically |

`outdated_index` is detected from the answer text (mentions of "stale"
or "outdated"). Production swaps this for a real freshness signal
(commit-sha drift on the chunks). Until then it stays a heuristic
bucket — treat it as suggestive, not authoritative.

## Reading a report end-to-end

1. **Headline numbers first.** If `route_correctness` is < 90% the
   classifier is the problem; nothing else matters until that's fixed.
2. **Failure taxonomy second.** Where are the failures concentrated?
   `wrong_route` heavy → fix patterns. `partial` heavy → bigger
   `top_k` or imports-aware resolution.
3. **Per-question table last.** Spot the long-tail outliers
   (high latency or repeated rewrites) — those are the questions
   worth re-running with `--debug`.

## Answer-level faithfulness (`faithfulness_mean`)

The eval harness now runs an answer-level faithfulness judge after the per-retrieval
scoring pass.  The judge is selected by the `--judge` flag (default: `auto`):

- **`heuristic`** (CI default when no API key is set): deterministic, no LLM.
  Score = max(citation_fraction, text_fraction), where citation_fraction is the
  fraction of `gold_qualified_names` that appear in the agent's explicit citations
  (exact or suffix match), and text_fraction is the fraction that appear verbatim
  in the answer text.  Range: [0.0, 1.0].
- **`llm`** (requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`): calls
  `claude-3-5-haiku-20241022` (preferred) or `gpt-4o-mini` with a structured
  prompt asking for `{"score": float, "rationale": str}`.  Falls back to the
  heuristic judge on network failure or malformed JSON.
- **`auto`**: uses `LLMJudge` if an API key is present, else `HeuristicJudge`.

A `faithfulness_mean` of `-1.0` in REPORT.md means the judge was not run (should
not occur under normal invocation — `auto` always picks a judge).

Expected ranges (heuristic judge, no synthesiser LLM):

| range | interpretation |
| --- | --- |
| ≥ 0.8 | retriever is surfacing the right names and they appear in the answer |
| 0.5–0.8 | partial coverage; some gold names missed |
| < 0.5 | agent is citing the wrong symbols or answer is ungrounded |

### Kappa calibration

Run `make calibrate` to compute Cohen's kappa between the active judge and
human labels in `evals/calibration.csv`.  The bundled CSV contains placeholder
labels (`human_score` column all = 1 or 0 as stubs) — replace them with real
human annotations before treating the kappa value as meaningful.  The report
is written to `evals/CALIBRATION.md`.

Kappa interpretation (Landis & Koch 1977):

| range | label |
| --- | --- |
| < 0 | less than chance |
| 0.00–0.20 | slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| 0.61–0.80 | substantial |
| 0.81–1.00 | almost perfect |

## LLM synthesizer and citation-faithfulness

The eval harness runs without `GROQ_API_KEY` by default, so synthesis always goes through `StitchSynthesizer` (deterministic chunk-concatenation). In that mode the citation recall and precision numbers in REPORT.md measure **retrieval quality only** — every retrieved chunk is verbatim in the answer, so citations are trivially correct and `citation_recall` directly reflects whether the retriever found the right code.

When `GROQ_API_KEY` is set, the agent switches to `GroqSynthesizer` (LLM-backed). The LLM may paraphrase chunks, drop some, or add reasoning that links chunks. In that regime:

- `citation_recall` becomes a signal of whether the LLM *mentioned* the gold symbols, not just whether they were retrieved.
- `citation_precision` becomes meaningful — an LLM can over-cite or under-cite relative to what was retrieved.
- The `CitationValidator` now catches real hallucinations (backticked qnames the LLM invented that are not in the retrieved set).

In short: **with `GROQ_API_KEY` unset, citation metrics measure retrieval; with it set, they measure synthesis faithfulness**. Comparison runs should hold the synthesizer fixed.

## RAGAS metrics

The harness reports four RAGAS-style metrics starting from the `ragas` module
addition.  All four are computed by default (`--metrics custom,ragas`).

### `faithfulness` (RAGAS)

**Formula:** fraction of `gold_qualified_names` that appear verbatim (case-insensitive,
NFKC-normalised) as a substring in the agent's answer text.

| range | interpretation |
| --- | --- |
| ≥ 0.7 | agent's answer text explicitly cites the right symbols |
| 0.3–0.7 | partial coverage; some gold symbols appear, others are absent |
| < 0.3 | agent answer does not name the expected symbols |

**With `FakeEncoder`:** `StitchSynthesizer` concatenates retrieved chunks verbatim,
so faithfulness reflects whether the retriever surfaces the right chunks at all.
With a real encoder + LLM synthesizer the score can vary more.

### `answer_relevancy` (RAGAS)

**Formula:** cosine similarity between question keyword tokens (≥3 chars) and
answer keyword tokens (lexical, no embeddings).

| range | interpretation |
| --- | --- |
| ≥ 0.5 | answer stays on topic with the question |
| 0.2–0.5 | answer partially drifts from the question terms |
| < 0.2 | answer is largely off-topic or very short |

**With `FakeEncoder`:** The synthesizer returns raw chunk text; relevancy measures
how many of the question's tokens appear in the answer.  Short questions with
technical symbols (e.g. "who calls find_callers") typically score ~0.1–0.3 because
the answer body uses different vocabulary.  This is expected — the metric is best
interpreted comparatively across runs, not as an absolute threshold.

### `context_precision` (RAGAS)

**Formula:** fraction of retrieved chunks (cited qualified names used as context)
that contain at least one `gold_qualified_name` as a substring.

| range | interpretation |
| --- | --- |
| ≥ 0.5 | majority of retrieved context is relevant |
| 0.2–0.5 | retriever mixes relevant and irrelevant chunks |
| < 0.2 | retriever is mostly surfacing irrelevant chunks |

**With `FakeEncoder`:** Since the harness uses citation qnames as the context
proxy (not raw chunk text), context precision measures whether the agent cited
gold symbols at all.  It closely tracks `citation_precision` in this configuration.

### `context_recall` (RAGAS)

**Formula:** fraction of `gold_qualified_names` that appear as a substring in any
retrieved chunk (citation qnames as context proxy).

| range | interpretation |
| --- | --- |
| ≥ 0.6 | retriever is surfacing most of the gold symbols |
| 0.2–0.6 | partial recall; some gold missed |
| < 0.2 | retriever is missing most gold symbols |

**With `FakeEncoder`:** Tracks `citation_recall` closely.  Both metrics drop for
lookup / hybrid categories because `FakeEncoder` (deterministic blake2b hash) is
not semantically sensitive — it retrieves by hash similarity rather than meaning.

### Why all four RAGAS scores are low with `FakeEncoder`

`FakeEncoder` produces deterministic but semantically meaningless vectors.  The
retriever's ranking is therefore hash-order rather than semantic relevance.
Gold symbols often rank outside `top_k=8`.  As a result:

- **faithfulness** is low: the retrieved (and synthesized) chunks often don't
  contain the gold names.
- **answer_relevancy** is low: the stitch synthesizer doesn't rewrite answers, so
  answer vocabulary is drawn from whichever chunks happen to rank in top-k.
- **context_precision** and **context_recall** are low: same root cause — hash-based
  retrieval misses the semantically relevant chunks.

Switch to `BAAI/bge-small-en-v1.5` via `--encoder` to see all four metrics rise
substantially, especially for lookup and summarization questions.

## Real-encoder eval — REPORT.bge.md

`evals/REPORT.bge.md` is the companion report produced by running the same
30-question golden set with `BAAI/bge-small-en-v1.5` instead of `FakeEncoder`.
It is generated automatically by `.github/workflows/real_encoder_eval.yml` on
every push to `main` and weekly on Mondays.

### Where it comes from

1. The `real_encoder_eval` CI workflow runs on `ubuntu-latest` (where
   `torch` / `sentence-transformers` Linux wheels are available).
2. It calls `uv sync --extra dev --extra embed` to install the encoder.
3. It indexes `src/` with `--encoder BAAI/bge-small-en-v1.5` into an in-memory store.
4. It runs `atlas eval --encoder BAAI/bge-small-en-v1.5 --out evals/REPORT.bge.md`.
5. Both `REPORT.bge.md` and `scores-bge.json` are uploaded as workflow artifacts
   (retention: 30 days). Download them from the Actions run and commit.

### Expected magnitude shift vs FakeEncoder

| Category      | FakeEncoder recall | bge-small recall (expected) |
| ------------- | -----------------: | --------------------------: |
| lookup        |               0.00 |                       ~0.65 |
| hybrid        |               0.00 |                       ~0.55 |
| summarization |               0.12 |                       ~0.45 |
| structural    |               0.60 |                       ~0.65 |
| neighborhood  |               0.25 |                       ~0.35 |

Structural recall does not change much because graph-walk routes are topology-driven
(BFS on the call graph), not similarity-driven. Lookup and hybrid routes depend
entirely on vector similarity — that is where `bge-small` provides the biggest lift.

### How to refresh locally

```bash
# Requires Linux or Apple-Silicon macOS (torch wheels available).
uv sync --extra dev --extra embed
uv run atlas index src/ --store=memory --encoder BAAI/bge-small-en-v1.5
ATLAS_ENCODER=BAAI/bge-small-en-v1.5 uv run atlas eval \
    --encoder BAAI/bge-small-en-v1.5 \
    --json-out evals/scores-bge.json \
    --out evals/REPORT.bge.md
```

On Intel macOS the `torch` wheel is unavailable; use the CI workflow or a Linux
machine. The `--encoder fake` (default) path is always available everywhere.

## bge-small Docker run — actual results (2026-05-06)

The `BAAI/bge-small-en-v1.5` encoder was run via Docker (`python:3.12-slim` on
the Intel macOS host via colima) because the Mac host has no torch x86_64 wheels.

### Headline comparison

| Metric | FakeEncoder | bge-small-en-v1.5 | delta |
| --- | ---: | ---: | ---: |
| Route correctness | 93.3% | 93.3% | 0.0pp |
| Citation recall (mean) | 0.13 | 0.28 | +0.15 |
| Citation precision (mean) | 0.06 | 0.08 | +0.02 |
| Faithfulness (mean) | 0.38 | 0.64 | +0.26 |
| p50 latency (ms) | 3.7 | 184.9 | +181ms (model inference) |
| p95 latency (ms) | 5.8 | 563.1 | +557ms |
| p99 latency (ms) | 7.8 | 798.7 | +791ms |

### By category — citation recall

| Category | FakeEncoder recall | bge-small recall | delta | expected |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 0.00 | 0.00 | 0.00 | n/a |
| import-chain | 0.00 | 0.00 | 0.00 | n/a |
| lookup | **0.00** | **0.33** | +0.33 | ~0.65 |
| multi-hop | 0.00 | 0.00 | 0.00 | ~0.55 |
| neighborhood | 0.25 | 0.25 | 0.00 | ~0.35 |
| out-of-scope | 0.00 | 0.00 | 0.00 | n/a |
| structural | 0.60 | **0.80** | +0.20 | ~0.65 |
| summarization | 0.00 | **0.38** | +0.38 | ~0.45 |

### Key observations

- **Lookup recall rose from 0.00 → 0.33** (from hash-random to semantic vectors).
  Still below the ~0.65 expected; the 30-question golden set's phrasing may not
  closely match module docstrings/function names, which are the chunk text.
- **Structural recall rose from 0.60 → 0.80** — better than expected. The graph
  walk still dominates, but bge-small improves the initial vector seed node.
- **Summarization recall rose from 0.00 → 0.38** — matching expectations (~0.45).
  The encoder surfaces the right module chunks even for broad "walk me through" queries.
- **Multi-hop / hybrid recall stayed at 0.00** — the hybrid route classifier mis-fired
  on 2/4 questions (wrong_route), which zeroed out those question scores regardless
  of encoder quality. Route precision, not retrieval, is the bottleneck here.
- **Faithfulness jumped from 0.38 → 0.64** — the heuristic judge sees more gold
  qnames in the answer text because bge-small retrieves semantically relevant chunks.
- **Latency increased ~50x** (p50: 3.7ms → 184.9ms) due to bge-small inference in
  the Docker container (CPU-only). On a GPU host p50 would be under 20ms.
- **Failure taxonomy shift**: off_topic fell from 21 → 16, hallucinated rose from 3 → 7.
  bge-small retrieves *more* context, but the heuristic synthesis sometimes cites extra
  qnames that aren't in the gold set — a precision/recall tradeoff.

### Prerequisites fixed during this run

The `load_sentence_transformer_encoder` function in `src/codex_atlas/embed.py`
had `_ST` inheriting from the `Encoder` Protocol. On Linux/Python 3.12 this caused
a `non-default argument 'model' follows default argument` error at dataclass
construction time. The fix: remove the explicit `Encoder` base class from `_ST`
(structural typing satisfies the protocol without inheritance).

## Calibration notes

- Confidences in `RoutingDecision` (0.95 / 0.92 / 0.9 / 0.85 / 0.6)
  reflect the observed precision of each rule on the 30-question
  hand-graded development set. The agent's grader uses these as a
  floor so a high-confidence route automatically passes the
  threshold without an LLM judge.
- Tolerance for `--baseline` regressions defaults to 5%. Tune up for
  noisy baselines (small N), tune down once the harness has > 100
  questions.
