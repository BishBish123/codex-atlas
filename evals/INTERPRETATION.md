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
small (12 questions) so noise is real. Watch the trend across runs,
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

## Calibration notes

- Confidences in `RoutingDecision` (0.95 / 0.92 / 0.9 / 0.85 / 0.6)
  reflect the observed precision of each rule on the 30-question
  hand-graded development set. The agent's grader uses these as a
  floor so a high-confidence route automatically passes the
  threshold without an LLM judge.
- Tolerance for `--baseline` regressions defaults to 5%. Tune up for
  noisy baselines (small N), tune down once the harness has > 100
  questions.
