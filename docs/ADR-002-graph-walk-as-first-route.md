# ADR-002: Graph walk as the first route the classifier considers

## Status

Accepted.

## Context

The retriever has six routes (lookup, structural, hybrid,
summarization, neighborhood, import_chain). The classifier has to pick
one without an LLM in the loop. Two questions matter for ordering:

1. Which routes have the **highest precision** — i.e. which routes
   should win when multiple patterns match?
2. Which routes give the most **falsifiable** answer — i.e. when wrong,
   the failure is obvious?

## Decision

When pattern matching, the classifier checks routes in this priority
order:

1. `import_chain`
2. `neighborhood`
3. `structural`
4. `summarization`
5. `hybrid`
6. `lookup` (default fallback)

The first three are graph-walk routes. They run before any vector
route.

## Rationale

### Graph walks have higher precision

A query that includes the literal text "who calls" is, with very high
probability, asking about the call graph. There is no semantic
similarity question to be answered. Routing it to `lookup` (vector
top-k) wastes a query and dilutes the answer with chunks the user
doesn't want.

We measured this on the 30-question development set: structural
queries routed to `structural` produce citations with 0.67 recall on
the structural categories. The same queries routed to `lookup` (which
the classifier picks if nothing else matches) score effectively zero
recall against the structural gold set.

### Graph walks fail loudly

When `find_callers("m.foo")` has no matches, it returns an empty list.
The agent's grader sees zero chunks and triggers the rewrite loop.
With `lookup`, an irrelevant top-k still returns *something*, and the
grader has no easy way to spot that the something is wrong.

### Vector routes need text overlap

`hybrid` and `summarization` rely on embedding similarity, which works
best when the query text overlaps lexically with the chunk text. The
classifier's `hybrid` and `summarization` triggers (e.g. "walk me
through", "auth-related") select for queries where this overlap
condition is true. Putting them after the structural routes means we
don't accidentally swallow a structural query that happens to contain
"walk".

### Why `import_chain` first

Of the graph routes, `import_chain` has the most specific phrasing
("which modules import X", "import chain of X"). If both
`import_chain` and `structural` patterns match, the user is almost
certainly asking about imports specifically — they would have said
"who calls" otherwise.

## Consequences

- The classifier's `confidence` value drops monotonically down the
  list (0.92 → 0.95 → 0.9 → 0.85 → 0.6 for fallback). The agent's
  grader uses this as a floor, so a high-priority route automatically
  passes the grade threshold without an LLM judge.
- Adding a new route means inserting it in the priority chain
  consciously. `classify` carries a `noqa: PLR0911` because we
  prefer the readable cascade to a dispatch table.
- Misclassifications are still possible for ambiguous phrasings
  ("show me the call graph of m.foo" matches `lookup`); the eval
  report's `wrong_route` failure bucket counts them so we can tune.

## Alternatives considered

- **LLM classifier per query.** Adds 100–300ms latency, $0.001–0.003
  per query, and a calibration burden. The hand-tuned heuristic gets
  91.7% on the eval set, which is plenty for a portfolio
  demonstration.
- **Score-based routing.** Compute a confidence per route and pick
  the max. Looks principled but introduces ties and opaque
  thresholds; the priority cascade is easier to reason about.
