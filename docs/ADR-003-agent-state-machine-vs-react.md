# ADR-003: Hand-rolled state machine over LangGraph and ReAct

## Status

Accepted.

## Context

The agent loop has to do five things:

1. classify the query
2. retrieve via the right route
3. grade the result
4. rewrite + retry if the grade is low
5. synthesise and validate the final answer

Three obvious shapes:

- **LangGraph** (the Python library that gave the project its
  vocabulary)
- **ReAct loop** (LLM-driven thought / action / observation cycle)
- **Hand-rolled state machine** (one `Agent.run` method, async, no
  external scheduler)

## Decision

We hand-rolled the state machine. The `Node` enum
(`classify | retrieve | grade | rewrite_query | answer | validate |
cancel`) carries the LangGraph vocabulary in trace events without
taking on the dependency.

## Rationale

### Why not LangGraph

LangGraph is a great fit when the graph has dynamic branching that
depends on LLM output (e.g. tool-use loops where the LLM picks the
next node). Codex-Atlas's graph is *static*: the only conditional is
"grade < threshold AND attempts < max", and that's a `while` loop.
Adding LangGraph would mean importing a graph compiler to run a
five-line `while` loop — net cost is a dependency, training friction,
and harder mocking in tests.

### Why not ReAct

ReAct is built around an LLM-in-the-loop. The LLM picks tools, observes
results, and re-prompts itself. That's a great fit when the tool space
is open-ended (web search, calculators, code execution). Codex-Atlas's
tool space is *closed*: six retriever routes with deterministic
selection. Wrapping that in ReAct would replace a 12-line classifier
regex bank with an LLM call per query — slower, more expensive, and
less testable.

### Why hand-rolled wins here

1. **Mockable.** `StubRetriever`, `StubGrader`, and `StubSynthesizer`
   in tests cover the loop end-to-end without a network. The trace
   contract is just a list of `TraceEvent` dataclasses; tests assert
   directly against it.
2. **Observable.** `tool_calls` is a structured log of every retriever
   invocation. We can serialise it as Langfuse spans, OpenTelemetry
   events, or Datadog metrics with a 30-line adapter — no agent loop
   re-implementation.
3. **Cancellable.** `step_timeout_s` and `run_timeout_s` integrate with
   `asyncio.wait_for` cleanly because we control the call sites. With
   LangGraph cancellation goes through the graph compiler's API, which
   is fine but more indirect.

## Consequences

- The agent code is ~300 lines, including the validator + cancel
  control flow. That fits comfortably in a single file.
- When we want LLM-driven behaviour (e.g. an LLM grader replacing
  `HeuristicGrader`), we plug it in via the `Grader` Protocol —
  zero changes to the loop itself.
- If the loop ever needs *real* dynamic branching (e.g. an LLM that
  picks between two routes), we revisit. Until then the gain in
  control + testability is worth the missing graph compiler.

## Alternatives considered

- **AsyncIO `Task`-per-node.** Overkill; the nodes are sequential
  with one `while` loop.
- **LangGraph in test mode.** Possible, but the test surface still
  includes a graph compiler and the state graph YAML — a lot of
  ceremony for five nodes.
- **Pure ReAct via Anthropic tool-use.** The retriever is too cheap
  and too reliable for that to pay off; we save money and latency by
  classifying ourselves.
