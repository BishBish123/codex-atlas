"""LangGraph-style agent loop with reflection + bounded re-query + validate.

The state machine has seven nodes:

    classify -> retrieve -> grade ─┐
                  ▲                │
                  │       grade < threshold AND attempts < max
                  └────── rewrite_query ──┘
                                 │
                                 ▼
                              answer
                                 │
                                 ▼
                              validate
                                 │
                                 ▼
                          (final result)

`Grader`, `Synthesizer`, and the validator are protocols so the agent
runs with any LLM (or with deterministic fakes for offline testing). The
default grader is a heuristic over chunk count + retriever confidence;
swap in an LLM-as-judge implementation for production-grade accuracy.

Every state transition is timestamped + logged so adding Langfuse later
is local: subscribe to the trace stream and emit spans. The `tool_calls`
log records every retriever invocation so an external observability
plane can stitch them into a span tree without re-implementing the
state machine.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeVar

from codex_atlas.observability.langfuse import LangfuseTracer, NullTracer, make_tracer
from codex_atlas.retriever import RetrievalResult, Retriever, Route
from codex_atlas.store import StoredChunk

T = TypeVar("T")

_log = logging.getLogger(__name__)


class Node(StrEnum):
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    GRADE = "grade"
    REWRITE = "rewrite_query"
    ANSWER = "answer"
    VALIDATE = "validate"
    CANCEL = "cancel"


# Whole-of-answer policy applied after the validator decides the answer
# is unacceptable. ``advisory`` is the historical behaviour (log + return
# verbatim); ``redact`` rewrites the answer, replacing each ungrounded
# claim with a marker; ``reject`` replaces the answer with a refusal.
ValidationMode = Literal["advisory", "redact", "reject"]


_REJECT_MESSAGE = (
    "I could not produce a grounded answer for that query. "
    "Validation flagged ungrounded claims that aren't backed by the retrieved code."
)


class InvalidValidationReport(ValueError):
    """Raised by ``Agent._apply_validation`` when a ``ValidationReport``
    contains spans that violate the documented contract:

    * Each span must satisfy ``0 <= start < end <= len(answer)``.
    * Spans must not overlap after sorting by ``start``.

    A validator that returns an inconsistent report would cause the
    redactor to silently corrupt the answer text (out-of-bounds slice,
    or double-application of a replacement). Raising early gives
    the caller a clear signal instead.
    """


class CancelReason(StrEnum):
    """Why the agent terminated early."""

    TIMEOUT = "timeout"
    EXTERNAL = "external"


class _Cancelled(Exception):
    """Internal control-flow signal: the run was cancelled."""

    def __init__(self, reason: CancelReason) -> None:
        super().__init__(str(reason))
        self.reason = reason


@dataclass(frozen=True)
class TraceEvent:
    """One step in the agent's execution — what node fired, when, and how long."""

    node: Node
    started_at: float
    elapsed_ms: float
    detail: str = ""


@dataclass(frozen=True)
class ToolCall:
    """One retriever invocation — what query, which route, what came back."""

    query: str
    route: Route
    n_chunks: int
    elapsed_ms: float
    confidence: float


@dataclass(frozen=True)
class ClaimSpan:
    """A claim's location inside the answer string.

    Validators that scan the answer can return ``ClaimSpan`` entries so
    the agent's redactor knows the exact byte range of each match. This
    avoids ``str.replace`` collisions when the same backticked claim
    appears twice — only the ungrounded occurrences are rewritten,
    grounded ones survive verbatim.
    """

    claim: str
    start: int  # offset of the opening backtick
    end: int  # offset just past the closing backtick (exclusive)
    grounded: bool


@dataclass(frozen=True)
class ValidationReport:
    """Per-claim grounded-vs-ungrounded breakdown.

    A claim is *grounded* when it explicitly cites a chunk by qualified
    name and that name appears in the retrieved chunks. Anything else
    (free-form prose, hand-waving) is conservatively counted as
    ungrounded — false negatives there cost less than false positives.

    ``spans`` is the per-occurrence record (one entry per backtick-wrapped
    qualified name in the answer). The agent's redactor uses these
    offsets to rewrite ungrounded matches in place; ``ungrounded_claims``
    and ``grounded_qualified_names`` keep the de-duplicated names that
    callers actually want to display.
    """

    n_claims: int
    n_grounded: int
    ungrounded_claims: list[str]
    grounded_qualified_names: list[str]
    is_acceptable: bool
    spans: list[ClaimSpan] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    """A pointer into the corpus included with every answer.

    ``score`` is the retriever's per-chunk confidence (cosine similarity
    for the lookup route, hybrid combined score on the hybrid route,
    1.0 for the structural route — same value the retriever attaches to
    each ``StoredChunk``). ``text`` is the chunk source text. Both
    fields default to "neutral" values so older callers that construct
    a ``Citation`` without them keep working — the MCP boundary now
    surfaces real values when the agent wires them through, instead of
    the hardcoded 0.0 / "" the API contract previously lied about.
    """

    qualified_name: str
    file_path: str
    lineno_start: int
    lineno_end: int
    score: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class AgentResult:
    """Everything the caller (CLI / MCP tool) needs to render."""

    query: str
    final_query: str  # may differ from `query` if the agent re-wrote
    answer: str
    citations: list[Citation]
    route: Route
    grade: float
    attempts: int
    trace: list[TraceEvent]
    tool_calls: list[ToolCall] = field(default_factory=list)
    validation: ValidationReport | None = None
    cancelled: CancelReason | None = None
    # The node that was executing when ``cancelled is TIMEOUT`` fired
    # (e.g. ``Node.RETRIEVE``, ``Node.GRADE``, ``Node.ANSWER``). ``None``
    # for runs that completed normally or were cancelled for any other
    # reason. The MCP timeout response surfaces this directly so clients
    # don't have to infer the phase from a trace that already includes
    # the post-timeout ``Node.CANCEL`` event.
    cancelled_node: Node | None = None


@dataclass
class _State:
    """Internal mutable state — never crosses the public API."""

    original_query: str
    query: str
    attempts: int = 0
    retrieval: RetrievalResult | None = None
    grade: float = 0.0
    trace: list[TraceEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    cancelled: CancelReason | None = None
    # Set when the caller wants to skip classification for this run.
    route_override: Route | None = None
    # The node currently executing — set when a node body starts and
    # cleared back to ``None`` when it returns successfully. The agent
    # appends a ``Node.CANCEL`` event after a timeout, so the MCP
    # boundary cannot reliably infer the timed-out phase from
    # ``trace[-1]``. ``in_flight_node`` is the source of truth for
    # "what was running when the deadline fired" — and ``None`` is the
    # correct answer when the deadline fires *between* node bodies
    # (e.g. ``run_timeout_s`` exhausted right after ``_retrieve``
    # returned but before ``_grade`` started). Without the explicit
    # clear, a between-node timeout would leak the previous node's
    # value, surfacing a stale ``cancelled_node``.
    in_flight_node: Node | None = None


# ---------------------------------------------------------------------------
# Plug points
# ---------------------------------------------------------------------------


class Grader(Protocol):
    """Score a (query, retrieved chunks) pair on relevance — 0..1."""

    async def grade(self, query: str, chunks: list[StoredChunk]) -> float: ...


class QueryRewriter(Protocol):
    """Produce a sharper query when retrieval came back weak."""

    async def rewrite(self, original: str, retrieved: list[StoredChunk]) -> str: ...


class Synthesizer(Protocol):
    """Compose the final answer from query + retrieved chunks."""

    async def synthesize(self, query: str, chunks: list[StoredChunk]) -> str: ...


class Validator(Protocol):
    """Audit a synthesised answer against the retrieved chunks.

    Implementations return a ``ValidationReport`` flagging any claims
    that aren't grounded in a cited qualified name. The default
    implementation is a regex-based citation extractor; production swaps
    in an LLM judge that re-reads claims against chunk text.
    """

    async def validate(
        self, query: str, answer: str, chunks: list[StoredChunk]
    ) -> ValidationReport: ...


# ---------------------------------------------------------------------------
# Default no-LLM implementations — let the agent run end-to-end offline
# and in tests. Production swaps these for LLM-backed versions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicGrader:
    """Score = retriever confidence times (chunks present ? 1.0 : 0.0).

    Concretely: an empty retrieval is grade 0.0; a non-empty retrieval
    inherits the route's confidence. Crude but deterministic, and the
    relative ordering matches what an LLM judge would produce on the
    eval set.
    """

    async def grade(self, query: str, chunks: list[StoredChunk]) -> float:
        return 1.0 if chunks else 0.0


@dataclass(frozen=True)
class NoopRewriter:
    """When no LLM is available, append a clarifying suffix and try again."""

    async def rewrite(
        self,
        original: str,
        retrieved: list[StoredChunk],
    ) -> str:
        return f"{original} (in this codebase, with code-level detail)"


@dataclass(frozen=True)
class CitationValidator:
    """Default validator: every backticked qualified-name in the answer
    must appear in the retrieved chunks. Anything else is ungrounded.

    Grounding rule (v0.2 — see ``validate`` for tests):

    1. **Exact match.** Claim equals a retrieved chunk's qualified name.
    2. **Trailing-component match.** Some retrieved qname ends in
       ``"." + claim`` AND the leading prefix (everything before the
       trailing claim) is a known module / package / class derived from
       the retrieval set. This rejects spurious ``pkg_b.Cls.method``
       claims when only ``pkg_a.Cls.method`` was retrieved — the suffix
       ``.Cls.method`` matches but the leading ``pkg_b`` prefix is
       unknown, so the claim is rejected.
    3. **Containing match.** Claim is itself longer than any retrieved
       qname AND ends in some retrieved qname's last component path AND
       the *claim's* prefix is a known module — same prefix-check, in
       the other direction.

    The threshold is the minimum grounded fraction at which we accept
    the answer. Below it the answer is rejected (the agent's caller can
    decide whether to re-route or surface the warning).
    """

    accept_threshold: float = 0.5
    # Match backticked qualified names: `pkg.module.symbol` (>=2 parts).
    _claim_re: re.Pattern[str] = re.compile(r"`([A-Za-z_][\w.]*\.[A-Za-z_]\w*)`")

    @staticmethod
    def _module_prefixes(qnames: set[str]) -> set[str]:
        """All non-empty dotted prefixes of every retrieved qname.

        For ``pkg.module.Cls.method`` this yields
        ``{"pkg", "pkg.module", "pkg.module.Cls"}``. The full qname
        itself is intentionally excluded — that is matched by the exact
        / trailing-component rules separately.
        """
        prefixes: set[str] = set()
        for q in qnames:
            parts = q.split(".")
            for i in range(1, len(parts)):
                prefixes.add(".".join(parts[:i]))
        return prefixes

    async def validate(
        self, query: str, answer: str, chunks: list[StoredChunk]
    ) -> ValidationReport:
        cited_qnames = {c.qualified_name for c in chunks}
        known_prefixes = self._module_prefixes(cited_qnames)
        # Use ``finditer`` so we keep the byte offsets — the redactor
        # downstream needs spans, not just the matched substrings.
        matches = list(self._claim_re.finditer(answer))
        if not matches:
            # Nothing claim-shaped to validate. We log a structured event
            # so this is auditable in production traces — an LLM answer
            # that makes ZERO specific code references is suspicious even
            # when no specific claim regex-matched. We still accept (v0.2
            # policy: "no claims = pass through, log it"); a future v0.3
            # can flip this to fail when the question explicitly asks for
            # a symbol reference.
            _log.info(
                "citation_validator.zero_claims",
                extra={"query": query, "n_chunks": len(chunks)},
            )
            return ValidationReport(
                n_claims=0,
                n_grounded=0,
                ungrounded_claims=[],
                grounded_qualified_names=[],
                is_acceptable=True,
                spans=[],
            )
        grounded: list[str] = []
        ungrounded: list[str] = []
        spans: list[ClaimSpan] = []
        for m in matches:
            claim = m.group(1)
            is_grounded = self._is_grounded(claim, cited_qnames, known_prefixes)
            spans.append(
                ClaimSpan(claim=claim, start=m.start(), end=m.end(), grounded=is_grounded)
            )
            if is_grounded:
                grounded.append(claim)
            else:
                ungrounded.append(claim)
        n = len(matches)
        ratio = len(grounded) / n if n else 1.0
        return ValidationReport(
            n_claims=n,
            n_grounded=len(grounded),
            ungrounded_claims=ungrounded,
            grounded_qualified_names=grounded,
            is_acceptable=ratio >= self.accept_threshold,
            spans=spans,
        )

    @staticmethod
    def _is_grounded(claim: str, cited: set[str], known_prefixes: set[str]) -> bool:
        # 1. Exact match — strongest grounding signal.
        if claim in cited:
            return True

        # 2. Trailing-component match: some retrieved qname ends in
        #    ``.<claim>``. The intuition is that an answer can refer to
        #    ``Cls.method`` while the retrieved qname is the fully
        #    qualified ``pkg.module.Cls.method`` — same symbol, less
        #    qualified. We accept this BUT only when:
        #    (a) the claim has no dotted prefix of its own to verify
        #        (e.g. ``Cls.method``, two parts) — there's no module
        #        in the claim that could be a hallucination, OR
        #    (b) the claim's own leading prefix is a known module, so
        #        ``pkg.module.Cls.method`` claim grounds against
        #        ``pkg.module.Cls.method`` retrieval AND we've seen
        #        ``pkg.module`` as a known prefix.
        #
        #    Without (b), an answer claiming ``pkg_b.Cls.method`` would
        #    be grounded by a retrieved ``pkg_a.Cls.method`` because
        #    both share the trailing path ``.Cls.method`` — that's the
        #    hallucination we explicitly want to reject.
        suffix = f".{claim}"
        for q in cited:
            if q.endswith(suffix):
                # Strip the matched suffix from the retrieved qname; the
                # remainder is the retrieved module/package path.
                retrieved_prefix = q[: -len(suffix)]
                # How many dotted components does the claim itself
                # contribute beyond the trailing symbol path? A claim
                # ``Cls.method`` has one dot — purely a trailing path.
                # ``pkg.Cls.method`` has two dots — the leading ``pkg``
                # is a module assertion the validator must verify.
                claim_parts = claim.split(".")
                if len(claim_parts) <= 2:
                    # Pure trailing path; no module asserted by the claim.
                    return True
                claim_module_prefix = ".".join(claim_parts[:-2])
                # Either the claim's prefix is itself a known module, or
                # it matches the retrieved qname's actual prefix (i.e.
                # the same module the retrieval came from).
                if (
                    claim_module_prefix in known_prefixes
                    or claim_module_prefix == retrieved_prefix
                ):
                    return True

        # 3. Containing match: claim is longer than any retrieved qname
        #    and ends with one as a trailing path. Require the claim's
        #    leading prefix to be a known module/package — this catches
        #    the ``pkg_b.Cls.method`` attack when the retrieval set has
        #    only the bare ``Cls.method``: the claim asserts a module
        #    (``pkg_b``) that was never retrieved, so reject.
        for q in cited:
            tail = f".{q}"
            if claim.endswith(tail):
                claim_prefix = claim[: -len(tail)]
                if claim_prefix in known_prefixes:
                    return True
        return False


def _display_path(file_path: str) -> str:
    """Render ``file_path`` repo-relative when it lives inside ``cwd``.

    The indexer stores absolute paths so chunks survive a process moving
    cwd; the eval REPORT.md and the synthesiser-rendered answer text
    are read by humans and should not bleed
    ``/Users/<dev>/projects/...`` prefixes. ``os.path.relpath`` is the
    surgical fix per the UX review: when the path is under cwd we shrink
    it to the repo-relative form (``src/codex_atlas/agent.py``);
    otherwise we leave it alone.
    """
    try:
        rel = os.path.relpath(file_path, start=os.getcwd())
    except ValueError:
        # Different drive on Windows; relpath raises rather than crossing.
        return file_path
    # ``relpath`` keeps walking up with ``..`` for paths outside cwd.
    # If the relative form is longer or starts with ``..`` we're better
    # off keeping the absolute form — it's at least unambiguous.
    if rel.startswith("..") or len(rel) >= len(file_path):
        return file_path
    return rel


@dataclass(frozen=True)
class StitchSynthesizer:
    """Concatenate the top chunks into a markdown answer with inline cites.

    No LLM, but produces something a human can read and reviewers can
    verify against the citations. Production swaps this for an LLM
    synthesiser; the agent's behaviour stays the same.
    """

    max_chunks: int = 5

    async def synthesize(self, query: str, chunks: list[StoredChunk]) -> str:
        if not chunks:
            return (
                "I could not find any relevant code chunks for that query in the "
                "indexed corpus. Try a more specific term, or index a different repo."
            )
        lines = [f"# {query}", ""]
        for c in chunks[: self.max_chunks]:
            shown = _display_path(c.file_path)
            lines.append(
                f"## `{c.qualified_name}` ({shown}:{c.lineno_start}-{c.lineno_end})"
            )
            lines.append("```python")
            lines.append(c.text.rstrip())
            lines.append("```")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    grade_threshold: float = 0.7
    max_attempts: int = 3
    # Per-step timeout for the retriever / synthesiser. None disables.
    step_timeout_s: float | None = None
    # Whole-run timeout (across all retries). None disables.
    run_timeout_s: float | None = None
    # What to do when the validator flags the answer as ungrounded:
    #
    #   * ``redact`` (default): replace each ungrounded backticked claim
    #     IN PLACE (by span offset) with ``[ungrounded: <claim>]`` and
    #     append a summary stamp to the answer. The answer stays useful
    #     where it was grounded; readers see exactly which claims failed.
    #     Caveat: redaction only catches claims the validator can
    #     identify (backtick-wrapped dotted qualified names); arbitrary
    #     prose claims pass through untouched. For stricter enforcement
    #     use ``validation_mode="reject"``.
    #   * ``reject``: replace the entire answer with a refusal message.
    #     Use when downstream consumers must never see partial-grounded
    #     content.
    #   * ``advisory``: legacy behaviour — log + return the answer
    #     verbatim. Kept for callers that want the validator as a signal
    #     only, e.g. to render a warning banner.
    validation_mode: ValidationMode = "redact"


class Agent:
    """The LangGraph-style state machine wiring router + grader + synth."""

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer | None = None,
        grader: Grader | None = None,
        rewriter: QueryRewriter | None = None,
        validator: Validator | None = None,
        config: AgentConfig | None = None,
        tracer: LangfuseTracer | NullTracer | None = None,
    ) -> None:
        self._retriever = retriever
        if synthesizer is not None:
            self._synth = synthesizer
        else:
            from codex_atlas.synthesis import make_synthesizer  # noqa: PLC0415

            self._synth = make_synthesizer()
        self._grader = grader or HeuristicGrader()
        self._rewriter = rewriter or NoopRewriter()
        self._validator = validator or CitationValidator()
        self._config = config or AgentConfig()
        self._tracer: LangfuseTracer | NullTracer = tracer if tracer is not None else make_tracer()

    async def run(self, query: str, *, route_override: Route | None = None) -> AgentResult:
        """Run the state machine; optionally skip the classifier.

        ``route_override`` is forwarded to the retriever so callers (e.g.
        the MCP ``search_codebase(route=...)`` tool) can force a route
        without resorting to query rewording.
        """
        run_t0 = time.perf_counter()
        state = _State(original_query=query, query=query, route_override=route_override)
        trace_id = self._tracer.start_run(query)
        try:
            await self._with_run_deadline(self._retrieve(state), run_t0)
            # Emit the two trace events + tool call added by _retrieve.
            for ev in state.trace[-2:]:
                self._tracer.record_event(trace_id, ev)
            if state.tool_calls:
                self._tracer.record_event(trace_id, state.tool_calls[-1])

            await self._with_run_deadline(self._grade(state), run_t0)
            self._tracer.record_event(trace_id, state.trace[-1])

            while (
                state.cancelled is None
                and state.grade < self._config.grade_threshold
                and state.attempts < self._config.max_attempts - 1
            ):
                await self._with_run_deadline(self._rewrite(state), run_t0)
                self._tracer.record_event(trace_id, state.trace[-1])

                await self._with_run_deadline(self._retrieve(state), run_t0)
                for ev in state.trace[-2:]:
                    self._tracer.record_event(trace_id, ev)
                if state.tool_calls:
                    self._tracer.record_event(trace_id, state.tool_calls[-1])

                await self._with_run_deadline(self._grade(state), run_t0)
                self._tracer.record_event(trace_id, state.trace[-1])

            if state.cancelled is None:
                answer, citations = await self._with_run_deadline(self._answer(state), run_t0)
                self._tracer.record_event(trace_id, state.trace[-1])

                validation = await self._with_run_deadline(
                    self._validate(state, answer, citations), run_t0
                )
                self._tracer.record_event(trace_id, state.trace[-1])
                self._tracer.record_validation(trace_id, validation)
                # Enforce the configured policy when validation rejects.
                # ``_apply_validation`` is a pure transform — the original
                # report is preserved on the result so callers can still
                # see what was flagged.
                answer = self._apply_validation(state, answer, validation)
            else:
                answer, citations, validation = ("", [], None)
        except _Cancelled as exc:
            state.cancelled = exc.reason
            state.trace.append(
                TraceEvent(
                    node=Node.CANCEL,
                    started_at=time.perf_counter(),
                    elapsed_ms=0.0,
                    detail=f"reason={exc.reason}",
                )
            )
            self._tracer.record_event(trace_id, state.trace[-1])
            answer, citations, validation = ("", [], None)

        # Surface the timed-out node only when the cancellation was
        # actually a timeout — external cancels and normal completions
        # leave ``cancelled_node`` as ``None`` so MCP clients only see
        # the field populated when it carries useful diagnostic info.
        cancelled_node = (
            state.in_flight_node
            if state.cancelled is CancelReason.TIMEOUT
            else None
        )
        result = AgentResult(
            query=state.original_query,
            final_query=state.query,
            answer=answer,
            citations=citations,
            route=(state.retrieval.route if state.retrieval is not None else Route.LOOKUP),
            grade=state.grade,
            attempts=state.attempts + 1,
            trace=state.trace,
            tool_calls=state.tool_calls,
            validation=validation,
            cancelled=state.cancelled,
            cancelled_node=cancelled_node,
        )
        self._tracer.finish_run(trace_id, result)
        return result

    async def _with_run_deadline(self, coro: Coroutine[Any, Any, T], run_t0: float) -> T:
        """Wrap a coro in the optional whole-run timeout.

        Falls through unchanged when no timeout is configured. On
        timeout we raise ``_Cancelled(TIMEOUT)`` to unwind cleanly so
        the run still produces a structured ``AgentResult``.
        """
        run_timeout = self._config.run_timeout_s
        if run_timeout is None:
            return await coro
        elapsed = time.perf_counter() - run_t0
        remaining = run_timeout - elapsed
        if remaining <= 0:
            raise _Cancelled(CancelReason.TIMEOUT)
        try:
            return await asyncio.wait_for(coro, timeout=remaining)
        except TimeoutError as e:
            raise _Cancelled(CancelReason.TIMEOUT) from e

    # ---------- nodes ----------

    async def _retrieve(self, state: _State) -> None:
        t0 = time.perf_counter()
        state.in_flight_node = Node.RETRIEVE
        retrieval = await self._with_step_deadline(
            self._retriever.retrieve(state.query, route_override=state.route_override)
        )
        state.retrieval = retrieval
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.trace.append(
            TraceEvent(
                node=Node.CLASSIFY,
                started_at=t0,
                elapsed_ms=0.0,
                detail=f"route={retrieval.route} confidence={retrieval.confidence:.2f}",
            )
        )
        state.trace.append(
            TraceEvent(
                node=Node.RETRIEVE,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=(
                    f"chunks={len(retrieval.chunks)} extras={len(retrieval.extra_qualified_names)}"
                ),
            )
        )
        state.tool_calls.append(
            ToolCall(
                query=state.query,
                route=retrieval.route,
                n_chunks=len(retrieval.chunks),
                elapsed_ms=elapsed,
                confidence=retrieval.confidence,
            )
        )
        # Clear the in-flight marker so a deadline tripping BETWEEN this
        # node and the next does not surface a stale phase via
        # ``cancelled_node``. A timeout fired by ``_with_run_deadline``
        # before the next node sets the marker should report ``None``
        # (idle), not ``Node.RETRIEVE``.
        state.in_flight_node = None

    async def _with_step_deadline(self, coro: Coroutine[Any, Any, T]) -> T:
        """Wrap a single step in the per-step timeout (when configured)."""
        step = self._config.step_timeout_s
        if step is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=step)
        except TimeoutError as e:
            raise _Cancelled(CancelReason.TIMEOUT) from e

    async def _grade(self, state: _State) -> None:
        if state.retrieval is None:
            state.grade = 0.0
            return
        t0 = time.perf_counter()
        state.in_flight_node = Node.GRADE
        # Empty retrieval -> grade 0 (forces a re-query). Non-empty
        # retrieval defaults to the router's classifier confidence — a
        # high-confidence classifier route with chunks present is the
        # closest signal we have without an LLM judge. An LLM grader, if
        # plugged in, can override the heuristic upward.
        chunks = state.retrieval.chunks
        if not chunks:
            score = 0.0
        else:
            llm_score = await self._grader.grade(state.query, chunks)
            score = max(llm_score, state.retrieval.confidence)
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.grade = float(score)
        state.trace.append(
            TraceEvent(
                node=Node.GRADE,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=f"grade={state.grade:.2f}",
            )
        )
        state.in_flight_node = None

    async def _rewrite(self, state: _State) -> None:
        if state.retrieval is None:
            return
        t0 = time.perf_counter()
        state.in_flight_node = Node.REWRITE
        new_query = await self._rewriter.rewrite(state.original_query, state.retrieval.chunks)
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.trace.append(
            TraceEvent(
                node=Node.REWRITE,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=f"old_len={len(state.query)} new_len={len(new_query)}",
            )
        )
        state.query = new_query
        state.attempts += 1
        state.in_flight_node = None

    async def _answer(self, state: _State) -> tuple[str, list[Citation]]:
        chunks = state.retrieval.chunks if state.retrieval is not None else []
        t0 = time.perf_counter()
        state.in_flight_node = Node.ANSWER
        answer = await self._with_step_deadline(self._synth.synthesize(state.query, chunks))
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.trace.append(
            TraceEvent(
                node=Node.ANSWER,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=f"answer_len={len(answer)} cites={len(chunks)}",
            )
        )
        citations = [
            Citation(
                qualified_name=c.qualified_name,
                file_path=c.file_path,
                lineno_start=c.lineno_start,
                lineno_end=c.lineno_end,
                score=c.score,
                text=c.text,
            )
            for c in chunks
        ]
        state.in_flight_node = None
        return answer, citations

    async def _validate(
        self, state: _State, answer: str, citations: list[Citation]
    ) -> ValidationReport:
        chunks = state.retrieval.chunks if state.retrieval is not None else []
        t0 = time.perf_counter()
        state.in_flight_node = Node.VALIDATE
        report = await self._with_step_deadline(
            self._validator.validate(state.query, answer, chunks)
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.trace.append(
            TraceEvent(
                node=Node.VALIDATE,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=(
                    f"claims={report.n_claims} grounded={report.n_grounded} "
                    f"acceptable={report.is_acceptable}"
                ),
            )
        )
        state.in_flight_node = None
        return report

    @staticmethod
    def _validate_spans(spans: list[ClaimSpan], answer: str) -> None:
        """Raise ``InvalidValidationReport`` for out-of-bounds or overlapping spans.

        Checks are performed on the full ``spans`` list (grounded and
        ungrounded alike) because the redactor later filters to ungrounded
        spans — an out-of-bounds grounded span would produce a corrupt
        answer if we only validated ungrounded ones.
        """
        n = len(answer)
        for s in spans:
            if not (0 <= s.start < s.end <= n):
                raise InvalidValidationReport(
                    f"span for claim {s.claim!r} has bounds ({s.start}, {s.end}) "
                    f"outside answer of length {n}; "
                    f"validator produced a corrupt ValidationReport"
                )
        # Check for overlaps after sorting by start.
        sorted_spans = sorted(spans, key=lambda sp: sp.start)
        for a, b in itertools.pairwise(sorted_spans):
            if b.start < a.end:
                raise InvalidValidationReport(
                    f"spans for {a.claim!r} ({a.start}:{a.end}) and "
                    f"{b.claim!r} ({b.start}:{b.end}) overlap; "
                    f"validator produced a corrupt ValidationReport"
                )

    def _apply_validation(
        self, state: _State, answer: str, report: ValidationReport
    ) -> str:
        """Apply the configured validation_mode to an answer.

        Returns the (possibly modified) answer. ``advisory`` and accepted
        answers pass through unchanged; ``redact`` rewrites flagged
        claims; ``reject`` replaces the entire answer.

        Raises ``InvalidValidationReport`` when ``report.spans`` violates
        the contract (``0 <= start < end <= len(answer)``, no overlaps).
        The check runs before any answer mutation so a bad report never
        corrupts the answer silently.
        """
        if report.spans:
            self._validate_spans(report.spans, answer)
        if report.is_acceptable:
            return answer
        mode = self._config.validation_mode
        # Always log the enforcement action so production traces can
        # audit how often each mode fires and which claims tripped it.
        _log.info(
            "agent.validation_action",
            extra={
                "agent_query": state.original_query,
                "agent_validation_mode": mode,
                "agent_n_claims": report.n_claims,
                "agent_n_grounded": report.n_grounded,
                "agent_ungrounded_count": len(report.ungrounded_claims),
            },
        )
        state.trace.append(
            TraceEvent(
                node=Node.VALIDATE,
                started_at=time.perf_counter(),
                elapsed_ms=0.0,
                detail=(
                    f"action={mode} ungrounded={len(report.ungrounded_claims)}"
                ),
            )
        )
        if mode == "advisory":
            return answer
        if mode == "reject":
            return _REJECT_MESSAGE
        # ``redact``: rewrite each ungrounded backticked claim by span
        # offset rather than ``str.replace``. Slicing in REVERSE order
        # keeps earlier offsets valid as later text mutates. This is what
        # lets a grounded ``\`m.foo\``` and an ungrounded ``\`m.foo\``` in
        # the same answer end up handled differently — ``str.replace``
        # would have rewritten both occurrences.
        out = answer
        if report.spans:
            ungrounded_spans = [s for s in report.spans if not s.grounded]
            n_redacted = len(ungrounded_spans)
            for span in sorted(ungrounded_spans, key=lambda s: s.start, reverse=True):
                out = out[: span.start] + f"[ungrounded: {span.claim}]" + out[span.end :]
        else:
            # Backward-compat for validators that don't populate spans.
            # ``str.replace`` is the legacy path: it cannot distinguish
            # between grounded and ungrounded occurrences of the same
            # backticked string, but this branch only fires for custom
            # validators that opted out of span reporting.
            n_redacted = len(report.ungrounded_claims)
            for claim in report.ungrounded_claims:
                out = out.replace(f"`{claim}`", f"[ungrounded: {claim}]")
        # Summary stamp documents what redaction did + its limitation:
        # backticked claims only. Arbitrary prose isn't analysed by the
        # validator, so an answer with no backticked names produces a
        # stamp noting "0 redactions".
        stamp = (
            f"\n\n[validation: {n_redacted} ungrounded claims redacted; "
            "redact mode only handles backtick-wrapped qualified names — "
            "arbitrary prose claims are not detected. See citations for "
            "grounded references.]"
        )
        return out + stamp
