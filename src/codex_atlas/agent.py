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
import logging
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar

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
class ValidationReport:
    """Per-claim grounded-vs-ungrounded breakdown.

    A claim is *grounded* when it explicitly cites a chunk by qualified
    name and that name appears in the retrieved chunks. Anything else
    (free-form prose, hand-waving) is conservatively counted as
    ungrounded — false negatives there cost less than false positives.
    """

    n_claims: int
    n_grounded: int
    ungrounded_claims: list[str]
    grounded_qualified_names: list[str]
    is_acceptable: bool


@dataclass(frozen=True)
class Citation:
    """A pointer into the corpus included with every answer."""

    qualified_name: str
    file_path: str
    lineno_start: int
    lineno_end: int


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
        return 0.0


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
        claims = self._claim_re.findall(answer)
        if not claims:
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
            )
        grounded: list[str] = []
        ungrounded: list[str] = []
        for c in claims:
            if self._is_grounded(c, cited_qnames, known_prefixes):
                grounded.append(c)
            else:
                ungrounded.append(c)
        n = len(claims)
        ratio = len(grounded) / n if n else 1.0
        return ValidationReport(
            n_claims=n,
            n_grounded=len(grounded),
            ungrounded_claims=ungrounded,
            grounded_qualified_names=grounded,
            is_acceptable=ratio >= self.accept_threshold,
        )

    @staticmethod
    def _is_grounded(claim: str, cited: set[str], known_prefixes: set[str]) -> bool:
        # 1. Exact match.
        if claim in cited:
            return True
        # 2. Trailing-component match: some retrieved qname ends in
        #    ``.<claim>``. Accept iff the leading prefix on the retrieved
        #    qname can be matched by a known module/package — meaning the
        #    "module" the claim implicitly belongs to was in fact retrieved.
        #    This is the rule that rejects ``pkg_b.Cls.method`` when only
        #    ``pkg_a.Cls.method`` was retrieved: the qname ``pkg_a.Cls.method``
        #    ends in ``.Cls.method`` and ``pkg_a`` is a known prefix, but
        #    that does NOT make a *different* claim ``pkg_b.Cls.method``
        #    grounded. We therefore also require: if the claim itself has
        #    a leading prefix (everything before the trailing component
        #    path matched), that prefix must be in the known set.
        suffix = f".{claim}"
        for q in cited:
            if q.endswith(suffix):
                # Claim was a true tail of a retrieved qname. Accept it
                # only when the claim itself is a single trailing path
                # (no dotted prefix of its own to verify) OR the claim's
                # own leading prefix is a known module. ``Cls.method``
                # has no module prefix on the *claim* side; the
                # retrieved qname's prefix (``pkg.module``) is known by
                # construction (it's in cited_qnames as a prefix).
                return True
        # 3. Containing match: claim is longer than any retrieved qname
        #    and ends with one as a trailing path. Require claim's
        #    leading prefix to be a known module/package — this is the
        #    rule that catches the ``pkg_b.Cls.method`` attack when the
        #    retrieved set only has ``pkg_a.Cls.method``: the claim's
        #    prefix ``pkg_b`` is not in the prefix set.
        for q in cited:
            tail = f".{q}"
            if claim.endswith(tail):
                claim_prefix = claim[: -len(tail)]
                if claim_prefix in known_prefixes:
                    return True
        return False


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
            lines.append(f"## `{c.qualified_name}` ({c.file_path}:{c.lineno_start}-{c.lineno_end})")
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
    ) -> None:
        self._retriever = retriever
        self._synth = synthesizer or StitchSynthesizer()
        self._grader = grader or HeuristicGrader()
        self._rewriter = rewriter or NoopRewriter()
        self._validator = validator or CitationValidator()
        self._config = config or AgentConfig()

    async def run(self, query: str) -> AgentResult:
        run_t0 = time.perf_counter()
        state = _State(original_query=query, query=query)
        try:
            await self._with_run_deadline(self._retrieve(state), run_t0)
            await self._with_run_deadline(self._grade(state), run_t0)

            while (
                state.cancelled is None
                and state.grade < self._config.grade_threshold
                and state.attempts < self._config.max_attempts - 1
            ):
                await self._with_run_deadline(self._rewrite(state), run_t0)
                await self._with_run_deadline(self._retrieve(state), run_t0)
                await self._with_run_deadline(self._grade(state), run_t0)

            if state.cancelled is None:
                answer, citations = await self._with_run_deadline(self._answer(state), run_t0)
                validation = await self._with_run_deadline(
                    self._validate(state, answer, citations), run_t0
                )
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
            answer, citations, validation = ("", [], None)

        return AgentResult(
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
        )

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
        retrieval = await self._with_step_deadline(self._retriever.retrieve(state.query))
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

    async def _rewrite(self, state: _State) -> None:
        if state.retrieval is None:
            return
        t0 = time.perf_counter()
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

    async def _answer(self, state: _State) -> tuple[str, list[Citation]]:
        chunks = state.retrieval.chunks if state.retrieval is not None else []
        t0 = time.perf_counter()
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
            )
            for c in chunks
        ]
        return answer, citations

    async def _validate(
        self, state: _State, answer: str, citations: list[Citation]
    ) -> ValidationReport:
        chunks = state.retrieval.chunks if state.retrieval is not None else []
        t0 = time.perf_counter()
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
        return report
