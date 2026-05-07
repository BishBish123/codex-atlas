"""Real LangGraph backend for codex-atlas.

Builds a ``StateGraph`` with the same seven nodes as the hand-rolled
``Agent`` (classify, retrieve, grade, rewrite_query, answer, validate,
cancel) and wires identical conditional edges.

Enable with ``ATLAS_AGENT_BACKEND=langgraph`` — the default hand-rolled
path is unchanged.

Usage::

    import os
    os.environ["ATLAS_AGENT_BACKEND"] = "langgraph"
    from codex_atlas import make_agent
    agent = make_agent(retriever)
    result = await agent.run("who calls find_callers")

Install the optional dependency::

    uv sync --extra real   # or: pip install "codex-atlas[real]"

If ``langgraph`` is not installed, ``LangGraphAgent`` raises a
``RuntimeError`` at construction time with an actionable install hint.
The rest of the application is unaffected.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from codex_atlas.agent import (
    _REJECT_MESSAGE,
    AgentConfig,
    AgentResult,
    CancelReason,
    Citation,
    CitationValidator,
    ClaimSpan,
    Grader,
    HeuristicGrader,
    InvalidValidationReport,
    Node,
    NoopRewriter,
    QueryRewriter,
    Synthesizer,
    ToolCall,
    TraceEvent,
    ValidationReport,
    Validator,
)
from codex_atlas.observability.langfuse import LangfuseTracer, NullTracer, make_tracer
from codex_atlas.retriever import RetrievalResult, Retriever, Route
from codex_atlas.store import StoredChunk

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import guard
# ---------------------------------------------------------------------------


def _require_langgraph() -> Any:
    """Import langgraph or raise a clean RuntimeError with install hint."""
    try:
        import langgraph  # noqa: PLC0415
        return langgraph
    except ImportError as exc:
        raise RuntimeError(
            "langgraph is not installed. "
            "Install it with: uv sync --extra real  "
            "(or: pip install 'langgraph>=0.2')"
        ) from exc


# ---------------------------------------------------------------------------
# LangGraph state dataclass
# ---------------------------------------------------------------------------


@dataclass
class _LGState:
    """Mutable state threaded through every LangGraph node."""

    original_query: str
    query: str
    attempts: int = 0
    retrieval: RetrievalResult | None = None
    grade: float = 0.0
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    validation: ValidationReport | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    cancelled: CancelReason | None = None
    route_override: Route | None = None
    # flow control flags
    should_rewrite: bool = False
    done: bool = False


# ---------------------------------------------------------------------------
# State ↔ dict helpers (LangGraph works with dicts at runtime)
# ---------------------------------------------------------------------------


def _state_to_dict(s: _LGState) -> dict[str, Any]:
    return {
        "original_query": s.original_query,
        "query": s.query,
        "attempts": s.attempts,
        "retrieval": s.retrieval,
        "grade": s.grade,
        "answer": s.answer,
        "citations": s.citations,
        "validation": s.validation,
        "trace": s.trace,
        "tool_calls": s.tool_calls,
        "cancelled": s.cancelled,
        "route_override": s.route_override,
        "should_rewrite": s.should_rewrite,
        "done": s.done,
    }


def _dict_to_state(d: dict[str, Any]) -> _LGState:
    s = _LGState(
        original_query=d["original_query"],
        query=d["query"],
    )
    s.attempts = d.get("attempts", 0)
    s.retrieval = d.get("retrieval")
    s.grade = d.get("grade", 0.0)
    s.answer = d.get("answer", "")
    s.citations = d.get("citations", [])
    s.validation = d.get("validation")
    s.trace = d.get("trace", [])
    s.tool_calls = d.get("tool_calls", [])
    s.cancelled = d.get("cancelled")
    s.route_override = d.get("route_override")
    s.should_rewrite = d.get("should_rewrite", False)
    s.done = d.get("done", False)
    return s


# ---------------------------------------------------------------------------
# Validation helpers (mirrors Agent._validate_spans / _apply_validation)
# ---------------------------------------------------------------------------


def _validate_spans(spans: list[ClaimSpan], answer: str) -> None:
    """Raise ``InvalidValidationReport`` for out-of-bounds or overlapping spans."""
    n = len(answer)
    for s in spans:
        if not (0 <= s.start < s.end <= n):
            raise InvalidValidationReport(
                f"span for claim {s.claim!r} has bounds ({s.start}, {s.end}) "
                f"outside answer of length {n}; "
                f"validator produced a corrupt ValidationReport"
            )
    sorted_spans = sorted(spans, key=lambda sp: sp.start)
    for a, b in itertools.pairwise(sorted_spans):
        if b.start < a.end:
            raise InvalidValidationReport(
                f"spans for {a.claim!r} ({a.start}:{a.end}) and "
                f"{b.claim!r} ({b.start}:{b.end}) overlap; "
                f"validator produced a corrupt ValidationReport"
            )


def _apply_validation(
    answer: str,
    report: ValidationReport,
    config: AgentConfig,
    original_query: str,
    trace: list[TraceEvent],
) -> str:
    """Apply the configured validation_mode to an answer.

    Mirrors ``Agent._apply_validation`` without a dependency on ``_State``.
    """
    if report.spans:
        _validate_spans(report.spans, answer)
    if report.is_acceptable:
        return answer
    mode = config.validation_mode
    _log.info(
        "agent.validation_action",
        extra={
            "agent_query": original_query,
            "agent_validation_mode": mode,
            "agent_n_claims": report.n_claims,
            "agent_n_grounded": report.n_grounded,
            "agent_ungrounded_count": len(report.ungrounded_claims),
        },
    )
    trace.append(
        TraceEvent(
            node=Node.VALIDATE,
            started_at=time.perf_counter(),
            elapsed_ms=0.0,
            detail=f"action={mode} ungrounded={len(report.ungrounded_claims)}",
        )
    )
    if mode == "advisory":
        return answer
    if mode == "reject":
        return _REJECT_MESSAGE
    # redact
    out = answer
    if report.spans:
        ungrounded_spans = [s for s in report.spans if not s.grounded]
        n_redacted = len(ungrounded_spans)
        for span in sorted(ungrounded_spans, key=lambda s: s.start, reverse=True):
            out = out[: span.start] + f"[ungrounded: {span.claim}]" + out[span.end :]
    else:
        n_redacted = len(report.ungrounded_claims)
        for claim in report.ungrounded_claims:
            out = out.replace(f"`{claim}`", f"[ungrounded: {claim}]")
    stamp = (
        f"\n\n[validation: {n_redacted} ungrounded claims redacted; "
        "redact mode only handles backtick-wrapped qualified names — "
        "arbitrary prose claims are not detected. See citations for "
        "grounded references.]"
    )
    return out + stamp


# ---------------------------------------------------------------------------
# LangGraphAgent
# ---------------------------------------------------------------------------


class LangGraphAgent:
    """Agent backed by a real LangGraph ``StateGraph``.

    Implements the same public interface as ``Agent``:
    ``async def run(query, *, route_override=None) -> AgentResult``.

    Reuses all Protocol implementations (``Retriever``, ``Grader``,
    ``Synthesizer``, ``Validator``) — the swap is purely orchestration.
    """

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
        # Trigger the soft-import check eagerly so callers get a clean error
        # at construction time, not at the first ``run()`` call.
        _require_langgraph()
        self._retriever = retriever
        if synthesizer is not None:
            self._synth = synthesizer
        else:
            from codex_atlas.synthesis import make_synthesizer  # noqa: PLC0415
            self._synth = make_synthesizer()
        self._grader: Grader = grader or HeuristicGrader()
        self._rewriter: QueryRewriter = rewriter or NoopRewriter()
        self._validator: Validator = validator or CitationValidator()
        self._config = config or AgentConfig()
        self._tracer: LangfuseTracer | NullTracer = (
            tracer if tracer is not None else make_tracer()
        )

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _node_retrieve(self, state: _LGState) -> _LGState:
        t0 = time.perf_counter()
        retrieval = await self._retriever.retrieve(
            state.query, route_override=state.route_override
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
                    f"chunks={len(retrieval.chunks)} "
                    f"extras={len(retrieval.extra_qualified_names)}"
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
        return state

    async def _node_grade(self, state: _LGState) -> _LGState:
        if state.retrieval is None:
            state.grade = 0.0
            state.should_rewrite = False
            return state
        t0 = time.perf_counter()
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
        state.should_rewrite = (
            state.grade < self._config.grade_threshold
            and state.attempts < self._config.max_attempts - 1
        )
        return state

    async def _node_rewrite(self, state: _LGState) -> _LGState:
        if state.retrieval is None:
            return state
        t0 = time.perf_counter()
        chunks = state.retrieval.chunks
        new_query = await self._rewriter.rewrite(state.original_query, chunks)
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
        state.should_rewrite = False
        return state

    async def _node_answer(self, state: _LGState) -> _LGState:
        chunks: list[StoredChunk] = (
            state.retrieval.chunks if state.retrieval is not None else []
        )
        t0 = time.perf_counter()
        answer = await self._synth.synthesize(state.query, chunks)
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.answer = answer
        state.trace.append(
            TraceEvent(
                node=Node.ANSWER,
                started_at=t0,
                elapsed_ms=elapsed,
                detail=f"answer_len={len(answer)} cites={len(chunks)}",
            )
        )
        state.citations = [
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
        return state

    async def _node_validate(self, state: _LGState) -> _LGState:
        chunks: list[StoredChunk] = (
            state.retrieval.chunks if state.retrieval is not None else []
        )
        t0 = time.perf_counter()
        report = await self._validator.validate(state.query, state.answer, chunks)
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
        state.validation = report
        state.done = True
        return state

    async def _node_cancel(self, state: _LGState) -> _LGState:
        state.trace.append(
            TraceEvent(
                node=Node.CANCEL,
                started_at=time.perf_counter(),
                elapsed_ms=0.0,
                detail=f"reason={state.cancelled}",
            )
        )
        state.done = True
        return state

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:  # returns a compiled langgraph graph
        from langgraph.graph import END, StateGraph  # noqa: PLC0415

        # Wrap dataclass-based async nodes to dict-in / dict-out for LangGraph.
        def _wrap(node_fn: Any) -> Any:
            async def _wrapped(state: dict[str, Any]) -> dict[str, Any]:
                lg_state = _dict_to_state(state)
                updated = await node_fn(lg_state)
                return _state_to_dict(updated)
            return _wrapped

        def _route_after_grade(state: dict[str, Any]) -> str:
            if state.get("should_rewrite"):
                return "rewrite_query"
            return "answer"

        builder: Any = StateGraph(dict)

        builder.add_node("retrieve", _wrap(self._node_retrieve))
        builder.add_node("grade", _wrap(self._node_grade))
        builder.add_node("rewrite_query", _wrap(self._node_rewrite))
        builder.add_node("answer", _wrap(self._node_answer))
        builder.add_node("validate", _wrap(self._node_validate))
        builder.add_node("cancel", _wrap(self._node_cancel))

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            _route_after_grade,
            {"rewrite_query": "rewrite_query", "answer": "answer"},
        )
        # Conditional edge from grade leads back to retrieve via rewrite_query.
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("answer", "validate")
        builder.add_edge("validate", END)
        builder.add_edge("cancel", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public interface (same as Agent)
    # ------------------------------------------------------------------

    async def run(
        self, query: str, *, route_override: Route | None = None
    ) -> AgentResult:
        """Run the LangGraph state machine; return the same ``AgentResult``
        shape as the hand-rolled ``Agent``."""
        graph = self._build_graph()
        trace_id = self._tracer.start_run(query)

        initial: dict[str, Any] = _state_to_dict(
            _LGState(
                original_query=query,
                query=query,
                route_override=route_override,
            )
        )

        final_dict: dict[str, Any] = await graph.ainvoke(initial)
        state = _dict_to_state(final_dict)

        # Emit trace events and tool calls to observability plane.
        for ev in state.trace:
            self._tracer.record_event(trace_id, ev)
        for tc in state.tool_calls:
            self._tracer.record_event(trace_id, tc)
        if state.validation is not None:
            self._tracer.record_validation(trace_id, state.validation)

        # Apply validation policy (mirrors Agent._apply_validation).
        answer = state.answer
        if state.validation is not None and not state.cancelled:
            answer = _apply_validation(
                answer,
                state.validation,
                self._config,
                state.original_query,
                state.trace,
            )

        result = AgentResult(
            query=state.original_query,
            final_query=state.query,
            answer=answer,
            citations=state.citations,
            route=(
                state.retrieval.route
                if state.retrieval is not None
                else Route.LOOKUP
            ),
            grade=state.grade,
            attempts=state.attempts + 1,
            trace=state.trace,
            tool_calls=state.tool_calls,
            validation=state.validation,
            cancelled=state.cancelled,
            cancelled_node=None,
        )
        self._tracer.finish_run(trace_id, result)
        return result
