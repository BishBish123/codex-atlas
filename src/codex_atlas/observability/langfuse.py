"""Langfuse tracing adapter for codex-atlas.

The adapter is **env-key-gated**: when both ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are set the tracer attempts to import the
``langfuse`` SDK and initialise a live client. When either key is missing,
or when the SDK is not installed, all methods are no-ops and the rest of
the application continues unchanged.

Env vars
--------
LANGFUSE_PUBLIC_KEY  Required to enable tracing.
LANGFUSE_SECRET_KEY  Required to enable tracing.
LANGFUSE_HOST        Optional. Defaults to https://cloud.langfuse.com.

Typical usage
-------------
::

    from codex_atlas.observability.langfuse import make_tracer

    tracer = make_tracer()
    trace_id = tracer.start_run("what does Encoder do")
    tracer.record_event(trace_id, trace_event)
    tracer.record_validation(trace_id, report)
    tracer.finish_run(trace_id, result)

All calls are safe when env vars are absent — no exceptions, no side-effects.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codex_atlas.agent import AgentResult, ToolCall, TraceEvent, ValidationReport

_log = logging.getLogger(__name__)


class NullTracer:
    """No-op tracer used when env vars are absent or the SDK is not installed.

    Every method returns immediately without side-effects. The returned
    trace_id is a sentinel ``"null"`` string so callers can log it safely.
    """

    def start_run(self, query: str) -> str:
        return "null"

    def record_event(self, trace_id: str, event: Any) -> None:
        pass

    def record_validation(self, trace_id: str, report: Any) -> None:
        pass

    def finish_run(self, trace_id: str, result: Any) -> None:
        pass


class LangfuseTracer:
    """Langfuse-backed tracer.

    Constructed by :func:`make_tracer` when the required env vars are present.
    Falls back to no-op behaviour on any initialisation error (e.g. the
    ``langfuse`` package is not installed in the current venv).

    In **live mode** (SDK present and env vars set):

    * :meth:`start_run` — creates a Langfuse trace scoped to one agent run.
    * :meth:`record_event` — emits a child span for each ``TraceEvent`` or
      ``ToolCall`` attached as metadata.
    * :meth:`record_validation` — emits the ``ValidationReport`` result
      as a span so the Langfuse UI shows grounded vs ungrounded claim counts.
    * :meth:`finish_run` — attaches the final answer + citation count to the
      trace and calls ``langfuse.flush()`` so buffered events are delivered
      before the process exits.

    In **no-op mode** all methods return immediately without side-effects,
    matching the :class:`NullTracer` contract.
    """

    def __init__(self) -> None:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "")

        # Maps trace_id -> live trace object (v2/v3) or span handle (v4).
        self._traces: dict[str, Any] = {}

        if not (public_key and secret_key):
            _log.debug("langfuse.tracer_noop reason=env_vars_not_set")
            self._client: Any = None
            return

        try:
            import langfuse as _langfuse_mod  # type: ignore[import-untyped,unused-ignore]

            kwargs: dict[str, str] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            if host:
                kwargs["host"] = host

            # SDK v4 exposes get_client() as the preferred factory; fall
            # back to direct Langfuse() construction for SDK v2/v3 compat.
            if hasattr(_langfuse_mod, "get_client"):
                # v4: configure via env vars (already set) and fetch singleton.
                self._client = _langfuse_mod.get_client()
            else:
                # v2/v3: direct construction.
                self._client = _langfuse_mod.Langfuse(**kwargs)

            _log.info(
                "langfuse.tracer_live host=%s",
                host or "https://cloud.langfuse.com",
            )
        except ImportError:
            _log.warning(
                "langfuse.sdk_missing: "
                "install langfuse>=2.50 or codex-atlas[real] to enable tracing"
            )
            self._client = None
        except Exception as exc:
            _log.warning(
                "langfuse.init_failed exc_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            self._client = None

    @property
    def _live(self) -> bool:
        return self._client is not None

    def start_run(self, query: str) -> str:
        """Open a Langfuse trace for one agent run.

        Returns the trace ID string. In no-op mode returns ``"null"``.
        """
        if not self._live:
            return "null"
        try:
            # SDK v4 uses start_observation(); v2/v3 uses trace().
            if hasattr(self._client, "trace"):
                trace = self._client.trace(name="codex-atlas-run", input=query)
                trace_id: str = trace.id
            else:
                trace = self._client.start_observation(
                    name="codex-atlas-run", as_type="span"
                )
                trace.update(input=query)
                trace_id = getattr(trace, "id", "unknown")
            self._traces[trace_id] = trace
            return trace_id
        except Exception as exc:
            _log.warning(
                "langfuse.start_run_failed exc_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return "null"

    def record_event(self, trace_id: str, event: TraceEvent | ToolCall) -> None:
        """Emit a Langfuse child span for one agent step or tool call.

        Accepts both :class:`~codex_atlas.agent.TraceEvent` (state machine
        node transitions) and :class:`~codex_atlas.agent.ToolCall` (retriever
        invocations). In no-op mode or when the trace is absent returns
        immediately.
        """
        if not self._live:
            return
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        try:
            # Import here to avoid circular imports at module level.
            from codex_atlas.agent import ToolCall

            if isinstance(event, ToolCall):
                name = f"tool_call:{event.route}"
                metadata: dict[str, object] = {
                    "query": event.query,
                    "route": str(event.route),
                    "n_chunks": event.n_chunks,
                    "elapsed_ms": event.elapsed_ms,
                    "confidence": event.confidence,
                }
                output: dict[str, object] = {
                    "n_chunks": event.n_chunks,
                    "elapsed_ms": event.elapsed_ms,
                }
            else:
                # TraceEvent
                name = f"node:{event.node}"
                metadata = {
                    "node": str(event.node),
                    "elapsed_ms": event.elapsed_ms,
                    "detail": event.detail,
                }
                output = {
                    "node": str(event.node),
                    "elapsed_ms": event.elapsed_ms,
                }

            if hasattr(trace, "span"):
                # v2/v3 API
                span = trace.span(name=name, metadata=metadata)
                span.update(output=output)
                span.end()
            else:
                # v4 API
                obs = trace.start_observation(name=name, as_type="span")
                obs.update(metadata=metadata, output=output)
                obs.end()
        except Exception as exc:
            _log.warning(
                "langfuse.record_event_failed exc_type=%s error=%s",
                type(exc).__name__,
                exc,
            )

    def record_validation(self, trace_id: str, report: ValidationReport) -> None:
        """Emit the validation result as a Langfuse span.

        In no-op mode or when the trace is absent returns immediately.
        """
        if not self._live:
            return
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        try:
            metadata: dict[str, object] = {
                "n_claims": report.n_claims,
                "n_grounded": report.n_grounded,
                "ungrounded_claims": report.ungrounded_claims,
                "is_acceptable": report.is_acceptable,
            }
            output: dict[str, object] = {
                "is_acceptable": report.is_acceptable,
                "n_claims": report.n_claims,
                "n_grounded": report.n_grounded,
            }
            if hasattr(trace, "span"):
                span = trace.span(name="validation", metadata=metadata)
                span.update(output=output)
                span.end()
            else:
                obs = trace.start_observation(name="validation", as_type="span")
                obs.update(metadata=metadata, output=output)
                obs.end()
        except Exception as exc:
            _log.warning(
                "langfuse.record_validation_failed exc_type=%s error=%s",
                type(exc).__name__,
                exc,
            )

    def finish_run(self, trace_id: str, result: AgentResult) -> None:
        """Finalise the Langfuse trace and flush buffered events.

        Attaches the final answer length, citation count, grade, and
        attempts to the trace so the Langfuse UI shows a summary.
        In no-op mode or when the trace is absent returns immediately.
        """
        if not self._live:
            return
        trace = self._traces.pop(trace_id, None)
        if trace is None:
            return
        try:
            summary = {
                "answer_len": len(result.answer),
                "citations": len(result.citations),
                "grade": result.grade,
                "attempts": result.attempts,
                "route": str(result.route),
                "cancelled": str(result.cancelled) if result.cancelled else None,
            }
            if hasattr(trace, "update"):
                trace.update(output=summary)
            if hasattr(trace, "end"):
                trace.end()
            # Flush so short-lived CLI invocations deliver buffered events.
            if hasattr(self._client, "flush"):
                self._client.flush()
        except Exception as exc:
            _log.warning(
                "langfuse.finish_run_failed exc_type=%s error=%s",
                type(exc).__name__,
                exc,
            )


def make_tracer() -> LangfuseTracer | NullTracer:
    """Factory: return a :class:`LangfuseTracer` when env vars are set, else a :class:`NullTracer`.

    The returned tracer satisfies the same interface in both cases so
    callers need no conditional logic.

    Usage::

        tracer = make_tracer()
        trace_id = tracer.start_run(query)
        tracer.record_event(trace_id, event)
        tracer.record_validation(trace_id, report)
        tracer.finish_run(trace_id, result)
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (public_key and secret_key):
        return NullTracer()
    return LangfuseTracer()
