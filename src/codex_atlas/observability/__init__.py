"""Observability adapters for codex-atlas.

Exports the Langfuse tracer, NullTracer, and the ``make_tracer`` factory.
All are env-key-gated: tracing is active only when ``LANGFUSE_PUBLIC_KEY``
and ``LANGFUSE_SECRET_KEY`` are set; otherwise every call is a no-op.
"""

from codex_atlas.observability.langfuse import LangfuseTracer, NullTracer, make_tracer

__all__ = ["LangfuseTracer", "NullTracer", "make_tracer"]
