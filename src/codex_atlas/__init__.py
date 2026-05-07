"""Codex-Atlas: agentic GraphRAG over a codebase, exposed as an MCP server."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codex_atlas.agent import Agent, AgentConfig
    from codex_atlas.agent_langgraph import LangGraphAgent
    from codex_atlas.observability.langfuse import LangfuseTracer, NullTracer
    from codex_atlas.retriever import Retriever

try:
    __version__ = version("codex-atlas")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"


def make_agent(
    retriever: Retriever,
    *,
    backend: str | None = None,
    config: AgentConfig | None = None,
    tracer: LangfuseTracer | NullTracer | None = None,
) -> Agent | LangGraphAgent:
    """Factory that returns the right agent backend.

    Backend selection (first-wins):
    1. ``backend`` argument (``"hand-rolled"`` | ``"langgraph"``)
    2. ``ATLAS_AGENT_BACKEND`` env var
    3. Default: ``"hand-rolled"`` (the existing ``Agent`` — no extra deps)

    The ``"langgraph"`` backend requires ``langgraph>=0.2`` to be installed.
    If it is not, a ``RuntimeError`` is raised with an install hint.

    Example::

        from codex_atlas import make_agent
        agent = make_agent(retriever)                    # hand-rolled (default)
        agent = make_agent(retriever, backend="langgraph")  # LangGraph

    Or via env var::

        ATLAS_AGENT_BACKEND=langgraph uv run atlas ask "..."
    """
    resolved = backend or os.environ.get("ATLAS_AGENT_BACKEND", "hand-rolled")
    if resolved == "langgraph":
        from codex_atlas.agent_langgraph import LangGraphAgent  # noqa: PLC0415

        return LangGraphAgent(retriever, config=config, tracer=tracer)
    if resolved in ("hand-rolled", ""):
        from codex_atlas.agent import Agent  # noqa: PLC0415

        return Agent(retriever, config=config, tracer=tracer)
    raise RuntimeError(
        f"Unknown ATLAS_AGENT_BACKEND {resolved!r}. "
        "Valid values: 'hand-rolled' (default), 'langgraph'."
    )


__all__ = ["__version__", "make_agent"]
