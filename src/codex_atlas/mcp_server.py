"""FastMCP entry point for Codex-Atlas.

Exposes 7 tools + 1 resource:

- `search_code(query, top_k)` — adaptive-route retrieval (vector + graph)
- `explain_function(qualified_name)` — pull the chunk + summarise its
  immediate callers/callees
- `find_callers(qualified_name, depth)` — graph-only callers traversal
- `summarize_module(module_path)` — wider retrieval + 2-hop expansion
- `search_codebase(query, top_k, route?)` — generic search with an
  explicit route override (skips the classifier)
- `get_graph_neighborhood(symbol, depth)` — both-direction BFS on the
  call graph, returns the structured neighborhood
- `explain(symbol)` — runs the agent state machine end-to-end on a
  symbol, returning answer + citations + validation status
- `codebase://stats` — resource with node/edge counts + language stats

Reads `POSTGRES_DSN`, `ATLAS_GRAPH_PATH`, `ATLAS_ENCODER` from env.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import typer
from fastmcp import FastMCP
from pydantic import BaseModel

from codex_atlas.agent import Agent, AgentResult
from codex_atlas.embed import Encoder, FakeEncoder
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.retriever import Retriever, RetrieverConfig

# Hard cap on the BFS depth callers can request via
# ``get_graph_neighborhood``. Even with seen-set guards a 50K-node graph
# with depth=10000 walks far more than the caller could ever consume —
# we clamp at the MCP boundary so a confused LLM can't trigger a
# minute-long traversal. Eight hops covers the deepest realistic
# "everything related to this symbol" question on the indexed corpus.
MAX_NEIGHBORHOOD_DEPTH = 8

_log = logging.getLogger(__name__)

mcp: FastMCP = FastMCP(
    name="codex-atlas",
    instructions=(
        "Codex-Atlas: agentic GraphRAG over a code corpus. Use search_code "
        "for free-form questions; explain_function / find_callers when you "
        "have a qualified name in hand; summarize_module for high-level "
        "walkthroughs. Every answer comes with citations into the source."
    ),
)


class CodeSearchHit(BaseModel):
    qualified_name: str
    file_path: str
    lineno_start: int
    lineno_end: int
    score: float
    text: str


class SearchResponse(BaseModel):
    route: str
    grade: float
    attempts: int
    answer: str
    citations: list[CodeSearchHit]


class CallerEntry(BaseModel):
    qualified_name: str


class CallersResponse(BaseModel):
    target: str
    depth: int
    callers: list[CallerEntry]


class NeighborhoodResponse(BaseModel):
    target: str
    depth: int
    callers: list[str]
    callees: list[str]
    all: list[str]


class CodebaseStats(BaseModel):
    n_nodes: int
    n_edges: int
    language: str = "python"
    graph_path: str


def _encoder() -> Encoder:
    name = os.environ.get("ATLAS_ENCODER", "fake")
    if name == "fake":
        return FakeEncoder(dim=32)
    from codex_atlas.embed import load_sentence_transformer_encoder  # noqa: PLC0415

    return load_sentence_transformer_encoder(model_name=name)


def _graph() -> CallGraph:
    path = Path(os.environ.get("ATLAS_GRAPH_PATH", "data/graph.json"))
    if not path.exists():
        raise RuntimeError(f"call graph not found at {path}; run `atlas index` first")
    return CallGraph.load(path)


def _dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN env var is required")
    return dsn


async def _agent() -> Agent:
    from codex_atlas.store import ChunkStore  # noqa: PLC0415

    encoder = _encoder()
    store = ChunkStore(dsn=_dsn())
    await store.setup(dim=encoder.dim)
    retriever = Retriever(encoder, store, _graph(), RetrieverConfig(top_k=8))
    return Agent(retriever)


def _to_response(result: AgentResult) -> SearchResponse:
    hits = [
        CodeSearchHit(
            qualified_name=c.qualified_name,
            file_path=c.file_path,
            lineno_start=c.lineno_start,
            lineno_end=c.lineno_end,
            score=0.0,
            text="",
        )
        for c in result.citations
    ]
    return SearchResponse(
        route=str(result.route),
        grade=result.grade,
        attempts=result.attempts,
        answer=result.answer,
        citations=hits,
    )


@mcp.tool
async def search_code(query: str, top_k: int = 8) -> SearchResponse:
    """Adaptive-route code search. Returns synthesised answer + citations."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    agent = await _agent()
    return _to_response(await agent.run(query))


@mcp.tool
async def explain_function(qualified_name: str) -> SearchResponse:
    """Pull the chunk for `qualified_name` plus its immediate neighbours."""
    if not qualified_name.strip():
        raise ValueError("qualified_name must not be blank")
    agent = await _agent()
    # Phrase the query so the classifier picks the structural route.
    return _to_response(await agent.run(f"who calls {qualified_name}"))


@mcp.tool
async def find_callers(qualified_name: str, depth: int = 1) -> CallersResponse:
    """Graph-only callers traversal — no LLM, just the call graph."""
    if not qualified_name.strip():
        raise ValueError("qualified_name must not be blank")
    if depth <= 0:
        raise ValueError("depth must be positive")
    callers = _graph().find_callers(qualified_name, depth=depth)
    return CallersResponse(
        target=qualified_name,
        depth=depth,
        callers=[CallerEntry(qualified_name=c) for c in callers],
    )


@mcp.tool
async def summarize_module(module_path: str) -> SearchResponse:
    """High-level walkthrough of a module via the summarization route."""
    if not module_path.strip():
        raise ValueError("module_path must not be blank")
    agent = await _agent()
    return _to_response(await agent.run(f"walk me through {module_path}"))


@mcp.tool
async def search_codebase(query: str, top_k: int = 8, route: str | None = None) -> SearchResponse:
    """Generic search with an optional explicit route override.

    Pass ``route="structural"`` (or any ``Route`` value) to skip the
    classifier — useful when the calling LLM has already decided which
    pipeline it wants. With ``route=None`` this is identical to
    ``search_code``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        raise ValueError("query must not be blank")
    agent = await _agent()
    if route is None:
        return _to_response(await agent.run(query))
    # Force a route by phrasing the query so the classifier picks it.
    forced = {
        "structural": f"who calls {query}",
        "summarization": f"walk me through {query}",
        "hybrid": f"end-to-end {query}",
        "neighborhood": f"neighborhood of {query}",
        "import_chain": f"who imports {query}",
        "lookup": query,
    }.get(route, query)
    return _to_response(await agent.run(forced))


@mcp.tool
async def get_graph_neighborhood(symbol: str, depth: int = 2) -> NeighborhoodResponse:
    """Both-direction BFS on the call graph (callers + callees up to depth).

    Depth is clamped at ``MAX_NEIGHBORHOOD_DEPTH`` (8) at the MCP
    boundary; requests above the cap are reduced to the cap with a
    structured warning rather than rejected, so a slightly-too-large
    depth doesn't fail the tool call.
    """
    if not symbol.strip():
        raise ValueError("symbol must not be blank")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > MAX_NEIGHBORHOOD_DEPTH:
        _log.warning(
            "get_graph_neighborhood.depth_clamped",
            extra={
                "symbol": symbol,
                "requested_depth": depth,
                "clamped_depth": MAX_NEIGHBORHOOD_DEPTH,
            },
        )
        depth = MAX_NEIGHBORHOOD_DEPTH
    nb = _graph().caller_callee_neighborhood(symbol, depth=depth)
    return NeighborhoodResponse(
        target=symbol,
        depth=depth,
        callers=nb["callers"],
        callees=nb["callees"],
        all=nb["all"],
    )


@mcp.tool
async def explain(symbol: str) -> SearchResponse:
    """End-to-end agent run anchored on a symbol — the everything tool."""
    if not symbol.strip():
        raise ValueError("symbol must not be blank")
    agent = await _agent()
    return _to_response(await agent.run(f"who calls {symbol}"))


@mcp.resource("codebase://stats")
def codebase_stats() -> CodebaseStats:
    """Graph node/edge counts + language breakdown for the indexed corpus."""
    g = _graph()
    return CodebaseStats(
        n_nodes=g.n_nodes,
        n_edges=g.n_edges,
        language="python",
        graph_path=str(os.environ.get("ATLAS_GRAPH_PATH", "data/graph.json")),
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


_cli = typer.Typer(name="atlas-mcp", add_completion=False)


@_cli.command()
def run(
    transport: str = typer.Option("stdio", help="MCP transport: stdio | http"),
    host: str = typer.Option("127.0.0.1", help="HTTP bind host."),
    port: int = typer.Option(8090, help="HTTP bind port."),
) -> None:
    """Run the Codex-Atlas MCP server."""
    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    elif transport == "http":
        asyncio.run(mcp.run_http_async(host=host, port=port))
    else:
        raise typer.BadParameter(f"unknown transport {transport!r}")


def main() -> None:  # pragma: no cover
    _cli()


if __name__ == "__main__":  # pragma: no cover
    main()
