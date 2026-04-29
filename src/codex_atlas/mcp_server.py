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

from codex_atlas.agent import Agent, AgentConfig, AgentResult, CancelReason
from codex_atlas.embed import Encoder, FakeEncoder
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.retriever import Retriever, RetrieverConfig, Route

# Hard cap on the BFS depth callers can request via
# ``get_graph_neighborhood``. Even with seen-set guards a 50K-node graph
# with depth=10000 walks far more than the caller could ever consume —
# we clamp at the MCP boundary so a confused LLM can't trigger a
# minute-long traversal. Eight hops covers the deepest realistic
# "everything related to this symbol" question on the indexed corpus.
MAX_NEIGHBORHOOD_DEPTH = 8

# Hard caps applied in ``_to_response`` so a single runaway answer or a
# very wide retrieval can never flood an MCP client's context window.
# 200 000 bytes ≈ 150k tokens at 1.3 bytes/token — generous but bounded.
MAX_MCP_ANSWER_BYTES: int = 200_000
# Cap the citations list so response serialisation stays O(1) in size
# for the common "explain a large module" queries that pull 50+ chunks.
MAX_MCP_CITATIONS: int = 20

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
    # ``cancelled`` is the agent's ``CancelReason`` rendered as a string
    # (currently one of "timeout" or "external"), or ``None`` when the
    # run completed normally. Surfacing this at the MCP boundary lets
    # clients distinguish a cancelled run (empty answer + cancelled set)
    # from a normal "no results" answer (empty answer + cancelled None).
    cancelled: str | None = None


class AgentTimeoutResponse(BaseModel):
    """Returned by any MCP tool when the agent run is cancelled due to a timeout.

    ``error`` is always ``"agent_timeout"``.
    ``phase`` is the last node that was executing when the deadline fired
    (e.g. ``"retrieve"``, ``"grade"``, ``"answer"``).  Downstream clients
    can key on ``error == "agent_timeout"`` to distinguish a timeout from
    any other failure without parsing the message string.
    """

    error: str = "agent_timeout"
    phase: str


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
    """Load the call graph, reusing the process-wide cache populated by
    ``_agent()`` (or this function on first call).

    Three MCP tools — ``find_callers``, ``get_graph_neighborhood``, and
    ``codebase_stats`` — call ``_graph()`` directly without first going
    through ``_agent()``. Without this short-circuit each invocation
    paid a multi-second ``CallGraph.load`` (full JSON deserialisation)
    on every call, defeating the cache that ``_agent()`` populates for
    its own callers. Promoting the cache check to ``_graph()`` itself
    means *any* caller benefits, regardless of which tool they entered
    through. Concurrent first-callers may each load the graph and
    overwrite ``_cached_graph`` once — the result is identical so the
    race is benign and we avoid pulling the async ``_init_lock`` into
    a sync function.
    """
    global _cached_graph  # noqa: PLW0603
    if _cached_graph is not None:
        return _cached_graph
    path = Path(os.environ.get("ATLAS_GRAPH_PATH", "data/graph.json"))
    if not path.exists():
        raise RuntimeError(f"call graph not found at {path}; run `atlas index` first")
    _cached_graph = CallGraph.load(path)
    return _cached_graph


def _dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN env var is required")
    return dsn


_cached_store: object | None = None
_cached_encoder: Encoder | None = None
_cached_graph: CallGraph | None = None
# Serialises lazy initialisation of the process-wide caches above. Two
# concurrent ``_agent`` calls on a cold process used to each see
# ``_cached_store is None`` and each create a ``ChunkStore`` + run the
# DDL bootstrap; the loser's store + pool was orphaned. Acquire the
# lock + re-check (double-checked init) so the first caller wins and
# subsequent callers see the populated cache.
_init_lock: asyncio.Lock = asyncio.Lock()


def _store_backend() -> str:
    """Resolve the chunk-store backend from env.

    Defaults to ``memory`` so a fresh checkout with ``atlas index --store=memory``
    can serve MCP queries immediately, without a Postgres + pgvector
    container. ``postgres`` switches back to the pgvector adapter.
    """
    backend = os.environ.get("ATLAS_STORE", "memory")
    if backend not in {"memory", "postgres"}:
        raise RuntimeError(
            f"unknown ATLAS_STORE={backend!r}; expected one of: memory, postgres"
        )
    return backend


def _chunks_path() -> Path:
    return Path(os.environ.get("ATLAS_CHUNKS_PATH", "data/chunks.json"))


# Default timeout constants — overridable via environment variables
# ATLAS_STEP_TIMEOUT_S and ATLAS_RUN_TIMEOUT_S.
_DEFAULT_STEP_TIMEOUT_S: float = 10.0
_DEFAULT_RUN_TIMEOUT_S: float = 30.0


def _agent_config() -> AgentConfig:
    """Build an ``AgentConfig`` with defaults + optional env overrides.

    ``ATLAS_STEP_TIMEOUT_S`` overrides the per-step (retriever /
    synthesiser) timeout; ``ATLAS_RUN_TIMEOUT_S`` overrides the
    whole-run timeout.  Both default to 10 s / 30 s respectively so
    a hung retrieval or synthesiser doesn't stall the MCP client
    indefinitely.  Set either variable to ``0`` to disable that
    timeout (not recommended in production).
    """
    step_s = float(os.environ.get("ATLAS_STEP_TIMEOUT_S", _DEFAULT_STEP_TIMEOUT_S))
    run_s = float(os.environ.get("ATLAS_RUN_TIMEOUT_S", _DEFAULT_RUN_TIMEOUT_S))
    return AgentConfig(
        step_timeout_s=step_s if step_s > 0 else None,
        run_timeout_s=run_s if run_s > 0 else None,
    )


async def _agent(top_k: int = 8) -> Agent:
    """Build an agent for one request, reusing a process-wide store + graph.

    ``top_k`` is built into the per-request ``RetrieverConfig`` rather
    than overlaid on a cached agent — clearer ownership and no shared
    mutable state across concurrent calls. The expensive bits — the
    pgvector connection pool, the encoder, the call graph — are cached
    on first use so subsequent requests don't re-run DDL or re-load
    the graph from disk.

    Per-step and whole-run timeouts default to
    ``ATLAS_STEP_TIMEOUT_S`` (10 s) and ``ATLAS_RUN_TIMEOUT_S`` (30 s)
    respectively; set either env var to ``0`` to disable.
    """
    from codex_atlas.store import ChunkStore, InMemoryChunkStore  # noqa: PLC0415

    global _cached_store, _cached_encoder, _cached_graph  # noqa: PLW0603
    # Fast path: every cache populated, no lock needed.
    if (
        _cached_encoder is not None
        and _cached_store is not None
        and _cached_graph is not None
    ):
        retriever = Retriever(
            _cached_encoder,
            _cached_store,  # type: ignore[arg-type]
            _cached_graph,
            RetrieverConfig(top_k=top_k),
        )
        return Agent(retriever, config=_agent_config())
    # Cold path: serialise so concurrent first-callers don't each
    # create a duplicate ChunkStore / pool / graph.
    async with _init_lock:
        if _cached_encoder is None:
            _cached_encoder = _encoder()
        if _cached_store is None:
            backend = _store_backend()
            if backend == "memory":
                _cached_store = await InMemoryChunkStore.load_from_path(_chunks_path())
            else:
                store = ChunkStore(dsn=_dsn())
                await store.setup(dim=_cached_encoder.dim)
                _cached_store = store
        if _cached_graph is None:
            _cached_graph = _graph()
    retriever = Retriever(
        _cached_encoder,
        _cached_store,  # type: ignore[arg-type]
        _cached_graph,
        RetrieverConfig(top_k=top_k),
    )
    return Agent(retriever, config=_agent_config())


def _to_response(result: AgentResult) -> SearchResponse | AgentTimeoutResponse:
    """Convert an ``AgentResult`` to the appropriate MCP response.

    Returns an ``AgentTimeoutResponse`` (``{"error": "agent_timeout",
    "phase": "<last_node>"}`` ) when the run was cancelled due to a
    timeout.  The phase is taken from the last trace event so callers
    can identify which step hit the deadline.  All other runs return the
    normal ``SearchResponse``.
    """
    if result.cancelled is CancelReason.TIMEOUT:
        # ``Agent.run`` records ``Node.CANCEL`` AFTER the timeout fires,
        # so ``trace[-1]`` is always ``cancel`` for a real timeout —
        # using it as the phase produced misleading "phase=cancel"
        # responses that hid which step actually hit the deadline. The
        # agent now tracks the in-flight node explicitly on its result
        # (``AgentResult.cancelled_node``); fall through to the trace
        # only for older results that predate the field.
        if result.cancelled_node is not None:
            phase = str(result.cancelled_node)
        else:
            phase = str(result.trace[-1].node) if result.trace else "unknown"
        return AgentTimeoutResponse(error="agent_timeout", phase=phase)
    # Surface the per-chunk score + text the agent threaded through the
    # ``Citation`` record. Earlier the MCP layer hardcoded score=0.0 and
    # text="" — the schema advertised score: float and text: str but
    # every response flatlined those fields, which made score-aware
    # downstream code (re-ranking, snippet rendering) impossible.
    #
    # Apply hard caps so a runaway answer or very wide retrieval cannot
    # flood the MCP client's context window.
    answer = result.answer
    if len(answer.encode("utf-8")) > MAX_MCP_ANSWER_BYTES:
        # Reserve the marker bytes BEFORE slicing the original answer
        # so the FINAL annotated string stays at or below
        # ``MAX_MCP_ANSWER_BYTES``. Earlier we sliced to the cap and
        # then appended the marker, so the annotated answer always
        # exceeded the advertised cap by ``len(marker)`` bytes — a
        # silent overrun for any client relying on the byte budget.
        marker = "\n\n[answer truncated at MAX_MCP_ANSWER_BYTES]"
        marker_bytes = len(marker.encode("utf-8"))
        budget = MAX_MCP_ANSWER_BYTES - marker_bytes
        if budget < 0:
            # Marker alone exceeds the cap (would only happen if the
            # cap were configured below the marker length). Drop the
            # marker — preserving the cap is more important than the
            # truncation hint, and downstream readers can still see
            # the answer was clipped via the byte length.
            answer = answer.encode("utf-8")[:MAX_MCP_ANSWER_BYTES].decode(
                "utf-8", errors="ignore"
            )
        else:
            truncated = answer.encode("utf-8")[:budget].decode(
                "utf-8", errors="ignore"
            )
            answer = truncated + marker
        # Belt-and-braces: ``errors="ignore"`` may have produced a
        # shorter byte string than ``budget`` if the slice landed mid-
        # codepoint, but the marker re-add could in theory push past
        # the cap if a future change widens the marker. Re-clamp.
        encoded = answer.encode("utf-8")
        if len(encoded) > MAX_MCP_ANSWER_BYTES:
            answer = encoded[:MAX_MCP_ANSWER_BYTES].decode(
                "utf-8", errors="ignore"
            )
    citations = result.citations[:MAX_MCP_CITATIONS]
    hits = [
        CodeSearchHit(
            qualified_name=c.qualified_name,
            file_path=c.file_path,
            lineno_start=c.lineno_start,
            lineno_end=c.lineno_end,
            score=c.score,
            text=c.text,
        )
        for c in citations
    ]
    return SearchResponse(
        route=str(result.route),
        grade=result.grade,
        attempts=result.attempts,
        answer=answer,
        citations=hits,
        cancelled=str(result.cancelled) if result.cancelled is not None else None,
    )


@mcp.tool
async def search_code(query: str, top_k: int = 8) -> SearchResponse | AgentTimeoutResponse:
    """Adaptive-route code search. Returns synthesised answer + citations."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > 50:
        raise ValueError("top_k must be <= 50")
    agent = await _agent(top_k=top_k)
    return _to_response(await agent.run(query))


@mcp.tool
async def explain_function(qualified_name: str) -> SearchResponse | AgentTimeoutResponse:
    """Pull the chunk for `qualified_name` plus its immediate neighbours."""
    if not qualified_name.strip():
        raise ValueError("qualified_name must not be blank")
    agent = await _agent()
    # Phrase the query so the classifier picks the structural route.
    return _to_response(await agent.run(f"who calls {qualified_name}"))


@mcp.tool
async def find_callers(qualified_name: str, depth: int = 1) -> CallersResponse:
    """Graph-only callers traversal — no LLM, just the call graph.

    Depth is clamped at ``MAX_NEIGHBORHOOD_DEPTH`` (8) at the MCP
    boundary, matching ``get_graph_neighborhood``. Requests above the
    cap are reduced rather than rejected so a slightly-too-large depth
    doesn't fail the tool call.
    """
    if not qualified_name.strip():
        raise ValueError("qualified_name must not be blank")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > MAX_NEIGHBORHOOD_DEPTH:
        _log.warning(
            "find_callers.depth_clamped",
            extra={
                "qualified_name": qualified_name,
                "requested_depth": depth,
                "clamped_depth": MAX_NEIGHBORHOOD_DEPTH,
            },
        )
        depth = MAX_NEIGHBORHOOD_DEPTH
    callers = _graph().find_callers(qualified_name, depth=depth)
    return CallersResponse(
        target=qualified_name,
        depth=depth,
        callers=[CallerEntry(qualified_name=c) for c in callers],
    )


@mcp.tool
async def summarize_module(module_path: str) -> SearchResponse | AgentTimeoutResponse:
    """High-level walkthrough of a module via the summarization route."""
    if not module_path.strip():
        raise ValueError("module_path must not be blank")
    agent = await _agent()
    return _to_response(await agent.run(f"walk me through {module_path}"))


@mcp.tool
async def search_codebase(query: str, top_k: int = 8, route: str | None = None) -> SearchResponse | AgentTimeoutResponse:
    """Generic search with an optional explicit route override.

    Pass ``route="structural"`` (or any ``Route`` value) to skip the
    classifier and run the named retrieval pipeline directly — useful
    when the calling LLM has already decided which pipeline it wants.
    With ``route=None`` this is identical to ``search_code``. Unknown
    route names raise ``ValueError`` rather than silently falling
    through to the classifier.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > 50:
        raise ValueError("top_k must be <= 50")
    if not query.strip():
        raise ValueError("query must not be blank")
    override: Route | None = None
    if route is not None:
        try:
            override = Route(route)
        except ValueError as e:
            valid = ", ".join(r.value for r in Route)
            raise ValueError(f"unknown route {route!r}; expected one of: {valid}") from e
    agent = await _agent(top_k=top_k)
    return _to_response(await agent.run(query, route_override=override))


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
async def explain(symbol: str) -> SearchResponse | AgentTimeoutResponse:
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


def validate_startup_config() -> None:
    """Reject unrunnable configurations BEFORE the MCP server opens a transport.

    Without this check, a server started with ``--store=postgres`` and
    no ``POSTGRES_DSN`` would happily accept the connect handshake from
    its MCP client and only blow up on the first ``search_code`` /
    ``explain`` / ``find_callers`` tool call — at which point the
    client sees an opaque ``RuntimeError: POSTGRES_DSN env var is
    required``. Failing fast at startup turns that into a clean,
    actionable error before any tool is invoked.

    Memory mode does not require this check at startup because the
    snapshot is read lazily; a missing snapshot will surface the same
    ``FileNotFoundError`` whether we check now or on first use, and the
    error message already points at ``atlas index --store=memory``.
    """
    backend = _store_backend()
    if backend == "postgres" and not os.environ.get("POSTGRES_DSN"):
        raise RuntimeError(
            "atlas-mcp --store=postgres requires POSTGRES_DSN to be set "
            "(e.g. postgresql://bench:bench@localhost:5433/bench). "
            "Either export POSTGRES_DSN or run with --store=memory."
        )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


# The ``atlas-mcp`` entry point intentionally has fewer flags than the
# ``atlas mcp`` subcommand — the MCP server reads its store / chunks /
# graph configuration from environment variables so wrappers like
# Claude Desktop's ``mcpServers`` block (which only sets ``env``,
# never ``args``) work without a flag-passing shim. Document the env
# contract in the help epilog so ``atlas-mcp --help`` doesn't read as
# a feature gap relative to ``atlas mcp --help``.
_ATLAS_MCP_EPILOG = (
    "Environment variables (read at first-tool-call time):\n"
    "  ATLAS_STORE         memory (default; reads ATLAS_CHUNKS_PATH) | postgres\n"
    "  ATLAS_GRAPH_PATH    persisted call graph; default data/graph.json\n"
    "  ATLAS_CHUNKS_PATH   chunk snapshot when ATLAS_STORE=memory; default data/chunks.json\n"
    "  ATLAS_ENCODER       fake (default; deterministic blake2b) | sentence-transformers model id\n"
    "  ATLAS_STEP_TIMEOUT_S per-step (retriever/synth) timeout; default 10, 0 disables\n"
    "  ATLAS_RUN_TIMEOUT_S  whole-run timeout; default 30, 0 disables\n"
    "  POSTGRES_DSN        required when ATLAS_STORE=postgres\n"
    "Use absolute paths under MCP — clients launch the server with an unpredictable cwd."
)

_cli = typer.Typer(name="atlas-mcp", add_completion=False)


@_cli.command(epilog=_ATLAS_MCP_EPILOG)
def run(
    transport: str = typer.Option("stdio", help="MCP transport: stdio | http"),
    host: str = typer.Option("127.0.0.1", help="HTTP bind host."),
    port: int = typer.Option(8090, help="HTTP bind port."),
) -> None:
    """Run the Codex-Atlas MCP server."""
    # Fail-fast on unrunnable configs (e.g. postgres backend with no DSN)
    # before opening a transport, so clients never see the misconfig
    # surface as an opaque per-tool-call error.
    validate_startup_config()
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
