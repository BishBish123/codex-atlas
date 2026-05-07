"""End-to-end example client for the Codex-Atlas MCP server.

Spawns ``atlas-mcp`` over stdio using the MCP SDK, calls three representative
tools, and prints the results.

Prerequisites
-------------
1. Index the repo first so the server has data to serve::

       cd /path/to/codex-atlas
       uv run atlas index src/ --store=memory

2. Run this script::

       uv run python examples/mcp_client.py

Flags
-----
--dry-run   Skip the stdio server spawn; return fixture data instead.
            Used by ``tests/test_examples.py`` so CI never starts a
            real subprocess.
--json      Emit raw JSON instead of the Rich table (pipe-friendly).

The script exits 0 on success and non-zero on any error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ---------------------------------------------------------------------------
# Formatter helpers (testable without a network / server)
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int = 120) -> str:
    """Truncate *text* to *width* characters for display."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def format_search_result(tool_name: str, result: dict[str, Any]) -> str:
    """Format a SearchResponse or AgentTimeoutResponse as a human-readable string."""
    lines: list[str] = [f"[{tool_name}]"]
    if result.get("error") == "agent_timeout":
        lines.append(f"  ERROR: agent_timeout (phase={result.get('phase', '?')})")
        return "\n".join(lines)
    lines.append(f"  route    : {result.get('route', '?')}")
    lines.append(f"  grade    : {result.get('grade', 0):.2f}")
    lines.append(f"  attempts : {result.get('attempts', '?')}")
    answer = _truncate(result.get("answer", ""), 200)
    lines.append(f"  answer   : {answer!r}")
    citations = result.get("citations", [])
    lines.append(f"  citations: {len(citations)}")
    for c in citations[:3]:
        lines.append(
            f"    - {c['qualified_name']} ({c['file_path']}:{c['lineno_start']})"
        )
    return "\n".join(lines)


def format_callers_result(tool_name: str, result: dict[str, Any]) -> str:
    """Format a CallersResponse as a human-readable string."""
    lines: list[str] = [f"[{tool_name}]"]
    lines.append(f"  target  : {result.get('target', '?')}")
    lines.append(f"  depth   : {result.get('depth', '?')}")
    callers = result.get("callers", [])
    lines.append(f"  callers : {len(callers)}")
    for entry in callers[:5]:
        lines.append(f"    - {entry['qualified_name']}")
    if len(callers) > 5:
        lines.append(f"    ... and {len(callers) - 5} more")
    return "\n".join(lines)


def format_neighborhood_result(tool_name: str, result: dict[str, Any]) -> str:
    """Format a NeighborhoodResponse as a human-readable string."""
    lines: list[str] = [f"[{tool_name}]"]
    lines.append(f"  target  : {result.get('target', '?')}")
    lines.append(f"  depth   : {result.get('depth', '?')}")
    callers = result.get("callers", [])
    callees = result.get("callees", [])
    lines.append(f"  callers : {callers[:5]}")
    lines.append(f"  callees : {callees[:5]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixture data for --dry-run (lets tests exercise formatters)
# ---------------------------------------------------------------------------

_FIXTURE_SEARCH: dict[str, Any] = {
    "route": "structural",
    "grade": 0.75,
    "attempts": 1,
    "answer": "codex_atlas.agent.Agent.run is the main agent entry point.",
    "citations": [
        {
            "qualified_name": "codex_atlas.agent.Agent.run",
            "file_path": "src/codex_atlas/agent.py",
            "lineno_start": 500,
            "lineno_end": 540,
            "score": 0.91,
            "text": "async def run(self, query: str ...",
        }
    ],
    "cancelled": None,
}

_FIXTURE_CALLERS: dict[str, Any] = {
    "target": "codex_atlas.agent.Agent.run",
    "depth": 1,
    "callers": [
        {"qualified_name": "codex_atlas.mcp_server.search_code"},
        {"qualified_name": "codex_atlas.mcp_server.explain_function"},
        {"qualified_name": "codex_atlas.cli.ask"},
    ],
}

_FIXTURE_NEIGHBORHOOD: dict[str, Any] = {
    "target": "codex_atlas.retriever.Retriever.retrieve",
    "depth": 2,
    "callers": ["codex_atlas.agent.Agent._retrieve"],
    "callees": ["codex_atlas.store.ChunkStoreProtocol.search"],
    "all": ["codex_atlas.agent.Agent._retrieve", "codex_atlas.store.ChunkStoreProtocol.search"],
}


# ---------------------------------------------------------------------------
# Live MCP calls
# ---------------------------------------------------------------------------


async def _run_live(repo_root: Path, emit_json: bool) -> int:
    """Spawn atlas-mcp and call three tools over stdio."""
    chunks_path = repo_root / "data" / "chunks.json"
    graph_path = repo_root / "data" / "graph.json"

    if not graph_path.exists():
        print(
            f"ERROR: {graph_path} missing. Run: uv run atlas index src/ --store=memory",
            file=sys.stderr,
        )
        return 1
    if not chunks_path.exists():
        print(
            f"ERROR: {chunks_path} missing. Run: uv run atlas index src/ --store=memory",
            file=sys.stderr,
        )
        return 1

    # Locate the atlas-mcp binary inside the venv
    venv_bin = repo_root / ".venv" / "bin" / "atlas-mcp"
    if not venv_bin.exists():
        print(
            f"ERROR: {venv_bin} not found. Run: uv sync",
            file=sys.stderr,
        )
        return 1

    server_params = StdioServerParameters(
        command=str(venv_bin),
        args=[],
        env={
            "ATLAS_STORE": "memory",
            "ATLAS_GRAPH_PATH": str(graph_path),
            "ATLAS_CHUNKS_PATH": str(chunks_path),
            "ATLAS_ENCODER": "fake",
            # Disable timeouts so a cold start doesn't race
            "ATLAS_STEP_TIMEOUT_S": "0",
            "ATLAS_RUN_TIMEOUT_S": "0",
            # PATH so the venv's Python is reachable if atlas-mcp needs it
            "PATH": os.environ.get("PATH", ""),
        },
        cwd=str(repo_root),
    )

    results: list[tuple[str, dict[str, Any]]] = []

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()

        # Tool 1: search_code — free-form question, exercises the full
        # classify → retrieve → grade → answer pipeline.
        r1 = await session.call_tool(
            "search_code",
            {"query": "who calls Agent.run", "top_k": 4},
        )
        raw1: dict[str, Any] = {}
        if r1.content:
            raw1 = json.loads(r1.content[0].text)  # type: ignore[union-attr]
        results.append(("search_code", raw1))

        # Tool 2: find_callers — pure graph traversal, no LLM.
        r2 = await session.call_tool(
            "find_callers",
            {"qualified_name": "codex_atlas.agent.Agent.run", "depth": 1},
        )
        raw2: dict[str, Any] = {}
        if r2.content:
            raw2 = json.loads(r2.content[0].text)  # type: ignore[union-attr]
        results.append(("find_callers", raw2))

        # Tool 3: get_graph_neighborhood — BFS in both directions.
        r3 = await session.call_tool(
            "get_graph_neighborhood",
            {"symbol": "codex_atlas.retriever.Retriever.retrieve", "depth": 2},
        )
        raw3: dict[str, Any] = {}
        if r3.content:
            raw3 = json.loads(r3.content[0].text)  # type: ignore[union-attr]
        results.append(("get_graph_neighborhood", raw3))

    return _print_results(results, emit_json)


def _print_results(results: list[tuple[str, dict[str, Any]]], emit_json: bool) -> int:
    """Pretty-print results. Returns exit code."""
    if emit_json:
        print(json.dumps({name: data for name, data in results}, indent=2))
        return 0

    for tool_name, data in results:
        if tool_name in ("search_code", "explain_function", "summarize_module", "search_codebase"):
            print(format_search_result(tool_name, data))
        elif tool_name == "find_callers":
            print(format_callers_result(tool_name, data))
        elif tool_name == "get_graph_neighborhood":
            print(format_neighborhood_result(tool_name, data))
        else:
            print(f"[{tool_name}]\n  {data}")
        print()
    return 0


def _run_dry(emit_json: bool) -> int:
    """Return fixture data without spawning any subprocess."""
    results = [
        ("search_code", _FIXTURE_SEARCH),
        ("find_callers", _FIXTURE_CALLERS),
        ("get_graph_neighborhood", _FIXTURE_NEIGHBORHOOD),
    ]
    return _print_results(results, emit_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the example client. Returns exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use fixture data; do not spawn atlas-mcp. For testing.",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit raw JSON output.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to codex-atlas repo root (defaults to parent of examples/).",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        return _run_dry(args.emit_json)

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).parent.parent
    return asyncio.run(_run_live(repo_root, args.emit_json))


if __name__ == "__main__":
    sys.exit(main())
