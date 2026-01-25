"""Smoke tests for the MCP server's tool / resource shapes.

We don't spin up a real MCP transport here — we exercise the underlying
async functions directly to verify (1) they accept the documented
inputs, (2) they return the documented Pydantic models, and (3) input
validation rejects bad arguments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.mcp_server import (
    CallersResponse,
    CodebaseStats,
    NeighborhoodResponse,
    codebase_stats,
    find_callers,
    get_graph_neighborhood,
)


def _sym(qname: str) -> Symbol:
    return Symbol(
        qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
    )


@pytest.fixture
def fixture_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    g = CallGraph()
    g.ingest(
        [
            ParsedFile(
                file_path="m.py",
                module_name="m",
                symbols=[_sym("m"), _sym("m.a"), _sym("m.b"), _sym("m.c")],
                chunks=[],
                imports=[],
                calls=[("m.a", "b"), ("m.b", "c")],
            )
        ]
    )
    out = tmp_path / "graph.json"
    g.save(out)
    monkeypatch.setenv("ATLAS_GRAPH_PATH", str(out))
    return out


class TestFindCallersTool:
    async def test_returns_callers_response(self, fixture_graph: Path) -> None:
        resp = await find_callers("m.b", depth=1)
        assert isinstance(resp, CallersResponse)
        assert resp.target == "m.b"
        assert resp.depth == 1
        names = [c.qualified_name for c in resp.callers]
        assert "m.a" in names

    async def test_blank_qualified_name_rejected(self, fixture_graph: Path) -> None:
        with pytest.raises(ValueError, match="blank"):
            await find_callers("   ")

    async def test_zero_depth_rejected(self, fixture_graph: Path) -> None:
        with pytest.raises(ValueError, match="depth"):
            await find_callers("m.b", depth=0)


class TestGetGraphNeighborhoodTool:
    async def test_returns_neighborhood_response(self, fixture_graph: Path) -> None:
        resp = await get_graph_neighborhood("m.b", depth=1)
        assert isinstance(resp, NeighborhoodResponse)
        assert "m.a" in resp.callers
        assert "m.c" in resp.callees
        assert sorted(resp.all) == ["m.a", "m.c"]

    async def test_blank_symbol_rejected(self, fixture_graph: Path) -> None:
        with pytest.raises(ValueError, match="blank"):
            await get_graph_neighborhood("", depth=1)

    async def test_zero_depth_rejected(self, fixture_graph: Path) -> None:
        with pytest.raises(ValueError, match="depth"):
            await get_graph_neighborhood("m.b", depth=0)

    async def test_depth_two_includes_grandparents(self, fixture_graph: Path) -> None:
        resp = await get_graph_neighborhood("m.c", depth=2)
        assert "m.a" in resp.callers
        assert "m.b" in resp.callers


class TestCodebaseStatsResource:
    def test_returns_stats(self, fixture_graph: Path) -> None:
        stats = codebase_stats()
        assert isinstance(stats, CodebaseStats)
        assert stats.n_nodes >= 4
        assert stats.language == "python"


class TestModelShapes:
    """Confirm the Pydantic models documented in the README/docstring exist."""

    def test_neighborhood_response_fields(self) -> None:
        resp = NeighborhoodResponse(target="x", depth=1, callers=[], callees=[], all=[])
        assert resp.target == "x"
        assert resp.depth == 1

    def test_codebase_stats_default_language(self) -> None:
        stats = CodebaseStats(n_nodes=0, n_edges=0, graph_path="x")
        assert stats.language == "python"
