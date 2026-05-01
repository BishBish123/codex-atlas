"""Smoke tests for the MCP server's tool / resource shapes.

We don't spin up a real MCP transport here — we exercise the underlying
async functions directly to verify (1) they accept the documented
inputs, (2) they return the documented Pydantic models, and (3) input
validation rejects bad arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codex_atlas import mcp_server
from codex_atlas.agent import AgentResult
from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.mcp_server import (
    MAX_NEIGHBORHOOD_DEPTH,
    CallersResponse,
    CodebaseStats,
    NeighborhoodResponse,
    codebase_stats,
    find_callers,
    get_graph_neighborhood,
    search_code,
    search_codebase,
)
from codex_atlas.retriever import Route


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

    async def test_depth_clamped_at_max(self, fixture_graph: Path) -> None:
        # Requests above MAX_NEIGHBORHOOD_DEPTH (8) are clamped at the MCP
        # boundary so the call can't drag a 50K-node graph into a
        # minute-long traversal. The response surfaces the clamped depth.
        resp = await get_graph_neighborhood("m.b", depth=20)
        assert resp.depth == MAX_NEIGHBORHOOD_DEPTH


class TestCodebaseStatsResource:
    def test_returns_stats(self, fixture_graph: Path) -> None:
        stats = codebase_stats()
        assert isinstance(stats, CodebaseStats)
        assert stats.n_nodes >= 4
        assert stats.language == "python"


@dataclass
class _StubAgent:
    """Records the query it received so tests can assert on it."""

    queries: list[str] = field(default_factory=list)
    chunks_returned: int = 0

    async def run(self, query: str) -> AgentResult:
        self.queries.append(query)
        # Return an AgentResult that respects the recorded top_k cap by
        # producing N empty citations — the MCP layer caps these via the
        # retriever, not here, so we just verify the shape.
        return AgentResult(
            query=query,
            final_query=query,
            answer="stub",
            citations=[],
            route=Route.LOOKUP,
            grade=1.0,
            attempts=1,
            trace=[],
        )


@pytest.fixture
def stub_agent_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``_agent`` with a stub that records the ``top_k`` passed in."""
    captured: dict[str, object] = {"top_k": None, "agent": None}

    async def fake_agent(top_k: int = 8) -> _StubAgent:
        captured["top_k"] = top_k
        agent = _StubAgent()
        captured["agent"] = agent
        return agent

    monkeypatch.setattr(mcp_server, "_agent", fake_agent)
    return captured


class TestSearchCodeTopK:
    async def test_search_code_respects_top_k(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        # The MCP tool must thread the user-supplied top_k all the way down
        # to ``_agent``; previously it was hardcoded to 8 and the user
        # parameter was validated then ignored.
        await search_code("anything", top_k=3)
        assert stub_agent_factory["top_k"] == 3

    async def test_search_code_top_k_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            await search_code("anything", top_k=0)

    async def test_search_code_top_k_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            await search_code("anything", top_k=51)


class TestSearchCodebaseTopK:
    async def test_search_codebase_respects_top_k(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        await search_codebase("anything", top_k=5)
        assert stub_agent_factory["top_k"] == 5

    async def test_search_codebase_top_k_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            await search_codebase("anything", top_k=200)


class TestModelShapes:
    """Confirm the Pydantic models documented in the README/docstring exist."""

    def test_neighborhood_response_fields(self) -> None:
        resp = NeighborhoodResponse(target="x", depth=1, callers=[], callees=[], all=[])
        assert resp.target == "x"
        assert resp.depth == 1

    def test_codebase_stats_default_language(self) -> None:
        stats = CodebaseStats(n_nodes=0, n_edges=0, graph_path="x")
        assert stats.language == "python"
