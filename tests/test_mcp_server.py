"""Smoke tests for the MCP server's tool / resource shapes.

We don't spin up a real MCP transport here — we exercise the underlying
async functions directly to verify (1) they accept the documented
inputs, (2) they return the documented Pydantic models, and (3) input
validation rejects bad arguments.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from codex_atlas import mcp_server
from codex_atlas.agent import AgentResult, CancelReason
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

    async def test_clamps_oversized_depth(self, fixture_graph: Path) -> None:
        # Mirror get_graph_neighborhood: requests above the cap clamp
        # rather than fail. The response surfaces the clamped depth so
        # callers can tell their request was reduced.
        resp = await find_callers("m.b", depth=20)
        assert resp.depth == MAX_NEIGHBORHOOD_DEPTH


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
    """Records the query + route_override it received so tests can assert."""

    queries: list[str] = field(default_factory=list)
    overrides: list[Route | None] = field(default_factory=list)
    chunks_returned: int = 0

    async def run(self, query: str, *, route_override: Route | None = None) -> AgentResult:
        self.queries.append(query)
        self.overrides.append(route_override)
        return AgentResult(
            query=query,
            final_query=query,
            answer="stub",
            citations=[],
            route=route_override or Route.LOOKUP,
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

    async def test_search_code_rejects_blank_query(self) -> None:
        # Mirrors the existing search_codebase guard. A whitespace-only
        # query has no useful retrieval signal and the classifier would
        # silently fall through to the LOOKUP default.
        with pytest.raises(ValueError, match="blank"):
            await search_code("   ")
        with pytest.raises(ValueError, match="blank"):
            await search_code("")


class TestSearchCodebaseTopK:
    async def test_search_codebase_respects_top_k(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        await search_codebase("anything", top_k=5)
        assert stub_agent_factory["top_k"] == 5

    async def test_search_codebase_top_k_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            await search_codebase("anything", top_k=200)


class TestSearchCodebaseRouteOverride:
    async def test_route_override_skips_classifier(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        # Pick a query the heuristic classifier would NOT normally route to
        # graph_walk: a bare lookup-style "what does X do" question. With
        # ``route="structural"`` the override must be threaded through —
        # the agent receives the verbatim query (no rewording) AND the
        # explicit route_override.
        await search_codebase("what does helper do", route="structural")
        agent = stub_agent_factory["agent"]
        assert isinstance(agent, _StubAgent)
        assert agent.overrides == [Route.STRUCTURAL]
        # Verbatim query — no classifier-bait rewording.
        assert agent.queries == ["what does helper do"]

    async def test_route_override_none_passes_through(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        await search_codebase("anything", route=None)
        agent = stub_agent_factory["agent"]
        assert isinstance(agent, _StubAgent)
        assert agent.overrides == [None]

    async def test_unknown_route_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown route"):
            await search_codebase("anything", route="not_a_route")


class TestCachedStoreInitIdempotence:
    async def test_cached_store_init_idempotent_under_concurrency(
        self, monkeypatch: pytest.MonkeyPatch, fixture_graph: Path
    ) -> None:
        # Two concurrent ``_agent`` callers on a cold process used to
        # each create a ``ChunkStore`` and run the DDL bootstrap; the
        # loser's store was orphaned. The double-checked ``_init_lock``
        # makes setup() fire exactly once.
        # Reset module-level caches so this test exercises the cold path.
        monkeypatch.setattr(mcp_server, "_cached_store", None)
        monkeypatch.setattr(mcp_server, "_cached_encoder", None)
        monkeypatch.setattr(mcp_server, "_cached_graph", None)

        setup_calls = 0

        class _FakeStore:
            def __init__(self, dsn: str) -> None:
                self.dsn = dsn

            async def setup(self, dim: int) -> None:
                nonlocal setup_calls
                # Yield so a concurrent caller has the chance to race
                # past the ``is None`` check — exactly the interleaving
                # the lock must defeat.
                await asyncio.sleep(0)
                setup_calls += 1

        # Patch the lazy import target inside ``_agent``.
        import codex_atlas.store as store_mod  # noqa: PLC0415

        monkeypatch.setattr(store_mod, "ChunkStore", _FakeStore)

        # Provide a DSN so ``_dsn()`` doesn't raise.
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://stub")
        # The encoder factory + graph loader should still work; the
        # graph fixture has set ATLAS_GRAPH_PATH.

        async def call_agent() -> Any:
            return await mcp_server._agent(top_k=8)

        await asyncio.gather(*(call_agent() for _ in range(10)))
        assert setup_calls == 1


class TestCancelledSurface:
    async def test_cancelled_run_surfaces_in_mcp_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the agent reports a cancelled run, the MCP response must
        # carry the cancel reason as a string so clients can distinguish
        # "no results" from "the run was killed by the deadline".

        async def fake_agent(top_k: int = 8) -> _StubAgent:
            agent = _StubAgent()
            # Replace ``run`` to return a cancelled result.
            async def _run(query: str, *, route_override: Route | None = None) -> AgentResult:
                return AgentResult(
                    query=query,
                    final_query=query,
                    answer="",
                    citations=[],
                    route=Route.LOOKUP,
                    grade=0.0,
                    attempts=1,
                    trace=[],
                    cancelled=CancelReason.TIMEOUT,
                )

            agent.run = _run  # type: ignore[method-assign]
            return agent

        monkeypatch.setattr(mcp_server, "_agent", fake_agent)
        resp = await search_code("anything")
        assert resp.cancelled == "timeout"
        assert resp.answer == ""

    async def test_completed_run_has_null_cancelled(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        # Sanity check: a normal run leaves ``cancelled`` as None.
        resp = await search_code("anything")
        assert resp.cancelled is None


class TestModelShapes:
    """Confirm the Pydantic models documented in the README/docstring exist."""

    def test_neighborhood_response_fields(self) -> None:
        resp = NeighborhoodResponse(target="x", depth=1, callers=[], callees=[], all=[])
        assert resp.target == "x"
        assert resp.depth == 1

    def test_codebase_stats_default_language(self) -> None:
        stats = CodebaseStats(n_nodes=0, n_edges=0, graph_path="x")
        assert stats.language == "python"
