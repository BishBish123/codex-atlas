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
from codex_atlas.agent import AgentResult, CancelReason, Citation, TraceEvent
from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.mcp_server import (
    MAX_NEIGHBORHOOD_DEPTH,
    AgentTimeoutResponse,
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

        # Provide a DSN so ``_dsn()`` doesn't raise. Force the postgres
        # backend explicitly — the default backend is now ``memory`` so
        # this test would otherwise exercise the in-memory path and skip
        # the ``ChunkStore`` patch entirely.
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://stub")
        monkeypatch.setenv("ATLAS_STORE", "postgres")
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
        # When the agent reports a cancelled run due to TIMEOUT, the MCP
        # response must be an AgentTimeoutResponse with error="agent_timeout".

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
                    trace=[TraceEvent(node="retrieve", started_at=0.0, elapsed_ms=10.0)],
                    cancelled=CancelReason.TIMEOUT,
                )

            agent.run = _run  # type: ignore[method-assign]
            return agent

        monkeypatch.setattr(mcp_server, "_agent", fake_agent)
        resp = await search_code("anything")
        assert isinstance(resp, AgentTimeoutResponse)
        assert resp.error == "agent_timeout"
        assert resp.phase == "retrieve"

    async def test_completed_run_has_null_cancelled(
        self, stub_agent_factory: dict[str, object]
    ) -> None:
        # Sanity check: a normal run leaves ``cancelled`` as None.
        resp = await search_code("anything")
        assert resp.cancelled is None


class TestMcpAgentTimeoutTypedError:
    """test_mcp_agent_timeout_returns_typed_error

    When the agent's run_timeout_s or step_timeout_s fires, the MCP tool
    must return an ``AgentTimeoutResponse`` with ``error="agent_timeout"``
    and a ``phase`` string identifying the last executing node.  This is a
    typed error signal — not a raised exception — so MCP clients can key
    on the ``error`` field without parsing exception messages.
    """

    async def test_mcp_agent_timeout_returns_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out agent run produces AgentTimeoutResponse, not SearchResponse."""
        async def fake_agent(top_k: int = 8) -> _StubAgent:
            agent = _StubAgent()

            async def _run(query: str, *, route_override: Route | None = None) -> AgentResult:
                # Simulate a run that was cancelled mid-retrieve.
                return AgentResult(
                    query=query,
                    final_query=query,
                    answer="",
                    citations=[],
                    route=Route.LOOKUP,
                    grade=0.0,
                    attempts=1,
                    trace=[
                        TraceEvent(node="classify", started_at=0.0, elapsed_ms=1.0),
                        TraceEvent(node="retrieve", started_at=1.0, elapsed_ms=8999.0),
                    ],
                    cancelled=CancelReason.TIMEOUT,
                )

            agent.run = _run  # type: ignore[method-assign]
            return agent

        monkeypatch.setattr(mcp_server, "_agent", fake_agent)
        resp = await search_code("find all auth routes")
        assert isinstance(resp, AgentTimeoutResponse), (
            f"expected AgentTimeoutResponse, got {type(resp).__name__}: {resp!r}"
        )
        assert resp.error == "agent_timeout"
        # Phase should reflect the last trace node (retrieve in this case).
        assert resp.phase == "retrieve"

    async def test_timeout_env_overrides_are_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ATLAS_STEP_TIMEOUT_S and ATLAS_RUN_TIMEOUT_S are forwarded to AgentConfig."""
        import codex_atlas.mcp_server as srv  # noqa: PLC0415

        monkeypatch.setenv("ATLAS_STEP_TIMEOUT_S", "5")
        monkeypatch.setenv("ATLAS_RUN_TIMEOUT_S", "20")
        cfg = srv._agent_config()
        assert cfg.step_timeout_s == 5.0
        assert cfg.run_timeout_s == 20.0

    async def test_zero_timeout_env_disables_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting ATLAS_STEP_TIMEOUT_S=0 disables the step timeout."""
        import codex_atlas.mcp_server as srv  # noqa: PLC0415

        monkeypatch.setenv("ATLAS_STEP_TIMEOUT_S", "0")
        monkeypatch.setenv("ATLAS_RUN_TIMEOUT_S", "0")
        cfg = srv._agent_config()
        assert cfg.step_timeout_s is None
        assert cfg.run_timeout_s is None


class TestModelShapes:
    """Confirm the Pydantic models documented in the README/docstring exist."""

    def test_neighborhood_response_fields(self) -> None:
        resp = NeighborhoodResponse(target="x", depth=1, callers=[], callees=[], all=[])
        assert resp.target == "x"
        assert resp.depth == 1

    def test_codebase_stats_default_language(self) -> None:
        stats = CodebaseStats(n_nodes=0, n_edges=0, graph_path="x")
        assert stats.language == "python"


class TestCodeSearchHitScoreAndText:
    """``CodeSearchHit.score`` and ``.text`` must reflect the actual chunk.

    The MCP layer historically hardcoded ``score=0.0`` and ``text=""``
    even though the Pydantic schema advertised real fields. Wire the
    agent's ``Citation`` (which now carries score + text from the
    retrieved chunks) through to the response so score-aware clients
    (re-rankers, snippet renderers) see real numbers.
    """

    async def test_codesearchhit_has_real_score_and_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_agent(top_k: int = 8) -> _StubAgent:
            agent = _StubAgent()

            async def _run(
                query: str, *, route_override: Route | None = None
            ) -> AgentResult:
                return AgentResult(
                    query=query,
                    final_query=query,
                    answer="stub answer",
                    citations=[
                        Citation(
                            qualified_name="m.foo",
                            file_path="m.py",
                            lineno_start=1,
                            lineno_end=3,
                            score=0.87,
                            text="def foo():\n    return 1\n",
                        ),
                        Citation(
                            qualified_name="m.bar",
                            file_path="m.py",
                            lineno_start=5,
                            lineno_end=7,
                            score=0.42,
                            text="def bar():\n    return 2\n",
                        ),
                    ],
                    route=Route.LOOKUP,
                    grade=1.0,
                    attempts=1,
                    trace=[],
                )

            agent.run = _run  # type: ignore[method-assign]
            return agent

        monkeypatch.setattr(mcp_server, "_agent", fake_agent)
        resp = await search_code("anything")
        assert len(resp.citations) == 2
        # Real score from the retrieved chunk, not the hardcoded 0.0.
        assert resp.citations[0].score == pytest.approx(0.87)
        assert resp.citations[1].score == pytest.approx(0.42)
        # Non-empty text matching the chunk source.
        assert resp.citations[0].text == "def foo():\n    return 1\n"
        assert resp.citations[1].text == "def bar():\n    return 2\n"


class TestMcpAgentTimeoutPhaseFromCancelledNode:
    """The MCP timeout response must read phase from
    ``AgentResult.cancelled_node`` — NOT ``trace[-1]``.

    `Agent.run` appends ``Node.CANCEL`` to the trace AFTER the timeout
    fires, so a real timed-out run has ``trace[-1].node == "cancel"``.
    Reading phase from there hid which step actually hit the deadline.
    """

    async def test_phase_uses_cancelled_node_over_trace_last(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from codex_atlas.agent import Node  # noqa: PLC0415

        async def fake_agent(top_k: int = 8) -> _StubAgent:
            agent = _StubAgent()

            async def _run(
                query: str, *, route_override: Route | None = None
            ) -> AgentResult:
                # Mimic the real shape: the trace ends with Node.CANCEL
                # (the post-timeout sentinel) but the agent recorded the
                # actual in-flight node as ANSWER on the result.
                return AgentResult(
                    query=query,
                    final_query=query,
                    answer="",
                    citations=[],
                    route=Route.LOOKUP,
                    grade=0.0,
                    attempts=1,
                    trace=[
                        TraceEvent(node=Node.RETRIEVE, started_at=0.0, elapsed_ms=1.0),
                        TraceEvent(node=Node.GRADE, started_at=1.0, elapsed_ms=1.0),
                        TraceEvent(node=Node.ANSWER, started_at=2.0, elapsed_ms=999.0),
                        TraceEvent(node=Node.CANCEL, started_at=3.0, elapsed_ms=0.0),
                    ],
                    cancelled=CancelReason.TIMEOUT,
                    cancelled_node=Node.ANSWER,
                )

            agent.run = _run  # type: ignore[method-assign]
            return agent

        monkeypatch.setattr(mcp_server, "_agent", fake_agent)
        resp = await search_code("anything")
        assert isinstance(resp, AgentTimeoutResponse)
        # Phase reflects the in-flight node, NOT trace[-1] (cancel).
        assert resp.phase == "answer"

    async def test_real_timed_out_agent_run_phase_is_retrieve(self) -> None:
        # End-to-end: stand up a real Agent with a slow retriever and a
        # short step timeout, then push the AgentResult through
        # ``_to_response`` and assert the phase reflects the actual
        # node in flight when the deadline fired.
        from codex_atlas.agent import Agent, AgentConfig, Node  # noqa: PLC0415

        @dataclass
        class _SlowRetriever:
            async def retrieve(
                self, query: str, *, route_override: Route | None = None
            ) -> Any:
                await asyncio.sleep(0.5)
                # Never reached — the step timeout fires first.
                raise AssertionError("expected timeout before retrieve returned")

        agent = Agent(
            _SlowRetriever(),  # type: ignore[arg-type]
            config=AgentConfig(step_timeout_s=0.02),
        )
        result = await agent.run("q")
        assert result.cancelled is CancelReason.TIMEOUT
        assert result.cancelled_node is Node.RETRIEVE
        # Push through the MCP boundary translator.
        resp = mcp_server._to_response(result)
        assert isinstance(resp, AgentTimeoutResponse)
        assert resp.phase == "retrieve"


class TestMcpAnswerTruncationCap:
    """`_to_response` must keep the FINAL answer (with truncation marker)
    at or below ``MAX_MCP_ANSWER_BYTES``.

    Earlier the truncation logic clipped to the cap and THEN appended
    the marker, so the annotated answer always exceeded the advertised
    byte cap by ``len(marker)``.
    """

    async def test_truncated_answer_with_marker_stays_under_cap(self) -> None:
        from codex_atlas.agent import Node  # noqa: PLC0415

        # Build an oversized answer (well past the cap).
        oversize = "x" * (mcp_server.MAX_MCP_ANSWER_BYTES + 5_000)
        result = AgentResult(
            query="q",
            final_query="q",
            answer=oversize,
            citations=[],
            route=Route.LOOKUP,
            grade=1.0,
            attempts=1,
            trace=[TraceEvent(node=Node.ANSWER, started_at=0.0, elapsed_ms=1.0)],
        )
        resp = mcp_server._to_response(result)
        # Successful path produces SearchResponse, not AgentTimeoutResponse.
        from codex_atlas.mcp_server import SearchResponse  # noqa: PLC0415

        assert isinstance(resp, SearchResponse)
        # The marker is present so callers know the answer was clipped.
        assert "[answer truncated at MAX_MCP_ANSWER_BYTES]" in resp.answer
        # The FINAL annotated string (answer + marker) must respect the cap.
        final_bytes = len(resp.answer.encode("utf-8"))
        assert final_bytes <= mcp_server.MAX_MCP_ANSWER_BYTES, (
            f"final answer is {final_bytes} bytes, "
            f"exceeds cap {mcp_server.MAX_MCP_ANSWER_BYTES}"
        )

    async def test_short_answer_left_unchanged(self) -> None:
        from codex_atlas.agent import Node  # noqa: PLC0415

        result = AgentResult(
            query="q",
            final_query="q",
            answer="hi",
            citations=[],
            route=Route.LOOKUP,
            grade=1.0,
            attempts=1,
            trace=[TraceEvent(node=Node.ANSWER, started_at=0.0, elapsed_ms=1.0)],
        )
        resp = mcp_server._to_response(result)
        from codex_atlas.mcp_server import SearchResponse  # noqa: PLC0415

        assert isinstance(resp, SearchResponse)
        assert resp.answer == "hi"


class TestStartupValidation:
    """``validate_startup_config`` rejects unrunnable configs at startup.

    Pre-fix the MCP server happily accepted a ``--store=postgres`` start
    with no DSN and only blew up on the first tool call. Failing fast at
    startup turns that into a clean error before any client connects.
    """

    def test_postgres_without_dsn_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_STORE", "postgres")
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
            mcp_server.validate_startup_config()

    def test_postgres_with_dsn_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_STORE", "postgres")
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://stub")
        # Should not raise.
        mcp_server.validate_startup_config()

    def test_memory_without_dsn_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Memory mode never needs a DSN — the snapshot is read lazily.
        monkeypatch.setenv("ATLAS_STORE", "memory")
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        mcp_server.validate_startup_config()

    def test_default_backend_passes_without_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default ATLAS_STORE is memory — fresh checkout shouldn't need a DSN.
        monkeypatch.delenv("ATLAS_STORE", raising=False)
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        mcp_server.validate_startup_config()


class TestGraphCacheReuse:
    """``_graph()`` must cache the loaded ``CallGraph`` so direct callers
    (``find_callers``, ``get_graph_neighborhood``, ``codebase_stats``)
    don't re-deserialise ``data/graph.json`` on every tool invocation.

    The earlier implementation only populated ``_cached_graph`` from
    inside ``_agent()``. A client that drove the graph-only tools
    without going through the LLM-backed agent paid full
    ``CallGraph.load`` cost on every call. The fix promotes the cache
    check into ``_graph()`` itself; this test pins it.
    """

    def test_repeated_calls_reuse_cache(self, fixture_graph: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the cache to a known empty starting point — other tests in
        # this module run ``find_callers`` etc. and may have populated it.
        monkeypatch.setattr(mcp_server, "_cached_graph", None, raising=False)
        load_calls = {"n": 0}
        real_load = CallGraph.load

        def counting_load(path: Path) -> CallGraph:
            load_calls["n"] += 1
            return real_load(path)

        monkeypatch.setattr(CallGraph, "load", staticmethod(counting_load))
        # First call populates cache, second call must reuse it.
        g1 = mcp_server._graph()
        g2 = mcp_server._graph()
        g3 = mcp_server._graph()
        assert load_calls["n"] == 1, (
            f"expected one CallGraph.load over three _graph() calls; got {load_calls['n']}"
        )
        # Identity check: cache must hand back the *same* object, not a clone.
        assert g1 is g2 is g3
