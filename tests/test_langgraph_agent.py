"""Tests for the LangGraph agent backend.

Strategy
--------
* ``langgraph`` is an optional dependency; these tests run in CI without it.
  We install a minimal in-process stub that satisfies the ``StateGraph``
  interface the agent uses, then monkey-patch ``sys.modules`` before any
  code under test imports the real package.
* Tests that verify *result equivalence* between ``Agent`` and
  ``LangGraphAgent`` use deterministic stubs for retriever / grader /
  synth so the comparison is exact (modulo trace timestamps).
* ``make_agent()`` factory tests exercise env-var dispatch and unknown
  backend rejection.
* Soft-import test verifies the correct ``RuntimeError`` when langgraph
  is absent.

Note: ``asyncio_mode = "auto"`` is set in ``pyproject.toml``; every async
method is automatically run by pytest-asyncio.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from codex_atlas.agent import (
    AgentConfig,
    AgentResult,
    CitationValidator,
    HeuristicGrader,
    Node,
    NoopRewriter,
    StitchSynthesizer,
)
from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.retriever import RetrievalResult, Route
from codex_atlas.store import StoredChunk

# ---------------------------------------------------------------------------
# Minimal langgraph stub installed before any agent_langgraph import
# ---------------------------------------------------------------------------


def _build_langgraph_stub() -> types.ModuleType:  # type: ignore[type-arg]
    """Construct a minimal langgraph package stub.

    The stub satisfies exactly the surface that ``LangGraphAgent._build_graph``
    uses:
    - ``langgraph.graph.StateGraph``
    - ``langgraph.graph.END``

    ``StateGraph.compile()`` returns an object whose ``ainvoke`` method runs
    all registered nodes sequentially, respecting the conditional edge on
    ``grade`` and the ``rewrite_query -> retrieve`` back-edge.
    """

    class _CompiledGraph:
        """Minimal compiled graph: runs nodes depth-first, honouring the
        conditional edge after ``grade`` (should_rewrite → rewrite_query →
        retrieve; else → answer) and the ``rewrite_query → retrieve`` loop."""

        def __init__(
            self,
            nodes: dict[str, Any],
            entry: str,
            edges: dict[str, str],
            cond_edges: dict[str, Any],
        ) -> None:
            self._nodes = nodes
            self._entry = entry
            self._edges = edges           # static edges: source -> dest
            self._cond = cond_edges       # conditional: source -> (fn, map)

        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:  # type: ignore[type-arg]
            END = "__end__"
            current = self._entry
            while current and current != END:
                node_fn = self._nodes[current]
                update = await node_fn(state)
                state.update(update)
                if current in self._cond:
                    cond_fn, mapping = self._cond[current]
                    next_node = mapping[cond_fn(state)]
                elif current in self._edges:
                    next_node = self._edges[current]
                else:
                    break
                current = next_node
            return state

    class StateGraph:
        def __init__(self, schema: Any) -> None:
            self._nodes: dict[str, Any] = {}
            self._entry: str = ""
            self._edges: dict[str, str] = {}
            self._cond: dict[str, Any] = {}

        def add_node(self, name: str, fn: Any) -> None:
            self._nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self._entry = name

        def add_edge(self, src: str, dst: str) -> None:
            self._edges[src] = dst

        def add_conditional_edges(
            self, src: str, fn: Any, mapping: dict[str, str]
        ) -> None:
            self._cond[src] = (fn, mapping)

        def compile(self) -> _CompiledGraph:
            return _CompiledGraph(
                nodes=dict(self._nodes),
                entry=self._entry,
                edges=dict(self._edges),
                cond_edges=dict(self._cond),
            )

    # Build the module tree: langgraph + langgraph.graph
    lg_pkg = types.ModuleType("langgraph")
    lg_graph = types.ModuleType("langgraph.graph")
    lg_graph.StateGraph = StateGraph  # type: ignore[attr-defined]
    lg_graph.END = "__end__"  # type: ignore[attr-defined]
    lg_pkg.graph = lg_graph  # type: ignore[attr-defined]

    return lg_pkg


# Install the stub before any test can trigger a real langgraph import.
_LG_STUB = _build_langgraph_stub()
sys.modules.setdefault("langgraph", _LG_STUB)
sys.modules.setdefault("langgraph.graph", _LG_STUB.graph)  # type: ignore[attr-defined]

# Now it's safe to import agent_langgraph (will see the stub).
from codex_atlas import make_agent  # noqa: E402
from codex_atlas.agent_langgraph import LangGraphAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _stored(qname: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=f"def {qname.rsplit('.', maxsplit=1)[-1]}(): pass\n",
        score=0.9,
    )


@dataclass
class StubRetriever:
    """Returns one canned RetrievalResult per call, cycling through ``responses``."""

    responses: list[RetrievalResult]
    calls: int = 0

    async def retrieve(
        self, query: str, *, route_override: Route | None = None
    ) -> RetrievalResult:
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _result(
    chunks: list[StoredChunk], confidence: float = 0.9, route: Route = Route.LOOKUP
) -> RetrievalResult:
    return RetrievalResult(
        route=route,
        confidence=confidence,
        signals=["stub"],
        chunks=chunks,
        extra_qualified_names=[],
    )


# ---------------------------------------------------------------------------
# 1. LangGraphAgent basic interface tests
# ---------------------------------------------------------------------------


class TestLangGraphAgentInterface:
    async def test_returns_agent_result(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        assert isinstance(result, AgentResult)

    async def test_answer_contains_chunk_name(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        assert "m.foo" in result.answer

    async def test_citations_populated(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        assert len(result.citations) == 1
        assert result.citations[0].qualified_name == "m.foo"

    async def test_attempts_is_one_on_first_success(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q")
        assert result.attempts == 1

    async def test_trace_contains_expected_nodes(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q")
        nodes = [e.node for e in result.trace]
        assert Node.CLASSIFY in nodes
        assert Node.RETRIEVE in nodes
        assert Node.GRADE in nodes
        assert Node.ANSWER in nodes
        assert Node.VALIDATE in nodes

    async def test_tool_calls_recorded(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q")
        assert len(result.tool_calls) >= 1
        tc = result.tool_calls[0]
        assert tc.route is Route.LOOKUP
        assert tc.n_chunks == 1

    async def test_validation_report_returned(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q")
        assert result.validation is not None

    async def test_route_override_passed_through(self) -> None:
        retriever = StubRetriever(
            responses=[_result([_stored("m.foo")], route=Route.STRUCTURAL)]
        )
        agent = LangGraphAgent(retriever)  # type: ignore[arg-type]
        result = await agent.run("q", route_override=Route.STRUCTURAL)
        assert result.route is Route.STRUCTURAL


# ---------------------------------------------------------------------------
# 2. Reflection loop (rewrite_query edge)
# ---------------------------------------------------------------------------


class TestLangGraphReflection:
    async def test_rewrites_when_first_retrieval_empty(self) -> None:
        retriever = StubRetriever(
            responses=[
                _result([], confidence=0.9),
                _result([_stored("m.foo")], confidence=0.9),
            ]
        )
        agent = LangGraphAgent(  # type: ignore[arg-type]
            retriever, config=AgentConfig(max_attempts=3)
        )
        result = await agent.run("question")
        assert result.attempts >= 2
        assert any(e.node is Node.REWRITE for e in result.trace)
        assert "m.foo" in result.answer

    async def test_terminates_after_max_attempts(self) -> None:
        retriever = StubRetriever(responses=[_result([])])
        agent = LangGraphAgent(  # type: ignore[arg-type]
            retriever, config=AgentConfig(max_attempts=3)
        )
        result = await agent.run("question")
        # max_attempts=3 → up to 3 retrievals
        assert result.attempts == 3

    async def test_tool_calls_count_matches_attempts(self) -> None:
        retriever = StubRetriever(
            responses=[
                _result([], confidence=0.9),
                _result([], confidence=0.9),
                _result([_stored("m.foo")], confidence=0.9),
            ]
        )
        agent = LangGraphAgent(  # type: ignore[arg-type]
            retriever, config=AgentConfig(max_attempts=3)
        )
        result = await agent.run("q")
        assert len(result.tool_calls) == result.attempts


# ---------------------------------------------------------------------------
# 3. Result equivalence: Agent vs LangGraphAgent
# ---------------------------------------------------------------------------


class TestResultEquivalence:
    """Both agents must produce structurally equivalent AgentResult when
    given the same retriever / grader / synth / validator stubs."""

    async def _run_both(
        self, retriever_responses: list[RetrievalResult], config: AgentConfig | None = None
    ) -> tuple[AgentResult, AgentResult]:
        from codex_atlas.agent import Agent  # noqa: PLC0415

        synth = StitchSynthesizer()
        grader = HeuristicGrader()
        rewriter = NoopRewriter()
        validator = CitationValidator()
        cfg = config or AgentConfig()

        r1 = StubRetriever(responses=list(retriever_responses))
        r2 = StubRetriever(responses=list(retriever_responses))

        hand_rolled = Agent(  # type: ignore[arg-type]
            r1,
            synthesizer=synth,
            grader=grader,
            rewriter=rewriter,
            validator=validator,
            config=cfg,
        )
        lg_agent = LangGraphAgent(  # type: ignore[arg-type]
            r2,
            synthesizer=synth,
            grader=grader,
            rewriter=rewriter,
            validator=validator,
            config=cfg,
        )
        result_hr = await hand_rolled.run("test query")
        result_lg = await lg_agent.run("test query")
        return result_hr, result_lg

    async def test_same_answer_single_attempt(self) -> None:
        responses = [_result([_stored("m.foo")])]
        hr, lg = await self._run_both(responses)
        assert hr.answer == lg.answer

    async def test_same_citations(self) -> None:
        responses = [_result([_stored("m.foo"), _stored("m.bar")])]
        hr, lg = await self._run_both(responses)
        hr_qnames = {c.qualified_name for c in hr.citations}
        lg_qnames = {c.qualified_name for c in lg.citations}
        assert hr_qnames == lg_qnames

    async def test_same_attempts_on_rewrite(self) -> None:
        responses = [
            _result([], confidence=0.9),
            _result([_stored("m.foo")], confidence=0.9),
        ]
        hr, lg = await self._run_both(responses, config=AgentConfig(max_attempts=3))
        assert hr.attempts == lg.attempts

    async def test_same_route(self) -> None:
        responses = [_result([_stored("m.foo")], route=Route.HYBRID)]
        hr, lg = await self._run_both(responses)
        assert hr.route == lg.route

    async def test_both_return_agent_result_type(self) -> None:
        responses = [_result([_stored("m.foo")])]
        hr, lg = await self._run_both(responses)
        assert isinstance(hr, AgentResult)
        assert isinstance(lg, AgentResult)


# ---------------------------------------------------------------------------
# 4. make_agent() factory
# ---------------------------------------------------------------------------


class TestMakeAgentFactory:
    def test_default_returns_hand_rolled(self) -> None:
        from codex_atlas.agent import Agent  # noqa: PLC0415

        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever)  # type: ignore[arg-type]
        assert isinstance(agent, Agent)

    def test_explicit_hand_rolled(self) -> None:
        from codex_atlas.agent import Agent  # noqa: PLC0415

        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever, backend="hand-rolled")  # type: ignore[arg-type]
        assert isinstance(agent, Agent)

    def test_explicit_langgraph(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever, backend="langgraph")  # type: ignore[arg-type]
        assert isinstance(agent, LangGraphAgent)

    def test_env_var_selects_langgraph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_AGENT_BACKEND", "langgraph")
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever)  # type: ignore[arg-type]
        assert isinstance(agent, LangGraphAgent)

    def test_env_var_selects_hand_rolled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from codex_atlas.agent import Agent  # noqa: PLC0415

        monkeypatch.setenv("ATLAS_AGENT_BACKEND", "hand-rolled")
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever)  # type: ignore[arg-type]
        assert isinstance(agent, Agent)

    def test_backend_arg_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit ``backend=`` arg takes priority over the env var."""
        from codex_atlas.agent import Agent  # noqa: PLC0415

        monkeypatch.setenv("ATLAS_AGENT_BACKEND", "langgraph")
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = make_agent(retriever, backend="hand-rolled")  # type: ignore[arg-type]
        assert isinstance(agent, Agent)

    def test_unknown_backend_raises(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        with pytest.raises(RuntimeError, match="Unknown ATLAS_AGENT_BACKEND"):
            make_agent(retriever, backend="foobar")  # type: ignore[arg-type]

    def test_env_unknown_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_AGENT_BACKEND", "unknown-backend")
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        with pytest.raises(RuntimeError, match="Unknown ATLAS_AGENT_BACKEND"):
            make_agent(retriever)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Soft-import: RuntimeError with install hint when langgraph is absent
# ---------------------------------------------------------------------------


class TestSoftImport:
    def test_raises_runtime_error_with_install_hint(self) -> None:
        """When langgraph is not importable, ``LangGraphAgent.__init__``
        raises ``RuntimeError`` with an actionable message pointing to
        ``uv sync --extra real``."""
        # Temporarily remove the stub so _require_langgraph sees ImportError.
        saved = {k: v for k, v in sys.modules.items() if "langgraph" in k}
        for k in list(saved):
            del sys.modules[k]

        # Also patch the built-in import mechanism so ``import langgraph``
        # raises ImportError even if the real package happens to be installed.
        import builtins  # noqa: PLC0415
        original_import = builtins.__import__

        def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("langgraph"):
                raise ImportError("No module named 'langgraph'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = _failing_import  # type: ignore[assignment]
            from codex_atlas.agent_langgraph import _require_langgraph  # noqa: PLC0415

            with pytest.raises(RuntimeError, match="uv sync --extra real"):
                _require_langgraph()
        finally:
            builtins.__import__ = original_import
            # Restore stubs
            sys.modules.update(saved)

    def test_error_message_mentions_langgraph(self) -> None:
        """The RuntimeError message explicitly names langgraph."""
        saved = {k: v for k, v in sys.modules.items() if "langgraph" in k}
        for k in list(saved):
            del sys.modules[k]

        import builtins  # noqa: PLC0415
        original_import = builtins.__import__

        def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("langgraph"):
                raise ImportError("No module named 'langgraph'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = _failing_import  # type: ignore[assignment]
            from codex_atlas.agent_langgraph import _require_langgraph  # noqa: PLC0415

            with pytest.raises(RuntimeError, match="langgraph is not installed"):
                _require_langgraph()
        finally:
            builtins.__import__ = original_import
            sys.modules.update(saved)
