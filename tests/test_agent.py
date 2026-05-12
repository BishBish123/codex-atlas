"""Agent loop tests using stub retriever / synthesiser / grader."""

from __future__ import annotations

from dataclasses import dataclass

from codex_atlas.agent import (
    Agent,
    AgentConfig,
    HeuristicGrader,
    Node,
    NoopRewriter,
    StitchSynthesizer,
)
from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.retriever import RetrievalResult, Route
from codex_atlas.store import StoredChunk


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
    """Returns one canned RetrievalResult per call, cycling through `responses`."""

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
# Happy path
# ---------------------------------------------------------------------------


class TestAgentHappyPath:
    async def test_returns_answer_and_citations(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = Agent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        assert "m.foo" in result.answer
        assert result.citations[0].qualified_name == "m.foo"
        assert result.attempts == 1
        assert result.route is Route.LOOKUP

    async def test_trace_records_every_node(self) -> None:
        retriever = StubRetriever(responses=[_result([_stored("m.foo")])])
        agent = Agent(retriever)  # type: ignore[arg-type]
        result = await agent.run("what does foo do")
        nodes = [e.node for e in result.trace]
        assert Node.CLASSIFY in nodes
        assert Node.RETRIEVE in nodes
        assert Node.GRADE in nodes
        assert Node.ANSWER in nodes


# ---------------------------------------------------------------------------
# Reflection / rewrite loop
# ---------------------------------------------------------------------------


class TestReflection:
    async def test_rewrites_when_first_retrieval_empty(self) -> None:
        # First retrieval is empty (grade 0) — agent should rewrite + retry.
        retriever = StubRetriever(
            responses=[
                _result([], confidence=0.9),  # empty -> grade 0
                _result([_stored("m.foo")], confidence=0.9),  # second call hits
            ]
        )
        agent = Agent(retriever, config=AgentConfig(max_attempts=3))  # type: ignore[arg-type]
        result = await agent.run("question")
        assert result.attempts >= 2
        assert any(e.node is Node.REWRITE for e in result.trace)
        assert "m.foo" in result.answer

    async def test_terminates_after_max_attempts(self) -> None:
        # Always-empty retriever — agent must stop after max_attempts and
        # render the empty-result message instead of looping forever.
        retriever = StubRetriever(responses=[_result([])])
        agent = Agent(
            retriever,  # type: ignore[arg-type]
            config=AgentConfig(max_attempts=3),
        )
        result = await agent.run("question")
        # max_attempts == 3 means up to 3 retrievals total.
        assert result.attempts == 3
        assert "could not find any relevant code chunks" in result.answer.lower()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    async def test_heuristic_grader_returns_zero(self) -> None:
        # The default grader returns 0; the agent's per-route confidence
        # floor is what makes non-empty retrievals pass the threshold.
        assert await HeuristicGrader().grade("q", [_stored("m.x")]) == 0.0

    async def test_noop_rewriter_appends_clarifier(self) -> None:
        rewriter = NoopRewriter()
        new = await rewriter.rewrite("original", [])
        assert new != "original"
        assert "original" in new

    async def test_stitch_synthesizer_includes_qualified_name(self) -> None:
        synth = StitchSynthesizer()
        out = await synth.synthesize("q", [_stored("m.foo"), _stored("m.bar")])
        assert "m.foo" in out
        assert "m.bar" in out

    async def test_stitch_synthesizer_caps_chunks(self) -> None:
        synth = StitchSynthesizer(max_chunks=2)
        out = await synth.synthesize("q", [_stored(f"m.x{i}") for i in range(10)])
        assert "m.x0" in out
        assert "m.x1" in out
        assert "m.x9" not in out

    async def test_stitch_synthesizer_renders_repo_relative_path(
        self, tmp_path: object, monkeypatch: object  # type: ignore[no-untyped-def]
    ) -> None:
        # When the chunk's file_path lives under cwd, the synthesiser must
        # render the path repo-relative so REPORT.md / answer text don't
        # leak absolute /Users/<dev>/... prefixes.
        import os as _os  # noqa: PLC0415

        # Build a chunk whose file_path is under whatever cwd we set.
        cwd = tmp_path  # type: ignore[assignment]
        sub = cwd / "src" / "codex_atlas"  # type: ignore[operator]
        sub.mkdir(parents=True)
        abs_path = sub / "agent.py"
        abs_path.write_text("def fn(): ...\n")
        monkeypatch.chdir(cwd)  # type: ignore[attr-defined]

        chunk = StoredChunk(
            chunk_id="x",
            qualified_name="codex_atlas.agent.fn",
            file_path=str(abs_path),
            lineno_start=1,
            lineno_end=1,
            kind=SymbolKind.FUNCTION,
            text="def fn(): ...\n",
            score=1.0,
        )
        synth = StitchSynthesizer()
        out = await synth.synthesize("q", [chunk])
        # Absolute path must not appear; relative form must.
        assert str(abs_path) not in out
        assert _os.path.join("src", "codex_atlas", "agent.py") in out
