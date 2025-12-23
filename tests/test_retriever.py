"""Tests for the routing classifier + retriever wiring (no DB)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from codex_atlas.embed import FakeEncoder
from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.retriever import (
    Retriever,
    RetrieverConfig,
    Route,
    classify,
)
from codex_atlas.store import StoredChunk

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            ("what does Depends do", Route.LOOKUP),
            ("explain the Pydantic dependency injection", Route.LOOKUP),
            ("who calls APIRouter.add_api_route", Route.STRUCTURAL),
            ("which functions call serve_static_files", Route.STRUCTURAL),
            ("what calls helper", Route.STRUCTURAL),
            ("subclasses of BaseDepends", Route.STRUCTURAL),
            ("walk me through dependency injection", Route.SUMMARIZATION),
            ("overview of the routing module", Route.SUMMARIZATION),
            ("show me all auth-related endpoints", Route.HYBRID),
            ("show me all functions which handle payments", Route.HYBRID),
            ("end-to-end request flow", Route.HYBRID),
        ],
    )
    def test_route_assignments(self, q: str, expected: Route) -> None:
        decision = classify(q)
        assert decision.route is expected
        assert 0.0 <= decision.confidence <= 1.0

    def test_empty_defaults_to_lookup(self) -> None:
        decision = classify("   ")
        assert decision.route is Route.LOOKUP
        assert "empty-query" in decision.signals


# ---------------------------------------------------------------------------
# Retriever — uses a mock ChunkStore so no DB is needed.
# ---------------------------------------------------------------------------


def _sym(qname: str) -> Symbol:
    return Symbol(
        qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
    )


def _mk_graph() -> CallGraph:
    g = CallGraph()
    g.ingest(
        [
            ParsedFile(
                file_path="m.py",
                module_name="m",
                symbols=[_sym("m"), _sym("m.a"), _sym("m.b"), _sym("m.c")],
                chunks=[],
                imports=[],
                calls=[("m.a", "b"), ("m.a", "c")],
            )
        ]
    )
    return g


def _stored(qname: str, score: float = 0.9) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=f"def {qname.rsplit('.', maxsplit=1)[-1]}(): pass\n",
        score=score,
    )


class TestRetriever:
    async def test_lookup_uses_vector_topk(self) -> None:
        store = AsyncMock()
        store.search.return_value = [_stored("m.a"), _stored("m.b")]
        r = Retriever(FakeEncoder(dim=16), store, _mk_graph(), RetrieverConfig(top_k=2))
        result = await r.retrieve("what does a do")
        assert result.route is Route.LOOKUP
        assert [c.qualified_name for c in result.chunks] == ["m.a", "m.b"]
        store.search.assert_awaited_once()
        # encoder should have produced a 1-D vector of dim 16
        args, kwargs = store.search.call_args
        assert args[0].shape == (16,)
        assert kwargs["k"] == 2

    async def test_structural_uses_graph(self) -> None:
        store = AsyncMock()
        # fetch_by_qualified_name returns a chunk per qname requested.
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=10))
        result = await r.retrieve("who calls m.a")
        assert result.route is Route.STRUCTURAL
        # m.a has no callers in the toy graph; callees should be present.
        names = {c.qualified_name for c in result.chunks}
        assert "m.b" in names or "m.c" in names
        # And the extras list surfaces the graph traversal.
        assert "m.a" in result.extra_qualified_names

    async def test_structural_falls_back_when_no_symbol_match(self) -> None:
        store = AsyncMock()
        store.search.return_value = [_stored("m.a")]
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=2))
        result = await r.retrieve("who calls something_unknown_xyz")
        # Heuristic still classifies as structural by phrasing.
        assert result.route is Route.STRUCTURAL
        # But because no symbol was matched, we fall back to vector.
        store.search.assert_awaited_once()
        assert result.chunks[0].qualified_name == "m.a"

    async def test_hybrid_expands_via_graph(self) -> None:
        store = AsyncMock()

        async def mock_search(vec: np.ndarray, k: int = 8) -> list[StoredChunk]:
            return [_stored("m.a")]  # single seed; expansion brings in m.b, m.c

        store.search.side_effect = mock_search
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=10))
        result = await r.retrieve("show me all auth-related code")
        assert result.route is Route.HYBRID
        names = {c.qualified_name for c in result.chunks}
        assert "m.a" in names
        # graph expansion adds neighbours
        assert "m.b" in names or "m.c" in names

    async def test_summarization_uses_wider_top_k(self) -> None:
        store = AsyncMock()
        store.search.return_value = [_stored("m.a"), _stored("m.b")]
        r = Retriever(
            FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=2, summary_top_k=12)
        )
        await r.retrieve("walk me through m")
        # First call should request the wider summary top-k.
        store.search.assert_awaited_once()
        _, kwargs = store.search.call_args
        assert kwargs["k"] == 12
