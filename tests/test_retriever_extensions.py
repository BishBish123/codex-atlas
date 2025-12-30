"""Tests for the retriever's neighborhood / import-chain routes + hybrid scorer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from codex_atlas.embed import FakeEncoder
from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.retriever import (
    HybridScore,
    Retriever,
    RetrieverConfig,
    Route,
    classify,
    hybrid_score,
)
from codex_atlas.store import StoredChunk


def _sym(qname: str) -> Symbol:
    return Symbol(
        qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
    )


def _stored(qname: str, score: float = 0.9, text: str = "") -> StoredChunk:
    return StoredChunk(
        chunk_id=f"x.py::{qname}::L1",
        qualified_name=qname,
        file_path="x.py",
        lineno_start=1,
        lineno_end=2,
        kind=SymbolKind.FUNCTION,
        text=text or f"def {qname.rsplit('.', 1)[-1]}(): pass\n",
        score=score,
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
                calls=[("m.a", "b"), ("m.b", "c")],
            )
        ]
    )
    return g


class TestNewClassifierRoutes:
    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            ("neighborhood of m.a", Route.NEIGHBORHOOD),
            ("everything around m.b", Route.NEIGHBORHOOD),
            ("import chain for util", Route.IMPORT_CHAIN),
            ("which modules import m.a", Route.IMPORT_CHAIN),
        ],
    )
    def test_routes(self, q: str, expected: Route) -> None:
        assert classify(q).route is expected


class TestNeighborhoodRoute:
    async def test_returns_caller_and_callee_chunks(self) -> None:
        store = AsyncMock()
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(
            FakeEncoder(dim=8),
            store,
            _mk_graph(),
            RetrieverConfig(top_k=10, neighborhood_depth=2),
        )
        result = await r.retrieve("neighborhood of m.b")
        assert result.route is Route.NEIGHBORHOOD
        names = {c.qualified_name for c in result.chunks}
        # m.a is caller; m.c is callee; m.b is the centre.
        assert "m.a" in names
        assert "m.c" in names
        assert "m.b" in names

    async def test_falls_back_to_lookup_when_unresolved(self) -> None:
        store = AsyncMock()
        store.search.return_value = [_stored("m.a")]
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=2))
        result = await r.retrieve("neighborhood of ghost_xyz")
        assert result.route is Route.NEIGHBORHOOD
        # No symbol resolution — fell through to vector top-k.
        store.search.assert_awaited_once()


class TestImportChainRoute:
    async def test_returns_modules_along_chain(self) -> None:
        store = AsyncMock()
        store.fetch_by_qualified_name.side_effect = _stored
        # Build a graph where m2 imports m1 imports util.
        g = CallGraph()
        files = [
            ParsedFile(
                file_path="util.py",
                module_name="util",
                symbols=[_sym("util")],
                chunks=[],
            ),
            ParsedFile(
                file_path="m1.py",
                module_name="m1",
                symbols=[_sym("m1")],
                chunks=[],
                imports=["util"],
            ),
            ParsedFile(
                file_path="m2.py",
                module_name="m2",
                symbols=[_sym("m2")],
                chunks=[],
                imports=["m1"],
            ),
        ]
        g.ingest(files)
        r = Retriever(FakeEncoder(dim=8), store, g, RetrieverConfig(top_k=5))
        result = await r.retrieve("which modules import util")
        assert result.route is Route.IMPORT_CHAIN
        assert "m1" in result.extra_qualified_names
        assert "m2" in result.extra_qualified_names


class TestHybridScorer:
    def test_seed_distance_zero(self) -> None:
        chunks = [_stored("a.b", score=0.9, text="hello world")]
        out = hybrid_score(
            chunks,
            query="hello",
            seeds={"a.b"},
            graph_distances={},
        )
        assert out[0].graph_distance == 0

    def test_distance_decay(self) -> None:
        chunks = [
            _stored("a.b", score=0.5, text="x"),
            _stored("a.c", score=0.5, text="x"),
        ]
        out = hybrid_score(
            chunks,
            query="anything",
            seeds=set(),
            graph_distances={"a.b": 0, "a.c": 4},
        )
        # Both have the same cosine + (irrelevant) text overlap; distance
        # makes the closer one win.
        ranked = sorted(out, key=lambda s: s.combined, reverse=True)
        assert ranked[0].qualified_name == "a.b"

    def test_text_overlap_boosts_score(self) -> None:
        # Cosine is identical; only one chunk's text contains the query word.
        chunks = [
            _stored("a.match", score=0.5, text="def match(): return query"),
            _stored("a.other", score=0.5, text="def other(): return foo"),
        ]
        out = hybrid_score(
            chunks,
            query="query token",
            seeds=set(),
            graph_distances={"a.match": 1, "a.other": 1},
        )
        ranked = sorted(out, key=lambda s: s.combined, reverse=True)
        assert ranked[0].qualified_name == "a.match"

    def test_zero_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            hybrid_score([], query="q", seeds=set(), graph_distances={}, weights=(0, 0, 0))

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            hybrid_score([], query="q", seeds=set(), graph_distances={}, weights=(-1, 1, 1))

    def test_weight_normalisation(self) -> None:
        # Weights (3, 1, 1) and (0.6, 0.2, 0.2) should produce identical
        # combined scores up to float noise (both normalise to 0.6/0.2/0.2).
        chunks = [_stored("a.b", score=0.7, text="hello")]
        a = hybrid_score(
            chunks, query="hello", seeds={"a.b"}, graph_distances={}, weights=(3, 1, 1)
        )
        b = hybrid_score(
            chunks, query="hello", seeds={"a.b"}, graph_distances={}, weights=(0.6, 0.2, 0.2)
        )
        assert pytest.approx(a[0].combined, rel=1e-9) == b[0].combined

    def test_empty_chunks_returns_empty(self) -> None:
        out = hybrid_score([], query="q", seeds=set(), graph_distances={})
        assert out == []

    def test_returns_hybridscore_dataclass(self) -> None:
        out = hybrid_score(
            [_stored("a.b", score=0.5, text="")],
            query="q",
            seeds=set(),
            graph_distances={},
        )
        assert isinstance(out[0], HybridScore)
