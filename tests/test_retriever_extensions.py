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


class TestHybridUsesScorer:
    """``_hybrid`` ranks via ``hybrid_score``, not discovery order."""

    async def test_hybrid_prefers_scorer_top_match(self) -> None:
        # Two seeds with very different cosine. Discovery order would
        # have put both into the result in seed order; the scorer should
        # rank the higher-cosine chunk first.
        seeds = [
            _stored("m.a", score=0.10, text="alpha"),
            _stored("m.b", score=0.95, text="beta"),
        ]
        store = AsyncMock()
        store.search.return_value = seeds
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(
            FakeEncoder(dim=8),
            store,
            _mk_graph(),
            RetrieverConfig(
                top_k=2,
                # All weight on cosine for a clean assertion.
                hybrid_weight_cosine=1.0,
                hybrid_weight_graph=0.0,
                hybrid_weight_fulltext=0.0,
            ),
        )
        result = await r.retrieve("show me all auth-related code")
        assert result.route is Route.HYBRID
        # Highest-cosine seed wins regardless of original ordering.
        assert result.chunks[0].qualified_name == "m.b"

    async def test_hybrid_score_decays_with_graph_depth(self) -> None:
        # Graph: m.a -> m.b -> m.c. Seed is m.a only. With
        # ``graph_depth=2`` the 1-hop neighbour ``m.b`` and the 2-hop
        # neighbour ``m.c`` are both expanded; under the previous
        # implementation both got the SAME ``graph_depth`` distance, so
        # the scorer couldn't tell them apart. With per-node BFS depth,
        # the 1-hop neighbour gets distance=1 and outranks the 2-hop
        # neighbour at distance=2 (decay 1/(1+d)).
        seeds = [_stored("m.a", score=0.0, text="")]
        store = AsyncMock()
        store.search.return_value = seeds
        # Equal cosine for the expansions so only graph distance
        # differentiates them.
        store.fetch_by_qualified_name.side_effect = lambda q: _stored(
            q, score=0.0, text=""
        )
        r = Retriever(
            FakeEncoder(dim=8),
            store,
            _mk_graph(),
            RetrieverConfig(
                top_k=3,
                graph_depth=2,
                # Score purely on graph distance for a clean assertion.
                hybrid_weight_cosine=0.0,
                hybrid_weight_graph=1.0,
                hybrid_weight_fulltext=0.0,
            ),
        )
        result = await r.retrieve("auth-related findings here")
        assert result.route is Route.HYBRID
        names = [c.qualified_name for c in result.chunks]
        # Seed (distance 0) first, then 1-hop, then 2-hop. With
        # collapsed depths, m.b and m.c would have been tied — the
        # decay only differentiates them when real BFS depth flows
        # through.
        assert names.index("m.b") < names.index("m.c")


class TestSummarizationMaterialisesExpansions:
    """``_summarization`` returns chunks for both seed AND graph neighbours."""

    async def test_returns_neighbour_chunks(self) -> None:
        # Graph: m.a -> m.b -> m.c. Seed = m.a; neighbours within 2 hops
        # are m.b and m.c. The summarisation route must fetch chunks for
        # the neighbours so the synthesiser sees their code.
        seed = [_stored("m.a")]
        store = AsyncMock()
        store.search.return_value = seed
        # Return a stored chunk for any qname asked for so the test can
        # observe which neighbours were materialised.
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(
            FakeEncoder(dim=8),
            store,
            _mk_graph(),
            RetrieverConfig(summary_top_k=8, top_k=4),
        )
        result = await r.retrieve("walk me through m")
        assert result.route is Route.SUMMARIZATION
        names = {c.qualified_name for c in result.chunks}
        # Seed plus 2-hop neighbours.
        assert "m.a" in names
        assert "m.b" in names
        assert "m.c" in names
        # The store was asked to fetch each of the expansion qnames.
        fetched = {call.args[0] for call in store.fetch_by_qualified_name.call_args_list}
        assert {"m.b", "m.c"}.issubset(fetched)


class TestHybridScoreZeroCosine:
    """score==0.0 is the no-cosine sentinel; a chunk with a perfect cosine
    score should outrank a graph-only (score=0.0) chunk even when the
    graph-only chunk is a direct neighbour of the seed."""

    def test_graph_neighbors_dont_dominate_hybrid_with_perfect_cosine(self) -> None:
        # graph-only chunk: direct neighbour (gd=1), no cosine.
        graph_neighbour = _stored("m.neighbor", score=0.0, text="")
        # vector chunk: not a seed/neighbour, but has a perfect cosine match.
        vector_hit = _stored("m.best", score=1.0, text="")
        out = hybrid_score(
            [graph_neighbour, vector_hit],
            query="anything",
            seeds=set(),
            graph_distances={"m.neighbor": 1, "m.best": 99},
        )
        ranked = [s.qualified_name for s in out]
        # The perfect-cosine chunk must outrank the graph-only neighbour.
        assert ranked[0] == "m.best", (
            f"expected m.best first, got {ranked}; "
            f"scores: {[(s.qualified_name, s.combined) for s in out]}"
        )

    def test_zero_score_treated_as_missing_not_zero_cosine(self) -> None:
        """Two chunks with score=0.0 — combined is derived from graph/text only."""
        a = _stored("m.a", score=0.0, text="hello world query")
        b = _stored("m.b", score=0.0, text="unrelated stuff here")
        out = hybrid_score(
            [a, b],
            query="hello query",
            seeds=set(),
            graph_distances={},
        )
        # a has more text overlap → should rank higher than b.
        assert out[0].qualified_name == "m.a"
        # cosine field is 0.0 for both since score was 0.0.
        assert out[0].cosine == 0.0
        assert out[1].cosine == 0.0


class TestExtractQualifiedNameAmbiguity:
    """Short-name extraction must reject ambiguous matches deterministically.

    Earlier ``_extract_qualified_name`` returned the first node whose
    qualified name ended in the bare short token, which made the result
    depend on graph node insertion order. Two functions named ``run`` in
    different modules silently picked whichever was indexed first.
    """

    def _graph_with_two_runs(self) -> CallGraph:
        g = CallGraph()
        g.ingest(
            [
                ParsedFile(
                    file_path="a.py",
                    module_name="a",
                    symbols=[_sym("a"), _sym("a.run")],
                    chunks=[],
                    imports=[],
                    calls=[],
                ),
                ParsedFile(
                    file_path="b.py",
                    module_name="b",
                    symbols=[_sym("b"), _sym("b.run")],
                    chunks=[],
                    imports=[],
                    calls=[],
                ),
            ]
        )
        return g

    async def test_ambiguous_short_name_falls_back_to_lookup(self) -> None:
        # Both ``a.run`` and ``b.run`` end in ``.run``; the structural
        # extractor must refuse to guess and the retriever must fall
        # back to vector lookup.
        g = self._graph_with_two_runs()
        store = AsyncMock()
        store.search.return_value = [_stored("a.run")]
        r = Retriever(FakeEncoder(dim=8), store, g, RetrieverConfig(top_k=2))
        result = await r.retrieve("who calls run")
        # Classifier still picks structural by phrasing.
        assert result.route is Route.STRUCTURAL
        # But the extractor refused to resolve — fall back to vector.
        store.search.assert_awaited_once()
        # No structural extras leaked because no qname was selected.
        assert result.extra_qualified_names == []

    async def test_unique_short_name_still_resolves(self) -> None:
        # Only one node ends in ``.run``; resolution should still work.
        g = CallGraph()
        g.ingest(
            [
                ParsedFile(
                    file_path="a.py",
                    module_name="a",
                    symbols=[_sym("a"), _sym("a.run"), _sym("a.helper")],
                    chunks=[],
                    imports=[],
                    calls=[("a.helper", "run")],
                )
            ]
        )
        store = AsyncMock()
        store.fetch_by_qualified_name.side_effect = _stored
        r = Retriever(FakeEncoder(dim=8), store, g, RetrieverConfig(top_k=4))
        result = await r.retrieve("who calls run")
        assert result.route is Route.STRUCTURAL
        # Extras include the resolved target.
        assert "a.run" in result.extra_qualified_names


class TestRouteOverride:
    """Explicit ``route_override`` skips the classifier and runs the named route."""

    async def test_override_skips_classifier(self) -> None:
        # The query ``what does b do`` would normally classify as LOOKUP
        # (no structural triggers, no import-chain phrasing). With the
        # override the retriever runs the structural pipeline anyway.
        store = AsyncMock()
        store.fetch_by_qualified_name.side_effect = _stored
        store.search.return_value = [_stored("m.a")]
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=10))
        result = await r.retrieve("what does m.b do", route_override=Route.STRUCTURAL)
        assert result.route is Route.STRUCTURAL
        # Classifier was bypassed — confidence is the synthesised 1.0 and
        # signals carry the override marker.
        assert result.confidence == 1.0
        assert "route_override" in result.signals

    async def test_no_override_uses_classifier(self) -> None:
        store = AsyncMock()
        store.search.return_value = [_stored("m.a")]
        r = Retriever(FakeEncoder(dim=8), store, _mk_graph(), RetrieverConfig(top_k=2))
        result = await r.retrieve("what does m.b do")
        # Default lookup classification.
        assert result.route is Route.LOOKUP
