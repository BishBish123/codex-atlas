"""Unit tests for the CallGraph."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph


def _sym(qname: str, kind: SymbolKind = SymbolKind.FUNCTION) -> Symbol:
    return Symbol(qualified_name=qname, kind=kind, file_path="x.py", lineno_start=1, lineno_end=1)


def _parsed(*, module: str, syms: list[Symbol], calls: list[tuple[str, str]]) -> ParsedFile:
    return ParsedFile(
        file_path=f"{module}.py",
        module_name=module,
        symbols=[_sym(module, SymbolKind.MODULE), *syms],
        chunks=[],
        imports=[],
        calls=calls,
    )


class TestIngest:
    def test_basic_call_graph(self) -> None:
        a = _parsed(
            module="m",
            syms=[_sym("m.caller"), _sym("m.callee")],
            calls=[("m.caller", "callee")],
        )
        g = CallGraph()
        g.ingest([a])
        assert g.find_callees("m.caller") == ["m.callee"]
        assert g.find_callers("m.callee") == ["m.caller"]

    def test_self_call_ignored(self) -> None:
        a = _parsed(
            module="m",
            syms=[_sym("m.recurse")],
            calls=[("m.recurse", "recurse")],
        )
        g = CallGraph()
        g.ingest([a])
        assert g.find_callees("m.recurse") == []

    def test_multiple_resolutions_for_same_short_name(self) -> None:
        # `helper` exists in two modules; calling `helper()` from a third
        # module should record an edge to *both* candidates.
        a = _parsed(module="util_a", syms=[_sym("util_a.helper")], calls=[])
        b = _parsed(module="util_b", syms=[_sym("util_b.helper")], calls=[])
        c = _parsed(
            module="caller",
            syms=[_sym("caller.run")],
            calls=[("caller.run", "helper")],
        )
        g = CallGraph()
        g.ingest([a, b, c])
        callees = sorted(g.find_callees("caller.run"))
        assert callees == ["util_a.helper", "util_b.helper"]


class TestQueries:
    def test_depth_two_traversal(self) -> None:
        # a -> b -> c -> d
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b"), _sym("m.c"), _sym("m.d")],
                calls=[("m.a", "b"), ("m.b", "c"), ("m.c", "d")],
            )
        ]
        g = CallGraph()
        g.ingest(files)
        # depth=1 sees only direct callees.
        assert g.find_callees("m.a", depth=1) == ["m.b"]
        # depth=2 includes c.
        assert sorted(g.find_callees("m.a", depth=2)) == ["m.b", "m.c"]
        # depth=10 — graph is a chain so we go to the end.
        assert sorted(g.find_callees("m.a", depth=10)) == ["m.b", "m.c", "m.d"]

    def test_unknown_symbol_returns_empty(self) -> None:
        g = CallGraph()
        assert g.find_callees("nope") == []
        assert g.find_callers("nope") == []
        assert g.neighbors("nope") == []

    def test_zero_depth_rejected(self) -> None:
        g = CallGraph()
        g.ingest([_parsed(module="m", syms=[_sym("m.a")], calls=[])])
        with pytest.raises(ValueError, match="depth"):
            g.find_callees("m.a", depth=0)
        with pytest.raises(ValueError, match="depth"):
            g.find_callers("m.a", depth=0)

    def test_neighbors_combines_directions(self) -> None:
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b"), _sym("m.c")],
                calls=[("m.a", "b"), ("m.c", "b")],
            )
        ]
        g = CallGraph()
        g.ingest(files)
        # b's neighbors: callers (a, c)
        assert sorted(g.neighbors("m.b")) == ["m.a", "m.c"]


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b")],
                calls=[("m.a", "b")],
            )
        ]
        g = CallGraph()
        g.ingest(files)

        out = g.save(tmp_path / "graph.json")
        loaded = CallGraph.load(out)

        assert loaded.n_nodes == g.n_nodes
        assert loaded.n_edges == g.n_edges
        assert loaded.find_callees("m.a") == ["m.b"]
        assert loaded.find_callers("m.b") == ["m.a"]
