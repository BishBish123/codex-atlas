"""Tests for the call graph's imports-aware resolution + neighborhood + import_chain."""

from __future__ import annotations

import logging

import pytest

from codex_atlas.indexer.ast_parser import ImportRef, ParsedFile, Symbol, SymbolKind
from codex_atlas.indexer.graph import CallGraph


def _sym(qname: str, kind: SymbolKind = SymbolKind.FUNCTION) -> Symbol:
    return Symbol(qualified_name=qname, kind=kind, file_path="x.py", lineno_start=1, lineno_end=1)


def _parsed(
    *,
    module: str,
    syms: list[Symbol],
    calls: list[tuple[str, str]],
    import_refs: list[ImportRef] | None = None,
    imports: list[str] | None = None,
) -> ParsedFile:
    return ParsedFile(
        file_path=f"{module}.py",
        module_name=module,
        symbols=[_sym(module, SymbolKind.MODULE), *syms],
        chunks=[],
        imports=imports or [],
        calls=calls,
        import_refs=import_refs or [],
    )


class TestImportsAwareResolution:
    def test_imported_name_resolves_to_exact_target(self) -> None:
        # caller.run calls `helper`, but `helper` is bound to `lib_a.helper`
        # via the import. The graph should record an edge to lib_a.helper
        # only — NOT to lib_b.helper.
        a = _parsed(module="lib_a", syms=[_sym("lib_a.helper")], calls=[])
        b = _parsed(module="lib_b", syms=[_sym("lib_b.helper")], calls=[])
        c = _parsed(
            module="caller",
            syms=[_sym("caller.run")],
            calls=[("caller.run", "helper")],
            import_refs=[ImportRef(local="helper", target="lib_a.helper", level=0)],
        )
        g = CallGraph()
        g.ingest([a, b, c])
        callees = sorted(g.find_callees("caller.run"))
        assert callees == ["lib_a.helper"]

    def test_unresolved_import_target_falls_back_to_short_name(self) -> None:
        # The import points to a target that isn't in the graph; the
        # short-name fallback should still record edges to all matches.
        a = _parsed(module="lib_a", syms=[_sym("lib_a.helper")], calls=[])
        b = _parsed(module="lib_b", syms=[_sym("lib_b.helper")], calls=[])
        c = _parsed(
            module="caller",
            syms=[_sym("caller.run")],
            calls=[("caller.run", "helper")],
            import_refs=[ImportRef(local="helper", target="ghost.helper", level=0)],
        )
        g = CallGraph()
        g.ingest([a, b, c])
        callees = sorted(g.find_callees("caller.run"))
        assert callees == ["lib_a.helper", "lib_b.helper"]

    def test_no_import_uses_short_name_fallback(self) -> None:
        a = _parsed(module="lib_a", syms=[_sym("lib_a.helper")], calls=[])
        c = _parsed(
            module="caller",
            syms=[_sym("caller.run")],
            calls=[("caller.run", "helper")],
        )
        g = CallGraph()
        g.ingest([a, c])
        # No import_refs; we still match by short name.
        assert g.find_callees("caller.run") == ["lib_a.helper"]

    def test_aliased_import_local_name_is_used(self) -> None:
        # `import lib_a.helper as h` -> caller calls `h` -> should resolve.
        a = _parsed(module="lib_a", syms=[_sym("lib_a.helper")], calls=[])
        c = _parsed(
            module="caller",
            syms=[_sym("caller.run")],
            calls=[("caller.run", "h")],
            import_refs=[ImportRef(local="h", target="lib_a.helper", level=0)],
        )
        g = CallGraph()
        g.ingest([a, c])
        assert g.find_callees("caller.run") == ["lib_a.helper"]


class TestNeighborhood:
    def test_basic_two_hop_neighborhood(self) -> None:
        # a -> b -> c; b is the centre.
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b"), _sym("m.c")],
                calls=[("m.a", "b"), ("m.b", "c")],
            )
        ]
        g = CallGraph()
        g.ingest(files)
        nb = g.caller_callee_neighborhood("m.b", depth=1)
        assert "m.a" in nb["callers"]
        assert "m.c" in nb["callees"]
        assert sorted(nb["all"]) == ["m.a", "m.c"]

    def test_depth_two_includes_grandparents(self) -> None:
        # a -> b -> c
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b"), _sym("m.c")],
                calls=[("m.a", "b"), ("m.b", "c")],
            )
        ]
        g = CallGraph()
        g.ingest(files)
        nb = g.caller_callee_neighborhood("m.c", depth=2)
        assert "m.a" in nb["callers"]
        assert "m.b" in nb["callers"]

    def test_depth_one_excludes_grandparents(self) -> None:
        files = [
            _parsed(
                module="m",
                syms=[_sym("m.a"), _sym("m.b"), _sym("m.c")],
                calls=[("m.a", "b"), ("m.b", "c")],
            )
        ]
        g = CallGraph()
        g.ingest(files)
        nb = g.caller_callee_neighborhood("m.c", depth=1)
        assert "m.b" in nb["callers"]
        assert "m.a" not in nb["callers"]

    def test_neighborhood_zero_depth_rejected(self) -> None:
        g = CallGraph()
        with pytest.raises(ValueError, match="depth"):
            g.caller_callee_neighborhood("m.a", depth=0)

    def test_neighborhood_unknown_symbol_returns_empty(self) -> None:
        g = CallGraph()
        nb = g.caller_callee_neighborhood("ghost", depth=2)
        assert nb["callers"] == []
        assert nb["callees"] == []
        assert nb["all"] == []


class TestImportChain:
    def test_module_import_chain(self) -> None:
        # m1 imports util; m2 imports m1. Walking from util backward at
        # depth 2 should surface m1 and m2.
        files = [
            _parsed(module="util", syms=[], calls=[], imports=[]),
            _parsed(module="m1", syms=[], calls=[], imports=["util"]),
            _parsed(module="m2", syms=[], calls=[], imports=["m1"]),
        ]
        g = CallGraph()
        g.ingest(files)
        chain = g.import_chain("util", max_depth=3)
        assert "m1" in chain
        assert "m2" in chain

    def test_import_chain_unknown_returns_empty(self) -> None:
        g = CallGraph()
        assert g.import_chain("nope") == []

    def test_import_chain_depth_zero_rejected(self) -> None:
        g = CallGraph()
        with pytest.raises(ValueError, match="max_depth"):
            g.import_chain("util", max_depth=0)


class TestRelativeImportResolution:
    """``from . import x`` / ``from ..pkg import y`` resolve to absolute targets."""

    def test_relative_import_level_1_resolves(self) -> None:
        # ``pkg.mod`` doing ``from . import other`` must resolve to ``pkg.other``.
        # Without relative-import resolution the local name ``other`` would
        # never reach a graph node, so the imports-aware path silently fails.
        # We register the symbol ``pkg.other`` directly so a call to ``other()``
        # records the exact imports-aware edge.
        other = _parsed(module="pkg.other", syms=[_sym("pkg.other")], calls=[])
        # Decoy: a top-level ``other`` symbol exists in the corpus. With the
        # bug present, the short-name fallback would record TWO edges
        # (``pkg.other`` AND ``other``); with relative-import resolution the
        # imports-aware path wins and records exactly one.
        decoy = _parsed(module="other", syms=[_sym("other")], calls=[])
        mod = _parsed(
            module="pkg.mod",
            syms=[_sym("pkg.mod.run")],
            calls=[("pkg.mod.run", "other")],
            import_refs=[ImportRef(local="other", target="other", level=1)],
        )
        g = CallGraph()
        g.ingest([other, decoy, mod])
        # Only pkg.other — the relative import resolves to that exact node.
        assert g.find_callees("pkg.mod.run") == ["pkg.other"]

    def test_relative_import_level_2_resolves(self) -> None:
        # ``pkg.sub.mod`` doing ``from .. import other`` should resolve to
        # ``pkg.other``. We register a callable on pkg.other so the
        # imports-aware path can record the exact edge.
        other = _parsed(module="pkg.other", syms=[_sym("pkg.other")], calls=[])
        mod = _parsed(
            module="pkg.sub.mod",
            syms=[_sym("pkg.sub.mod.run")],
            calls=[("pkg.sub.mod.run", "other")],
            import_refs=[ImportRef(local="other", target="other", level=2)],
        )
        g = CallGraph()
        g.ingest([other, mod])
        callees = sorted(g.find_callees("pkg.sub.mod.run"))
        assert "pkg.other" in callees

    def test_relative_import_with_target(self) -> None:
        # ``pkg.sub.mod`` doing ``from ..util import helper`` -> ``pkg.util.helper``.
        util = _parsed(module="pkg.util", syms=[_sym("pkg.util.helper")], calls=[])
        # An unrelated module also has a ``helper`` so the short-name fallback
        # would over-match — the test asserts the imports-aware path WINS.
        decoy = _parsed(module="lib_b", syms=[_sym("lib_b.helper")], calls=[])
        mod = _parsed(
            module="pkg.sub.mod",
            syms=[_sym("pkg.sub.mod.run")],
            calls=[("pkg.sub.mod.run", "helper")],
            import_refs=[ImportRef(local="helper", target="util.helper", level=2)],
        )
        g = CallGraph()
        g.ingest([util, decoy, mod])
        # Only pkg.util.helper — the relative import disambiguates.
        assert g.find_callees("pkg.sub.mod.run") == ["pkg.util.helper"]

    def test_relative_import_escaping_package_warns_and_skips(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ``pkg`` (one part) doing ``from .. import x`` escapes the package —
        # we can't reify it. The resolver should log a warning and skip.
        # Short-name fallback still records edges so callers don't go silent.
        other = _parsed(module="other", syms=[_sym("other.x")], calls=[])
        mod = _parsed(
            module="pkg",
            syms=[_sym("pkg.run")],
            calls=[("pkg.run", "x")],
            import_refs=[ImportRef(local="x", target="x", level=2)],
        )
        g = CallGraph()
        with caplog.at_level(logging.WARNING, logger="codex_atlas.indexer.graph"):
            g.ingest([other, mod])
        assert any("relative_import_escapes_package" in rec.message for rec in caplog.records)
        # Short-name fallback still works.
        assert g.find_callees("pkg.run") == ["other.x"]
