"""Tests for the call graph's imports-aware resolution."""

from __future__ import annotations

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
        assert g.find_callees("caller.run") == ["lib_a.helper"]

    def test_aliased_import_local_name_is_used(self) -> None:
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
