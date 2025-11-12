"""Unit tests for the AST parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_atlas.indexer.ast_parser import (
    SymbolKind,
    parse_python_file,
    resolve_module_name,
)

# ---------------------------------------------------------------------------
# Module name resolution
# ---------------------------------------------------------------------------


class TestResolveModuleName:
    def test_simple_module(self, tmp_path: Path) -> None:
        f = tmp_path / "foo" / "bar.py"
        f.parent.mkdir(parents=True)
        f.write_text("")
        assert resolve_module_name(f, tmp_path) == "foo.bar"

    def test_init_collapses_to_parent(self, tmp_path: Path) -> None:
        f = tmp_path / "foo" / "__init__.py"
        f.parent.mkdir(parents=True)
        f.write_text("")
        assert resolve_module_name(f, tmp_path) == "foo"


# ---------------------------------------------------------------------------
# Symbol + chunk extraction
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_module_symbol_always_first(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.symbols[0].kind is SymbolKind.MODULE
        assert pf.symbols[0].qualified_name == "m"
        assert pf.chunks == []

    def test_class_and_method_qualnames(self, tmp_path: Path) -> None:
        f = tmp_path / "pkg" / "shapes.py"
        f.parent.mkdir()
        f.write_text(
            "class Square:\n"
            "    def area(self):\n"
            "        return self.side * self.side\n"
            "\n"
            "    def perimeter(self):\n"
            "        return 4 * self.side\n"
        )
        pf = parse_python_file(f, tmp_path)
        qnames = {s.qualified_name for s in pf.symbols}
        assert "pkg.shapes" in qnames
        assert "pkg.shapes.Square" in qnames
        assert "pkg.shapes.Square.area" in qnames
        assert "pkg.shapes.Square.perimeter" in qnames

        # Both methods became chunks; the class itself did not (we chunk at
        # function granularity).
        chunk_qnames = {c.qualified_name for c in pf.chunks}
        assert chunk_qnames == {
            "pkg.shapes.Square.area",
            "pkg.shapes.Square.perimeter",
        }

    def test_method_kind_distinguished_from_function(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def free(): pass\n\nclass C:\n    def bound(self): pass\n")
        pf = parse_python_file(f, tmp_path)
        kinds = {s.qualified_name: s.kind for s in pf.symbols}
        assert kinds["m.free"] is SymbolKind.FUNCTION
        assert kinds["m.C.bound"] is SymbolKind.METHOD

    def test_async_function_recorded(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("async def fetch(url): return url\n")
        pf = parse_python_file(f, tmp_path)
        assert any(s.qualified_name == "m.fetch" for s in pf.symbols)
        assert any(c.qualified_name == "m.fetch" for c in pf.chunks)

    def test_chunk_text_includes_def_line(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def add(a, b):\n    return a + b\n")
        pf = parse_python_file(f, tmp_path)
        c = pf.chunks[0]
        assert c.text.startswith("def add")
        assert "return a + b" in c.text

    def test_chunk_id_unique_per_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def a(): pass\ndef b(): pass\n")
        pf = parse_python_file(f, tmp_path)
        ids = {c.chunk_id() for c in pf.chunks}
        assert len(ids) == len(pf.chunks)


class TestImportsAndCalls:
    def test_imports_recorded(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("import os\nfrom collections import OrderedDict\n")
        pf = parse_python_file(f, tmp_path)
        assert "os" in pf.imports
        assert "collections.OrderedDict" in pf.imports

    def test_from_import_records_base_module(self, tmp_path: Path) -> None:
        # ``from foo.bar import baz`` must record BOTH ``foo.bar`` and
        # ``foo.bar.baz`` in ``imports``. Recording only the symbol form
        # left ``CallGraph.import_chain("foo.bar")`` blind to consumers
        # that used the standard ``from ... import ...`` shape — by far
        # the most common Python import.
        f = tmp_path / "m.py"
        f.write_text("from foo.bar import baz\n")
        pf = parse_python_file(f, tmp_path)
        assert "foo.bar" in pf.imports
        assert "foo.bar.baz" in pf.imports

    def test_from_import_base_module_lets_import_chain_find_module(
        self, tmp_path: Path
    ) -> None:
        # End-to-end: parse ``from codex_atlas.store import ChunkStore``
        # and assert ``import_chain("codex_atlas.store")`` finds the
        # consumer module via the EDGE_IMPORTS reverse walk.
        from codex_atlas.indexer.graph import CallGraph  # noqa: PLC0415
        from codex_atlas.indexer.walker import parse_corpus  # noqa: PLC0415

        # Stand up a tiny package that mirrors the real-world shape.
        (tmp_path / "codex_atlas").mkdir()
        (tmp_path / "codex_atlas" / "__init__.py").write_text("")
        (tmp_path / "codex_atlas" / "store.py").write_text(
            "class ChunkStore: ...\n"
        )
        (tmp_path / "consumer.py").write_text(
            "from codex_atlas.store import ChunkStore\n"
        )
        parsed = parse_corpus(tmp_path)
        g = CallGraph()
        g.ingest(parsed)
        chain = g.import_chain("codex_atlas.store")
        assert "consumer" in chain

    def test_relative_import_keeps_dots(self, tmp_path: Path) -> None:
        f = tmp_path / "pkg" / "m.py"
        f.parent.mkdir()
        f.write_text("from . import sibling\n")
        pf = parse_python_file(f, tmp_path)
        assert any(i.startswith(".") and i.endswith("sibling") for i in pf.imports)

    def test_calls_attributed_to_caller(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def helper(): pass\ndef main():\n    helper()\n    print('hi')\n")
        pf = parse_python_file(f, tmp_path)
        callers = {(caller, callee) for caller, callee in pf.calls}
        assert ("m.main", "helper") in callers
        assert ("m.main", "print") in callers

    def test_attribute_call_keeps_only_method_name(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def main():\n    foo.bar.baz()\n")
        pf = parse_python_file(f, tmp_path)
        assert ("m.main", "baz") in {(c, n) for c, n in pf.calls}

    def test_module_level_calls_not_attributed(self, tmp_path: Path) -> None:
        """Calls outside any function shouldn't get attributed to a phantom caller."""
        f = tmp_path / "m.py"
        f.write_text("print('top-level')\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.calls == []


class TestErrorHandling:
    def test_syntax_error_returns_empty_parsed_file(self, tmp_path: Path) -> None:
        f = tmp_path / "broken.py"
        f.write_text("def (broken: syntax\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.module_name == "broken"
        assert pf.symbols == []
        assert pf.chunks == []

    @pytest.mark.parametrize("encoding_bytes", [b"\xff\xfe non-utf8 \x00\x00"])
    def test_non_utf8_falls_back_to_replace(self, tmp_path: Path, encoding_bytes: bytes) -> None:
        f = tmp_path / "weird.py"
        f.write_bytes(encoding_bytes)
        pf = parse_python_file(f, tmp_path)
        # Either the parse fails (empty result) or it succeeds; either way,
        # we don't crash on bad bytes.
        assert pf.module_name == "weird"
