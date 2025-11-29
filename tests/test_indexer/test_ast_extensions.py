"""Tests for the AST parser's async / decorator / type-alias / imports map extensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_atlas.indexer.ast_parser import (
    SymbolKind,
    parse_python_file,
)


class TestAsyncDetection:
    def test_async_def_marked(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("async def fetch(url): return url\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.fetch")
        assert sym.is_async is True

    def test_sync_def_not_marked_async(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def helper(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.helper")
        assert sym.is_async is False

    def test_async_method_inside_class(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("class C:\n    async def go(self): return 1\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.C.go")
        assert sym.is_async is True
        assert sym.kind is SymbolKind.METHOD


class TestDecoratorCapture:
    def test_simple_decorator(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("@staticmethod\ndef f(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.f")
        assert "staticmethod" in sym.decorators

    def test_dotted_decorator(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("@app.route\ndef view(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.view")
        assert "app.route" in sym.decorators

    def test_called_decorator(self, tmp_path: Path) -> None:
        # `@pytest.mark.asyncio` parses as a Call(Attribute(...)). The
        # parser should capture the dotted name of the call target.
        f = tmp_path / "m.py"
        f.write_text("@pytest.mark.asyncio\nasync def test_x(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.test_x")
        assert "pytest.mark.asyncio" in sym.decorators

    def test_multiple_decorators(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("@classmethod\n@cache\ndef f(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.f")
        assert "classmethod" in sym.decorators
        assert "cache" in sym.decorators

    def test_class_decorator(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("@dataclass\nclass C: pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.C")
        assert "dataclass" in sym.decorators


class TestDocstringCapture:
    def test_function_docstring(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text('def helper():\n    """Do the thing."""\n    return 1\n')
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.helper")
        assert sym.docstring == "Do the thing."

    def test_class_docstring(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text('class C:\n    """A widget."""\n    pass\n')
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.C")
        assert sym.docstring == "A widget."

    def test_no_docstring_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def f(): pass\n")
        pf = parse_python_file(f, tmp_path)
        sym = next(s for s in pf.symbols if s.qualified_name == "m.f")
        assert sym.docstring is None


class TestTypeAliases:
    def test_plain_assignment_alias(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("Path = str\n")
        pf = parse_python_file(f, tmp_path)
        targets = {a.qualified_name: a.target for a in pf.type_aliases}
        assert targets == {"m.Path": "str"}

    def test_subscripted_alias(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("Items = list[int]\n")
        pf = parse_python_file(f, tmp_path)
        names = {a.qualified_name for a in pf.type_aliases}
        assert "m.Items" in names

    def test_constant_assignment_skipped(self, tmp_path: Path) -> None:
        # `X = 42` is data, not a type alias — must be skipped.
        f = tmp_path / "m.py"
        f.write_text("X = 42\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.type_aliases == []

    def test_call_assignment_skipped(self, tmp_path: Path) -> None:
        # `X = make()` is also data; constants can't be type aliases.
        f = tmp_path / "m.py"
        f.write_text("X = make_thing()\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.type_aliases == []

    def test_pep_613_typealias_annotation(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("from typing import TypeAlias\nIds: TypeAlias = list[int]\n")
        pf = parse_python_file(f, tmp_path)
        names = {a.qualified_name for a in pf.type_aliases}
        assert "m.Ids" in names


class TestImportRefs:
    def test_simple_import(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("import os\n")
        pf = parse_python_file(f, tmp_path)
        refs = {(r.local, r.target) for r in pf.import_refs}
        assert ("os", "os") in refs

    def test_import_as_renames_local(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("import numpy as np\n")
        pf = parse_python_file(f, tmp_path)
        refs = {(r.local, r.target) for r in pf.import_refs}
        assert ("np", "numpy") in refs

    def test_from_import_records_full_target(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("from foo.bar import Baz\n")
        pf = parse_python_file(f, tmp_path)
        refs = {(r.local, r.target) for r in pf.import_refs}
        assert ("Baz", "foo.bar.Baz") in refs

    def test_from_import_with_asname(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("from foo.bar import Baz as B\n")
        pf = parse_python_file(f, tmp_path)
        refs = {(r.local, r.target) for r in pf.import_refs}
        assert ("B", "foo.bar.Baz") in refs

    def test_relative_import_records_level(self, tmp_path: Path) -> None:
        f = tmp_path / "pkg" / "m.py"
        f.parent.mkdir()
        f.write_text("from . import sibling\n")
        pf = parse_python_file(f, tmp_path)
        ref = next(r for r in pf.import_refs if r.local == "sibling")
        assert ref.level == 1


class TestLambdaAndExpressionCalls:
    def test_lambda_in_call_position_skipped(self, tmp_path: Path) -> None:
        # `(lambda x: x)()` should not produce a call edge — the callee
        # has no resolvable name.
        f = tmp_path / "m.py"
        f.write_text("def main():\n    (lambda x: x)(1)\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.calls == []

    def test_expression_call_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("def main():\n    (a + b)()\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.calls == []


@pytest.mark.parametrize(
    "src,expected_count",
    [
        ("def a(): pass\ndef b(): pass\n", 2),
        ("class C:\n    def m(self): pass\n", 1),
        ("async def go(): pass\n", 1),
    ],
)
def test_chunk_count_matches_expected(tmp_path: Path, src: str, expected_count: int) -> None:
    f = tmp_path / "m.py"
    f.write_text(src)
    pf = parse_python_file(f, tmp_path)
    assert len(pf.chunks) == expected_count
