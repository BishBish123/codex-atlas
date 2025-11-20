"""Tests for the AST parser's async / decorator / docstring extensions."""

from __future__ import annotations

from pathlib import Path

from codex_atlas.indexer.ast_parser import SymbolKind, parse_python_file


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
        f = tmp_path / "m.py"
        f.write_text("X = 42\n")
        pf = parse_python_file(f, tmp_path)
        assert pf.type_aliases == []

    def test_call_assignment_skipped(self, tmp_path: Path) -> None:
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
