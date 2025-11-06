"""Python AST parser that extracts the structural facts Codex-Atlas needs.

For each file we produce:

* `Symbol` records for every module, class, and function/method.
* `Chunk` records — one per function-or-method — that pair the source
  text with its location, kind, and the qualified name we'll embed it
  under.
* `imports` and `calls` lists feed the call-graph builder downstream.

We stick to the standard library `ast` module here on purpose: it covers
100% of valid Python without a tree-sitter dependency, and the chunker
only ever needs structural info (where things live), not full type
resolution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class SymbolKind(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass(frozen=True)
class Symbol:
    """One named entity in a module: module / class / function / method."""

    qualified_name: str  # e.g. "fastapi.routing.APIRouter.add_api_route"
    kind: SymbolKind
    file_path: str
    lineno_start: int
    lineno_end: int


@dataclass(frozen=True)
class Chunk:
    """A piece of source the embedder will see — one per function-or-method."""

    qualified_name: str
    kind: SymbolKind
    file_path: str
    lineno_start: int
    lineno_end: int
    text: str

    def chunk_id(self) -> str:
        return f"{self.file_path}::{self.qualified_name}::L{self.lineno_start}"


@dataclass(frozen=True)
class ParsedFile:
    """Everything we extract from one source file."""

    file_path: str
    module_name: str
    symbols: list[Symbol]
    chunks: list[Chunk]
    imports: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(
        default_factory=list,
        metadata={"description": "(caller_qualified_name, callee_unqualified_name) pairs"},
    )


# ---------------------------------------------------------------------------
# Module name resolution
# ---------------------------------------------------------------------------


def resolve_module_name(file_path: Path, root: Path) -> str:
    """Convert `<root>/foo/bar/baz.py` to `foo.bar.baz`.

    `__init__.py` collapses to its parent directory's name.
    """
    rel = file_path.resolve().relative_to(root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Parser entry point
# ---------------------------------------------------------------------------


def parse_python_file(file_path: Path, root: Path) -> ParsedFile:
    """Parse one .py file into the AST-derived facts the indexer needs."""
    source = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        # Syntax errors are common in real corpora (vendored generators,
        # stub files, etc.). Skip them quietly with an empty result so the
        # indexer keeps making progress.
        return ParsedFile(
            file_path=str(file_path),
            module_name=resolve_module_name(file_path, root),
            symbols=[],
            chunks=[],
        )

    module_name = resolve_module_name(file_path, root)
    visitor = _Collector(
        file_path=str(file_path),
        module_name=module_name,
        source_lines=source.splitlines(keepends=True),
    )
    visitor.visit(tree)
    return ParsedFile(
        file_path=str(file_path),
        module_name=module_name,
        symbols=visitor.symbols,
        chunks=visitor.chunks,
        imports=visitor.imports,
        calls=visitor.calls,
    )


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class _Collector(ast.NodeVisitor):
    """Walks the AST, accumulates symbols + chunks + imports + calls."""

    def __init__(self, file_path: str, module_name: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.module_name = module_name
        self.source_lines = source_lines
        self.symbols: list[Symbol] = []
        self.chunks: list[Chunk] = []
        self.imports: list[str] = []
        self.calls: list[tuple[str, str]] = []
        # Stack of qualified names so nested classes / methods get the right prefix.
        self._scope: list[str] = [module_name]
        # The qualified name of the function we're currently inside (or None
        # at module scope), used to attribute calls back to their caller.
        self._current_callable: str | None = None

        self.symbols.append(
            Symbol(
                qualified_name=module_name,
                kind=SymbolKind.MODULE,
                file_path=file_path,
                lineno_start=1,
                lineno_end=len(source_lines) or 1,
            )
        )

    # ---------- imports ----------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # `from foo.bar import baz` -> we record `foo.bar.baz`.
        # `from . import sibling` -> we record `.sibling` so the graph
        # builder can tell relative imports apart from absolute ones.
        dots = "." * (node.level or 0)
        module = node.module or ""
        for alias in node.names:
            if dots and not module:
                target = f"{dots}{alias.name}"
            elif module:
                target = f"{dots}{module}.{alias.name}"
            else:
                target = alias.name
            self.imports.append(target)

    # ---------- classes ----------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = ".".join((*self._scope, node.name))
        self.symbols.append(
            Symbol(
                qualified_name=qname,
                kind=SymbolKind.CLASS,
                file_path=self.file_path,
                lineno_start=node.lineno,
                lineno_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            )
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    # ---------- functions + methods ----------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_callable(node)

    def _handle_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qname = ".".join((*self._scope, node.name))
        # If the immediate parent scope is a class, this is a method.
        kind = (
            SymbolKind.METHOD
            if len(self._scope) > 1 and self._is_in_class()
            else SymbolKind.FUNCTION
        )
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        text = "".join(self.source_lines[node.lineno - 1 : end])
        self.symbols.append(
            Symbol(
                qualified_name=qname,
                kind=kind,
                file_path=self.file_path,
                lineno_start=node.lineno,
                lineno_end=end,
            )
        )
        self.chunks.append(
            Chunk(
                qualified_name=qname,
                kind=kind,
                file_path=self.file_path,
                lineno_start=node.lineno,
                lineno_end=end,
                text=text,
            )
        )

        prev_callable = self._current_callable
        self._current_callable = qname
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()
        self._current_callable = prev_callable

    # ---------- calls ----------

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_callable is not None:
            callee = _name_of_call(node.func)
            if callee:
                self.calls.append((self._current_callable, callee))
        self.generic_visit(node)

    # ---------- helpers ----------

    def _is_in_class(self) -> bool:
        """True if the immediate enclosing scope is a class (not a function)."""
        # Walk up our own symbol list to find the enclosing scope kind.
        # Cheap and avoids a parallel stack: the most-recently-added symbol
        # whose qualified name is exactly our parent scope is the parent.
        parent_qname = ".".join(self._scope)
        for sym in reversed(self.symbols):
            if sym.qualified_name == parent_qname:
                return sym.kind is SymbolKind.CLASS
        return False


def _name_of_call(node: ast.AST) -> str | None:
    """Recover a textual callee name from `Call.func`. None for unhandled shapes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        # `foo.bar.baz()` -> "baz"; the receiver type is unknown without
        # type inference, so we keep just the rightmost name.
        return node.attr
    return None
