"""Stub: parse_python_file lands in the next commit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedFile:
    file_path: str
    module_name: str
    symbols: list = field(default_factory=list)
    chunks: list = field(default_factory=list)


def parse_python_file(file_path: Path, root: Path) -> ParsedFile:
    """Stub: replaced in the AST visitor commit."""
    return ParsedFile(file_path=str(file_path), module_name=file_path.stem)
