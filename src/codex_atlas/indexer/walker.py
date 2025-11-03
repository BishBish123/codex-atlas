"""Walk a directory tree and parse every Python file we find."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from codex_atlas.indexer.ast_parser import ParsedFile, parse_python_file

# Directories we always skip — vendored deps and build artifacts mostly add
# noise without changing the architectural picture.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
    }
)


def iter_python_files(root: Path, max_files: int | None = None) -> Iterator[Path]:
    """Yield every `.py` file under `root`, skipping known noise directories."""
    root = root.resolve()
    yielded = 0
    for path in sorted(root.rglob("*.py")):
        if _is_in_skip_dir(path, root):
            continue
        yield path
        yielded += 1
        if max_files is not None and yielded >= max_files:
            return


def _is_in_skip_dir(path: Path, root: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.relative_to(root).parts)


def parse_corpus(root: Path, max_files: int | None = None) -> list[ParsedFile]:
    """Parse every Python file under `root` (capped at `max_files`)."""
    root = root.resolve()
    out: list[ParsedFile] = []
    for path in iter_python_files(root, max_files=max_files):
        out.append(parse_python_file(path, root))
    return out
