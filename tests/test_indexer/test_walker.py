"""Tests for the corpus walker."""

from __future__ import annotations

from pathlib import Path

from codex_atlas.indexer.walker import iter_python_files, parse_corpus


def test_walker_finds_py_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n")
    (tmp_path / "README.md").write_text("not python\n")

    files = list(iter_python_files(tmp_path))
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
    assert rels == ["a.py", "pkg/b.py"]


def test_walker_skips_known_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("noop\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("noop\n")

    files = [p.name for p in iter_python_files(tmp_path)]
    assert files == ["a.py"]


def test_walker_max_files_caps_iteration(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n")
    files = list(iter_python_files(tmp_path, max_files=3))
    assert len(files) == 3


def test_parse_corpus_returns_one_record_per_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo(): pass\n")
    (tmp_path / "b.py").write_text("def bar(): pass\n")

    parsed = parse_corpus(tmp_path)
    modules = {pf.module_name for pf in parsed}
    assert modules == {"a", "b"}
    assert all(any(c.qualified_name.endswith((".foo", ".bar")) for c in pf.chunks) for pf in parsed)
