"""Tests for the portfolio documentation artifacts.

Covers:
- scripts/generate_cli_help.py is importable and has the expected functions.
- docs/CLI.md exists and contains all expected atlas subcommands.
- DEMO.md exists and references the asciicast, REPORT.md, and ADRs.
- The CLI help generator handles a missing/unknown subcommand gracefully.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parent.parent
CLI_MD = REPO_ROOT / "docs" / "CLI.md"
DEMO_MD = REPO_ROOT / "DEMO.md"
GENERATOR = REPO_ROOT / "scripts" / "generate_cli_help.py"


# ---------------------------------------------------------------------------
# Helper: load the generator module without executing __main__
# ---------------------------------------------------------------------------

def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_cli_help", GENERATOR)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# 1. Generator script is importable and has the expected public functions
# ---------------------------------------------------------------------------

class TestGeneratorModule:
    def test_generator_exists(self) -> None:
        assert GENERATOR.exists(), f"Expected {GENERATOR} to exist"

    def test_importable(self) -> None:
        mod = _load_generator()
        assert mod is not None

    def test_has_generate_function(self) -> None:
        mod = _load_generator()
        assert callable(getattr(mod, "generate", None)), "generate() function missing"

    def test_has_extract_subcommands(self) -> None:
        mod = _load_generator()
        assert callable(getattr(mod, "extract_subcommands", None))

    def test_has_strip_ansi(self) -> None:
        mod = _load_generator()
        assert callable(getattr(mod, "strip_ansi", None))

    def test_has_build_section(self) -> None:
        mod = _load_generator()
        assert callable(getattr(mod, "build_section", None))

    def test_strip_ansi_removes_escapes(self) -> None:
        mod = _load_generator()
        raw = "\x1b[32mgreen\x1b[0m text"
        assert mod.strip_ansi(raw) == "green text"

    def test_extract_subcommands_parses_typer_output(self) -> None:
        """Regression: continuation lines (word-wrap) must not be parsed as commands."""
        mod = _load_generator()
        # Minimal fake top-level help output mimicking Rich/Typer layout
        fake_help = (
            "╭─ Commands ───────────────────────────────────────────────────────────────────╮\n"
            "│ index      Walk `corpus`, parse every .py, embed every chunk, persist the    │\n"
            "│            graph.                                                            │\n"
            "│ ask        Run a single agent query end-to-end.                              │\n"
            "│ eval       Run the golden test set and write a markdown report.              │\n"
            "╰──────────────────────────────────────────────────────────────────────────────╯\n"
        )
        names = mod.extract_subcommands(fake_help)
        assert names == ["index", "ask", "eval"], f"Got: {names}"
        # Continuation tokens like 'graph.' must not appear
        assert "graph" not in names


# ---------------------------------------------------------------------------
# 2. docs/CLI.md exists and covers all atlas subcommands
# ---------------------------------------------------------------------------

EXPECTED_SUBCOMMANDS = ["atlas index", "atlas ask", "atlas search", "atlas explain",
                        "atlas mcp", "atlas eval", "atlas calibrate"]


class TestCliMd:
    def test_cli_md_exists(self) -> None:
        assert CLI_MD.exists(), f"Expected {CLI_MD} to exist (run `make cli-help`)"

    @pytest.mark.parametrize("subcmd", EXPECTED_SUBCOMMANDS)
    def test_contains_subcommand(self, subcmd: str) -> None:
        content = CLI_MD.read_text(encoding="utf-8")
        assert subcmd in content, f"docs/CLI.md missing section for '{subcmd}'"

    def test_has_code_blocks(self) -> None:
        content = CLI_MD.read_text(encoding="utf-8")
        assert "```" in content, "docs/CLI.md should contain fenced code blocks"

    def test_has_auto_generated_notice(self) -> None:
        content = CLI_MD.read_text(encoding="utf-8")
        assert "generate_cli_help" in content

    def test_has_synopsis_lines(self) -> None:
        content = CLI_MD.read_text(encoding="utf-8")
        assert "Synopsis" in content


# ---------------------------------------------------------------------------
# 3. DEMO.md exists and references key artifacts
# ---------------------------------------------------------------------------

class TestDemoMd:
    def test_demo_md_exists(self) -> None:
        assert DEMO_MD.exists(), "Expected DEMO.md at repo root"

    def test_references_asciinema(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        # Either direct mention of asciinema or the demo.cast file
        assert "asciinema" in content or "demo.cast" in content

    def test_references_report_md(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "REPORT.md" in content

    def test_references_adrs(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "ADR" in content

    def test_within_line_budget(self) -> None:
        lines = DEMO_MD.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 300, f"DEMO.md has {len(lines)} lines (limit 300)"

    def test_has_what_this_is_section(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "What this is" in content

    def test_has_demo_section(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "demo" in content.lower()

    def test_has_what_to_evaluate_section(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "evaluate" in content.lower() or "Evaluate" in content

    def test_has_where_to_look_section(self) -> None:
        content = DEMO_MD.read_text(encoding="utf-8")
        assert "Where to look" in content or "where to look" in content.lower()


# ---------------------------------------------------------------------------
# 4. Generator handles missing/unknown subcommand gracefully
# ---------------------------------------------------------------------------

class TestGeneratorGracefulFailure:
    def test_missing_subcommand_produces_warning_not_crash(self, tmp_path: Path) -> None:
        """build_section for a nonexistent subcommand should return a string, not raise."""
        mod = _load_generator()
        # Patch CLI_ENTRY to avoid invoking actual atlas binary
        # build_section calls run_help which calls subprocess — we can call it
        # with a fabricated subcommand name and verify the return is a string.
        section = mod.build_section("__nonexistent_subcommand__", "fallback description")
        assert isinstance(section, str)
        # Should include the subcommand name in the section header
        assert "__nonexistent_subcommand__" in section

    def test_run_help_returns_tuple_on_failure(self) -> None:
        mod = _load_generator()
        ok, output = mod.run_help(["uv", "run", "atlas", "__no_such_cmd__"])
        assert isinstance(ok, bool)
        assert isinstance(output, str)
        # The command should have failed
        assert ok is False

    def test_generate_writes_file(self, tmp_path: Path) -> None:
        """generate() must write a non-empty file to the given path."""
        mod = _load_generator()
        out = tmp_path / "CLI.md"
        content = mod.generate(output_path=out)
        assert out.exists()
        assert len(content) > 100
        assert "atlas" in content.lower()
