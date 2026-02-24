"""Smoke tests for the `atlas` CLI via typer.testing.CliRunner.

These don't touch a real Postgres — they exercise the CLI surface (arg
parsing, --format json, --debug, --help) and the few code paths that
work without a DB. The full `atlas index` and `atlas eval` are covered
by integration tests against the bundled fixture corpus when a DSN is
provided.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from codex_atlas.cli import app

runner = CliRunner()


class TestHelp:
    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "atlas" in result.stdout
        assert "index" in result.stdout
        assert "eval" in result.stdout

    def test_index_help(self) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
        assert "corpus" in result.stdout.lower()

    def test_search_help(self) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0

    def test_explain_help(self) -> None:
        result = runner.invoke(app, ["explain", "--help"])
        assert result.exit_code == 0

    def test_mcp_help(self) -> None:
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        # The new flags should show up.
        assert "baseline" in result.stdout.lower()
        assert "failure-report" in result.stdout.lower()


class TestIndexJsonFormat:
    def test_index_json_skips_pgvector(self, tmp_path: Path) -> None:
        # `--skip-embed` lets us run without POSTGRES_DSN.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo(): pass\n")
        graph_out = tmp_path / "graph.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--format",
                "json",
                "--skip-embed",
            ],
        )
        assert result.exit_code == 0, result.stdout
        # The JSON payload spans multiple stdout lines; find the first '{'
        # and parse from there to the matching '}'.
        out = result.stdout.strip()
        first_brace = out.index("{")
        # The CLI emits exactly one JSON object then exits, so the rest
        # of stdout from that point is parseable.
        payload = json.loads(out[first_brace:])
        assert payload["files_parsed"] >= 1
        assert payload["graph"]["n_nodes"] >= 1
        assert payload["chunks"]["total"] >= 1
        assert graph_out.exists()


class TestDebugFlag:
    def test_debug_flag_accepted(self) -> None:
        result = runner.invoke(app, ["--debug", "--help"])
        assert result.exit_code == 0


class TestErrorHandling:
    def test_index_skip_embed_without_dsn_succeeds(self, tmp_path: Path) -> None:
        # `--skip-embed` lets us index without POSTGRES_DSN. This documents
        # the contract that the embed step is the only DSN consumer in the
        # `index` path.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo(): pass\n")
        graph_out = tmp_path / "graph.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--skip-embed",
                "--format",
                "rich",
            ],
        )
        assert result.exit_code == 0


def test_unknown_command_fails() -> None:
    result = runner.invoke(app, ["totally-not-a-command"])
    assert result.exit_code != 0
