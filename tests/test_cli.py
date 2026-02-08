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

import typer
from typer.testing import CliRunner

from codex_atlas import cli as cli_mod
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


class TestIndexCorpusValidation:
    """`atlas index <file.py>` (single file) used to silently produce a
    0-file walk and overwrite ``data/graph.json`` with an empty graph,
    bricking every structural query route. The CLI now rejects any
    non-directory corpus argument before it touches state."""

    def test_index_rejects_file_path(self, tmp_path: Path) -> None:
        single_file = tmp_path / "loose.py"
        single_file.write_text("def x(): pass\n")
        graph_out = tmp_path / "graph.json"
        # Pre-write a graph that the test will assert is preserved.
        graph_out.write_text('{"sentinel": "do not overwrite"}')
        result = runner.invoke(
            app,
            [
                "index",
                str(single_file),
                "--graph-out",
                str(graph_out),
                "--skip-embed",
            ],
        )
        assert result.exit_code != 0, result.stdout
        # Graph file from a previous run must still be intact.
        assert graph_out.read_text() == '{"sentinel": "do not overwrite"}'

    def test_index_rejects_missing_path(self, tmp_path: Path) -> None:
        graph_out = tmp_path / "graph.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(tmp_path / "does-not-exist"),
                "--graph-out",
                str(graph_out),
                "--skip-embed",
            ],
        )
        assert result.exit_code != 0, result.stdout
        assert not graph_out.exists()

    def test_index_rejects_empty_dir(self, tmp_path: Path) -> None:
        """A directory with no .py files must not silently wipe graph.json."""
        empty = tmp_path / "empty"
        empty.mkdir()
        graph_out = tmp_path / "graph.json"
        graph_out.write_text('{"sentinel": "preserve"}')
        result = runner.invoke(
            app,
            [
                "index",
                str(empty),
                "--graph-out",
                str(graph_out),
                "--skip-embed",
            ],
        )
        assert result.exit_code != 0, result.stdout
        assert graph_out.read_text() == '{"sentinel": "preserve"}'


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


class TestMemoryStore:
    """``--store=memory`` removes the Postgres requirement from index/ask/search/explain.

    ``atlas index --store=memory`` must dump a ``data/chunks.json`` snapshot
    that subsequent commands can read back; ``atlas ask --store=memory``
    must succeed end-to-end with no DSN env var set.
    """

    def test_index_memory_store_persists_to_disk(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # No DSN — the memory backend must not touch Postgres.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.py").write_text("def alpha():\n    return 1\n")
        graph_out = tmp_path / "graph.json"
        chunks_out = tmp_path / "chunks.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--store",
                "memory",
                "--chunks-out",
                str(chunks_out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert chunks_out.exists()
        # The persisted snapshot is a JSON object with rows + dim.
        payload = json.loads(chunks_out.read_text())
        assert payload["dim"] >= 1
        assert any(c["qualified_name"].endswith(".alpha") for c in payload["chunks"])

    def test_ask_memory_store_reads_persisted_chunks(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # Build the snapshot first via `atlas index --store=memory`.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.py").write_text(
            "def alpha():\n    return 1\n\n"
            "def beta():\n    alpha()\n    return 2\n"
        )
        graph_out = tmp_path / "graph.json"
        chunks_out = tmp_path / "chunks.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--store",
                "memory",
                "--chunks-out",
                str(chunks_out),
            ],
        )
        assert result.exit_code == 0, result.stdout

        # Now ``atlas ask --store=memory`` must work without POSTGRES_DSN.
        ask_result = runner.invoke(
            app,
            [
                "ask",
                "who calls alpha",
                "--graph",
                str(graph_out),
                "--store",
                "memory",
                "--chunks",
                str(chunks_out),
            ],
        )
        assert ask_result.exit_code == 0, ask_result.stdout
        # The agent's "Route:" header lands on stdout regardless of route.
        assert "Route:" in ask_result.stdout

    def test_full_index_then_ask_no_postgres(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # End-to-end: a fresh checkout with no DSN env var should be able
        # to index + ask. Search also goes through the same path; cover it.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.py").write_text("def gamma():\n    return 'g'\n")
        graph_out = tmp_path / "graph.json"
        chunks_out = tmp_path / "chunks.json"
        idx = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--store",
                "memory",
                "--chunks-out",
                str(chunks_out),
            ],
        )
        assert idx.exit_code == 0, idx.stdout

        srch = runner.invoke(
            app,
            [
                "search",
                "gamma",
                "--graph",
                str(graph_out),
                "--store",
                "memory",
                "--chunks",
                str(chunks_out),
                "--format",
                "json",
            ],
        )
        assert srch.exit_code == 0, srch.stdout
        payload = json.loads(srch.stdout[srch.stdout.index("{") :])
        assert "route" in payload
        assert "chunks" in payload

    def test_ask_memory_store_missing_snapshot_errors_clearly(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # No snapshot -> exit 2 + actionable message ("run `atlas index ...`").
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        graph_out = tmp_path / "graph.json"
        # Even without a graph the snapshot check fires first if it exists.
        # Use a real graph so we don't conflate the two error paths.
        graph_out.write_text('{"nodes": [], "edges": []}')
        result = runner.invoke(
            app,
            [
                "ask",
                "anything",
                "--graph",
                str(graph_out),
                "--store",
                "memory",
                "--chunks",
                str(tmp_path / "missing.json"),
            ],
        )
        assert result.exit_code == 2
        assert "atlas index" in result.stdout


class TestExplainForcesStructuralRoute:
    """`atlas explain` must force ``route_override=Route.STRUCTURAL`` so the
    classifier can't downgrade a structural intent to vector-only lookup.
    """

    def test_explain_passes_structural_route_override(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # Build the minimal hermetic env: a tiny corpus, indexed via memory.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "src"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo():\n    return 1\n")
        graph_out = tmp_path / "graph.json"
        chunks_out = tmp_path / "chunks.json"
        idx = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--graph-out",
                str(graph_out),
                "--store",
                "memory",
                "--chunks-out",
                str(chunks_out),
            ],
        )
        assert idx.exit_code == 0, idx.stdout

        # Patch Agent.run to capture the route_override kw.
        captured: dict[str, object] = {}

        from codex_atlas import agent as agent_mod  # noqa: PLC0415

        original_run = agent_mod.Agent.run

        async def _spy_run(
            self: agent_mod.Agent,
            query: str,
            *,
            route_override: object = None,
        ) -> object:
            captured["route_override"] = route_override
            captured["query"] = query
            return await original_run(self, query, route_override=route_override)  # type: ignore[arg-type]

        monkeypatch.setattr(agent_mod.Agent, "run", _spy_run)

        result = runner.invoke(
            app,
            [
                "explain",
                "a.foo",
                "--graph",
                str(graph_out),
                "--store",
                "memory",
                "--chunks",
                str(chunks_out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        # The CLI must hand the agent an explicit structural override.
        from codex_atlas.retriever import Route as _Route  # noqa: PLC0415

        assert captured["route_override"] == _Route.STRUCTURAL
        # And it must pass the BARE qualified name — no "who calls X"
        # rewrite. Under STRUCTURAL the rewrite was dead weight (the
        # extractor only needs the qname token); dropping it also keeps
        # the synthesizer's H1 header tidy.
        assert captured["query"] == "a.foo"


class TestMcpStartupFailFast:
    """`atlas mcp --store=postgres` exits 2 before opening a transport when DSN is missing."""

    def test_mcp_postgres_without_dsn_exits_2(
        self, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        result = runner.invoke(app, ["mcp", "--store", "postgres"])
        assert result.exit_code == 2, result.stdout
        flat = " ".join(result.stdout.split())
        assert "POSTGRES_DSN" in flat


class TestEvalGraphPreflight:
    """`atlas eval` must surface a clear error when graph.json is missing.

    Before this fix the `--store=memory` path crashed with a bare
    ``FileNotFoundError`` from inside ``CallGraph.load``. Users with no
    pre-built graph couldn't tell what was missing. The fix pre-flight
    checks the graph path AND adds an explicit ``--rebuild-graph`` flag
    that builds it from `--corpus` inline.
    """

    def test_eval_missing_graph_clear_error(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        # Provide a valid (non-empty) corpus so the corpus preflight
        # passes — we want to reach the missing-graph error path. The
        # corpus preflight runs BEFORE the graph check now (so a bad
        # ``--corpus`` short-circuits before the encoder model loads),
        # which means an empty-dir ``--corpus`` would surface a corpus
        # error first instead of the graph error this test exercises.
        corpus = tmp_path / "src"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo():\n    return 1\n")
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(tmp_path / "absent.json"),
                "--store",
                "memory",
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "REPORT.md"),
            ],
        )
        assert result.exit_code == 2, result.stdout
        # Rich wraps long lines; collapse whitespace before substring checks
        # so "atlas index --skip-embed" survives a soft-wrap.
        flat = " ".join(result.stdout.split())
        assert "graph.json missing" in flat
        assert "atlas index --skip-embed" in flat
        assert "--rebuild-graph" in flat

    def test_eval_rebuild_graph_rejects_missing_corpus(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # `--rebuild-graph` used to call `parse_corpus()` directly with no
        # validation, so a missing path silently produced an empty parse
        # and the harness scored every question against zero chunks.
        # The eval-time guard now mirrors `index()`'s checks.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(tmp_path / "graph.json"),
                "--store",
                "memory",
                "--corpus",
                str(tmp_path / "does-not-exist"),
                "--rebuild-graph",
                "--out",
                str(tmp_path / "REPORT.md"),
            ],
        )
        assert result.exit_code != 0, result.stdout
        flat = " ".join(result.stdout.split())
        assert "must be a directory" in flat or "missing path" in flat

    def test_eval_rebuild_graph_rejects_single_file_corpus(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # A single .py file passed as --corpus would slip past
        # `is_dir()` and produce an empty walk. Reject explicitly.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        single = tmp_path / "lone.py"
        single.write_text("def f(): pass\n")
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(tmp_path / "graph.json"),
                "--store",
                "memory",
                "--corpus",
                str(single),
                "--rebuild-graph",
                "--out",
                str(tmp_path / "REPORT.md"),
            ],
        )
        assert result.exit_code != 0, result.stdout
        flat = " ".join(result.stdout.split())
        assert "must be a directory" in flat

    def test_eval_memory_store_rejects_missing_corpus(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # The non-rebuild memory branch ALSO calls parse_corpus and used
        # to silently evaluate against an empty index. Build a graph
        # first so we get past the graph-existence preflight, then point
        # --corpus at a missing path.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "src"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo(): pass\n")
        graph_path = tmp_path / "graph.json"
        # Pre-build the graph so the eval reaches the memory-store path.
        from codex_atlas.indexer.graph import CallGraph  # noqa: PLC0415
        from codex_atlas.indexer.walker import parse_corpus  # noqa: PLC0415

        cg = CallGraph()
        cg.ingest(parse_corpus(corpus))
        cg.save(graph_path)
        # Now run eval with a non-existent --corpus.
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(graph_path),
                "--store",
                "memory",
                "--corpus",
                str(tmp_path / "does-not-exist"),
                "--out",
                str(tmp_path / "REPORT.md"),
            ],
        )
        assert result.exit_code != 0, result.stdout
        flat = " ".join(result.stdout.split())
        assert "must be a directory" in flat or "missing path" in flat

    def test_eval_bad_corpus_short_circuits_before_encoder_load(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # Regression: ``_resolve_encoder`` for any non-`fake` value loads
        # a sentence-transformers model (slow, may hit the network on a
        # cold cache). A typo'd ``--corpus`` should surface a path
        # error *before* paying that cost. Patch ``_resolve_encoder`` to
        # raise unconditionally — if the corpus check fires first
        # (correct behavior), the user sees the corpus error and the
        # encoder shim is never invoked. If it fires second (bug), the
        # encoder shim raises a generic RuntimeError mapped to exit 1.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        from codex_atlas import cli as cli_mod  # noqa: PLC0415

        encoder_called = {"n": 0}

        def boom(_name: str):  # type: ignore[no-untyped-def]
            encoder_called["n"] += 1
            raise RuntimeError("encoder load attempted before corpus validation")

        monkeypatch.setattr(cli_mod, "_resolve_encoder", boom)
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(tmp_path / "graph.json"),
                "--store",
                "memory",
                "--corpus",
                str(tmp_path / "missing-corpus"),
                "--encoder",
                "BAAI/bge-small-en-v1.5",  # would-be heavy load
                "--out",
                str(tmp_path / "REPORT.md"),
            ],
        )
        assert result.exit_code == 2, result.stdout
        flat = " ".join(result.stdout.split())
        assert "must be a directory" in flat or "missing path" in flat
        # The encoder shim must NOT have been invoked.
        assert encoder_called["n"] == 0, (
            f"_resolve_encoder ran before corpus validation "
            f"(called {encoder_called['n']}x); the heavy encoder model "
            "would have downloaded for what is just a CLI typo."
        )

    def test_eval_rebuild_graph_builds_inline(
        self, tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
    ) -> None:
        # With --rebuild-graph the harness parses `--corpus` and writes a
        # fresh graph file before evaluating. The eval then runs end-to-end
        # against the in-memory store; we just need it to exit 0 and write
        # a report.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        corpus = tmp_path / "src"
        corpus.mkdir()
        (corpus / "a.py").write_text("def foo():\n    return 1\n")
        graph_path = tmp_path / "graph.json"
        report_path = tmp_path / "REPORT.md"
        result = runner.invoke(
            app,
            [
                "eval",
                "--graph",
                str(graph_path),
                "--store",
                "memory",
                "--corpus",
                str(corpus),
                "--rebuild-graph",
                "--out",
                str(report_path),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert graph_path.exists()
        assert report_path.exists()


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


class TestDebugReraise:
    def test_bail_under_debug_reraises_original_exception(self) -> None:
        # In --debug mode, _bail() should re-raise the original exception
        # so that a debugger / runner can see the real traceback rather
        # than a bare typer.Exit(1). We toggle _DEBUG directly because
        # the only existing _bail() callsite (eval --baseline missing-file)
        # requires a live Postgres to reach.
        original = cli_mod._DEBUG
        cli_mod._DEBUG = True
        try:
            err = FileNotFoundError("missing baseline")
            try:
                cli_mod._bail("baseline missing", err)
            except FileNotFoundError as caught:
                assert caught is err
            else:  # pragma: no cover - defensive
                raise AssertionError("_bail did not re-raise under --debug")
        finally:
            cli_mod._DEBUG = original

    def test_bail_without_debug_raises_typer_exit(self) -> None:
        # In normal mode (no --debug) _bail() defaults to typer.Exit(2)
        # for user-input errors; an explicit exit_code can be passed for
        # internal errors (1).
        original = cli_mod._DEBUG
        cli_mod._DEBUG = False
        try:
            try:
                cli_mod._bail("plain user error")
            except typer.Exit as e:
                assert e.exit_code == 2
            else:  # pragma: no cover - defensive
                raise AssertionError("_bail did not raise typer.Exit")
            try:
                cli_mod._bail("plain internal error", exit_code=1)
            except typer.Exit as e:
                assert e.exit_code == 1
            else:  # pragma: no cover - defensive
                raise AssertionError("_bail did not raise typer.Exit")
        finally:
            cli_mod._DEBUG = original


class TestExitCodes:
    """User-input error paths must consistently exit 2."""

    def test_missing_postgres_dsn_exits_2(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # `_dsn()` reads POSTGRES_DSN; missing env var is a user error
        # and must surface as exit 2 (not 1). We call `_dsn()` directly
        # rather than driving a full command via the runner because the
        # async commands open extra resources that don't matter to this
        # contract.
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        original = cli_mod._DEBUG
        cli_mod._DEBUG = False
        try:
            try:
                cli_mod._dsn()
            except typer.Exit as e:
                assert e.exit_code == 2
            else:  # pragma: no cover - defensive
                raise AssertionError("_dsn did not raise typer.Exit")
        finally:
            cli_mod._DEBUG = original

    def test_unknown_mcp_transport_exits_2(self) -> None:
        # `atlas mcp --transport foo` is a bad CLI argument; typer's own
        # BadParameter handling already produces exit 2 — pin that here.
        result = runner.invoke(app, ["mcp", "--transport", "totally-bogus"])
        assert result.exit_code == 2


class TestCommandWrapper:
    """Unhandled exceptions surface as exit 1 normally; in --debug, propagate."""

    def _trigger_index_failure(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
        tmp_path: Path,
        *,
        debug: bool,
    ) -> object:
        # Inject a broken ``parse_corpus`` so the index command fails
        # while still going through the wrapper. Any unhandled exception
        # path would do — this is the most direct.
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("synthetic failure for wrapper test")

        monkeypatch.setattr(cli_mod, "parse_corpus", boom)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        graph_out = tmp_path / "graph.json"
        argv = [
            "index",
            str(corpus),
            "--graph-out",
            str(graph_out),
            "--skip-embed",
        ]
        if debug:
            argv = ["--debug", *argv]
        return runner.invoke(app, argv)

    def test_internal_error_exits_1(
        self, monkeypatch, tmp_path: Path  # type: ignore[no-untyped-def]
    ) -> None:
        result = self._trigger_index_failure(monkeypatch, tmp_path, debug=False)
        # Without --debug: clean exit 1 + friendly "internal error" message.
        assert result.exit_code == 1
        # The synthetic message is visible OR the wrapper's prefix.
        out = (result.stdout or "") + (result.stderr or "")
        assert "internal error" in out.lower() or "synthetic" in out.lower()

    def test_internal_error_in_debug_propagates(
        self, monkeypatch, tmp_path: Path  # type: ignore[no-untyped-def]
    ) -> None:
        result = self._trigger_index_failure(monkeypatch, tmp_path, debug=True)
        # With --debug: the wrapper re-raises; CliRunner surfaces the
        # original exception via ``result.exception``.
        assert result.exception is not None
        assert isinstance(result.exception, RuntimeError)
        assert "synthetic" in str(result.exception)
