"""`atlas` CLI: index a corpus, ask questions, run the golden test set."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from codex_atlas.agent import Agent
from codex_atlas.embed import Encoder, FakeEncoder
from codex_atlas.eval.golden import load_golden_set
from codex_atlas.eval.harness import (
    aggregate,
    evaluate_against_baseline,
    render_report,
    run_eval,
    write_failure_report,
)
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.indexer.walker import parse_corpus
from codex_atlas.retriever import Retriever, RetrieverConfig, Route
from codex_atlas.store import ChunkStore, ChunkStoreProtocol, InMemoryChunkStore

app = typer.Typer(
    name="atlas",
    help="Codex-Atlas: agentic GraphRAG over a codebase, exposed as an MCP server.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


_DEBUG = False


@app.callback()
def _root(
    debug: bool = typer.Option(False, "--debug", help="Print full tracebacks on error."),
) -> None:
    """Top-level option carrier — flips the global debug flag."""
    global _DEBUG  # noqa: PLW0603
    _DEBUG = debug


def _bail(message: str, exc: Exception | None = None, exit_code: int = 2) -> NoReturn:
    """Render an error nicely; in --debug mode re-raise so the original
    exception (and its full traceback) bubbles up for debuggers.

    Exit-code convention (matches the rest of the CLI):

      * ``2`` — user-input error (bad args, missing file, malformed JSON,
        baseline regression). This is the default.
      * ``1`` — unexpected internal error (bug, unhandled exception).
      * ``0`` — success (never reached via _bail).
    """
    console.print(f"[red bold]error[/] {message}")
    if _DEBUG and exc is not None:
        console.print(traceback.format_exc())
        # Re-raise so callers running under --debug get the original
        # exception in the runner / pdb / launcher rather than a bare
        # SystemExit. Wrapping with chaining preserves the message.
        raise exc
    raise typer.Exit(code=exit_code)


@contextmanager
def _command_wrapper() -> Iterator[None]:
    """Catch unhandled exceptions inside any CLI command body.

    Without this every command had its own scattered ``try/except`` and
    ``--debug`` only fired for the two early-exit paths in ``_bail``
    (missing DSN + missing baseline). Wrapping every command body
    surfaces tracebacks in ``--debug`` mode and falls back to a clean
    "internal error" message + exit 1 in normal mode for any other
    failure. ``typer.Exit`` and ``typer.BadParameter`` pass through —
    those are intentional, structured exits.
    """
    try:
        yield
    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:
        if _DEBUG:
            console.print(traceback.format_exc())
            raise
        console.print(f"[red bold]internal error[/] {e}")
        raise typer.Exit(code=1) from e


def _resolve_encoder(name: str) -> Encoder:
    if name == "fake":
        return FakeEncoder(dim=32)
    from codex_atlas.embed import load_sentence_transformer_encoder  # noqa: PLC0415

    return load_sentence_transformer_encoder(model_name=name)


def _dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        # User-input error — exit 2 (matches the convention in `_bail`).
        # We use _bail rather than typer.BadParameter because BadParameter
        # raised from inside an async coroutine propagates out as a
        # generic exception (exit 1) in typer; _bail standardises the
        # exit code regardless of where it is raised.
        _bail(
            "POSTGRES_DSN env var is required "
            "(e.g. postgresql://bench:bench@localhost:5433/bench)"
        )
    return dsn


# Default location for the on-disk InMemoryChunkStore snapshot. Kept next
# to ``data/graph.json`` so a memory-store user has a single ``data/``
# directory holding the whole hermetic index.
DEFAULT_MEMORY_STORE_PATH = Path("data/chunks.json")


async def _open_query_store(
    *, store_backend: str, dim: int, memory_path: Path
) -> ChunkStoreProtocol:
    """Open a chunk store for read-only query commands (ask/search/explain/mcp).

    ``memory`` reads a snapshot written by ``atlas index --store=memory``;
    ``postgres`` opens a pgvector pool via ``POSTGRES_DSN``. Anything
    else is rejected with a typed user error so the CLI surfaces a clean
    exit 2 rather than a stack trace.
    """
    if store_backend == "memory":
        if not memory_path.exists():
            _bail(
                f"in-memory chunk store snapshot not found at {memory_path}; "
                f"run `atlas index --store=memory` first to build it"
            )
        store = await InMemoryChunkStore.load_from_path(memory_path)
        return store
    if store_backend == "postgres":
        pg_store = ChunkStore(dsn=_dsn())
        await pg_store.setup(dim=dim)
        return pg_store
    _bail(f"unknown --store value {store_backend!r} (expected: memory, postgres)")


@app.command()
def index(
    corpus: Path = typer.Argument(..., help="Directory to index."),
    graph_out: Path = typer.Option(
        Path("data/graph.json"), help="Where to persist the call graph."
    ),
    encoder: str = typer.Option(
        "fake", help="`fake` (deterministic blake2b) or any sentence-transformers model id."
    ),
    max_files: int | None = typer.Option(None, help="Cap files for fast smoke runs."),
    drop_existing: bool = typer.Option(
        False, "--drop-existing", help="DROP TABLE before reinserting."
    ),
    output_format: str = typer.Option(
        "rich",
        "--format",
        help="`rich` (default; pretty Console output) or `json` (machine-readable index dump).",
    ),
    skip_embed: bool = typer.Option(
        False,
        "--skip-embed",
        help="Skip pgvector upsert. Useful with --format json for graph-only debug.",
    ),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; serialises to data/chunks.json — no DB) or "
        "`postgres` (pgvector via POSTGRES_DSN).",
    ),
    chunks_out: Path = typer.Option(
        DEFAULT_MEMORY_STORE_PATH,
        "--chunks-out",
        help="Where to persist the in-memory chunk snapshot when --store=memory.",
    ),
) -> None:
    """Walk `corpus`, parse every .py, embed every chunk, persist the graph."""
    # Reject non-directory corpus paths up front: passing a single .py
    # file (or a missing path) used to silently produce a 0-file walk
    # which then *overwrote* `data/graph.json` with an empty graph,
    # bricking every structural query route. Validate before any state
    # write so the user gets a typer-shaped error and the previous graph
    # stays intact.
    if not corpus.is_dir():
        what = "missing path" if not corpus.exists() else "file"
        raise typer.BadParameter(
            f"CORPUS must be a directory; got {what}: {corpus}. "
            "To index a single file, pass a directory containing only that file.",
            param_hint="CORPUS",
        )

    async def _run() -> None:  # noqa: PLR0912
        encoder_obj = _resolve_encoder(encoder)
        if output_format == "rich":
            console.print(f"[green]parsing[/] {corpus}")
        parsed = parse_corpus(corpus, max_files=max_files)
        if not parsed:
            # Defensive second line: even with a directory corpus the
            # walker can return zero files (pure non-Python tree, all
            # files filtered, etc.). We refuse to overwrite graph.json
            # with an empty graph in that case — the user almost
            # certainly intended a different path.
            raise typer.BadParameter(
                f"0 .py files found under {corpus}; aborting to avoid wiping "
                f"{graph_out} with an empty graph.",
                param_hint="CORPUS",
            )
        if output_format == "rich":
            console.print(f"[green]parsed[/] {len(parsed)} files")

        graph = CallGraph()
        graph.ingest(parsed)
        graph.save(graph_out)
        if output_format == "rich":
            console.print(
                f"[green]graph[/] {graph.n_nodes} nodes, {graph.n_edges} edges -> {graph_out}"
            )

        chunks = [c for pf in parsed for c in pf.chunks]

        if output_format == "json":
            payload = {
                "corpus": str(corpus),
                "files_parsed": len(parsed),
                "graph": {
                    "path": str(graph_out),
                    "n_nodes": graph.n_nodes,
                    "n_edges": graph.n_edges,
                },
                "chunks": {
                    "total": len(chunks),
                    "by_kind": {
                        k: sum(1 for c in chunks if str(c.kind) == k)
                        for k in {str(c.kind) for c in chunks}
                    },
                },
                "encoder": encoder_obj.name,
                "store": store_backend,
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        if skip_embed:
            return
        if not chunks:
            if output_format == "rich":
                console.print("[yellow]no chunks to embed[/]")
            return

        if output_format == "rich":
            console.print(f"[green]embedding[/] {len(chunks)} chunks via {encoder_obj.name}")
        vectors = encoder_obj.encode([c.text for c in chunks])

        if store_backend == "memory":
            mem_store = InMemoryChunkStore()
            await mem_store.setup(dim=encoder_obj.dim, drop_existing=drop_existing)
            n_written = await mem_store.upsert_chunks(chunks, vectors)
            await mem_store.save_to_path(chunks_out)
            if output_format == "rich":
                console.print(
                    f"[green]wrote[/] {n_written} chunks to {chunks_out} (in-memory store)"
                )
            return
        if store_backend != "postgres":
            _bail(
                f"unknown --store value {store_backend!r} "
                f"(expected: memory, postgres)"
            )

        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim, drop_existing=drop_existing)
        # Tombstone every chunk row that belongs to a file we just
        # parsed BEFORE inserting the new chunks. ``Chunk.chunk_id`` is
        # now stable on (file_path, qualified_name) so symbols that
        # merely moved keep their row through the UPSERT — but symbols
        # that were renamed or deleted in this file would otherwise
        # leave ghost rows that surface during retrieval. Deleting by
        # file_path first turns reindex into a true idempotent
        # operation. ``drop_existing=True`` already wipes the whole
        # table, so the per-file delete is redundant in that mode.
        if not drop_existing:
            touched_files = {c.file_path for c in chunks}
            for fp in touched_files:
                await store.delete_by_file_path(fp)
        n_written = await store.upsert_chunks(chunks, vectors)
        if output_format == "rich":
            console.print(f"[green]upserted[/] {n_written} chunks into pgvector")

    with _command_wrapper():
        asyncio.run(_run())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-form question to ask the agent."),
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    top_k: int = typer.Option(8, help="Top-k retrieval cap."),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; reads data/chunks.json — no DB) or "
        "`postgres` (pgvector via POSTGRES_DSN).",
    ),
    chunks_path: Path = typer.Option(
        DEFAULT_MEMORY_STORE_PATH,
        "--chunks",
        help="Path to the chunk snapshot when --store=memory.",
    ),
) -> None:
    """Run a single agent query end-to-end."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = await _open_query_store(
            store_backend=store_backend,
            dim=encoder_obj.dim,
            memory_path=chunks_path,
        )
        retriever = Retriever(encoder_obj, store, cg, RetrieverConfig(top_k=top_k))
        agent = Agent(retriever)
        result = await agent.run(question)
        console.print(
            f"[bold]Route:[/] {result.route} (grade {result.grade:.2f}, attempts {result.attempts})"
        )
        console.print(Markdown(result.answer))
        if result.citations:
            t = Table(title="Citations")
            t.add_column("Symbol")
            t.add_column("File")
            t.add_column("Lines")
            for c in result.citations:
                t.add_row(c.qualified_name, c.file_path, f"{c.lineno_start}-{c.lineno_end}")
            console.print(t)

    with _command_wrapper():
        asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-form code search query."),
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    top_k: int = typer.Option(8, help="Top-k retrieval cap."),
    output_format: str = typer.Option(
        "rich",
        "--format",
        help="`rich` (default) or `json` for one-shot machine-readable output.",
    ),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; reads data/chunks.json — no DB) or "
        "`postgres` (pgvector via POSTGRES_DSN).",
    ),
    chunks_path: Path = typer.Option(
        DEFAULT_MEMORY_STORE_PATH,
        "--chunks",
        help="Path to the chunk snapshot when --store=memory.",
    ),
) -> None:
    """One-shot retrieval — runs the router but skips the answer synthesis."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = await _open_query_store(
            store_backend=store_backend,
            dim=encoder_obj.dim,
            memory_path=chunks_path,
        )
        retriever = Retriever(encoder_obj, store, cg, RetrieverConfig(top_k=top_k))
        result = await retriever.retrieve(query)
        if output_format == "json":
            sys.stdout.write(
                json.dumps(
                    {
                        "route": str(result.route),
                        "confidence": result.confidence,
                        "signals": result.signals,
                        "chunks": [
                            {
                                "qualified_name": c.qualified_name,
                                "file_path": c.file_path,
                                "lineno_start": c.lineno_start,
                                "lineno_end": c.lineno_end,
                                "score": c.score,
                            }
                            for c in result.chunks
                        ],
                        "extra_qualified_names": result.extra_qualified_names,
                    },
                    indent=2,
                )
                + "\n"
            )
            return
        console.print(f"[bold]Route:[/] {result.route} (confidence {result.confidence:.2f})")
        if not result.chunks:
            console.print("[yellow]no chunks[/]")
            return
        t = Table(title="Top chunks")
        t.add_column("Symbol")
        t.add_column("File")
        t.add_column("Lines")
        t.add_column("Score")
        for c in result.chunks:
            t.add_row(
                c.qualified_name,
                c.file_path,
                f"{c.lineno_start}-{c.lineno_end}",
                f"{c.score:.3f}",
            )
        console.print(t)

    with _command_wrapper():
        asyncio.run(_run())


@app.command()
def explain(
    qualified_name: str = typer.Argument(..., help="Qualified name to explain."),
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; reads data/chunks.json — no DB) or "
        "`postgres` (pgvector via POSTGRES_DSN).",
    ),
    chunks_path: Path = typer.Option(
        DEFAULT_MEMORY_STORE_PATH,
        "--chunks",
        help="Path to the chunk snapshot when --store=memory.",
    ),
) -> None:
    """One-shot agent run anchored on a qualified name.

    Forces ``route=structural`` (graph-walk) so the classifier never
    rewrites a structural intent — passing a bare qualified name like
    ``pkg.mod.Cls.method`` previously sometimes routed as ``lookup``
    (vector-only) when the heuristic missed the "who calls" prefix.
    The structural route walks the call graph directly, which is what
    every ``explain`` caller actually wants.

    Query: passed straight through as the bare ``qualified_name``. The
    structural extractor (``_extract_qualified_name``) only needs the
    qname token and ignores any prefix — the previous "who calls X"
    rewrite was inherited from the era when the classifier had to see
    the keyword phrase to pick the route. With ``route_override`` doing
    that selection up-front, the rewrite is dead weight; dropping it
    also produces a cleaner answer header (``# pkg.mod.fn`` vs
    ``# who calls pkg.mod.fn``) in the synth output.
    """

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = await _open_query_store(
            store_backend=store_backend,
            dim=encoder_obj.dim,
            memory_path=chunks_path,
        )
        retriever = Retriever(encoder_obj, store, cg, RetrieverConfig(top_k=8))
        agent = Agent(retriever)
        result = await agent.run(
            qualified_name,
            route_override=Route.STRUCTURAL,
        )
        console.print(
            f"[bold]Route:[/] {result.route} (grade {result.grade:.2f}, attempts {result.attempts})"
        )
        console.print(Markdown(result.answer))

    with _command_wrapper():
        asyncio.run(_run())


@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="MCP transport: stdio | http"),
    host: str = typer.Option("127.0.0.1", help="HTTP bind host."),
    port: int = typer.Option(8090, help="HTTP bind port."),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; reads data/chunks.json — no DB) or "
        "`postgres` (pgvector via POSTGRES_DSN).",
    ),
    chunks_path: Path = typer.Option(
        DEFAULT_MEMORY_STORE_PATH,
        "--chunks",
        help="Path to the chunk snapshot when --store=memory.",
    ),
) -> None:
    """Start the Codex-Atlas MCP server."""
    # Forward store choice to the MCP server via env. The server reads
    # ``ATLAS_STORE`` and ``ATLAS_CHUNKS_PATH`` at first-tool-call time.
    if store_backend not in {"memory", "postgres"}:
        _bail(f"unknown --store value {store_backend!r} (expected: memory, postgres)")
    os.environ["ATLAS_STORE"] = store_backend
    os.environ["ATLAS_CHUNKS_PATH"] = str(chunks_path)

    from codex_atlas.mcp_server import mcp as mcp_app  # noqa: PLC0415
    from codex_atlas.mcp_server import validate_startup_config  # noqa: PLC0415

    with _command_wrapper():
        # Fail-fast on bad startup config and map the error to exit 2,
        # instead of letting the wrapper turn it into the generic
        # "internal error" exit 1.
        try:
            validate_startup_config()
        except RuntimeError as e:
            _bail(str(e), e)

        if transport == "stdio":
            asyncio.run(mcp_app.run_stdio_async())
        elif transport == "http":
            asyncio.run(mcp_app.run_http_async(host=host, port=port))
        else:
            raise typer.BadParameter(f"unknown transport {transport!r}")


def _require_corpus_dir_and_parse(corpus: Path) -> list:  # type: ignore[type-arg]
    """Validate ``corpus`` is a non-empty directory and return its parse.

    Mirrors the same guards ``index()`` runs before any state write:
    a missing path or single ``.py`` file would otherwise produce an
    empty parse, which silently evaluates against zero chunks rather
    than failing — turning every retrieval into an unanswerable query.
    Both eval-time entry points (``--rebuild-graph`` and
    ``--store=memory``) must validate before parsing. ``_bail`` is used
    over ``typer.BadParameter`` because the call sites live inside
    ``asyncio.run(_run())`` and BadParameter raised through the asyncio
    boundary doesn't reach typer's pretty-printer — the user would see
    an empty stderr and a bare ``SystemExit(2)``.
    """
    if not corpus.is_dir():
        what = "missing path" if not corpus.exists() else "file"
        _bail(
            f"--corpus must be a directory; got {what}: {corpus}. "
            "To evaluate against a single file, pass a directory containing only that file."
        )
    parsed = parse_corpus(corpus)
    if not parsed:
        _bail(
            f"0 .py files found under {corpus}; aborting eval — "
            f"the harness would otherwise score every question against an empty index."
        )
    return parsed


@app.command(name="eval")
def eval_cmd(  # noqa: PLR0915
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    out: Path = typer.Option(Path("evals/REPORT.md"), help="Where to write the markdown report."),
    json_out: Path | None = typer.Option(None, help="Optional JSON dump of per-question scores."),
    failure_report: Path | None = typer.Option(
        None,
        "--failure-report",
        help="Optional JSONL dump for per-question manual debugging.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Path to baseline JSON (aggregate metrics). Exits non-zero on regression.",
    ),
    baseline_out: Path | None = typer.Option(
        None,
        "--baseline-out",
        help="Write the aggregate metrics of this run to a baseline JSON file. "
        "Use to refresh `evals/baseline.json` after material code changes.",
    ),
    tolerance: float = typer.Option(
        0.05, help="Fractional tolerance for the baseline regression gate."
    ),
    store_backend: str = typer.Option(
        "memory",
        "--store",
        help="`memory` (default; dict-backed, rebuilds the index from the "
        "corpus on the fly — no DB required) or `postgres` "
        "(pgvector via POSTGRES_DSN).",
    ),
    corpus: Path = typer.Option(
        Path("src"),
        "--corpus",
        help="Source tree to index when --store=memory.",
    ),
    rebuild_graph: bool = typer.Option(
        False,
        "--rebuild-graph",
        help="Parse `--corpus` and write `--graph` from scratch before evaluating.",
    ),
) -> None:
    """Run the golden test set and write a markdown + (optional) JSON report."""

    async def _run() -> None:  # noqa: PLR0912, PLR0915
        encoder_obj = _resolve_encoder(encoder)
        # If --rebuild-graph is set, parse the corpus and (re)write the
        # graph file BEFORE we try to load it. This is the explicit
        # opt-in that pairs with the actionable error below — users who
        # don't have a graph yet can run `atlas eval --rebuild-graph
        # --corpus src` instead of having to remember the two-step
        # `atlas index --skip-embed` invocation.
        if rebuild_graph:
            parsed = _require_corpus_dir_and_parse(corpus)
            cg_built = CallGraph()
            cg_built.ingest(parsed)
            cg_built.save(graph)
            console.print(
                f"[green]rebuilt[/] graph "
                f"({cg_built.n_nodes} nodes, {cg_built.n_edges} edges) -> {graph}"
            )
        if not graph.exists():
            _bail(
                f"graph.json missing at {graph} — run "
                f"`atlas index --skip-embed --corpus <path>` first, or pass "
                f"--rebuild-graph (with --corpus) to build it inline."
            )
        cg = CallGraph.load(graph)
        store: ChunkStoreProtocol
        if store_backend == "memory":
            # Dict-backed: rebuild the embedding index from `corpus` so
            # the harness stays hermetic. No DSN, no pgvector.
            store = InMemoryChunkStore()
            await store.setup(dim=encoder_obj.dim)
            parsed = _require_corpus_dir_and_parse(corpus)
            chunks = [c for pf in parsed for c in pf.chunks]
            if chunks:
                vectors = encoder_obj.encode([c.text for c in chunks])
                await store.upsert_chunks(chunks, vectors)
        elif store_backend == "postgres":
            store = ChunkStore(dsn=_dsn())
            await store.setup(dim=encoder_obj.dim)
        else:
            _bail(f"unknown --store value {store_backend!r} (expected: postgres, memory)")
        retriever = Retriever(encoder_obj, store, cg, RetrieverConfig(top_k=8))
        agent = Agent(retriever)
        questions = load_golden_set()
        results = await run_eval(agent, questions)
        report = render_report(results)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        console.print(f"[green]wrote[/] {out}")
        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    [
                        {
                            "qid": r.qid,
                            "category": r.category,
                            "expected_route": str(r.expected_route),
                            "actual_route": str(r.actual_route),
                            "route_correct": r.route_correct,
                            "citation_recall": r.citation_recall,
                            "citation_precision": r.citation_precision,
                            "latency_ms": r.latency_ms,
                            "attempts": r.attempts,
                            "tool_call_count": r.tool_call_count,
                            "cost_estimate_usd": r.cost_estimate_usd,
                            "failure_bucket": str(r.failure_bucket),
                            "cited_qualified_names": r.cited_qualified_names,
                        }
                        for r in results
                    ],
                    indent=2,
                )
            )
            console.print(f"[green]wrote[/] {json_out}")
        if failure_report is not None:
            write_failure_report(results, failure_report)
            console.print(f"[green]wrote[/] {failure_report}")
        if baseline_out is not None:
            # Write the aggregate metrics as the canonical baseline.
            # Pin a 5ms absolute slack on latency percentiles so the CI
            # gate doesn't flag wall-clock jitter on hot CPUs (single-
            # digit-ms latencies move ~25% from one run to the next).
            baseline_payload: dict[str, object] = {
                "_comment": (
                    "Real measurements from the canonical "
                    "`atlas eval --store=memory` run against the "
                    "bundled corpus. Re-generate after material code "
                    "changes with `atlas eval --store=memory "
                    "--baseline-out evals/baseline.json`."
                ),
                "latency_tolerance_ms": 5.0,
                **aggregate(results),
            }
            baseline_out.parent.mkdir(parents=True, exist_ok=True)
            baseline_out.write_text(json.dumps(baseline_payload, indent=2))
            console.print(f"[green]wrote[/] {baseline_out}")
        if baseline is not None:
            try:
                diff = evaluate_against_baseline(results, baseline, tolerance=tolerance)
            except FileNotFoundError as e:
                _bail(f"baseline file not found: {baseline}", e)
                return
            for metric, (b, c) in diff.regressions.items():
                console.print(f"[red]regression[/] {metric}: baseline={b:.4f} now={c:.4f}")
            for metric, (b, c) in diff.improvements.items():
                console.print(f"[green]improvement[/] {metric}: baseline={b:.4f} now={c:.4f}")
            if diff.is_regression:
                raise typer.Exit(code=2)
        # Print headline so the CLI invocation surfaces the numbers.
        console.print(report.split("## By category")[0])

    with _command_wrapper():
        asyncio.run(_run())
