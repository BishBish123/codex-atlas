"""`atlas` CLI: index a corpus, ask questions, run the golden test set."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
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
    evaluate_against_baseline,
    render_report,
    run_eval,
    write_failure_report,
)
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.indexer.walker import parse_corpus
from codex_atlas.retriever import Retriever, RetrieverConfig
from codex_atlas.store import ChunkStore

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
) -> None:
    """Walk `corpus`, parse every .py, embed every chunk, persist the graph."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        if output_format == "rich":
            console.print(f"[green]parsing[/] {corpus}")
        parsed = parse_corpus(corpus, max_files=max_files)
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

        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim, drop_existing=drop_existing)
        n_written = await store.upsert_chunks(chunks, vectors)
        if output_format == "rich":
            console.print(f"[green]upserted[/] {n_written} chunks into pgvector")

    asyncio.run(_run())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-form question to ask the agent."),
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    top_k: int = typer.Option(8, help="Top-k retrieval cap."),
) -> None:
    """Run a single agent query end-to-end."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim)
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
) -> None:
    """One-shot retrieval — runs the router but skips the answer synthesis."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim)
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

    asyncio.run(_run())


@app.command()
def explain(
    qualified_name: str = typer.Argument(..., help="Qualified name to explain."),
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
) -> None:
    """One-shot agent run anchored on a qualified name (uses structural route)."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim)
        retriever = Retriever(encoder_obj, store, cg, RetrieverConfig(top_k=8))
        agent = Agent(retriever)
        result = await agent.run(f"who calls {qualified_name}")
        console.print(
            f"[bold]Route:[/] {result.route} (grade {result.grade:.2f}, attempts {result.attempts})"
        )
        console.print(Markdown(result.answer))

    asyncio.run(_run())


@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="MCP transport: stdio | http"),
    host: str = typer.Option("127.0.0.1", help="HTTP bind host."),
    port: int = typer.Option(8090, help="HTTP bind port."),
) -> None:
    """Start the Codex-Atlas MCP server (alias for `atlas-mcp run`)."""
    from codex_atlas.mcp_server import mcp as mcp_app  # noqa: PLC0415

    if transport == "stdio":
        asyncio.run(mcp_app.run_stdio_async())
    elif transport == "http":
        asyncio.run(mcp_app.run_http_async(host=host, port=port))
    else:
        raise typer.BadParameter(f"unknown transport {transport!r}")


@app.command(name="eval")
def eval_cmd(
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
    tolerance: float = typer.Option(
        0.05, help="Fractional tolerance for the baseline regression gate."
    ),
) -> None:
    """Run the golden test set and write a markdown + (optional) JSON report."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        cg = CallGraph.load(graph)
        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim)
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

    asyncio.run(_run())
