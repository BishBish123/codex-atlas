"""`atlas` CLI: index a corpus, ask questions, run the golden test set.

(The `eval` subcommand here runs an evaluation harness — it does not
call Python's builtin `eval()` anywhere in this file.)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from codex_atlas.agent import Agent
from codex_atlas.embed import Encoder, FakeEncoder
from codex_atlas.eval.golden import load_golden_set
from codex_atlas.eval.harness import render_report, run_eval
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


def _resolve_encoder(name: str) -> Encoder:
    if name == "fake":
        return FakeEncoder(dim=32)
    from codex_atlas.embed import load_sentence_transformer_encoder  # noqa: PLC0415

    return load_sentence_transformer_encoder(model_name=name)


def _dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise typer.BadParameter(
            "POSTGRES_DSN env var is required (e.g. postgresql://bench:bench@localhost:5433/bench)"
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
) -> None:
    """Walk `corpus`, parse every .py, embed every chunk, persist the graph."""

    async def _run() -> None:
        encoder_obj = _resolve_encoder(encoder)
        console.print(f"[green]parsing[/] {corpus}")
        parsed = parse_corpus(corpus, max_files=max_files)
        console.print(f"[green]parsed[/] {len(parsed)} files")

        graph = CallGraph()
        graph.ingest(parsed)
        graph.save(graph_out)
        console.print(
            f"[green]graph[/] {graph.n_nodes} nodes, {graph.n_edges} edges -> {graph_out}"
        )

        chunks = [c for pf in parsed for c in pf.chunks]
        if not chunks:
            console.print("[yellow]no chunks to embed[/]")
            return

        console.print(f"[green]embedding[/] {len(chunks)} chunks via {encoder_obj.name}")
        vectors = encoder_obj.encode([c.text for c in chunks])

        store = ChunkStore(dsn=_dsn())
        await store.setup(dim=encoder_obj.dim, drop_existing=drop_existing)
        n_written = await store.upsert_chunks(chunks, vectors)
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


@app.command(name="eval")
def eval_cmd(
    graph: Path = typer.Option(Path("data/graph.json"), help="Persisted call graph."),
    encoder: str = typer.Option("fake", help="Encoder used at index time."),
    out: Path = typer.Option(Path("evals/REPORT.md"), help="Where to write the markdown report."),
    json_out: Path | None = typer.Option(None, help="Optional JSON dump of per-question scores."),
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
                            "cited_qualified_names": r.cited_qualified_names,
                        }
                        for r in results
                    ],
                    indent=2,
                )
            )
            console.print(f"[green]wrote[/] {json_out}")
        # Print headline so the CLI invocation surfaces the numbers.
        console.print(report.split("## By category")[0])

    asyncio.run(_run())
