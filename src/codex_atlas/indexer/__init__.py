"""Codex-Atlas indexer: AST parser + chunker + pgvector + NetworkX call graph."""

from __future__ import annotations

import os

from codex_atlas.indexer.ast_parser import (
    Chunk,
    ImportRef,
    ParsedFile,
    Symbol,
    SymbolKind,
    TypeAlias,
    parse_python_file,
)
from codex_atlas.indexer.graph import CallGraph

__all__ = [
    "CallGraph",
    "Chunk",
    "ImportRef",
    "ParsedFile",
    "Symbol",
    "SymbolKind",
    "TypeAlias",
    "make_call_graph",
    "parse_python_file",
]


def make_call_graph() -> object:
    """Return the appropriate call-graph backend based on env vars.

    Backend selection:

    * ``ATLAS_GRAPH_BACKEND=networkx`` (default) → :class:`CallGraph`
      (in-memory NetworkX; no external service required).
    * ``ATLAS_GRAPH_BACKEND=neo4j`` → :class:`~codex_atlas.indexer.neo4j_graph.Neo4jCallGraph`
      (async Neo4j driver; reads ``ATLAS_NEO4J_URI``, ``ATLAS_NEO4J_USERNAME``,
      ``ATLAS_NEO4J_PASSWORD``).

    The return type is ``object`` because the two backends share a family API
    but do not yet share a common Protocol — ``Neo4jCallGraph`` methods are all
    coroutines, whereas ``CallGraph`` is synchronous. Callers that need the full
    async API should cast to ``Neo4jCallGraph`` explicitly.

    Raises ``RuntimeError`` for unknown ``ATLAS_GRAPH_BACKEND`` values.
    """
    backend = os.environ.get("ATLAS_GRAPH_BACKEND", "networkx").lower()
    if backend == "networkx":
        return CallGraph()
    if backend == "neo4j":
        # Lazy import so the module stays importable when neo4j is absent.
        from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

        return Neo4jCallGraph()
    raise RuntimeError(
        f"Unknown ATLAS_GRAPH_BACKEND={backend!r}; "
        f"expected 'networkx' (default) or 'neo4j'."
    )
