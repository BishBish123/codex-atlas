"""In-memory call graph backed by NetworkX with a parquet/json persistence layer.

Why NetworkX over Neo4j? For a single-codebase, single-user portfolio
deployment a 50K-node directed graph fits in memory easily and saves
the user from booking a Neo4j instance. The query API is intentionally
limited to what the retriever actually needs:

* `find_callers(qualified_name, depth)` — who depends on this?
* `find_callees(qualified_name, depth)` — what does this depend on?
* `neighbors(qualified_name, depth)` — both directions, hop-bounded.

The persistence format is a single `.json` for portability + diffability
(the graph is small enough that binary doesn't pay).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from codex_atlas.indexer.ast_parser import ParsedFile, Symbol, SymbolKind

EDGE_CALLS = "calls"
EDGE_IMPORTS = "imports"
EDGE_DEFINES = "defines"


class CallGraph:
    """A directed multigraph of (symbol -> symbol) edges with edge kinds."""

    def __init__(self) -> None:
        # `MultiDiGraph[str]` keys nodes by string; networkx's stubs accept
        # the parameterisation though the runtime class is non-generic.
        self._g: nx.MultiDiGraph[str] = nx.MultiDiGraph()

    # ---------- mutation ----------

    def add_symbol(self, sym: Symbol) -> None:
        self._g.add_node(
            sym.qualified_name,
            kind=str(sym.kind),
            file_path=sym.file_path,
            lineno_start=sym.lineno_start,
            lineno_end=sym.lineno_end,
        )

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        self._g.add_edge(src, dst, kind=kind)

    def ingest(self, parsed: Iterable[ParsedFile]) -> None:  # noqa: PLR0912
        """Add every parsed-file's symbols + calls + imports to the graph.

        Calls are recorded as `caller -> callee` edges keyed `kind=calls`.
        Because the AST visitor only knows the unqualified name of the
        callee (we have no type inference), we resolve calls to whichever
        symbol matches the unqualified name *somewhere* in the corpus —
        recording an edge per match. For ambiguous callees (common name
        like `get`), the graph carries every plausible target so the
        retriever can dedupe / rank later.
        """
        # Index unqualified-name -> set of qualified names so call resolution
        # is O(1) per call.
        unqualified_index: dict[str, set[str]] = {}
        for pf in parsed:
            for sym in pf.symbols:
                self.add_symbol(sym)
                short = sym.qualified_name.rsplit(".", 1)[-1]
                unqualified_index.setdefault(short, set()).add(sym.qualified_name)

        # `defines`: module -> any class/function/method directly inside it.
        for pf in parsed:
            for sym in pf.symbols:
                if sym.kind is SymbolKind.MODULE:
                    continue
                if sym.qualified_name.startswith(f"{pf.module_name}."):
                    self.add_edge(pf.module_name, sym.qualified_name, EDGE_DEFINES)

        # `imports`: module -> imported module/symbol (best-effort, may dangle).
        for pf in parsed:
            for imp in pf.imports:
                if imp:
                    self.add_edge(pf.module_name, imp, EDGE_IMPORTS)

        # `calls`: caller -> every plausible callee (matched by short name).
        for pf in parsed:
            for caller, callee_short in pf.calls:
                for resolved in unqualified_index.get(callee_short, ()):
                    if resolved == caller:
                        continue  # ignore self-loops
                    self.add_edge(caller, resolved, EDGE_CALLS)

    # ---------- queries ----------

    @property
    def n_nodes(self) -> int:
        return int(self._g.number_of_nodes())

    @property
    def n_edges(self) -> int:
        return int(self._g.number_of_edges())

    def has_symbol(self, qualified_name: str) -> bool:
        return qualified_name in self._g

    def find_callers(self, qualified_name: str, depth: int = 1) -> list[str]:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if qualified_name not in self._g:
            return []
        # Predecessors traversal up to `depth` hops along EDGE_CALLS edges.
        return list(self._traverse(qualified_name, depth, predecessors=True, kind=EDGE_CALLS))

    def find_callees(self, qualified_name: str, depth: int = 1) -> list[str]:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if qualified_name not in self._g:
            return []
        return list(self._traverse(qualified_name, depth, predecessors=False, kind=EDGE_CALLS))

    def neighbors(self, qualified_name: str, depth: int = 1) -> list[str]:
        if qualified_name not in self._g:
            return []
        forward = set(self.find_callees(qualified_name, depth))
        backward = set(self.find_callers(qualified_name, depth))
        return sorted(forward | backward)

    def _traverse(self, start: str, depth: int, *, predecessors: bool, kind: str) -> Iterable[str]:
        seen: set[str] = {start}
        frontier: set[str] = {start}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                edges = (
                    self._g.in_edges(node, keys=True, data=True)
                    if predecessors
                    else self._g.out_edges(node, keys=True, data=True)
                )
                for u, v, _key, data in edges:
                    if data.get("kind") != kind:
                        continue
                    other = u if predecessors else v
                    if other in seen:
                        continue
                    seen.add(other)
                    next_frontier.add(other)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(seen - {start})

    # ---------- persistence ----------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": n, **dict(d)} for n, d in self._g.nodes(data=True)],
            "edges": [
                {"src": u, "dst": v, "kind": d.get("kind", "")}
                for u, v, d in self._g.edges(data=True)
            ],
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return out

    @classmethod
    def load(cls, path: str | Path) -> CallGraph:
        payload = json.loads(Path(path).read_text())
        g = cls()
        for n in payload["nodes"]:
            attrs = {k: v for k, v in n.items() if k != "id"}
            g._g.add_node(n["id"], **attrs)
        for e in payload["edges"]:
            g._g.add_edge(e["src"], e["dst"], kind=e.get("kind", ""))
        return g
