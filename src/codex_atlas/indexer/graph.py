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
import logging
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from codex_atlas.indexer.ast_parser import ImportRef, ParsedFile, Symbol, SymbolKind

_log = logging.getLogger(__name__)

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
        Resolution is two-stage:

        1. **Imports-map resolution.** For each caller's source module we
           build a `local-name -> dotted-target` map from `ImportRef`s.
           If the callee's short name is bound in that map and the target
           exists as a graph node, we record an edge to that exact target
           and skip the corpus-wide fallback. This is much sharper than
           short-name matching: it avoids over-matching common names like
           ``get`` / ``add``.
        2. **Short-name fallback.** If the callee is not bound by an
           import or its target is not a known symbol, we fall back to
           recording an edge per qualified-name match in the corpus
           (preserving the existing v1 behaviour for unresolved calls).
        """
        # Materialise once so we can iterate multiple times below.
        parsed_list = list(parsed)

        # Index unqualified-name -> set of qualified names so call resolution
        # is O(1) per call.
        unqualified_index: dict[str, set[str]] = {}
        for pf in parsed_list:
            for sym in pf.symbols:
                self.add_symbol(sym)
                short = sym.qualified_name.rsplit(".", 1)[-1]
                unqualified_index.setdefault(short, set()).add(sym.qualified_name)

        # Per-module local-name -> dotted-target map for imports resolution.
        # Relative imports (level >= 1) are reified to absolute targets here
        # using the parsed file's own module_name as the anchor — without
        # this, ``from . import x`` would never line up with the absolute
        # qname recorded for ``x`` elsewhere in the corpus.
        imports_map: dict[str, dict[str, str]] = {}
        for pf in parsed_list:
            local_to_target: dict[str, str] = {}
            for ref in pf.import_refs:
                target = _resolve_import_target(pf.module_name, ref)
                if target is None:
                    continue
                local_to_target[ref.local] = target
            imports_map[pf.module_name] = local_to_target

        # `defines`: module -> any class/function/method directly inside it.
        for pf in parsed_list:
            for sym in pf.symbols:
                if sym.kind is SymbolKind.MODULE:
                    continue
                if sym.qualified_name.startswith(f"{pf.module_name}."):
                    self.add_edge(pf.module_name, sym.qualified_name, EDGE_DEFINES)

        # `imports`: module -> imported module/symbol (best-effort, may dangle).
        for pf in parsed_list:
            for imp in pf.imports:
                if imp:
                    self.add_edge(pf.module_name, imp, EDGE_IMPORTS)

        # `calls`: caller -> callee, imports-aware first, short-name fallback.
        for pf in parsed_list:
            local_to_target = imports_map.get(pf.module_name, {})
            for caller, callee_short in pf.calls:
                resolved_via_import: str | None = None
                target = local_to_target.get(callee_short)
                if target is not None and self._g.has_node(target):
                    resolved_via_import = target
                if resolved_via_import is not None:
                    if resolved_via_import != caller:
                        self.add_edge(caller, resolved_via_import, EDGE_CALLS)
                    continue
                for candidate in unqualified_index.get(callee_short, ()):
                    if candidate == caller:
                        continue  # ignore self-loops
                    self.add_edge(caller, candidate, EDGE_CALLS)

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

    def neighbors_with_depth(
        self, qualified_name: str, depth: int = 1
    ) -> dict[str, int]:
        """Return BFS hop counts for every node within ``depth`` hops.

        Used by the hybrid retrieval scorer so 1-hop neighbours score
        higher than 2-hop neighbours under the ``1 / (1 + d)`` decay.
        ``neighbors`` collapses both depths to the same flat list and
        loses that information; this method preserves it.

        The result excludes ``qualified_name`` itself. Both call
        directions (callers + callees) are merged; the smaller hop
        count wins when a node is reachable in both directions at
        different depths.
        """
        if qualified_name not in self._g:
            return {}
        if depth <= 0:
            raise ValueError("depth must be positive")
        out: dict[str, int] = {}
        for predecessors in (True, False):
            seen: set[str] = {qualified_name}
            frontier: set[str] = {qualified_name}
            for hop in range(1, depth + 1):
                next_frontier: set[str] = set()
                for node in frontier:
                    edges = (
                        self._g.in_edges(node, keys=True, data=True)
                        if predecessors
                        else self._g.out_edges(node, keys=True, data=True)
                    )
                    for u, v, _key, data in edges:
                        if data.get("kind") != EDGE_CALLS:
                            continue
                        other = u if predecessors else v
                        if other in seen:
                            continue
                        seen.add(other)
                        next_frontier.add(other)
                        existing = out.get(other)
                        if existing is None or hop < existing:
                            out[other] = hop
                frontier = next_frontier
                if not frontier:
                    break
        return out

    def caller_callee_neighborhood(
        self, qualified_name: str, depth: int = 2
    ) -> dict[str, list[str]]:
        """Both-direction BFS up to `depth` hops, with results split by direction.

        Returns ``{"callers": [...], "callees": [...], "all": [...]}``.
        Useful for the retriever's `caller_callee_neighborhood` route, which
        wants to render "everything within K hops of this symbol" without
        re-running two queries.
        """
        if depth <= 0:
            raise ValueError("depth must be positive")
        callers = self.find_callers(qualified_name, depth=depth)
        callees = self.find_callees(qualified_name, depth=depth)
        union = sorted({*callers, *callees})
        return {"callers": callers, "callees": callees, "all": union}

    def import_chain(self, qualified_name: str, max_depth: int = 4) -> list[str]:
        """Walk imports edges backward from `qualified_name` to its source modules.

        Given a symbol, return modules that (transitively) import it via
        the `imports` edge kind. The depth is bounded so we don't traverse
        a 50K-node graph unbounded; in practice 3-4 hops is more than
        enough to show "who pulls this in".
        """
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if qualified_name not in self._g:
            return []
        return list(self._traverse(qualified_name, max_depth, predecessors=True, kind=EDGE_IMPORTS))

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


def _resolve_import_target(module_name: str, ref: ImportRef) -> str | None:
    """Reify a relative ImportRef to its absolute dotted target.

    Absolute imports (``level == 0``) are returned unchanged. Relative
    imports (``level >= 1``) are anchored on the importing module's own
    package: drop the last ``level`` parts of ``module_name``, then join
    with ``ref.target`` if it is non-empty.

    Returns ``None`` (and logs a warning) when a relative import escapes
    the package — i.e. the importing module doesn't have enough package
    parts to satisfy the requested level. The caller skips the entry so
    the rest of the imports map keeps building.
    """
    if ref.level <= 0:
        return ref.target
    parts = module_name.split(".") if module_name else []
    if ref.level > len(parts):
        # Relative import escapes the package root — there is no sensible
        # absolute target. Skip with a structured warning rather than
        # synthesising a name that can't possibly resolve.
        # ``module`` and ``level`` are reserved LogRecord attribute names —
        # use namespaced keys so the logging machinery can attach them to
        # the record without raising KeyError.
        _log.warning(
            "graph.relative_import_escapes_package",
            extra={
                "import_module": module_name,
                "import_level": ref.level,
                "import_target": ref.target,
                "import_local": ref.local,
            },
        )
        return None
    base_parts = parts[: len(parts) - ref.level]
    absolute_parts = [*base_parts, ref.target] if ref.target else base_parts
    absolute = ".".join(p for p in absolute_parts if p)
    return absolute or None
