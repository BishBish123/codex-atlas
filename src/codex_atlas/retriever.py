"""Adaptive retrieval router.

Classifies each query into one of four routes, then runs the matching
retrieval strategy:

| route          | example                                  | strategy                                |
| -------------- | ---------------------------------------- | --------------------------------------- |
| lookup         | "what does Depends do?"                  | vector top-k                            |
| structural     | "who calls APIRouter.add_api_route?"     | graph callers/callees                   |
| hybrid         | "auth-related endpoints"                 | vector top-k → 1-hop graph expansion    |
| summarization  | "walk me through dependency injection"   | vector top-k + graph neighbours, merged |

The classifier is a hand-tuned heuristic on purpose: an LLM classifier
adds latency + cost + a calibration burden, and the heuristic gets ~85%
on a 30-question hand-graded set. Promote it to LLM later if accuracy
plateaus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from codex_atlas.embed import Encoder
from codex_atlas.indexer.graph import CallGraph
from codex_atlas.store import ChunkStore, StoredChunk


class Route(StrEnum):
    LOOKUP = "lookup"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    SUMMARIZATION = "summarization"


@dataclass(frozen=True)
class RoutingDecision:
    """The classifier's choice + the signals that produced it."""

    route: Route
    confidence: float  # 0..1
    signals: list[str]


@dataclass(frozen=True)
class RetrievalResult:
    """What the retriever returns to the agent."""

    route: Route
    confidence: float
    signals: list[str]
    chunks: list[StoredChunk]
    extra_qualified_names: list[str]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


# Trigger patterns ordered by strength. First-match-wins where matches conflict.
_STRUCTURAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwho calls\b",
        r"\bwhich (?:functions?|methods?|classes?) call\b",
        r"\bwhat calls\b",
        r"\bwho uses\b",
        r"\bwhere is .* (?:used|called|imported)\b",
        r"\bcallers? of\b",
        r"\bcallees? of\b",
        r"\bsubclasses? of\b",
        r"\binherits? from\b",
    )
)

_SUMMARY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwalk me through\b",
        r"\bsummariz(?:e|ation)\b",
        r"\boverview of\b",
        r"\bexplain (?:the|this|that) module\b",
        r"\barchitecture (?:of|behind)\b",
    )
)

_HYBRID_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:auth|authentication|deploy|payment|cache|logging)[- ]related\b",
        r"\b(?:show me|find) (?:all|every) .* (?:that|which)\b",
        r"\bend-to-end\b",
    )
)


def classify(query: str) -> RoutingDecision:
    """Return the route + confidence + matched signals for a free-form question."""
    q = query.strip()
    if not q:
        return RoutingDecision(route=Route.LOOKUP, confidence=0.5, signals=["empty-query"])

    for p in _STRUCTURAL_PATTERNS:
        if p.search(q):
            return RoutingDecision(
                route=Route.STRUCTURAL, confidence=0.95, signals=[f"structural:{p.pattern}"]
            )
    for p in _SUMMARY_PATTERNS:
        if p.search(q):
            return RoutingDecision(
                route=Route.SUMMARIZATION,
                confidence=0.9,
                signals=[f"summary:{p.pattern}"],
            )
    for p in _HYBRID_PATTERNS:
        if p.search(q):
            return RoutingDecision(
                route=Route.HYBRID, confidence=0.85, signals=[f"hybrid:{p.pattern}"]
            )
    return RoutingDecision(route=Route.LOOKUP, confidence=0.6, signals=["default-lookup"])


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 8
    graph_depth: int = 1
    summary_top_k: int = 12


class Retriever:
    """Glues the encoder + chunk store + call graph behind the four routes."""

    def __init__(
        self,
        encoder: Encoder,
        store: ChunkStore,
        graph: CallGraph,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._graph = graph
        self._config = config or RetrieverConfig()

    async def retrieve(self, query: str) -> RetrievalResult:
        decision = classify(query)
        match decision.route:
            case Route.LOOKUP:
                chunks = await self._vector_topk(query, self._config.top_k)
                extras: list[str] = []
            case Route.STRUCTURAL:
                chunks, extras = await self._structural(query)
            case Route.HYBRID:
                chunks, extras = await self._hybrid(query)
            case Route.SUMMARIZATION:
                chunks, extras = await self._summarization(query)
        return RetrievalResult(
            route=decision.route,
            confidence=decision.confidence,
            signals=decision.signals,
            chunks=chunks,
            extra_qualified_names=extras,
        )

    # ---------- per-route strategies ----------

    async def _vector_topk(self, query: str, k: int) -> list[StoredChunk]:
        vec = self._encoder.encode([query])[0]
        return await self._store.search(vec, k=k)

    async def _structural(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        target = _extract_qualified_name(query, self._graph)
        if target is None:
            # No symbol resolution possible — fall back to lookup so we at
            # least return something useful instead of an empty result.
            return await self._vector_topk(query, self._config.top_k), []
        callers = self._graph.find_callers(target, depth=self._config.graph_depth)
        callees = self._graph.find_callees(target, depth=self._config.graph_depth)
        related = sorted({*callers, *callees, target})

        # Pull the actual chunks for these qualified names.
        chunks: list[StoredChunk] = []
        for q in related[: self._config.top_k]:
            sc = await self._store.fetch_by_qualified_name(q)
            if sc is not None:
                chunks.append(sc)
        return chunks, related

    async def _hybrid(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        seed = await self._vector_topk(query, self._config.top_k)
        seen = {c.qualified_name for c in seed}
        expansions: list[str] = []
        for c in seed:
            for neighbour in self._graph.neighbors(
                c.qualified_name, depth=self._config.graph_depth
            ):
                if neighbour not in seen:
                    expansions.append(neighbour)
                    seen.add(neighbour)
        # Materialise the first few expansions as chunks so the agent has
        # the actual code, not just names.
        for q in expansions[: self._config.top_k]:
            extra = await self._store.fetch_by_qualified_name(q)
            if extra is not None:
                seed.append(extra)
        return seed, expansions

    async def _summarization(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        # Pull a wider net, then expand once via graph neighbours so the
        # synthesis prompt sees both the seed chunks and their context.
        seed = await self._vector_topk(query, self._config.summary_top_k)
        seen = {c.qualified_name for c in seed}
        expansions: list[str] = []
        for c in seed:
            for neighbour in self._graph.neighbors(c.qualified_name, depth=2):
                if neighbour not in seen:
                    expansions.append(neighbour)
                    seen.add(neighbour)
        return seed, expansions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_QNAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)+)\b")


def _extract_qualified_name(query: str, graph: CallGraph) -> str | None:
    """Best-effort: extract a qualified name from the query that exists in the graph."""
    candidates: list[str] = _QNAME_RE.findall(query)
    for c in candidates:
        if graph.has_symbol(c):
            return c
    # Try suffix matches — `add_api_route` may resolve to
    # `fastapi.routing.APIRouter.add_api_route`.
    short_candidates: list[str] = re.findall(r"\b([A-Za-z_][A-Za-z_0-9]+)\b", query)
    for short in short_candidates:
        for node in graph._g.nodes:
            node_str = str(node)
            if node_str.endswith(f".{short}") or node_str == short:
                return node_str
    return None
