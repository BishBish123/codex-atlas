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
from codex_atlas.store import ChunkStoreProtocol, StoredChunk


class Route(StrEnum):
    LOOKUP = "lookup"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    SUMMARIZATION = "summarization"
    NEIGHBORHOOD = "neighborhood"
    IMPORT_CHAIN = "import_chain"


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

_NEIGHBORHOOD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bneighborhood (?:of|around)\b",
        r"\b(?:everything|all) (?:near|around|surrounding)\b",
        r"\b(?:both|caller and callee|callee and caller)\b",
    )
)

_IMPORT_CHAIN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bimport chain\b",
        r"\b(?:who|which modules?) imports?\b",
        r"\bwhere is .* imported from\b",
    )
)


def classify(query: str) -> RoutingDecision:  # noqa: PLR0911
    """Return the route + confidence + matched signals for a free-form question.

    Routes are tried in priority order; the first regex hit wins.
    Confidences are *calibrated*, not arbitrary: they reflect the
    observed precision of each rule on the 30-question hand-graded
    development set (see ``evals/INTERPRETATION.md``). The lookup
    fallback is intentionally low-confidence so the agent's grader can
    decide whether to re-route via the rewrite loop.
    """
    q = query.strip()
    if not q:
        return RoutingDecision(route=Route.LOOKUP, confidence=0.5, signals=["empty-query"])

    for p in _IMPORT_CHAIN_PATTERNS:
        if p.search(q):
            return RoutingDecision(
                route=Route.IMPORT_CHAIN,
                confidence=0.92,
                signals=[f"import_chain:{p.pattern}"],
            )
    for p in _NEIGHBORHOOD_PATTERNS:
        if p.search(q):
            return RoutingDecision(
                route=Route.NEIGHBORHOOD,
                confidence=0.9,
                signals=[f"neighborhood:{p.pattern}"],
            )
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
    neighborhood_depth: int = 2
    import_chain_max_depth: int = 4
    # Hybrid scorer weights — must sum to ~1.0 by convention but the
    # scorer normalises so any non-negative weights work in tests.
    hybrid_weight_cosine: float = 0.6
    hybrid_weight_graph: float = 0.25
    hybrid_weight_fulltext: float = 0.15


@dataclass(frozen=True)
class HybridScore:
    """Per-chunk scoring breakdown for the hybrid scorer."""

    qualified_name: str
    cosine: float
    graph_distance: int  # 0 = seed; 1 = direct neighbour; ...
    fulltext: float
    combined: float


def hybrid_score(
    chunks: list[StoredChunk],
    *,
    query: str,
    seeds: set[str],
    graph_distances: dict[str, int],
    weights: tuple[float, float, float] = (0.6, 0.25, 0.15),
) -> list[HybridScore]:
    """Combine cosine + graph-distance + tfidf-ish full-text into one score.

    `chunks` carry a cosine score from the vector store. Graph distance
    is taken from `graph_distances[qname]` (0 if missing — i.e. unknown,
    treated as "far"). Full-text is a coarse term-overlap fraction
    against the query — not a real TF-IDF, but cheap and monotonic with
    relevance. Weights are passed in so the agent / config can tune at
    runtime; we normalise so callers can pass un-normalised vectors.
    """
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    w_cos, w_graph, w_text = (w / total for w in weights)

    q_terms = {t for t in re.split(r"\W+", query.lower()) if len(t) >= 3}
    out: list[HybridScore] = []
    for c in chunks:
        cosine = max(0.0, min(1.0, c.score))
        # Seeds get distance 0 by convention; everything else uses the
        # explicit map (with a far-away default).
        gd = 0 if c.qualified_name in seeds else graph_distances.get(c.qualified_name, 99)
        # Distance score: 1.0 at distance 0, decays as 1 / (1 + d).
        gd_score = 1.0 / (1.0 + gd)
        text_terms = {t for t in re.split(r"\W+", c.text.lower()) if len(t) >= 3}
        overlap = len(q_terms & text_terms) / len(q_terms) if q_terms else 0.0
        combined = w_cos * cosine + w_graph * gd_score + w_text * overlap
        out.append(
            HybridScore(
                qualified_name=c.qualified_name,
                cosine=cosine,
                graph_distance=gd,
                fulltext=overlap,
                combined=combined,
            )
        )
    out.sort(key=lambda s: s.combined, reverse=True)
    return out


class Retriever:
    """Glues the encoder + chunk store + call graph behind the four routes."""

    def __init__(
        self,
        encoder: Encoder,
        store: ChunkStoreProtocol,
        graph: CallGraph,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._graph = graph
        self._config = config or RetrieverConfig()

    async def retrieve(self, query: str, *, route_override: Route | None = None) -> RetrievalResult:
        """Run retrieval; optionally skip the classifier with ``route_override``.

        When ``route_override`` is supplied the classifier is bypassed and
        the named route runs verbatim. The returned ``RoutingDecision`` is
        synthesised so downstream consumers (eval harness, traces) still
        see a structured route + confidence — confidence is fixed at 1.0
        because the caller asserted the choice.
        """
        if route_override is not None:
            decision = RoutingDecision(
                route=route_override,
                confidence=1.0,
                signals=["route_override"],
            )
        else:
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
            case Route.NEIGHBORHOOD:
                chunks, extras = await self._neighborhood(query)
            case Route.IMPORT_CHAIN:
                chunks, extras = await self._import_chain(query)
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
        # Stage 1: vector top-k for the seed. Stage 2: 1-hop graph
        # neighbours. Stage 3: rank the union with ``hybrid_score`` so
        # cosine, graph proximity, and term overlap all contribute —
        # otherwise the seed-then-append order produced by stages 1+2
        # would dominate even when an expanded neighbour is a better
        # match. We keep the discovery-order ``expansions`` list as the
        # auxiliary "extras" return so the agent can still see the
        # neighbour names, irrespective of the ranking cut.
        seed = await self._vector_topk(query, self._config.top_k)
        seed_qnames = {c.qualified_name for c in seed}
        graph_distances: dict[str, int] = dict.fromkeys(seed_qnames, 0)
        expansions: list[str] = []
        seen = set(seed_qnames)
        for c in seed:
            for neighbour in self._graph.neighbors(
                c.qualified_name, depth=self._config.graph_depth
            ):
                if neighbour not in seen:
                    expansions.append(neighbour)
                    seen.add(neighbour)
                    graph_distances[neighbour] = self._config.graph_depth
        # Materialise the expansions so the ranker sees real chunk text.
        candidates: list[StoredChunk] = list(seed)
        for q in expansions[: self._config.top_k]:
            extra = await self._store.fetch_by_qualified_name(q)
            if extra is not None:
                candidates.append(extra)
        weights = (
            self._config.hybrid_weight_cosine,
            self._config.hybrid_weight_graph,
            self._config.hybrid_weight_fulltext,
        )
        scored = hybrid_score(
            candidates,
            query=query,
            seeds=seed_qnames,
            graph_distances=graph_distances,
            weights=weights,
        )
        # Reorder ``candidates`` to match the ranked output, keep top-k.
        order = {s.qualified_name: i for i, s in enumerate(scored)}
        candidates.sort(key=lambda c: order.get(c.qualified_name, len(order)))
        return candidates[: self._config.top_k], expansions

    async def _summarization(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        # Pull a wider net, then expand 2 hops via graph neighbours so
        # the synthesis prompt sees both the seed chunks AND the actual
        # code of their context (not just the names). Without
        # materialisation the synthesiser only ever saw the seed, which
        # made the summarisation route indistinguishable from a wider
        # lookup — README and ADRs both promised "wider top-k + 2-hop
        # graph neighbours", so this delivers the second half.
        seed = await self._vector_topk(query, self._config.summary_top_k)
        seen = {c.qualified_name for c in seed}
        expansions: list[str] = []
        for c in seed:
            for neighbour in self._graph.neighbors(c.qualified_name, depth=2):
                if neighbour not in seen:
                    expansions.append(neighbour)
                    seen.add(neighbour)
        # Materialise expansion chunks via the store. Bounded by
        # ``summary_top_k`` so a hub symbol with many neighbours can't
        # blow up the prompt.
        materialised: list[StoredChunk] = list(seed)
        for q in expansions[: self._config.summary_top_k]:
            extra = await self._store.fetch_by_qualified_name(q)
            if extra is not None:
                materialised.append(extra)
        return materialised, expansions

    async def _neighborhood(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        target = _extract_qualified_name(query, self._graph)
        if target is None:
            return await self._vector_topk(query, self._config.top_k), []
        nb = self._graph.caller_callee_neighborhood(target, depth=self._config.neighborhood_depth)
        related = sorted({target, *nb["all"]})
        chunks: list[StoredChunk] = []
        for q in related[: self._config.top_k]:
            sc = await self._store.fetch_by_qualified_name(q)
            if sc is not None:
                chunks.append(sc)
        return chunks, related

    async def _import_chain(self, query: str) -> tuple[list[StoredChunk], list[str]]:
        target = _extract_qualified_name(query, self._graph)
        if target is None:
            return await self._vector_topk(query, self._config.top_k), []
        chain = self._graph.import_chain(target, max_depth=self._config.import_chain_max_depth)
        chunks: list[StoredChunk] = []
        for q in chain[: self._config.top_k]:
            sc = await self._store.fetch_by_qualified_name(q)
            if sc is not None:
                chunks.append(sc)
        return chunks, chain


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
