"""Neo4j AuraDB-backed call/import graph (async driver).

This module provides ``Neo4jCallGraph``, an alternative to the in-memory
``NetworkXCallGraph`` (``CallGraph`` in ``graph.py``) for deployments that
already run Neo4j AuraDB or a self-hosted Neo4j Community instance.

**Env-var contract**

| Variable | Default | Purpose |
| --- | --- | --- |
| ``ATLAS_NEO4J_URI`` | — | Neo4j bolt/neo4j+s URI (required to use this backend) |
| ``ATLAS_NEO4J_USERNAME`` | ``neo4j`` | Database username |
| ``ATLAS_NEO4J_PASSWORD`` | — | Database password |

The ``neo4j`` Python driver (``neo4j>=5.20``) is a soft dependency: if it is
not installed the module still imports cleanly; calling ``Neo4jCallGraph``
raises ``ImportError`` at instantiation time with an actionable message.

**Why soft-fail?** The ``[real]`` extras group is optional. Existing memory-
only deployments must not break because a driver they never use is absent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import of the neo4j driver
# ---------------------------------------------------------------------------

try:
    from neo4j import AsyncGraphDatabase

    _NEO4J_AVAILABLE = True
except ImportError:  # pragma: no cover - only missing in stripped installs
    _NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None

# ---------------------------------------------------------------------------
# Cypher constants
# ---------------------------------------------------------------------------

_SETUP_CONSTRAINT = (
    "CREATE CONSTRAINT symbol_qname IF NOT EXISTS "
    "FOR (s:Symbol) REQUIRE s.qualified_name IS UNIQUE"
)
_SETUP_INDEX = (
    "CREATE INDEX symbol_module IF NOT EXISTS "
    "FOR (s:Symbol) ON (s.module)"
)

_ADD_SYMBOL = (
    "MERGE (s:Symbol {qualified_name: $qn}) "
    "SET s.module = $m, s.kind = $k"
)

# ``CALLS`` / ``IMPORTS`` relationship type is parameterised at Python level;
# Cypher does not support parameterised relationship types so we use string
# formatting on a fixed allow-list rather than user-supplied data.
_ALLOWED_REL_TYPES = frozenset({"CALLS", "IMPORTS"})

_ADD_EDGE_TMPL = (
    "MATCH (a:Symbol {{qualified_name: $from_qn}}), "
    "(b:Symbol {{qualified_name: $to_qn}}) "
    "MERGE (a)-[:{rel_type}]->(b)"
)

_FIND_CALLERS = (
    "MATCH (caller:Symbol)-[:CALLS]->(target:Symbol {qualified_name: $qn}) "
    "RETURN caller.qualified_name AS qname"
)

_FIND_CALLEES = (
    "MATCH (source:Symbol {qualified_name: $qn})-[:CALLS]->(callee:Symbol) "
    "RETURN callee.qualified_name AS qname"
)

_IMPORT_CHAIN = (
    "MATCH path=()-[:IMPORTS*1..3]->(:Symbol {qualified_name: $qn}) "
    "RETURN path"
)

_GET_NEIGHBORHOOD_TMPL = (
    "MATCH (s:Symbol {{qualified_name: $qn}})-[*1..{depth}]-(n:Symbol) "
    "RETURN DISTINCT n.qualified_name AS qname"
)

_CLEAR = "MATCH (n:Symbol) DETACH DELETE n"


# ---------------------------------------------------------------------------
# Neo4jCallGraph
# ---------------------------------------------------------------------------


class Neo4jCallGraph:
    """Async Neo4j-backed call/import graph.

    Parameters are read from environment variables at construction time.
    Call ``await instance.setup()`` once before mutating or querying the
    graph — ``setup()`` creates the uniqueness constraint and the module index.

    All public methods are coroutines because the ``neo4j`` async driver
    requires an ``async with session`` context.
    """

    def __init__(self) -> None:
        if not _NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j driver is not installed. "
                "Run `uv sync --extra real` (or `pip install 'neo4j>=5.20'`) "
                "and then retry."
            )

        uri = os.environ.get("ATLAS_NEO4J_URI")
        if not uri:
            raise RuntimeError(
                "ATLAS_NEO4J_URI environment variable is required "
                "when using the neo4j graph backend."
            )
        username = os.environ.get("ATLAS_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("ATLAS_NEO4J_PASSWORD", "")

        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        _log.info(
            "neo4j_graph.driver_created",
            extra={"uri": uri, "username": username},
        )

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Apply the Cypher schema (constraint + index).

        Idempotent: uses ``IF NOT EXISTS`` guards so repeated calls are safe.
        """
        async with self._driver.session() as session:
            await session.run(_SETUP_CONSTRAINT)
            await session.run(_SETUP_INDEX)
        _log.info("neo4j_graph.schema_applied")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    async def add_symbol(self, qualified_name: str, module: str, kind: str) -> None:
        """Upsert a Symbol node.

        Uses ``MERGE`` so re-indexing the same symbol is idempotent.
        ``SET`` overwrites module/kind in case they changed.
        """
        async with self._driver.session() as session:
            await session.run(_ADD_SYMBOL, qn=qualified_name, m=module, k=kind)

    async def add_edge(self, from_qn: str, to_qn: str, kind: str) -> None:
        """Merge a CALLS or IMPORTS relationship between two Symbol nodes.

        ``kind`` must be one of ``"calls"`` or ``"imports"`` (case-insensitive).
        The relationship type is upper-cased and validated against an allow-list
        to prevent injection — Cypher does not support parameterised rel types.
        """
        rel_type = kind.upper()
        if rel_type not in _ALLOWED_REL_TYPES:
            raise ValueError(
                f"Unknown edge kind {kind!r}; expected one of {sorted(_ALLOWED_REL_TYPES)}"
            )
        cypher = _ADD_EDGE_TMPL.format(rel_type=rel_type)
        async with self._driver.session() as session:
            await session.run(cypher, from_qn=from_qn, to_qn=to_qn)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def find_callers(self, qualified_name: str) -> list[str]:
        """Return qualified names of symbols that directly call ``qualified_name``."""
        async with self._driver.session() as session:
            result = await session.run(_FIND_CALLERS, qn=qualified_name)
            records = await result.data()
        return [r["qname"] for r in records]

    async def find_callees(self, qualified_name: str) -> list[str]:
        """Return qualified names of symbols that ``qualified_name`` calls."""
        async with self._driver.session() as session:
            result = await session.run(_FIND_CALLEES, qn=qualified_name)
            records = await result.data()
        return [r["qname"] for r in records]

    async def import_chain(self, target_qn: str) -> list[Any]:
        """Return paths of up to 3 IMPORTS hops leading to ``target_qn``.

        Returns raw Neo4j ``Path`` objects. Callers inspect
        ``path.nodes`` / ``path.relationships`` as needed.
        """
        async with self._driver.session() as session:
            result = await session.run(_IMPORT_CHAIN, qn=target_qn)
            records = await result.data()
        return [r["path"] for r in records]

    async def get_neighborhood(self, qualified_name: str, depth: int = 1) -> list[str]:
        """Return qualified names of all nodes within ``depth`` hops.

        Both directions are traversed (any relationship type).
        The source node itself is excluded from the result.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")
        cypher = _GET_NEIGHBORHOOD_TMPL.format(depth=depth)
        async with self._driver.session() as session:
            result = await session.run(cypher, qn=qualified_name)
            records = await result.data()
        return [r["qname"] for r in records]

    async def clear(self) -> None:
        """Delete all Symbol nodes and their relationships."""
        async with self._driver.session() as session:
            await session.run(_CLEAR)
        _log.info("neo4j_graph.cleared")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying async driver (release connection pool)."""
        await self._driver.close()
