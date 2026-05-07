"""pgvector-backed chunk store (env-key-gated via ATLAS_PG_DSN).

Implements a simplified chunk-store interface where chunks are stored as
``StoredChunk`` dataclasses (already-embedded, score field ignored on write).
This backend is selected by ``ATLAS_CHUNK_STORE=pgvector`` via the
``make_chunk_store()`` factory in ``store/__init__.py``.

Embedding dimension is fixed at 384 (BGE-small-en-v1.5) by the DDL. Callers
must produce 384-dimensional vectors before calling ``upsert_with_embeddings``.

Imports of ``asyncpg`` and ``pgvector`` are soft-failed at module load so the
package stays importable when those libraries are absent (memory-only installs).
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    import asyncpg

    _ASYNCPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment,unused-ignore]
    _ASYNCPG_AVAILABLE = False

try:
    from pgvector.asyncpg import (
        register_vector as _register_vector,  # type: ignore[import-untyped,unused-ignore]
    )

    _PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _register_vector = None  # type: ignore[assignment,unused-ignore]
    _PGVECTOR_AVAILABLE = False

from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.store._core import StoredChunk

# Embedding dimension for bge-small-en-v1.5.
_EMBEDDING_DIM = 384

# DDL applied idempotently by ``setup()``. ivfflat with lists=100 is the
# right choice for the expected corpus size (tens of thousands of chunks):
#
#   * ivfflat (inverted file flat) partitions vectors into ``lists``
#     Voronoi cells at build time. At query time it probes only the
#     ``probes`` closest cells (default: 1-10), giving O(n/lists) instead
#     of O(n) scan. For n ~= 10k-100k rows, lists=100 keeps each cell at
#     100-1000 rows well inside asyncpg's comfortable row-fetch range
#     while giving >95% recall at probes=10.
#
#   * HNSW (the other pgvector index type) builds a multi-layer proximity
#     graph and gives higher recall at lower probes, but its build cost
#     is O(n * m * ef_construction) and its index size is 2-5x larger
#     than ivfflat. For an incremental-upsert workload (the indexer runs
#     repeatedly as the corpus grows) that build cost compounds; ivfflat
#     rebuilds cheaply because INSERT just appends to the nearest cell.
#
# The ``CREATE INDEX IF NOT EXISTS`` guards make re-running ``setup()``
# fully idempotent.
_DDL = f"""\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT    PRIMARY KEY,
    qualified_name TEXT    NOT NULL,
    file_path      TEXT    NOT NULL,
    lineno_start   INT     NOT NULL,
    lineno_end     INT     NOT NULL,
    text           TEXT    NOT NULL,
    embedding      vector({_EMBEDDING_DIM})
);

CREATE INDEX IF NOT EXISTS chunks_qname_idx
    ON chunks (qualified_name);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

_UPSERT_SQL = (
    "INSERT INTO chunks"
    "    (chunk_id, qualified_name, file_path, lineno_start, lineno_end, text, embedding)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7)"
    " ON CONFLICT (chunk_id) DO UPDATE SET"
    "    qualified_name = EXCLUDED.qualified_name,"
    "    file_path      = EXCLUDED.file_path,"
    "    lineno_start   = EXCLUDED.lineno_start,"
    "    lineno_end     = EXCLUDED.lineno_end,"
    "    text           = EXCLUDED.text,"
    "    embedding      = EXCLUDED.embedding"
)

_FETCH_BY_QNAME_SQL = (
    "SELECT chunk_id, qualified_name, file_path, lineno_start, lineno_end, text"
    " FROM   chunks"
    " WHERE  qualified_name = $1"
)

_SEARCH_SQL = (
    "SELECT chunk_id, qualified_name, file_path, lineno_start, lineno_end, text,"
    "       1 - (embedding <=> $1) AS score"
    " FROM   chunks"
    " ORDER BY embedding <=> $1"
    " LIMIT  $2"
)

_GET_SQL = (
    "SELECT chunk_id, qualified_name, file_path, lineno_start, lineno_end, text"
    " FROM   chunks"
    " WHERE  chunk_id = $1"
    " LIMIT  1"
)

_TRUNCATE_SQL = "TRUNCATE TABLE chunks"


def _row_to_stored_chunk(row: Any, score: float = 0.0) -> StoredChunk:
    return StoredChunk(
        chunk_id=row["chunk_id"],
        qualified_name=row["qualified_name"],
        file_path=row["file_path"],
        lineno_start=row["lineno_start"],
        lineno_end=row["lineno_end"],
        kind=SymbolKind.FUNCTION,  # pgvector store does not persist kind
        text=row["text"],
        score=score,
    )


class PgVectorChunkStore:
    """Async pgvector adapter gated behind ``ATLAS_PG_DSN``.

    Lifecycle
    ---------
    1. Construct: ``store = PgVectorChunkStore(dsn=os.environ["ATLAS_PG_DSN"])``
    2. Bootstrap: ``await store.setup()`` -- idempotent DDL + pool init.
    3. Use: ``upsert_chunks``, ``upsert_with_embeddings``, ``search``,
       ``fetch_by_qualified_name``, ``get``, ``clear``.

    All public methods require ``setup()`` to have completed; they raise
    ``RuntimeError("PgVectorChunkStore not initialised -- call setup() first")``
    otherwise.

    Pool sizing: min=1, max=5. The MCP server is single-process and the
    eval harness runs sequentially; a larger pool holds idle connections.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> None:
        if not _ASYNCPG_AVAILABLE:
            raise RuntimeError(  # pragma: no cover
                "asyncpg is not installed; "
                "pip install 'codex-atlas[real]' to use PgVectorChunkStore"
            )
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: Any = None
        self._bootstrapped: bool = False
        self._pool_lock: asyncio.Lock = asyncio.Lock()
        self._setup_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_pool(self) -> Any:
        """Lazily create the connection pool (double-checked init)."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self._dsn,
                        min_size=self._min_pool_size,
                        max_size=self._max_pool_size,
                    )
        return self._pool

    def _require_bootstrapped(self) -> None:
        if not self._bootstrapped:
            raise RuntimeError(
                "PgVectorChunkStore not initialised -- call setup() first"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Apply DDL (idempotent) and initialise the connection pool.

        Safe to call multiple times -- DDL uses ``IF NOT EXISTS`` guards
        and the pool is reused. The body runs under ``_setup_lock`` so
        concurrent first-callers don't each run the DDL block.
        """
        async with self._setup_lock:
            if self._bootstrapped:
                return
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Register the vector codec on this connection before
                # executing the DDL that references the vector type.
                if _register_vector is not None:
                    await _register_vector(conn)
                # Execute each DDL statement individually -- asyncpg does
                # not support semicolon-separated multi-statement strings
                # in a single ``execute()`` call.
                for stmt in _split_ddl(_DDL):
                    await conn.execute(stmt)
            self._bootstrapped = True

    async def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        """INSERT ... ON CONFLICT DO UPDATE for every chunk in ``chunks``.

        Embedding is inserted as NULL when the ``StoredChunk`` was not
        produced with an explicit embedding vector. Use
        ``upsert_with_embeddings`` when you have the raw float vectors.
        """
        self._require_bootstrapped()
        if not chunks:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if _register_vector is not None:
                await _register_vector(conn)
            rows = [
                (
                    c.chunk_id,
                    c.qualified_name,
                    c.file_path,
                    c.lineno_start,
                    c.lineno_end,
                    c.text,
                    None,  # embedding -- not available on StoredChunk directly
                )
                for c in chunks
            ]
            await conn.executemany(_UPSERT_SQL, rows)

    async def upsert_with_embeddings(
        self,
        chunks: list[StoredChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks paired with their pre-computed embedding vectors.

        ``embeddings[i]`` is the 384-dimensional float list for ``chunks[i]``.
        This is the primary write path when you have both the chunk metadata
        and the raw vector.
        """
        self._require_bootstrapped()
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if _register_vector is not None:
                await _register_vector(conn)
            rows = [
                (
                    c.chunk_id,
                    c.qualified_name,
                    c.file_path,
                    c.lineno_start,
                    c.lineno_end,
                    c.text,
                    emb,
                )
                for c, emb in zip(chunks, embeddings, strict=True)
            ]
            await conn.executemany(_UPSERT_SQL, rows)

    async def fetch_by_qualified_name(self, qname: str) -> list[StoredChunk]:
        """Return all stored chunks whose ``qualified_name`` matches ``qname``."""
        self._require_bootstrapped()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_BY_QNAME_SQL, qname)
        return [_row_to_stored_chunk(r) for r in rows]

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[StoredChunk]:
        """ANN search using cosine distance via ``embedding <=> $1``.

        Returns up to ``top_k`` chunks ordered by descending cosine
        similarity (1 - cosine_distance). The ``score`` field of each
        returned ``StoredChunk`` holds the similarity value in ``(0, 1]``.
        """
        self._require_bootstrapped()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if _register_vector is not None:
                await _register_vector(conn)
            rows = await conn.fetch(_SEARCH_SQL, query_embedding, top_k)
        return [_row_to_stored_chunk(r, score=float(r["score"])) for r in rows]

    async def get(self, chunk_id: str) -> StoredChunk | None:
        """Fetch a single chunk by primary key. Returns ``None`` if absent."""
        self._require_bootstrapped()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_GET_SQL, chunk_id)
        if row is None:
            return None
        return _row_to_stored_chunk(row)

    async def clear(self) -> None:
        """TRUNCATE the chunks table -- removes all rows instantly."""
        self._require_bootstrapped()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_TRUNCATE_SQL)

    async def close(self) -> None:
        """Close the connection pool. Idempotent."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._bootstrapped = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_ddl(ddl: str) -> list[str]:
    """Split a multi-statement DDL string into individual statements.

    asyncpg rejects semicolon-separated multi-statement strings in a
    single ``execute()`` call. We split on ``';'`` and strip whitespace /
    empty tokens.
    """
    return [s.strip() for s in ddl.split(";") if s.strip()]
