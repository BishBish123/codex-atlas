"""pgvector-backed chunk store.

One row per indexed chunk (function or method body): id + qualified_name +
file_path + lineno_start + lineno_end + kind + text + vector.

Lifecycle: ``setup`` is a one-shot bootstrap that creates the table +
HNSW index AND initialises a small ``asyncpg`` connection pool;
``upsert_chunks`` is idempotent on (file_path, qualified_name,
lineno_start); ``search`` runs a cosine top-k. Re-running ``setup`` is
safe — the underlying DDL is ``IF NOT EXISTS`` and the pool is reused.

Pool sizing is intentionally tiny (min=1, max=5). The MCP server is
single-process and the eval harness fires sequentially; a larger pool
would just hold idle connections.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

import asyncpg
import numpy as np

from codex_atlas.indexer.ast_parser import Chunk, SymbolKind

# Postgres unquoted identifier rule, conservatively narrowed: a leading
# letter or underscore, then up to 62 letters / digits / underscores
# (Postgres caps identifiers at NAMEDATALEN-1 = 63 chars). The pattern
# matches the pg lexer's NAMEDATALEN-1 rule and rejects everything that
# isn't a plain SQL name — quotes, semicolons, whitespace, dots.
_TABLE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@dataclass(frozen=True)
class StoredChunk:
    """A retrieved chunk with its similarity score."""

    chunk_id: str
    qualified_name: str
    file_path: str
    lineno_start: int
    lineno_end: int
    kind: SymbolKind
    text: str
    score: float


class ChunkStoreProtocol(Protocol):
    """Methods the retriever needs from any chunk store backend.

    The pgvector ``ChunkStore`` and the test-only ``InMemoryChunkStore``
    both conform. New backends only need to implement this surface.
    """

    async def setup(self, dim: int, drop_existing: bool = False) -> None: ...

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256
    ) -> int: ...

    async def search(self, query_vec: np.ndarray, k: int = 8) -> list[StoredChunk]: ...

    async def fetch_by_qualified_name(self, qualified_name: str) -> StoredChunk | None: ...


class ChunkStore:
    """Async pgvector adapter for code chunks (pooled connections)."""

    def __init__(
        self,
        dsn: str,
        table: str = "codex_atlas_chunks",
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> None:
        # Validate the ``table`` argument up-front so the f-string
        # interpolation downstream is bounded to a known-safe SQL
        # identifier shape. Without this the ``ruff S608`` suppression
        # at the use sites would be load-bearing on caller discipline.
        if not _TABLE_IDENT_RE.match(table):
            raise ValueError(
                f"invalid table identifier {table!r}; must match "
                f"{_TABLE_IDENT_RE.pattern}"
            )
        self._dsn = dsn
        self._table = table
        self._dim: int | None = None
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None
        # Set once setup() has run successfully; lets repeat callers
        # short-circuit the DDL without skipping pool initialisation.
        self._bootstrapped: bool = False

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[asyncpg.Connection]:
        # Lazy-init the pool on first acquisition so callers that only
        # ever pass through ``setup`` (e.g. tests using the InMemory
        # variant) don't pay for a connect they never need.
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
        async with self._pool.acquire() as conn:
            yield conn

    async def setup(self, dim: int, drop_existing: bool = False) -> None:
        """One-shot bootstrap: ensure schema + pool. Safe to skip after the first call.

        Re-running is a no-op for the DDL (every statement is
        ``IF NOT EXISTS``) but still cheap; chronic-path callers (MCP /
        agent loop) should call this once at process start and reuse
        the store across requests.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if self._bootstrapped and not drop_existing and self._dim == dim:
            return
        self._dim = dim
        async with self._connect() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if drop_existing:
                await conn.execute(f'DROP TABLE IF EXISTS "{self._table}"')
            await conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{self._table}" ('
                "id text PRIMARY KEY,"
                "qualified_name text NOT NULL,"
                "file_path text NOT NULL,"
                "lineno_start int NOT NULL,"
                "lineno_end int NOT NULL,"
                "kind text NOT NULL,"
                "text text NOT NULL,"
                f"vec vector({dim}) NOT NULL"
                ")"
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{self._table}_qname_idx" '
                f'ON "{self._table}" (qualified_name)'
            )
            # HNSW index — created idempotently. m + ef_construction picked
            # for the typical 10k-50k chunk range a single-codebase ingest.
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{self._table}_vec_idx" '
                f'ON "{self._table}" USING hnsw (vec vector_cosine_ops) '
                "WITH (m = 16, ef_construction = 64)"
            )
        self._bootstrapped = True

    async def close(self) -> None:
        """Close the pool. Idempotent — safe to call multiple times."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._bootstrapped = False

    async def teardown(self) -> None:
        async with self._connect() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS "{self._table}"')

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256
    ) -> int:
        if self._dim is None:
            raise RuntimeError("upsert_chunks() called before setup()")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunk/vector length mismatch: {len(chunks)} vs {vectors.shape[0]}")
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != setup dim {self._dim}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        async with self._connect() as conn:
            from pgvector.asyncpg import register_vector as register_async  # noqa: PLC0415

            await register_async(conn)
            sql = (
                f'INSERT INTO "{self._table}" '
                "(id, qualified_name, file_path, lineno_start, lineno_end, kind, text, vec) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (id) DO UPDATE SET "
                "  qualified_name = EXCLUDED.qualified_name, "
                "  file_path = EXCLUDED.file_path, "
                "  lineno_start = EXCLUDED.lineno_start, "
                "  lineno_end = EXCLUDED.lineno_end, "
                "  kind = EXCLUDED.kind, "
                "  text = EXCLUDED.text, "
                "  vec = EXCLUDED.vec"
            )
            n_written = 0
            for i in range(0, len(chunks), batch_size):
                rows = []
                for j in range(i, min(i + batch_size, len(chunks))):
                    c = chunks[j]
                    rows.append(
                        (
                            c.chunk_id(),
                            c.qualified_name,
                            c.file_path,
                            c.lineno_start,
                            c.lineno_end,
                            str(c.kind),
                            c.text,
                            vectors[j],
                        )
                    )
                await conn.executemany(sql, rows)
                n_written += len(rows)
            return n_written

    async def search(self, query_vec: np.ndarray, k: int = 8) -> list[StoredChunk]:
        if self._dim is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query_vec.ndim != 1 or query_vec.shape[0] != self._dim:
            raise ValueError(f"query_vec must be 1-D of dim {self._dim}, got {query_vec.shape!r}")
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32, copy=False)

        async with self._connect() as conn:
            from pgvector.asyncpg import register_vector as register_async  # noqa: PLC0415

            await register_async(conn)
            rows = await conn.fetch(
                f"SELECT id, qualified_name, file_path, lineno_start, lineno_end, "
                f"kind, text, 1 - (vec <=> $1) AS score "
                f'FROM "{self._table}" ORDER BY vec <=> $1 LIMIT $2',
                query_vec,
                k,
            )
        return [
            StoredChunk(
                chunk_id=r["id"],
                qualified_name=r["qualified_name"],
                file_path=r["file_path"],
                lineno_start=r["lineno_start"],
                lineno_end=r["lineno_end"],
                kind=SymbolKind(r["kind"]),
                text=r["text"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def fetch_by_qualified_name(self, qualified_name: str) -> StoredChunk | None:
        async with self._connect() as conn:
            row = await conn.fetchrow(
                f"SELECT id, qualified_name, file_path, lineno_start, lineno_end, "
                f'kind, text FROM "{self._table}" WHERE qualified_name = $1 LIMIT 1',
                qualified_name,
            )
        if not row:
            return None
        return StoredChunk(
            chunk_id=row["id"],
            qualified_name=row["qualified_name"],
            file_path=row["file_path"],
            lineno_start=row["lineno_start"],
            lineno_end=row["lineno_end"],
            kind=SymbolKind(row["kind"]),
            text=row["text"],
            score=1.0,
        )


class InMemoryChunkStore:
    """Dict-backed chunk store for tests + the eval harness.

    Same protocol surface as ``ChunkStore`` (see ``ChunkStoreProtocol``).
    Cosine similarity is computed in NumPy on every search — fine for
    the 12-question eval set + small fixture corpora; do NOT use for
    > a few thousand chunks. The pgvector backend exists for that.
    """

    def __init__(self) -> None:
        self._dim: int | None = None
        # Keyed by chunk id; an OrderedDict-style insertion order is
        # preserved by plain dict in 3.7+.
        self._rows: dict[str, StoredChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    async def setup(self, dim: int, drop_existing: bool = False) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if drop_existing:
            self._rows.clear()
            self._vectors.clear()
        self._dim = dim

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256
    ) -> int:
        if self._dim is None:
            raise RuntimeError("upsert_chunks() called before setup()")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector length mismatch: {len(chunks)} vs {vectors.shape[0]}"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != setup dim {self._dim}")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        for i, chunk in enumerate(chunks):
            cid = chunk.chunk_id()
            self._rows[cid] = StoredChunk(
                chunk_id=cid,
                qualified_name=chunk.qualified_name,
                file_path=chunk.file_path,
                lineno_start=chunk.lineno_start,
                lineno_end=chunk.lineno_end,
                kind=chunk.kind,
                text=chunk.text,
                score=1.0,
            )
            self._vectors[cid] = vectors[i].copy()
        return len(chunks)

    async def search(self, query_vec: np.ndarray, k: int = 8) -> list[StoredChunk]:
        if self._dim is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query_vec.ndim != 1 or query_vec.shape[0] != self._dim:
            raise ValueError(
                f"query_vec must be 1-D of dim {self._dim}, got {query_vec.shape!r}"
            )
        if not self._rows:
            return []
        # Cosine similarity = (a . b) / (|a| |b|). Compute against every
        # stored vector — O(n) per query, fine for the in-memory backend.
        q = query_vec.astype(np.float32, copy=False)
        q_norm = float(np.linalg.norm(q)) or 1.0
        results: list[tuple[float, StoredChunk]] = []
        for cid, vec in self._vectors.items():
            v_norm = float(np.linalg.norm(vec)) or 1.0
            cos = float(np.dot(q, vec) / (q_norm * v_norm))
            row = self._rows[cid]
            results.append(
                (
                    cos,
                    StoredChunk(
                        chunk_id=row.chunk_id,
                        qualified_name=row.qualified_name,
                        file_path=row.file_path,
                        lineno_start=row.lineno_start,
                        lineno_end=row.lineno_end,
                        kind=row.kind,
                        text=row.text,
                        score=cos,
                    ),
                )
            )
        results.sort(key=lambda t: t[0], reverse=True)
        return [r[1] for r in results[:k]]

    async def fetch_by_qualified_name(self, qualified_name: str) -> StoredChunk | None:
        for row in self._rows.values():
            if row.qualified_name == qualified_name:
                return row
        return None
