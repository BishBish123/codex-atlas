"""pgvector-backed chunk store.

One row per indexed chunk (function or method body): id + qualified_name +
file_path + lineno_start + lineno_end + kind + text + vector.

Lifecycle: `setup` creates the table + HNSW index; `upsert_chunks` is
idempotent on (file_path, qualified_name, lineno_start); `search` runs a
cosine top-k.

Connection-per-call so the store works behind a process pool. For
production use, swap in a `psycopg_pool` — the API stays the same.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import numpy as np

from codex_atlas.indexer.ast_parser import Chunk, SymbolKind


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


class ChunkStore:
    """Async pgvector adapter for code chunks."""

    def __init__(self, dsn: str, table: str = "codex_atlas_chunks") -> None:
        self._dsn = dsn
        self._table = table
        self._dim: int | None = None

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[asyncpg.Connection]:
        conn = await asyncpg.connect(self._dsn)
        try:
            yield conn
        finally:
            await conn.close()

    async def setup(self, dim: int, drop_existing: bool = False) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
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
