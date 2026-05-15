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

import asyncio
import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
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


class CorruptedChunkStoreError(RuntimeError):
    """Raised when ``InMemoryChunkStore.load_from_path`` finds a malformed snapshot.

    The on-disk JSON exists but its bytes are unparseable, missing
    required keys, or contain values of the wrong shape (e.g. a
    string where an int is required, or an unknown ``SymbolKind``).
    The error message instructs the caller to rebuild the snapshot
    rather than to attempt a recovery — partial loads silently
    poison every downstream search ranking.
    """


@dataclass(frozen=True)
class StoredChunk:
    """A retrieved chunk with its similarity score.

    ``score`` semantics:
    * **0.0** — no cosine similarity is available (graph-traversal upserts,
      ``fetch_by_qualified_name`` lookups, or any path that didn't run a
      vector search).  Downstream scorers (e.g. ``hybrid_score``) treat
      exactly 0.0 as a missing-cosine signal and omit the cosine component
      from the combined score rather than treating it as a genuinely
      terrible match.
    * **positive** — a real cosine similarity in ``(0, 1]``.  The vector
      store sets this from the dot-product search; values produced by
      ``InMemoryChunkStore.search`` are always in this range.
    """

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

    async def delete_by_file_path(self, file_path: str) -> int: ...


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
        # Public ops (upsert, search, fetch, delete) refuse to run until
        # this flag is True so a concurrent caller can't slip past a
        # half-initialised store.
        self._bootstrapped: bool = False
        # Serialises lazy pool creation. Without this, two concurrent
        # ``_connect`` callers can both observe ``_pool is None`` and
        # each fire ``asyncpg.create_pool`` — the second one wins the
        # assignment and the first one's pool is silently leaked. The
        # double-checked init under the lock fixes the race.
        self._pool_lock: asyncio.Lock = asyncio.Lock()
        # Serialises bootstrap. The earlier code set ``self._dim`` BEFORE
        # the DDL block, so a concurrent first upsert/search would see
        # ``_dim is not None``, proceed, and hit a missing-table error.
        # All state mutations move INSIDE the locked section, after the
        # DDL commits, so observers either see "not bootstrapped" (and
        # raise) or "fully bootstrapped" (and proceed).
        self._setup_lock: asyncio.Lock = asyncio.Lock()

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[asyncpg.Connection]:
        # Lazy-init the pool on first acquisition so callers that only
        # ever pass through ``setup`` (e.g. tests using the InMemory
        # variant) don't pay for a connect they never need. Two
        # concurrent first-callers race on the ``is None`` check, so
        # acquire the lock and re-check before creating the pool.
        if self._pool is None:
            async with self._pool_lock:
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

        Concurrency: the body runs under ``_setup_lock``. Visible state
        (``_dim``, ``_bootstrapped``) is set AFTER the DDL block
        finishes so a concurrent ``upsert_chunks`` / ``search`` either
        sees "not bootstrapped" and raises (rather than racing into a
        missing-table error) or sees a fully-initialised store.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        async with self._setup_lock:
            # Idempotent fast path: a previous winning caller already
            # bootstrapped the store at the same dim. Skip the DDL.
            if self._bootstrapped and not drop_existing and self._dim == dim:
                return
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
            # ONLY after the DDL block commits do we publish "ready" state.
            # Concurrent observers that lost the lock now see a fully-set-up
            # store rather than a half-initialised one.
            self._dim = dim
            self._bootstrapped = True

    def _require_bootstrapped(self) -> None:
        """Reject public-op calls that happen before ``setup()`` finishes.

        Concurrent setup + upsert/search used to race: the old code set
        ``_dim`` BEFORE the DDL block, so a parallel caller saw "dim is
        set, must be ready" and proceeded to hit a missing-table error.
        ``_bootstrapped`` flips to True only AFTER the DDL block
        commits, so this check rejects the early caller cleanly with a
        message they can act on.
        """
        if not self._bootstrapped:
            raise RuntimeError("ChunkStore not initialized — call setup() first")

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
        self._require_bootstrapped()
        assert self._dim is not None  # narrowed by _require_bootstrapped
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
        self._require_bootstrapped()
        assert self._dim is not None  # narrowed by _require_bootstrapped
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
        self._require_bootstrapped()
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
            score=0.0,  # no cosine — graph-path lookup, not a vector search
        )

    async def delete_by_file_path(self, file_path: str) -> int:
        """Drop every chunk row owned by ``file_path``. Returns rows deleted.

        Called by the indexer right before re-inserting the chunks for
        a file. Combined with the now-line-stable ``Chunk.chunk_id`` this
        guarantees reindex idempotency: symbols that move keep their row
        (UPSERT updates the lineno fields), and symbols that were renamed
        or deleted are tombstoned by this DELETE rather than left behind
        as ghost rows.
        """
        self._require_bootstrapped()
        async with self._connect() as conn:
            tag = await conn.execute(
                f'DELETE FROM "{self._table}" WHERE file_path = $1',
                file_path,
            )
        # asyncpg returns the command tag, e.g. "DELETE 7"; pull out the
        # row count so callers can report it. Falls back to 0 if the tag
        # is unexpectedly shaped.
        try:
            return int(tag.rsplit(" ", 1)[-1])
        except (ValueError, AttributeError):
            return 0


# ---------------------------------------------------------------------------
# InMemoryChunkStore load helpers (module-level to stay within complexity
# thresholds and to be unit-testable independently of the class).
# ---------------------------------------------------------------------------

_KNOWN_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


def _validate_schema_version(sv: int, src: Path) -> None:
    """Raise ``CorruptedChunkStoreError`` for unknown schema versions."""
    if sv not in _KNOWN_SCHEMA_VERSIONS:
        raise CorruptedChunkStoreError(
            f"snapshot at {src} has unknown schema_version={sv} "
            f"(known: {sorted(_KNOWN_SCHEMA_VERSIONS)}); "
            f"rerun `atlas index --store=memory` to rebuild"
        )


def _load_chunk_entry(
    store: InMemoryChunkStore,
    i: int,
    entry: object,
    dim: int,
) -> None:
    """Validate one chunk entry and insert it into ``store`` (without the lock)."""
    if not isinstance(entry, dict):
        raise ValueError(f"entry {i} must be a JSON object, got {type(entry).__name__}")
    cid = str(entry["chunk_id"])
    store._rows[cid] = StoredChunk(
        chunk_id=cid,
        qualified_name=str(entry["qualified_name"]),
        file_path=str(entry["file_path"]),
        lineno_start=int(entry["lineno_start"]),
        lineno_end=int(entry["lineno_end"]),
        kind=SymbolKind(entry["kind"]),
        text=str(entry["text"]),
        score=0.0,  # no cosine — will be set per-query in search()
    )
    vec = np.asarray(entry["vector"], dtype=np.float32)
    # Validate vector shape and finiteness. A NaN or Inf in a stored vector
    # silently corrupts cosine similarity for every query (dot-product
    # propagates NaN through the whole ranking). A wrong-dim vector
    # crashes search().
    if vec.ndim != 1 or vec.shape[0] != dim:
        raise ValueError(
            f"entry {i} (chunk_id={cid!r}): vector has shape "
            f"{vec.shape!r}, expected ({dim},)"
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError(
            f"entry {i} (chunk_id={cid!r}): vector contains "
            f"non-finite values (NaN or Inf)"
        )
    store._vectors[cid] = vec


class InMemoryChunkStore:
    """Dict-backed chunk store for tests + the eval harness.

    Same protocol surface as ``ChunkStore`` (see ``ChunkStoreProtocol``).
    Cosine similarity is computed in NumPy on every search — fine for
    the 12-question eval set + small fixture corpora; do NOT use for
    > a few thousand chunks. The pgvector backend exists for that.

    Concurrency contract: ``setup``, ``upsert_chunks``, ``search`` and
    ``fetch_by_qualified_name`` are all serialised through an
    ``asyncio.Lock``. Concurrent callers see a consistent view of the
    store — a ``search`` started while an ``upsert_chunks`` is in flight
    blocks until the upsert finishes. Without the lock the concurrent
    iteration over ``_vectors.items()`` raised ``RuntimeError: dictionary
    changed size during iteration``.
    """

    def __init__(self) -> None:
        self._dim: int | None = None
        # Keyed by chunk id; an OrderedDict-style insertion order is
        # preserved by plain dict in 3.7+.
        self._rows: dict[str, StoredChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        # All public methods that touch ``_rows``/``_vectors`` acquire
        # this lock so the in-memory store is concurrent-safe — see
        # class docstring. Constructed lazily on first use because an
        # ``asyncio.Lock`` requires a running loop in some Python
        # versions, and tests construct the store outside one.
        self._lock: asyncio.Lock = asyncio.Lock()

    async def setup(self, dim: int, drop_existing: bool = False) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        async with self._lock:
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
        async with self._lock:
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
                    score=0.0,  # no cosine — will be set per-query in search()
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
        async with self._lock:
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
        async with self._lock:
            for row in self._rows.values():
                if row.qualified_name == qualified_name:
                    return row
            return None

    async def delete_by_file_path(self, file_path: str) -> int:
        """In-memory mirror of ``ChunkStore.delete_by_file_path``."""
        async with self._lock:
            doomed = [cid for cid, row in self._rows.items() if row.file_path == file_path]
            for cid in doomed:
                del self._rows[cid]
                self._vectors.pop(cid, None)
            return len(doomed)

    # Threshold above which save_to_path switches from a single-object
    # JSON file to JSONL (one chunk per line).  At ~10k chunks, a
    # monolithic JSON object serialises to several hundred MB; JSONL
    # lets consumers stream individual rows without holding the entire
    # document in memory.
    _JSONL_CHUNK_THRESHOLD: int = 10_000

    async def save_to_path(self, path: str | Path) -> Path:
        """Serialise the in-memory state to ``path``.

        Persistence is the missing half of ``--store=memory``: without
        it, ``atlas index --store=memory`` would build the embedding
        table inside the indexer process and immediately discard it,
        forcing every ``ask`` / ``search`` to reindex inline. We dump
        rows + vectors + dim to disk so the next CLI invocation can
        ``load_from_path`` and serve queries with no DB.

        **Format selection** — for ≤ 10 000 chunks the file is a
        single JSON object (backwards-compatible with earlier snapshots);
        for > 10 000 chunks the file switches to JSONL: a one-line JSON
        header followed by one chunk-entry per line.  ``load_from_path``
        detects the format automatically.

        **Disk-space preflight** — before writing, we check that the
        destination filesystem has at least as many free bytes as the
        serialised payload.  The check is best-effort; if the stat call
        fails (network FS, special device) we proceed without it.

        Crash- and concurrency-safe: the payload is written to a
        sibling tempfile (``<path>.tmp.<pid>.<ts>``), ``fsync``'d, then
        ``os.replace``'d into place. Readers either see the previous
        complete snapshot or the new complete snapshot — never the
        truncated mid-write JSON that ``Path.write_text`` would leave
        behind on a process crash or a concurrent writer interleave.
        Two concurrent writers each rename their own tempfile; the last
        rename wins and the loser's tempfile is silently superseded
        (``os.replace`` is atomic on POSIX and Windows).
        """
        if self._dim is None:
            raise RuntimeError("save_to_path() called before setup()")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            n_chunks = len(self._rows)
            use_jsonl = n_chunks > self._JSONL_CHUNK_THRESHOLD
            chunk_entries = [
                {
                    "chunk_id": row.chunk_id,
                    "qualified_name": row.qualified_name,
                    "file_path": row.file_path,
                    "lineno_start": row.lineno_start,
                    "lineno_end": row.lineno_end,
                    "kind": str(row.kind),
                    "text": row.text,
                    "vector": self._vectors[row.chunk_id].astype(float).tolist(),
                }
                for row in self._rows.values()
            ]
        # The CLI is the sole caller (one-shot write at end of `atlas
        # index`); blocking I/O here is fine and matches the pattern
        # already used by ``CallGraph.save``.
        # Build the payload eagerly so a serialisation error (e.g. NaN in a
        # vector) raises BEFORE we touch the filesystem — leaves the
        # previous snapshot untouched.
        header = {"schema_version": 1, "dim": self._dim, "format": "jsonl" if use_jsonl else "json"}
        if use_jsonl:
            # JSONL: header line + one chunk entry per line.
            lines = [json.dumps(header)]
            lines.extend(json.dumps(e) for e in chunk_entries)
            encoded = ("\n".join(lines) + "\n").encode("utf-8")
        else:
            payload = {**header, "chunks": chunk_entries}
            encoded = json.dumps(payload).encode("utf-8")
        # Disk-space preflight: estimate free bytes on the destination
        # filesystem before writing so we get a clear error rather than
        # a mysterious partial write or ENOSPC mid-rename.
        try:
            usage = shutil.disk_usage(out.parent)
            if usage.free < len(encoded):
                raise OSError(
                    f"not enough disk space: need {len(encoded):,} bytes, "
                    f"only {usage.free:,} free on {out.parent}"
                )
        except OSError as exc:
            if "not enough disk space" in str(exc):
                raise
            # stat failed (network FS, special device, etc.) — proceed.
        # Capture the destination's current permissions so we can restore
        # them after the atomic replace.  ``os.replace`` inherits the
        # tempfile's mode (0o644) rather than the destination's mode, which
        # would silently downgrade a 0o600 snapshot to world-readable.
        # Read the mode BEFORE the write so a concurrent chmod between
        # stat and replace still wins — the worst case is we restore the
        # pre-write mode, which is no worse than the old behaviour.
        dest_mode: int | None = None
        with contextlib.suppress(OSError):
            dest_mode = os.stat(out).st_mode & 0o777
        # Sibling tempfile keeps the rename on the same filesystem so
        # ``os.replace`` is guaranteed atomic. PID + monotonic ns
        # disambiguates concurrent writers in the same process AND
        # across processes.
        tmp_path = out.with_name(f"{out.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
        try:
            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
            try:
                os.write(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, out)
            # Restore the pre-existing permissions after the atomic swap.
            # Best-effort: a chmod failure is not fatal — the snapshot is
            # already written and readable.
            if dest_mode is not None:
                with contextlib.suppress(OSError):
                    os.chmod(out, dest_mode)
        except BaseException:
            # Best-effort cleanup so a failed write doesn't leak the
            # tempfile next to the snapshot. ``missing_ok`` swallows the
            # already-renamed case where ``os.replace`` succeeded but a
            # later step (there isn't one today, but defensively) raised.
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise
        return out

    @classmethod
    async def load_from_path(cls, path: str | Path) -> InMemoryChunkStore:
        """Build an ``InMemoryChunkStore`` from a JSON file written by ``save_to_path``.

        Raises ``FileNotFoundError`` when the snapshot is missing.
        Raises ``CorruptedChunkStoreError`` when the file is present
        but unparseable, missing required keys, or contains values of
        the wrong shape (e.g. a non-numeric ``dim``, a missing
        ``vector`` field, or an unknown ``kind``). The error message
        names the offending path and tells the caller to rerun
        ``atlas index --store=memory`` to rebuild — partial loads
        would silently poison search rankings, so we refuse them.
        """
        src = Path(path)
        if not src.exists():  # noqa: ASYNC240
            raise FileNotFoundError(
                f"in-memory chunk store snapshot not found at {src}; "
                f"run `atlas index --store=memory` first"
            )
        try:
            raw = src.read_text()  # noqa: ASYNC240
            # Detect format: JSONL files start with a single-line JSON header
            # that includes "format": "jsonl"; plain JSON is the historic default.
            first_line = raw.split("\n", 1)[0]
            is_jsonl = False
            header_obj: dict[str, object] = {}
            with contextlib.suppress(json.JSONDecodeError):
                candidate = json.loads(first_line)
                if isinstance(candidate, dict) and candidate.get("format") == "jsonl":
                    is_jsonl = True
                    header_obj = candidate

            if is_jsonl:
                sv = int(str(header_obj.get("schema_version", 1)))
                _validate_schema_version(sv, src)
                dim = int(str(header_obj["dim"]))
                store = cls()
                await store.setup(dim=dim)
                async with store._lock:
                    for i, raw_line in enumerate(raw.splitlines()[1:]):
                        stripped = raw_line.strip()
                        if not stripped:
                            continue
                        _load_chunk_entry(store, i, json.loads(stripped), dim)
            else:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"snapshot root must be a JSON object, got {type(payload).__name__}"
                    )
                # schema_version defaults to 1 for backwards compatibility with
                # snapshots written before FIX 4 introduced the field.
                sv = int(payload.get("schema_version", 1))
                _validate_schema_version(sv, src)
                dim = int(payload["dim"])
                entries = payload["chunks"]
                if not isinstance(entries, list):
                    raise ValueError(
                        f"`chunks` must be a list, got {type(entries).__name__}"
                    )
                store = cls()
                await store.setup(dim=dim)
                async with store._lock:
                    for i, entry in enumerate(entries):
                        _load_chunk_entry(store, i, entry, dim)
        except CorruptedChunkStoreError:
            raise
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise CorruptedChunkStoreError(
                f"snapshot at {src} is corrupt or malformed ({exc}); "
                f"rerun `atlas index --store=memory` to rebuild"
            ) from exc
        return store
