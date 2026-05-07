"""Unit tests for ``PgVectorChunkStore`` and the ``make_chunk_store()`` factory.

All tests use mocked asyncpg — no real Postgres required.
The integration path (real DB) is gated behind ``@pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest

from codex_atlas.indexer.ast_parser import SymbolKind
from codex_atlas.store import InMemoryChunkStore, StoredChunk, make_chunk_store
from codex_atlas.store.pgvector import PgVectorChunkStore, _split_ddl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stored_chunk(
    chunk_id: str = "f.py::m.fn",
    qname: str = "m.fn",
    file_path: str = "f.py",
    lineno_start: int = 1,
    lineno_end: int = 5,
    text: str = "def fn(): pass",
    score: float = 0.0,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        qualified_name=qname,
        file_path=file_path,
        lineno_start=lineno_start,
        lineno_end=lineno_end,
        kind=SymbolKind.FUNCTION,
        text=text,
        score=score,
    )


FIXTURE_CHUNKS = [
    _stored_chunk("f.py::m.alpha", "m.alpha", "f.py", 1, 4, "def alpha(): pass"),
    _stored_chunk("f.py::m.beta", "m.beta", "f.py", 10, 14, "def beta(): pass"),
    _stored_chunk("g.py::m.gamma", "m.gamma", "g.py", 1, 3, "def gamma(): pass"),
]

FIXTURE_EMBEDDINGS = [
    [0.1] * 384,
    [0.2] * 384,
    [0.3] * 384,
]


# ---------------------------------------------------------------------------
# Asyncpg connection/pool stubs
# ---------------------------------------------------------------------------


class _FakeRecord(dict):  # type: ignore[type-arg]
    """asyncpg ``Record`` stand-in that supports attribute access."""

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return super().__getitem__(key)


def _make_record(**kwargs: Any) -> _FakeRecord:
    return _FakeRecord(kwargs)


class _FakeConn:
    """Minimal asyncpg connection stub."""

    def __init__(self) -> None:
        self.executed_sqls: list[str] = []
        self.executemany_calls: list[tuple[str, list[Any]]] = []
        self._fetch_rows: list[_FakeRecord] = []
        self._fetchrow_result: _FakeRecord | None = None

    async def execute(self, sql: str, *_args: Any) -> str:
        self.executed_sqls.append(sql)
        return "OK"

    async def executemany(self, sql: str, rows: list[Any]) -> None:
        self.executemany_calls.append((sql, list(rows)))

    async def fetch(self, sql: str, *_args: Any) -> list[_FakeRecord]:
        self.executed_sqls.append(sql)
        return self._fetch_rows

    async def fetchrow(self, sql: str, *_args: Any) -> _FakeRecord | None:
        self.executed_sqls.append(sql)
        return self._fetchrow_result


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:  # type: ignore[misc]
        yield self._conn

    async def close(self) -> None:
        self._closed = True


def _make_store_with_stub(fake_conn: _FakeConn) -> PgVectorChunkStore:
    """Build a ``PgVectorChunkStore`` whose pool is pre-stubbed."""
    store = PgVectorChunkStore.__new__(PgVectorChunkStore)
    store._dsn = "postgresql://stub"
    store._min_pool_size = 1
    store._max_pool_size = 5
    store._pool = _FakePool(fake_conn)  # type: ignore[assignment]
    store._bootstrapped = False
    store._pool_lock = asyncio.Lock()
    store._setup_lock = asyncio.Lock()
    return store


async def _bootstrap(store: PgVectorChunkStore, fake_conn: _FakeConn) -> None:
    """Call ``setup()`` while suppressing the pgvector codec registration."""
    with patch("codex_atlas.store.pgvector._register_vector", new=None):
        await store.setup()


# ---------------------------------------------------------------------------
# _split_ddl helper
# ---------------------------------------------------------------------------


class TestSplitDdl:
    def test_splits_on_semicolons(self) -> None:
        sql = "CREATE EXTENSION IF NOT EXISTS vector; CREATE TABLE IF NOT EXISTS t (id TEXT)"
        parts = _split_ddl(sql)
        assert len(parts) == 2
        assert parts[0] == "CREATE EXTENSION IF NOT EXISTS vector"
        assert "CREATE TABLE" in parts[1]

    def test_ignores_blank_segments(self) -> None:
        sql = "  ;  CREATE TABLE t (id TEXT);  ;  "
        parts = _split_ddl(sql)
        assert len(parts) == 1

    def test_empty_string(self) -> None:
        assert _split_ddl("") == []


# ---------------------------------------------------------------------------
# setup() — schema DDL
# ---------------------------------------------------------------------------


class TestSetup:
    async def test_ddl_applied_on_setup(self) -> None:
        """``setup()`` must execute the four DDL statements idempotently."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        ddl_text = " ".join(conn.executed_sqls)
        assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl_text
        assert "CREATE TABLE IF NOT EXISTS chunks" in ddl_text
        assert "chunks_qname_idx" in ddl_text
        assert "chunks_embedding_idx" in ddl_text
        assert "ivfflat" in ddl_text
        assert "lists = 100" in ddl_text

    async def test_setup_idempotent(self) -> None:
        """Calling ``setup()`` twice must not re-execute the DDL."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        first_count = len(conn.executed_sqls)
        # Second call — should short-circuit.
        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.setup()
        assert len(conn.executed_sqls) == first_count

    async def test_bootstrapped_flag_set_after_setup(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        assert not store._bootstrapped
        await _bootstrap(store, conn)
        assert store._bootstrapped

    async def test_public_ops_require_setup(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        # _bootstrapped starts False — every public op must raise.
        with pytest.raises(RuntimeError, match="not initialised"):
            await store.upsert_chunks([])
        with pytest.raises(RuntimeError, match="not initialised"):
            await store.fetch_by_qualified_name("m.fn")
        with pytest.raises(RuntimeError, match="not initialised"):
            await store.search([0.0] * 384)
        with pytest.raises(RuntimeError, match="not initialised"):
            await store.get("some_id")
        with pytest.raises(RuntimeError, match="not initialised"):
            await store.clear()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


class TestUpsertChunks:
    async def test_upsert_round_trip_via_executemany(self) -> None:
        """``upsert_chunks`` must issue ``executemany`` with ON CONFLICT DO UPDATE."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executemany_calls.clear()

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.upsert_chunks(FIXTURE_CHUNKS[:2])

        assert len(conn.executemany_calls) == 1
        sql, rows = conn.executemany_calls[0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql
        assert len(rows) == 2

    async def test_upsert_empty_list_is_noop(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executemany_calls.clear()

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.upsert_chunks([])

        assert conn.executemany_calls == []

    async def test_idempotent_upsert_same_chunk_id(self) -> None:
        """Re-upserting the same chunk_id must use ON CONFLICT DO UPDATE (not insert twice)."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executemany_calls.clear()

        chunk = _stored_chunk(chunk_id="unique_id", qname="m.fn")
        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.upsert_chunks([chunk])
            await store.upsert_chunks([chunk])

        # Two calls to executemany, each with ON CONFLICT DO UPDATE
        for sql, _ in conn.executemany_calls:
            assert "ON CONFLICT" in sql and "DO UPDATE" in sql

    async def test_upsert_with_embeddings_length_mismatch_raises(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        with (
            pytest.raises(ValueError, match="length mismatch"),
            patch("codex_atlas.store.pgvector._register_vector", new=None),
        ):
            await store.upsert_with_embeddings(FIXTURE_CHUNKS, [[0.1] * 384])

    async def test_upsert_with_embeddings_passes_vectors(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executemany_calls.clear()

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.upsert_with_embeddings(FIXTURE_CHUNKS, FIXTURE_EMBEDDINGS)

        assert len(conn.executemany_calls) == 1
        _, rows = conn.executemany_calls[0]
        assert len(rows) == 3
        # Embedding should be passed as the last element of each row tuple.
        assert rows[0][-1] == FIXTURE_EMBEDDINGS[0]
        assert rows[1][-1] == FIXTURE_EMBEDDINGS[1]
        assert rows[2][-1] == FIXTURE_EMBEDDINGS[2]


# ---------------------------------------------------------------------------
# fetch_by_qualified_name
# ---------------------------------------------------------------------------


class TestFetchByQualifiedName:
    async def test_returns_matching_rows(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        conn._fetch_rows = [
            _make_record(
                chunk_id="f.py::m.alpha",
                qualified_name="m.alpha",
                file_path="f.py",
                lineno_start=1,
                lineno_end=4,
                text="def alpha(): pass",
            )
        ]
        results = await store.fetch_by_qualified_name("m.alpha")
        assert len(results) == 1
        assert results[0].qualified_name == "m.alpha"
        assert results[0].file_path == "f.py"
        assert results[0].score == 0.0  # fetch path returns score=0

    async def test_returns_empty_list_when_no_match(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        conn._fetch_rows = []
        results = await store.fetch_by_qualified_name("m.ghost")
        assert results == []

    async def test_uses_indexed_lookup_sql(self) -> None:
        """The SQL must reference the ``qualified_name`` column with a parameter."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executed_sqls.clear()
        conn._fetch_rows = []

        await store.fetch_by_qualified_name("m.fn")

        assert len(conn.executed_sqls) == 1
        sql = conn.executed_sqls[0]
        assert "qualified_name" in sql
        assert "$1" in sql


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_sql_shape(self) -> None:
        """``search`` must use cosine distance operator ``<=>`` with the embedding column."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executed_sqls.clear()
        conn._fetch_rows = []

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            await store.search([0.1] * 384, top_k=5)

        assert len(conn.executed_sqls) == 1
        sql = conn.executed_sqls[0]
        assert "<=>" in sql
        assert "embedding" in sql
        assert "$1" in sql  # query vector is parameterised
        assert "$2" in sql  # top_k is parameterised

    async def test_search_result_parsing(self) -> None:
        """Results from the DB must be parsed into ``StoredChunk`` with the cosine score."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        conn._fetch_rows = [
            _make_record(
                chunk_id="f.py::m.alpha",
                qualified_name="m.alpha",
                file_path="f.py",
                lineno_start=1,
                lineno_end=4,
                text="def alpha(): pass",
                score=0.92,
            ),
            _make_record(
                chunk_id="g.py::m.gamma",
                qualified_name="m.gamma",
                file_path="g.py",
                lineno_start=1,
                lineno_end=3,
                text="def gamma(): pass",
                score=0.71,
            ),
        ]

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            results = await store.search([0.1] * 384, top_k=2)

        assert len(results) == 2
        assert results[0].qualified_name == "m.alpha"
        assert results[0].score == pytest.approx(0.92)
        assert results[1].qualified_name == "m.gamma"
        assert results[1].score == pytest.approx(0.71)

    async def test_search_invalid_top_k_raises(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        with pytest.raises(ValueError, match="top_k"):
            await store.search([0.1] * 384, top_k=0)

    async def test_search_empty_store_returns_empty(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn._fetch_rows = []

        with patch("codex_atlas.store.pgvector._register_vector", new=None):
            results = await store.search([0.1] * 384, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_existing_chunk(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        conn._fetchrow_result = _make_record(
            chunk_id="f.py::m.alpha",
            qualified_name="m.alpha",
            file_path="f.py",
            lineno_start=1,
            lineno_end=4,
            text="def alpha(): pass",
        )
        result = await store.get("f.py::m.alpha")
        assert result is not None
        assert result.chunk_id == "f.py::m.alpha"
        assert result.qualified_name == "m.alpha"

    async def test_get_missing_chunk_returns_none(self) -> None:
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)

        conn._fetchrow_result = None
        result = await store.get("nonexistent_id")
        assert result is None


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_issues_truncate(self) -> None:
        """``clear()`` must execute ``TRUNCATE TABLE chunks``."""
        conn = _FakeConn()
        store = _make_store_with_stub(conn)
        await _bootstrap(store, conn)
        conn.executed_sqls.clear()

        await store.clear()

        assert len(conn.executed_sqls) == 1
        assert "TRUNCATE" in conn.executed_sqls[0].upper()
        assert "chunks" in conn.executed_sqls[0]


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_marks_pool_closed(self) -> None:
        conn = _FakeConn()
        pool = _FakePool(conn)
        store = PgVectorChunkStore.__new__(PgVectorChunkStore)
        store._dsn = "postgresql://stub"
        store._pool = pool  # type: ignore[assignment]
        store._bootstrapped = True
        store._pool_lock = asyncio.Lock()
        store._setup_lock = asyncio.Lock()
        store._min_pool_size = 1
        store._max_pool_size = 5

        await store.close()

        assert pool._closed
        assert store._pool is None
        assert not store._bootstrapped

    async def test_close_idempotent(self) -> None:
        store = PgVectorChunkStore.__new__(PgVectorChunkStore)
        store._dsn = "postgresql://stub"
        store._pool = None
        store._bootstrapped = False
        store._pool_lock = asyncio.Lock()
        store._setup_lock = asyncio.Lock()
        store._min_pool_size = 1
        store._max_pool_size = 5
        # Must not raise even when pool is None.
        await store.close()


# ---------------------------------------------------------------------------
# make_chunk_store() factory
# ---------------------------------------------------------------------------


class TestMakeChunkStoreFactory:
    def test_default_returns_in_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATLAS_CHUNK_STORE", raising=False)
        store = make_chunk_store()
        assert isinstance(store, InMemoryChunkStore)

    def test_memory_explicit_returns_in_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_CHUNK_STORE", "memory")
        store = make_chunk_store()
        assert isinstance(store, InMemoryChunkStore)

    def test_pgvector_with_dsn_returns_pgvector_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_CHUNK_STORE", "pgvector")
        monkeypatch.setenv("ATLAS_PG_DSN", "postgresql://u:p@localhost/db")
        store = make_chunk_store()
        assert isinstance(store, PgVectorChunkStore)

    def test_pgvector_without_dsn_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_CHUNK_STORE", "pgvector")
        monkeypatch.delenv("ATLAS_PG_DSN", raising=False)
        with pytest.raises(RuntimeError, match="ATLAS_PG_DSN"):
            make_chunk_store()

    def test_unknown_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_CHUNK_STORE", "redis")
        with pytest.raises(RuntimeError, match="unknown ATLAS_CHUNK_STORE"):
            make_chunk_store()
