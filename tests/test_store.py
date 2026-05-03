"""Unit tests for ``ChunkStore`` helpers that don't need a live Postgres.

End-to-end pgvector behaviour is tested by the integration tests gated
behind ``POSTGRES_DSN``; these cover the pure-Python invariants:
identifier validation, setup idempotence, and pool lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from codex_atlas.indexer.ast_parser import Chunk, SymbolKind
from codex_atlas.store import ChunkStore, InMemoryChunkStore


class _FakeConnection:
    """Minimal connection stub that records every executed SQL string."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> str:
        self.executed.append(sql)
        return "OK"


@asynccontextmanager
async def _stub_connect(conn: _FakeConnection) -> AsyncIterator[_FakeConnection]:
    yield conn


class TestTableIdentifierValidation:
    """``table`` is interpolated into DDL/DML f-strings — validate it."""

    @pytest.mark.parametrize(
        "bad",
        [
            "; DROP TABLE foo; --",
            "foo; DROP TABLE bar",
            "foo bar",
            'foo"bar',
            "1foo",  # leading digit not allowed
            "foo.bar",  # dot
            "",
            "a" * 64,  # exceeds NAMEDATALEN-1
        ],
    )
    def test_rejects_invalid_table_name(self, bad: str) -> None:
        with pytest.raises(ValueError, match="invalid table identifier"):
            ChunkStore(dsn="postgresql://stub", table=bad)

    def test_accepts_default_table(self) -> None:
        # Default name and any plain SQL identifier should pass.
        ChunkStore(dsn="postgresql://stub")
        ChunkStore(dsn="postgresql://stub", table="my_table")
        ChunkStore(dsn="postgresql://stub", table="_underscored")
        ChunkStore(dsn="postgresql://stub", table="MixedCase123")


class TestSetupIdempotence:
    async def test_setup_twice_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The second ``setup`` call with the same dim must NOT re-issue
        # the DDL — chronic-path callers (MCP, agent loop) call setup
        # before each retrieval, and re-running DDL was the whole point
        # of the connection-overhead complaint.
        store = ChunkStore(dsn="postgresql://stub", table="test_chunks")
        conn = _FakeConnection()
        # Replace _connect with a no-pool stub so we don't need a server.
        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(conn)
        )
        await store.setup(dim=8)
        first_call_count = len(conn.executed)
        assert first_call_count > 0  # bootstrap ran
        await store.setup(dim=8)
        # No additional DDL on the second call.
        assert len(conn.executed) == first_call_count

    async def test_setup_with_different_dim_re_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Changing dim is a real schema change; setup must run again.
        store = ChunkStore(dsn="postgresql://stub")
        conn = _FakeConnection()
        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(conn)
        )
        await store.setup(dim=8)
        first_call_count = len(conn.executed)
        await store.setup(dim=16)
        assert len(conn.executed) > first_call_count

    async def test_setup_with_drop_existing_re_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``drop_existing`` always re-runs — the caller asked for a fresh schema.
        store = ChunkStore(dsn="postgresql://stub")
        conn = _FakeConnection()
        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(conn)
        )
        await store.setup(dim=8)
        first_call_count = len(conn.executed)
        await store.setup(dim=8, drop_existing=True)
        assert len(conn.executed) > first_call_count

    async def test_invalid_dim_rejected(self) -> None:
        store = ChunkStore(dsn="postgresql://stub")
        # Negative dim is clearly invalid; zero is the corner case the
        # constraint ``dim > 0`` is for.
        with pytest.raises(ValueError, match="dim"):
            await store.setup(dim=0)


def _chunk(qname: str, idx: int) -> Chunk:
    return Chunk(
        qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        file_path=f"f{idx}.py",
        lineno_start=1,
        lineno_end=2,
        text=f"def {qname.rsplit('.', 1)[-1]}(): pass",
    )


class TestInMemoryChunkStoreRoundTrip:
    async def test_round_trip_100_chunks(self) -> None:
        # The fundamental contract: write N chunks, search returns ranked
        # StoredChunks, fetch_by_qualified_name finds them by qname.
        store = InMemoryChunkStore()
        await store.setup(dim=8)
        rng = np.random.default_rng(seed=42)
        n = 100
        chunks = [_chunk(f"m.fn_{i}", i) for i in range(n)]
        vectors = rng.standard_normal((n, 8)).astype(np.float32)
        written = await store.upsert_chunks(chunks, vectors)
        assert written == n
        # Search using one of the actual stored vectors — that vector
        # must rank itself first (cosine = 1.0 with self).
        query = vectors[42]
        results = await store.search(query, k=5)
        assert results[0].qualified_name == "m.fn_42"
        assert results[0].score == pytest.approx(1.0, rel=1e-4)
        assert len(results) == 5
        # fetch_by_qualified_name returns the row for known qnames.
        row = await store.fetch_by_qualified_name("m.fn_7")
        assert row is not None
        assert row.qualified_name == "m.fn_7"
        # And None for unknown.
        assert (await store.fetch_by_qualified_name("m.ghost")) is None

    async def test_setup_with_drop_existing_clears(self) -> None:
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        chunks = [_chunk("m.a", 0)]
        vectors = np.zeros((1, 4), dtype=np.float32)
        await store.upsert_chunks(chunks, vectors)
        # Re-setup with drop_existing wipes prior content.
        await store.setup(dim=4, drop_existing=True)
        results = await store.search(vectors[0], k=1)
        assert results == []

    async def test_search_before_setup_errors(self) -> None:
        store = InMemoryChunkStore()
        with pytest.raises(RuntimeError, match="setup"):
            await store.search(np.zeros(4, dtype=np.float32))

    async def test_dim_mismatch_rejected(self) -> None:
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        chunks = [_chunk("m.a", 0)]
        bad_vectors = np.zeros((1, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="dim"):
            await store.upsert_chunks(chunks, bad_vectors)
