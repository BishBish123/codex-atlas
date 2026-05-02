"""Unit tests for ``ChunkStore`` helpers that don't need a live Postgres.

End-to-end pgvector behaviour is tested by the integration tests gated
behind ``POSTGRES_DSN``; these cover the pure-Python invariants:
identifier validation, setup idempotence, and pool lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from codex_atlas.store import ChunkStore


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
