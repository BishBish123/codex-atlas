"""Unit tests for ``ChunkStore`` helpers that don't need a live Postgres.

End-to-end pgvector behaviour is tested by the integration tests gated
behind ``POSTGRES_DSN``; these cover the pure-Python invariants:
identifier validation, setup idempotence, and pool lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from codex_atlas.indexer.ast_parser import Chunk, SymbolKind
from codex_atlas.store import ChunkStore, CorruptedChunkStoreError, InMemoryChunkStore


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


class TestPoolInitIdempotence:
    async def test_pool_init_idempotent_under_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two concurrent first-callers used to each observe ``_pool is
        # None`` and each fire ``asyncpg.create_pool``; the loser's pool
        # was leaked. With the double-checked init lock, ``create_pool``
        # is invoked exactly once even under N concurrent ``_connect``
        # callers.
        store = ChunkStore(dsn="postgresql://stub")
        create_calls = 0

        class _StubConn:
            async def execute(self, *_args: Any, **_kwargs: Any) -> str:
                return "OK"

        class _StubAcquire:
            async def __aenter__(self) -> _StubConn:
                return _StubConn()

            async def __aexit__(self, *_exc: Any) -> None:
                return None

        class _StubPool:
            def acquire(self) -> _StubAcquire:
                return _StubAcquire()

            async def close(self) -> None:  # pragma: no cover
                pass

        async def fake_create_pool(*_args: Any, **_kwargs: Any) -> _StubPool:
            nonlocal create_calls
            # Yield to the loop so a concurrent caller has the chance
            # to observe the pre-assignment ``_pool is None`` state —
            # this is exactly the interleaving the lock must defeat.
            await asyncio.sleep(0)
            create_calls += 1
            return _StubPool()

        import codex_atlas.store as store_mod  # noqa: PLC0415

        monkeypatch.setattr(store_mod.asyncpg, "create_pool", fake_create_pool)

        async def open_and_close() -> None:
            async with store._connect():
                pass

        await asyncio.gather(*(open_and_close() for _ in range(10)))
        # Without the double-checked lock, two of these would race past
        # the ``is None`` check and call create_pool twice.
        assert create_calls == 1


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

    async def test_in_memory_store_concurrent_upsert_and_search(self) -> None:
        # Without the per-store lock, ``search`` iterating ``_vectors``
        # while ``upsert_chunks`` mutated it raised
        # ``RuntimeError: dictionary changed size during iteration``.
        # Fire many upserts + searches concurrently and assert (1) no
        # exceptions escape, (2) the final state is consistent (all
        # upserted chunks present, search results don't reference dropped
        # ids).
        store = InMemoryChunkStore()
        await store.setup(dim=8)
        rng = np.random.default_rng(seed=7)
        # Pre-seed so search has rows to iterate over from the start.
        seed_chunks = [_chunk(f"m.seed_{i}", i) for i in range(20)]
        seed_vecs = rng.standard_normal((20, 8)).astype(np.float32)
        await store.upsert_chunks(seed_chunks, seed_vecs)

        async def upsert_batch(start: int) -> None:
            chunks = [_chunk(f"m.batch_{start}_{i}", start * 100 + i) for i in range(10)]
            vecs = rng.standard_normal((10, 8)).astype(np.float32)
            await store.upsert_chunks(chunks, vecs)

        async def do_search() -> list[str]:
            qv = rng.standard_normal(8).astype(np.float32)
            res = await store.search(qv, k=5)
            return [r.qualified_name for r in res]

        # Mix 10 upsert batches with 20 searches, all firing in parallel.
        tasks: list[asyncio.Future[Any]] = []
        for i in range(10):
            tasks.append(asyncio.ensure_future(upsert_batch(i)))
        for _ in range(20):
            tasks.append(asyncio.ensure_future(do_search()))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # No RuntimeError leaked from concurrent dict iteration.
        for r in results:
            assert not isinstance(r, BaseException), r

        # Final state: all 20 seed chunks + 10 batches * 10 = 120 rows.
        for i in range(20):
            assert (await store.fetch_by_qualified_name(f"m.seed_{i}")) is not None
        for batch in range(10):
            for i in range(10):
                assert (
                    await store.fetch_by_qualified_name(f"m.batch_{batch}_{i}")
                ) is not None


class TestChunkIdStability:
    """``Chunk.chunk_id`` must be stable on (file_path, qualified_name).

    The earlier scheme keyed on ``lineno_start`` too, which meant any
    line shift (a new import, a docstring tweak) re-minted a fresh row
    instead of UPSERTing the existing one. Ghost rows piled up across
    reindexes; retrieval surfaced them. The fix drops lineno from the
    identity, so the same symbol at a different line keeps the same id.
    """

    def test_chunk_id_stable_under_line_shift(self) -> None:
        a = Chunk(
            qualified_name="pkg.m.fn",
            kind=SymbolKind.FUNCTION,
            file_path="pkg/m.py",
            lineno_start=10,
            lineno_end=12,
            text="def fn(): pass",
        )
        b = Chunk(
            qualified_name="pkg.m.fn",
            kind=SymbolKind.FUNCTION,
            file_path="pkg/m.py",
            lineno_start=42,  # symbol moved (new import added above it)
            lineno_end=44,
            text="def fn(): pass",
        )
        assert a.chunk_id() == b.chunk_id()

    def test_chunk_id_distinct_per_file(self) -> None:
        # Same qname in two different files is a real conflict-of-name
        # situation that the store must keep separate.
        a = Chunk(
            qualified_name="pkg.helper",
            kind=SymbolKind.FUNCTION,
            file_path="pkg/a.py",
            lineno_start=1,
            lineno_end=2,
            text="def helper(): pass",
        )
        b = Chunk(
            qualified_name="pkg.helper",
            kind=SymbolKind.FUNCTION,
            file_path="pkg/b.py",
            lineno_start=1,
            lineno_end=2,
            text="def helper(): pass",
        )
        assert a.chunk_id() != b.chunk_id()


class TestReindexReplacesChunks:
    """`InMemoryChunkStore.delete_by_file_path` + UPSERT == idempotent reindex.

    The stable-id scheme alone isn't enough: when a symbol is renamed
    or removed the OLD chunk_id never collides with anything on the
    next pass. We delete-by-file before reinserting so renames /
    deletions tombstone cleanly.
    """

    async def test_reindex_replaces_chunks_for_changed_file(self) -> None:
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        # v1: three symbols in m.py.
        v1 = [
            Chunk(
                qualified_name=f"m.{name}",
                kind=SymbolKind.FUNCTION,
                file_path="m.py",
                lineno_start=i + 1,
                lineno_end=i + 2,
                text=f"def {name}(): pass",
            )
            for i, name in enumerate(("alpha", "beta", "gamma"))
        ]
        await store.upsert_chunks(v1, np.ones((3, 4), dtype=np.float32))
        for name in ("alpha", "beta", "gamma"):
            assert (await store.fetch_by_qualified_name(f"m.{name}")) is not None
        # v2: ``beta`` was removed, ``gamma`` was renamed to ``delta``,
        # ``alpha`` shifted down a few lines.
        v2 = [
            Chunk(
                qualified_name="m.alpha",
                kind=SymbolKind.FUNCTION,
                file_path="m.py",
                lineno_start=20,
                lineno_end=21,
                text="def alpha(): pass",
            ),
            Chunk(
                qualified_name="m.delta",
                kind=SymbolKind.FUNCTION,
                file_path="m.py",
                lineno_start=30,
                lineno_end=31,
                text="def delta(): pass",
            ),
        ]
        # Mirror the cli's reindex sequence: delete by file, then upsert.
        n_deleted = await store.delete_by_file_path("m.py")
        assert n_deleted == 3
        await store.upsert_chunks(v2, np.ones((2, 4), dtype=np.float32))
        # alpha + delta survive; beta + gamma are tombstoned.
        assert (await store.fetch_by_qualified_name("m.alpha")) is not None
        assert (await store.fetch_by_qualified_name("m.delta")) is not None
        assert (await store.fetch_by_qualified_name("m.beta")) is None
        assert (await store.fetch_by_qualified_name("m.gamma")) is None
        # alpha's stored lineno reflects the new position — the row was
        # updated in place, not re-minted as a duplicate.
        alpha = await store.fetch_by_qualified_name("m.alpha")
        assert alpha is not None and alpha.lineno_start == 20

    async def test_delete_by_file_path_only_touches_named_file(self) -> None:
        # A file's delete must NOT take other files' rows with it.
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        chunks = [
            Chunk(
                qualified_name="m.a",
                kind=SymbolKind.FUNCTION,
                file_path="a.py",
                lineno_start=1,
                lineno_end=2,
                text="def a(): pass",
            ),
            Chunk(
                qualified_name="m.b",
                kind=SymbolKind.FUNCTION,
                file_path="b.py",
                lineno_start=1,
                lineno_end=2,
                text="def b(): pass",
            ),
        ]
        await store.upsert_chunks(chunks, np.ones((2, 4), dtype=np.float32))
        n = await store.delete_by_file_path("a.py")
        assert n == 1
        assert (await store.fetch_by_qualified_name("m.a")) is None
        assert (await store.fetch_by_qualified_name("m.b")) is not None


class TestChunkStoreDeleteByFilePath:
    """`ChunkStore.delete_by_file_path` issues a single parameterised DELETE.

    The DELETE hits the live pgvector table at integration time; we
    verify the SQL shape and the command-tag parsing here so the unit
    suite can lock down regressions without a Postgres dependency.
    """

    async def test_delete_by_file_path_issues_parameterised_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = ChunkStore(dsn="postgresql://stub", table="chunks_test")

        executed: list[tuple[str, tuple[Any, ...]]] = []

        class _Conn:
            async def execute(self, sql: str, *args: Any, **_kwargs: Any) -> str:
                executed.append((sql, args))
                # Mimic asyncpg's command tag for DELETE.
                return "DELETE 4"

        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(_Conn())
        )
        await store.setup(dim=4)
        # Drop the bootstrap DDL captures so we only see the DELETE.
        executed.clear()
        rows = await store.delete_by_file_path("pkg/m.py")
        assert rows == 4
        assert len(executed) == 1
        sql, args = executed[0]
        # The file_path is parameterised — no string interpolation of
        # caller-controlled values.
        assert "DELETE FROM" in sql
        assert "WHERE file_path = $1" in sql
        assert args == ("pkg/m.py",)

    async def test_delete_by_file_path_handles_zero_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = ChunkStore(dsn="postgresql://stub", table="chunks_test")

        class _Conn:
            async def execute(self, *_args: Any, **_kwargs: Any) -> str:
                return "DELETE 0"

        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(_Conn())
        )
        await store.setup(dim=4)
        assert (await store.delete_by_file_path("nope.py")) == 0

    async def test_delete_before_setup_errors(self) -> None:
        store = ChunkStore(dsn="postgresql://stub")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.delete_by_file_path("a.py")


class TestSetupRaceProtection:
    """``setup()`` and concurrent first-callers must not race.

    The earlier code set ``self._dim = dim`` BEFORE the DDL block.
    A concurrent first ``upsert_chunks`` / ``search`` saw "dim is set"
    and proceeded to hit a missing-table error. The fix moves the state
    publication to AFTER the DDL block AND gates every public op on a
    ``_bootstrapped`` flag set in the same atomic step.
    """

    async def test_concurrent_setup_and_upsert_does_not_race(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = ChunkStore(dsn="postgresql://stub")

        ddl_started = asyncio.Event()
        ddl_can_finish = asyncio.Event()

        class _Conn:
            async def execute(self, sql: str, *_args: Any, **_kwargs: Any) -> str:
                # Pause INSIDE the DDL block so a concurrent caller has
                # the maximum opportunity to see the half-initialised
                # state. Without the lock + post-DDL flag, the caller
                # would race past ``_dim is not None`` and try to UPSERT
                # before the table exists.
                if "CREATE TABLE" in sql:
                    ddl_started.set()
                    await ddl_can_finish.wait()
                return "OK"

            async def fetch(self, *_args: Any, **_kwargs: Any) -> list[Any]:
                return []

            async def executemany(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(_Conn())
        )
        # pgvector's register_vector hits the conn with type-codec
        # registration; stub it out so we don't depend on a live conn.
        from pgvector import asyncpg as pgv_async  # noqa: PLC0415

        async def fake_register_vector(_conn: Any) -> None:
            return None

        monkeypatch.setattr(pgv_async, "register_vector", fake_register_vector)

        async def setup_task() -> None:
            await store.setup(dim=4)

        async def upsert_task() -> Exception | int | None:
            await ddl_started.wait()
            # At this exact point the OLD implementation already had
            # ``_dim`` set; the upsert would slip past and fail at the
            # database. The new implementation rejects with a clean
            # RuntimeError that says "call setup() first".
            chunks = [
                Chunk(
                    qualified_name="m.fn",
                    kind=SymbolKind.FUNCTION,
                    file_path="m.py",
                    lineno_start=1,
                    lineno_end=2,
                    text="def fn(): pass",
                )
            ]
            try:
                return await store.upsert_chunks(
                    chunks, np.zeros((1, 4), dtype=np.float32)
                )
            except RuntimeError as e:
                return e

        async def driver() -> tuple[None, Exception | int | None]:
            tasks = asyncio.gather(setup_task(), upsert_task())
            # Let upsert_task park on ddl_started, then release the DDL.
            await ddl_started.wait()
            ddl_can_finish.set()
            return await tasks

        results: tuple[None, Exception | int | None] = await driver()
        _, upsert_outcome = results
        # Upsert either errored cleanly with the not-initialised
        # message OR completed successfully (if the lock interleaving
        # made it run AFTER setup finished). What MUST NOT happen is
        # asyncpg.UndefinedTableError or similar racing-into-DDL noise.
        if isinstance(upsert_outcome, Exception):
            assert "not initialized" in str(upsert_outcome)
        else:
            assert upsert_outcome == 1

    async def test_setup_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The lock + flag must NOT make repeat setup() calls re-run the
        # DDL block. ``test_setup_twice_short_circuits`` exists already
        # — this version asserts the same property explicitly under the
        # new locking regime.
        store = ChunkStore(dsn="postgresql://stub")
        conn = _FakeConnection()
        monkeypatch.setattr(
            ChunkStore, "_connect", lambda self: _stub_connect(conn)
        )
        await store.setup(dim=8)
        n_after_first = len(conn.executed)
        for _ in range(5):
            await store.setup(dim=8)
        # Idempotent re-entry: no new DDL across the next 5 calls.
        assert len(conn.executed) == n_after_first

    async def test_public_ops_before_setup_error(self) -> None:
        store = ChunkStore(dsn="postgresql://stub")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.upsert_chunks([], np.zeros((0, 4), dtype=np.float32))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.search(np.zeros(4, dtype=np.float32))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.fetch_by_qualified_name("anything")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.delete_by_file_path("anything.py")


class TestInMemoryStorePersistence:
    """``InMemoryChunkStore.save_to_path`` + ``load_from_path`` round-trip.

    The hermetic ``--store=memory`` flow depends on the indexer
    serialising the in-memory rows to disk so subsequent ``ask`` /
    ``search`` invocations can read them back without re-embedding.
    """

    async def test_round_trip_preserves_search_ranking(self, tmp_path: Any) -> None:
        store = InMemoryChunkStore()
        await store.setup(dim=8)
        rng = np.random.default_rng(seed=11)
        chunks = [_chunk(f"m.fn_{i}", i) for i in range(20)]
        vectors = rng.standard_normal((20, 8)).astype(np.float32)
        await store.upsert_chunks(chunks, vectors)
        path = tmp_path / "chunks.json"
        await store.save_to_path(path)
        assert path.exists()

        # Load into a fresh store and verify search returns the same top-1.
        loaded = await InMemoryChunkStore.load_from_path(path)
        results = await loaded.search(vectors[7], k=3)
        assert results[0].qualified_name == "m.fn_7"
        assert results[0].score == pytest.approx(1.0, rel=1e-4)
        # fetch_by_qualified_name keeps working after the round-trip.
        row = await loaded.fetch_by_qualified_name("m.fn_3")
        assert row is not None
        assert row.qualified_name == "m.fn_3"

    async def test_save_before_setup_errors(self, tmp_path: Any) -> None:
        store = InMemoryChunkStore()
        with pytest.raises(RuntimeError, match="setup"):
            await store.save_to_path(tmp_path / "chunks.json")

    async def test_load_missing_file_errors(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError, match="atlas index"):
            await InMemoryChunkStore.load_from_path(tmp_path / "nope.json")

    async def test_save_to_path_is_atomic(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-write crash must leave the previous snapshot intact.

        Concurrent ``atlas index`` + ``atlas ask`` used to be able to
        observe a half-written ``chunks.json`` because ``Path.write_text``
        truncates first and writes second. The atomic-rename refit
        writes to a sibling tempfile and only ``os.replace``'s after the
        bytes are durable; a crash mid-encode (modelled here by making
        ``json.dumps`` raise) must therefore not perturb the existing
        file.
        """
        # First, write a known-good snapshot — the file we're protecting.
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        chunks = [_chunk("m.fn_0", 0)]
        vectors = np.eye(1, 4, dtype=np.float32)
        await store.upsert_chunks(chunks, vectors)
        path = tmp_path / "chunks.json"
        await store.save_to_path(path)
        good_bytes = path.read_bytes()

        # Now make ``json.dumps`` blow up to simulate a fault during the
        # next write. The old code would have called ``write_text`` and
        # left the file truncated; the atomic path raises BEFORE any
        # filesystem write because we encode first, so the existing
        # file is bit-for-bit unchanged.
        import codex_atlas.store as store_mod  # noqa: PLC0415

        def _boom(*_a: Any, **_kw: Any) -> str:
            raise RuntimeError("simulated mid-write fault")

        monkeypatch.setattr(store_mod.json, "dumps", _boom)

        with pytest.raises(RuntimeError, match="simulated"):
            await store.save_to_path(path)

        assert path.read_bytes() == good_bytes
        # And no orphan tempfile next to the snapshot.
        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert siblings == ["chunks.json"], siblings

    async def test_load_raises_on_truncated_json(self, tmp_path: Any) -> None:
        """A half-written file (e.g. a power-loss snapshot) raises a typed error.

        Before the typed-error refit, ``load_from_path`` let the bare
        ``json.JSONDecodeError`` bubble straight to the CLI top level,
        producing a stack trace rather than an actionable message. The
        new contract: detect it, wrap it, point the caller at
        ``atlas index --store=memory``.
        """
        path = tmp_path / "chunks.json"
        # Truncate a real-looking payload mid-key. The previous
        # implementation would have stopped at the JSONDecodeError.
        path.write_text('{"dim": 8, "chunks": [{"chunk_id": "a", ')
        with pytest.raises(CorruptedChunkStoreError, match="atlas index"):
            await InMemoryChunkStore.load_from_path(path)

    async def test_load_raises_on_unknown_schema_keys(self, tmp_path: Any) -> None:
        """Missing required keys / wrong types yield ``CorruptedChunkStoreError``.

        The schema is implicit (positional dict access in
        ``load_from_path``); validating it explicitly here pins the
        contract so a future schema bump can't quietly load a
        partially-populated store.
        """
        path = tmp_path / "chunks.json"
        # Valid JSON, but ``dim`` is a string and ``chunks`` lacks
        # ``qualified_name`` — both used to raise raw ValueError /
        # KeyError.
        path.write_text(
            '{"dim": "not-an-int", "chunks": [{"chunk_id": "a"}]}'
        )
        with pytest.raises(CorruptedChunkStoreError, match="atlas index"):
            await InMemoryChunkStore.load_from_path(path)

        # Missing ``chunks`` key entirely.
        path.write_text('{"dim": 8}')
        with pytest.raises(CorruptedChunkStoreError, match="atlas index"):
            await InMemoryChunkStore.load_from_path(path)

        # Unknown SymbolKind — the str-to-enum coercion used to raise
        # raw ValueError. Now wrapped.
        path.write_text(
            '{"dim": 4, "chunks": [{"chunk_id": "a", "qualified_name": "m.a", '
            '"file_path": "f.py", "lineno_start": 1, "lineno_end": 2, '
            '"kind": "alien_kind", "text": "x", "vector": [0,0,0,0]}]}'
        )
        with pytest.raises(CorruptedChunkStoreError, match="atlas index"):
            await InMemoryChunkStore.load_from_path(path)

    async def test_load_succeeds_on_valid_round_trip(self, tmp_path: Any) -> None:
        """Sanity: the typed-error wrapper doesn't break the happy path."""
        store = InMemoryChunkStore()
        await store.setup(dim=4)
        await store.upsert_chunks(
            [_chunk("m.fn_0", 0)],
            np.eye(1, 4, dtype=np.float32),
        )
        path = tmp_path / "chunks.json"
        await store.save_to_path(path)

        # Round-trip cleanly — no CorruptedChunkStoreError.
        loaded = await InMemoryChunkStore.load_from_path(path)
        row = await loaded.fetch_by_qualified_name("m.fn_0")
        assert row is not None
        assert row.qualified_name == "m.fn_0"

    async def test_save_to_path_handles_concurrent_writers(self, tmp_path: Any) -> None:
        """Two concurrent ``save_to_path`` calls both succeed; final file is intact.

        With the old non-atomic write, two concurrent writers could
        interleave bytes and produce malformed JSON. Under the
        rename-into-place implementation each writer renames its own
        tempfile; the last rename wins and the file is one of the two
        valid snapshots — never a torn mix.
        """
        # Two distinct stores so the two writes have meaningfully
        # different payloads — lets us assert the final file matches
        # exactly one of them, not a chimera.
        store_a = InMemoryChunkStore()
        await store_a.setup(dim=4)
        await store_a.upsert_chunks(
            [_chunk("m.a", 0)],
            np.eye(1, 4, dtype=np.float32),
        )
        store_b = InMemoryChunkStore()
        await store_b.setup(dim=4)
        await store_b.upsert_chunks(
            [_chunk("m.b", 1)],
            np.eye(1, 4, dtype=np.float32),
        )

        path = tmp_path / "chunks.json"
        await asyncio.gather(
            store_a.save_to_path(path),
            store_b.save_to_path(path),
        )

        # Final file is parseable JSON (no torn write), and the chunk
        # list matches one of the two writers exactly.
        import json as _json  # noqa: PLC0415

        payload = _json.loads(path.read_text())
        qnames = [c["qualified_name"] for c in payload["chunks"]]
        assert qnames in (["m.a"], ["m.b"]), qnames
        # No tempfiles linger.
        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert siblings == ["chunks.json"], siblings
