"""Chunk store package — public API.

All names that existed in the flat ``store.py`` module are re-exported here
so existing imports (``from codex_atlas.store import ChunkStore``, etc.) keep
working without changes.

``make_chunk_store()`` is the new factory added by this release.
"""

from __future__ import annotations

import json  # noqa: F401  (re-exported so tests can monkeypatch store_mod.json)
import os
from typing import TYPE_CHECKING, Union

import asyncpg  # noqa: F401  (re-exported so tests can monkeypatch store_mod.asyncpg)

from codex_atlas.store._core import (
    _KNOWN_SCHEMA_VERSIONS,
    _TABLE_IDENT_RE,
    ChunkStore,
    ChunkStoreProtocol,
    CorruptedChunkStoreError,
    InMemoryChunkStore,
    StoredChunk,
    _load_chunk_entry,
    _validate_schema_version,
)

if TYPE_CHECKING:
    from codex_atlas.store.pgvector import PgVectorChunkStore

__all__ = [
    "_KNOWN_SCHEMA_VERSIONS",
    "_TABLE_IDENT_RE",
    "ChunkStore",
    "ChunkStoreProtocol",
    "CorruptedChunkStoreError",
    "InMemoryChunkStore",
    "StoredChunk",
    "_load_chunk_entry",
    "_validate_schema_version",
    "make_chunk_store",
]

# ``make_chunk_store`` may return any of the three concrete store types.
# The Union annotation lets mypy type-check call sites without importing
# PgVectorChunkStore eagerly (which would create a hard dependency on
# asyncpg at module load time).
_AnyStore = Union[InMemoryChunkStore, ChunkStore, "PgVectorChunkStore"]


def make_chunk_store() -> _AnyStore:
    """Factory — instantiates the chunk store backend named by ``ATLAS_CHUNK_STORE``.

    Environment variables
    ---------------------
    ``ATLAS_CHUNK_STORE`` (default ``memory``)
        * ``memory``   -> ``InMemoryChunkStore``  (no DB; call ``setup()`` then
                          ``load_from_path()`` / ``upsert_chunks()`` as normal)
        * ``pgvector`` -> ``PgVectorChunkStore``  (asyncpg pool; reads ``ATLAS_PG_DSN``)

    ``ATLAS_PG_DSN``
        Required when ``ATLAS_CHUNK_STORE=pgvector``.

    Raises ``RuntimeError`` for unknown backend names or when
    ``ATLAS_PG_DSN`` is missing in pgvector mode.
    """
    backend = os.environ.get("ATLAS_CHUNK_STORE", "memory")
    if backend == "memory":
        return InMemoryChunkStore()
    if backend == "pgvector":
        from codex_atlas.store.pgvector import PgVectorChunkStore  # noqa: PLC0415

        dsn = os.environ.get("ATLAS_PG_DSN")
        if not dsn:
            raise RuntimeError(
                "ATLAS_CHUNK_STORE=pgvector requires ATLAS_PG_DSN to be set "
                "(e.g. postgresql://user:pass@localhost/mydb). "
                "Export ATLAS_PG_DSN or switch to ATLAS_CHUNK_STORE=memory."
            )
        return PgVectorChunkStore(dsn=dsn)
    raise RuntimeError(
        f"unknown ATLAS_CHUNK_STORE={backend!r}; "
        f"expected one of: memory, pgvector"
    )
