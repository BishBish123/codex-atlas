# ADR-001: pgvector over FAISS

## Status

Accepted.

## Context

Codex-Atlas needs a vector index that supports HNSW-style approximate
nearest-neighbour search over function-grain embeddings (10K–50K rows
for a typical Python repo, up to ~500K for a monorepo). The two
realistic options at the time of writing are:

- **pgvector** on Postgres 17 (HNSW + IVF-flat indexes, SQL surface)
- **FAISS** (Facebook AI Similarity Search, CPU/GPU, file-based index)

## Decision

We use pgvector on Postgres 17.

## Consequences

### Why pgvector won

1. **Operational story.** Reviewers, recruiters, and most production
   teams already run Postgres. Adding a vector column to an existing
   database is one extension install; standing up a dedicated FAISS
   service introduces a second moving part.
2. **Idempotent upsert.** pgvector inherits Postgres's `INSERT ... ON
   CONFLICT DO UPDATE`, which lets us re-index a corpus without
   reasoning about deletion semantics. FAISS indexes are append-only;
   re-indexing means rebuilding from scratch or maintaining a parallel
   tombstone set.
3. **Joins.** Once embeddings live next to per-chunk metadata in the
   same row, the structural retriever can `JOIN` on `qualified_name`
   to pull a chunk by name without a second round-trip. With FAISS we
   would carry that metadata in a separate process or sidecar store.
4. **Portability.** Postgres is the lowest-common-denominator
   datastore for any production deployment; the same DSN that runs CI
   runs prod.

### Why FAISS lost

1. **Two stores to operate.** FAISS handles vectors well but you still
   need a metadata store next to it. That's two backups, two backups
   to test, two failure modes.
2. **Index rebuild cost.** FAISS HNSW indexes are not designed for
   in-place updates. The codebase being indexed is not static — every
   commit invalidates a few chunks — so we want incremental upsert.
3. **No SQL.** A senior reviewer reaches for SQL when they want to
   debug "why did `m.foo` come back at rank 3?". With FAISS that's a
   custom Python script.

### What we give up

- **Speed at the very large end.** FAISS GPU indexes outperform
  pgvector on 10M+ vector workloads. Codex-Atlas is built for single
  codebases, where 50K is the realistic upper bound — we are deep
  inside the regime where pgvector is plenty fast.
- **In-process embedding.** pgvector requires a network round-trip per
  query. With FAISS we could embed and search in one process. We
  consider this an acceptable cost for the operational simplicity.

## Revisit

If we ever index multiple codebases in one deployment and the row
count crosses ~5M, re-evaluate. Until then, pgvector is the cleaner
choice.
