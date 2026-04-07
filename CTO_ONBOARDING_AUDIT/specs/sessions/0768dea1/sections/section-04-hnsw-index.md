# Section 04: Add HNSW Index on Embeddings (Item 0.4)

## Background

The `embeddings` table in PostgreSQL (defined in `src/quantstack/rag/query.py` lines 20-33) stores vector embeddings with dimension 1024. It has B-tree and GIN indexes on `collection` and `metadata->ticker`, but no vector index on the `embedding` column. Every call to `search_similar()` in `rag/query.py` performs a full sequential scan using the cosine distance operator (`embedding <=> %s::vector`). At 500+ rows this degrades from sub-10ms to 100ms+. An HNSW (Hierarchical Navigable Small World) index provides approximate nearest neighbor search in logarithmic time.

This section adds a single migration function to `src/quantstack/db.py` that creates the HNSW index. The index is idempotent (`CREATE INDEX IF NOT EXISTS`), consistent with every other migration in the file.

## Dependencies

- None. This section can be implemented in isolation.
- **Blocks:** section-05-rag-fix (the RAG fix benefits from this index being present).

## Tests First

**Test file:** `tests/unit/test_hnsw_migration.py`

Four tests, all following the existing pytest + psycopg2 conventions in the project:

```python
"""Tests for HNSW vector index migration."""

# Test: migration creates HNSW index on fresh database
# Setup: Use a test DB connection with the embeddings table created (via rag/query.py _INIT_SQL)
#        but no HNSW index present. Call _migrate_hnsw_index_pg(conn).
# Assert: Query pg_indexes where tablename='embeddings' and indexname='idx_embeddings_hnsw'.
#         Exactly one row returned.

# Test: migration is idempotent — running twice does not error
# Setup: Call _migrate_hnsw_index_pg(conn) twice on the same database.
# Assert: No exception on the second call. Index still exists (one row in pg_indexes).

# Test: index uses the correct operator class (vector_cosine_ops)
# Setup: Run migration, then query pg_am and pg_opclass for the index.
# Assert: The index's access method is 'hnsw' and operator class is vector_cosine_ops.
#         Query: SELECT am.amname, opc.opcname
#                FROM pg_index idx
#                JOIN pg_class cl ON cl.oid = idx.indexrelid
#                JOIN pg_am am ON am.oid = cl.relam
#                JOIN pg_opclass opc ON opc.oid = idx.indclass[0]
#                WHERE cl.relname = 'idx_embeddings_hnsw'

# Test: index has correct storage parameters (m=16, ef_construction=100)
# Setup: Run migration, then query pg_class for reloptions.
# Assert: reloptions contains 'm=16' and 'ef_construction=100'.
#         Query: SELECT reloptions FROM pg_class WHERE relname = 'idx_embeddings_hnsw'
```

All four tests require a live PostgreSQL instance with the `vector` extension. If the project's test infrastructure uses a shared test database (check `tests/conftest.py` or `tests/unit/conftest.py` for DB fixtures), use that. If not, these may need to run as integration tests against a Docker PostgreSQL with pgvector. The `CREATE EXTENSION IF NOT EXISTS vector` must execute before the embeddings table is created.

## Implementation

### File: `src/quantstack/db.py`

Two changes are needed in this file:

**Change 1 — Add the migration function.** Place it after the last existing migration function (`_migrate_ewf_pg`, which ends around line 2383), before the `run_migrations()` wrapper at line 2385.

Function signature and body:

```python
def _migrate_hnsw_index_pg(conn: PgConnection) -> None:
    """Add HNSW vector index on embeddings.embedding for cosine similarity search.

    Without this index, every semantic search does a full sequential scan.
    HNSW provides approximate nearest neighbor search in O(log n) time.

    Parameters: m=16 (default, sufficient for <10k rows),
    ef_construction=100 (above default 64 for better recall at negligible build cost).
    Operator class vector_cosine_ops matches the <=> operator used in rag/query.py.
    """
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
            ON embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 100)
    """)
    logger.debug("[DB] HNSW index on embeddings.embedding migrated")
```

**Change 2 — Register the migration in `run_migrations_pg()`.** Add `_migrate_hnsw_index_pg(conn)` to the migration call list inside `run_migrations_pg()`, after `_migrate_ewf_pg(conn)` (line 536). This is the last position in the chain, which is correct because the `embeddings` table is created by `rag/query.py`'s `_INIT_SQL` (not by db.py migrations), so ordering within db.py doesn't matter for table existence — but placing it last keeps the pattern of appending new migrations at the end.

The relevant block in `run_migrations_pg()` currently ends:

```python
            _migrate_institutional_gaps_pg(conn)
            _migrate_ewf_pg(conn)

            logger.info("[DB] PostgreSQL migrations complete")
```

After the change:

```python
            _migrate_institutional_gaps_pg(conn)
            _migrate_ewf_pg(conn)
            _migrate_hnsw_index_pg(conn)

            logger.info("[DB] PostgreSQL migrations complete")
```

### No changes to `src/quantstack/rag/query.py`

The existing B-tree and GIN indexes in `rag/query.py`'s `_INIT_SQL` remain untouched. The HNSW index is added via `db.py` because:

1. It is a migration (runs once on upgrade), not a table definition.
2. All other schema migrations are centralized in `db.py`.
3. Adding it to `_INIT_SQL` would run the `CREATE INDEX IF NOT EXISTS` check on every process start — harmless but unnecessary overhead, and inconsistent with where all other index additions go.

## Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `m` | 16 | Default. Sufficient for tables under 10k rows. Higher values increase memory per node with negligible recall improvement at this scale. |
| `ef_construction` | 100 | Above default (64). Cheap insurance — the index build is sub-second at current table size, but recall improves measurably. |
| `vector_cosine_ops` | — | Matches the `<=>` cosine distance operator used in `rag/query.py` line 122 (`ORDER BY embedding <=> %s::vector`). Using the wrong operator class would make the index invisible to the query planner. |

## Considerations

- **Build time:** Sub-second for the current scale (<10k rows). No need to tune `maintenance_work_mem` or run the build in a background session.
- **`ef_search` tuning:** The default `hnsw.ef_search` is 40. Setting it to 100 at connection init would improve recall at negligible latency cost for this table size. This is an optional follow-up — it does not block this section. If pursued, the right place is in `rag/query.py`'s `_get_connection()` function: `cur.execute("SET hnsw.ef_search = 100")`.
- **pgvector extension:** The `CREATE EXTENSION IF NOT EXISTS vector` statement already runs in `rag/query.py`'s `_INIT_SQL`. The HNSW index depends on this extension being present. If `db.py` migrations run before any RAG code initializes, the `CREATE INDEX` will fail because the extension doesn't exist yet. Mitigation: add `CREATE EXTENSION IF NOT EXISTS vector` as the first statement in `_migrate_hnsw_index_pg` before the `CREATE INDEX`. This is idempotent and avoids ordering fragility.
- **Future scaling:** At 100k+ rows the current parameters remain adequate. Only at 1M+ embeddings would increasing `m` to 24 or switching to IVFFlat be worth evaluating.
- **Table existence:** The `embeddings` table is created by `rag/query.py`, not by `db.py` migrations. If `_migrate_hnsw_index_pg` runs before the table exists, the `CREATE INDEX` will fail. The `IF NOT EXISTS` clause does not help here — it prevents duplicate indexes, not missing tables. Mitigation: wrap the index creation in a guard that checks for the table first, or accept that the migration will be a no-op error on first startup (before any embedding is written). The pragmatic approach: add a `CREATE TABLE IF NOT EXISTS` for the embeddings table before the index, or simply let the error be caught by the existing exception handler in `run_migrations_pg()` and log a warning. Given that the embeddings table is created early in the system lifecycle (RAG init runs at import time), this edge case is unlikely in practice.

## Verification

After deploying, confirm the index exists:

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'embeddings' AND indexname = 'idx_embeddings_hnsw';
```

Expected: one row with `indexdef` containing `USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100)`.

To verify the query planner uses the index:

```sql
EXPLAIN ANALYZE
SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM embeddings
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

The plan should show `Index Scan using idx_embeddings_hnsw` rather than `Seq Scan`.
