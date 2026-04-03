# Section 09: RAG Pipeline Migration (ChromaDB to pgvector)

## Background

QuantStack's RAG pipeline currently uses ChromaDB as a separate Docker service for vector similarity search across three collections: `trade_outcomes`, `strategy_knowledge`, and `market_research`. The embedding model is Ollama's `mxbai-embed-large`. The migration replaces ChromaDB with pgvector, an extension on the existing PostgreSQL database. This eliminates a separate service (container, network hop, failure domain) and gives transactional consistency with application data.

The affected files are:

- `src/quantstack/rag/embeddings.py` -- ChromaDB-compatible `OllamaEmbeddingFunction` class
- `src/quantstack/rag/query.py` -- `search_knowledge_base()`, `remember_knowledge()`, `get_chromadb_client()` singleton
- `src/quantstack/rag/ingest.py` -- `ingest_memory_files()`, `chunk_markdown()`, `file_to_collection()`
- `src/quantstack/rag/migrate_memory.py` -- one-time ChromaDB ingestion script
- `src/quantstack/rag/__init__.py` -- re-exports
- `src/quantstack/tools/knowledge_tools.py` -- CrewAI BaseTool wrappers for knowledge store (separate concern, migrated in section-05)

The embedding model (`mxbai-embed-large` via Ollama) does not change. Only the storage backend changes.

## Dependencies

- **section-01-scaffolding**: `pgvector>=0.3.0` and `psycopg[binary]>=3.1.0` / `psycopg-pool>=3.1.0` must be in `pyproject.toml`. The PostgreSQL Docker image must support pgvector (see section-12 for `pgvector/pgvector:pg16`).

## Tests First

All tests go in `tests/unit/test_rag_pipeline.py`, replacing the existing ChromaDB-based tests. The existing test file uses an in-memory ChromaDB client and a `FakeEmbeddingFunction`. The new tests use a test PostgreSQL database with pgvector enabled.

For unit tests that should not require a live database, mock `psycopg` at the connection level. For integration tests that verify actual pgvector behavior (similarity ordering, filtering), use a test database fixture.

```python
# tests/unit/test_rag_pipeline.py

"""Tests for RAG Pipeline migration (ChromaDB -> pgvector).

Tests are written BEFORE implementation per TDD convention.
"""

import pytest


# ---------------------------------------------------------------------------
# pgvector extension availability
# ---------------------------------------------------------------------------

# Test: pgvector extension is installed in test database
# Connect to test DB, run "SELECT * FROM pg_extension WHERE extname = 'vector'",
# assert one row returned. This is a gate — if this fails, nothing else works.


# ---------------------------------------------------------------------------
# Embeddings table schema
# ---------------------------------------------------------------------------

# Test: embeddings table created with correct schema
#   Columns: id (text PK), collection (text), content (text),
#            embedding (vector), metadata (jsonb), created_at (timestamptz)
# Verify by querying information_schema.columns for the embeddings table.


# ---------------------------------------------------------------------------
# store_embedding()
# ---------------------------------------------------------------------------

# Test: store_embedding() inserts record with correct vector dimension
#   Call store_embedding() with a known vector (e.g., 10-dimensional float list).
#   Query the row back. Assert the vector dimension matches.
#   Assert content, collection, and metadata are stored correctly.


# ---------------------------------------------------------------------------
# search_similar()
# ---------------------------------------------------------------------------

# Test: search_similar() returns top-N results ordered by cosine similarity
#   Insert 3 embeddings with known vectors. Query with a vector close to one of them.
#   Assert the closest vector is returned first. Assert len(results) <= n_results.

# Test: search_similar() filters by collection
#   Insert embeddings into "strategy_knowledge" and "trade_outcomes".
#   Search with collection="strategy_knowledge".
#   Assert all results have collection == "strategy_knowledge".

# Test: search_similar() filters by metadata (e.g., ticker)
#   Insert two embeddings, one with metadata {"ticker": "AAPL"}, one with {"ticker": "SPY"}.
#   Search with metadata filter ticker="AAPL".
#   Assert only the AAPL result is returned.


# ---------------------------------------------------------------------------
# delete_collection()
# ---------------------------------------------------------------------------

# Test: delete_collection() removes all records for a collection
#   Insert 3 records into "trade_outcomes". Call delete_collection("trade_outcomes").
#   Assert count of records with collection="trade_outcomes" is 0.
#   Assert records in other collections are untouched.


# ---------------------------------------------------------------------------
# No chromadb imports
# ---------------------------------------------------------------------------

# Test: no code imports chromadb (after migration)
#   Scan all .py files under src/quantstack/rag/ for "import chromadb" or
#   "from chromadb". Assert zero matches. This is a static check, not a runtime test.


# ---------------------------------------------------------------------------
# Migration script
# ---------------------------------------------------------------------------

# Test: migration script exports and imports embeddings with matching counts
#   Seed a mock ChromaDB with known documents across 3 collections.
#   Run migration function. Assert pgvector row counts match per collection.

# Test: migration script verifies vector dimensions are consistent
#   Seed ChromaDB with vectors of dimension 1024 in one collection and
#   dimension 768 in another. Migration should detect and report this.

# Test: migration script sample similarity comparison produces equivalent results
#   Seed ChromaDB with 10 documents. Run migration. Pick 3 query vectors.
#   Run similarity search on both ChromaDB and pgvector. Assert top-3 results
#   contain the same document IDs (order may differ slightly due to float precision).


# ---------------------------------------------------------------------------
# Preserved public API
# ---------------------------------------------------------------------------

# Test: search_knowledge_base() still works with same signature
#   search_knowledge_base(query, collection, ticker, n_results) returns
#   list[dict] with keys "text", "metadata", "distance", "collection".

# Test: remember_knowledge() still works with same signature
#   remember_knowledge(text, collection, metadata) returns a document ID string.

# Test: remember_knowledge() rejects unknown collection names
#   Raises ValueError for collection not in COLLECTIONS.

# Test: chunk_markdown() is unchanged (pure function, no backend dependency)
#   Existing tests for chunk_markdown pass without modification.

# Test: file_to_collection() is unchanged (pure function)
#   Existing tests for file_to_collection pass without modification.
```

## Implementation Details

### 1. Database Setup -- pgvector Extension and Table

The `quantstack` PostgreSQL database needs the `vector` extension enabled and an `embeddings` table created. This should be done via a migration script or an init function called at application startup.

**SQL schema:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector NOT NULL,       -- dimension set dynamically or fixed (e.g., 1024 for mxbai-embed-large)
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings (collection);
CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_ticker ON embeddings USING GIN ((metadata -> 'ticker'));
```

The vector dimension for `mxbai-embed-large` is 1024. Use `vector(1024)` if you want strict enforcement, or plain `vector` for flexibility. Strict is safer -- it catches dimension mismatches at insert time rather than at query time.

Add an IVFFlat or HNSW index for similarity search performance once the table exceeds ~10k rows:

```sql
-- Add after data is loaded (IVFFlat requires data to build lists)
CREATE INDEX IF NOT EXISTS idx_embeddings_cosine
    ON embeddings USING hnsw (embedding vector_cosine_ops);
```

### 2. Refactor `src/quantstack/rag/embeddings.py`

The `OllamaEmbeddingFunction` class stays largely the same -- it calls Ollama to get vectors. However, it no longer needs to conform to ChromaDB's embedding function interface. Simplify:

- Keep the `__call__(self, input: list[str]) -> list[list[float]]` method (still useful).
- Remove the `name()` method (ChromaDB-specific, not needed by pgvector).
- Update the error message to remove the reference to `pip install 'quantstack[crewai]'` -- change to `pip install 'quantstack[langgraph]'` or just `pip install ollama`.

### 3. Refactor `src/quantstack/rag/query.py`

This is the core of the migration. Replace ChromaDB client with a psycopg connection pool.

**Current public API to preserve:**

```python
COLLECTIONS = ("trade_outcomes", "strategy_knowledge", "market_research")

def search_knowledge_base(
    query: str,
    collection: str | None = None,
    ticker: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Returns list of dicts with 'text', 'metadata', 'distance', 'collection' keys."""

def remember_knowledge(
    text: str,
    collection: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Returns document ID string."""
```

Remove the `client` and `embedding_fn` test-injection parameters. For testing, inject the database connection instead (or use a test database).

**New internal functions:**

- `_get_pool() -> psycopg_pool.AsyncConnectionPool` -- lazy-initialized async connection pool singleton. Reads `TRADER_PG_URL` env var. Bounded pool size (min=1, max=5 is reasonable for RAG workload).
- `store_embedding(id, collection, content, embedding, metadata)` -- INSERT into embeddings table using pgvector's vector type.
- `search_similar(query_embedding, collection, n_results, metadata_filter)` -- SELECT with `ORDER BY embedding <=> query_vector LIMIT n`. The `<=>` operator is cosine distance in pgvector.
- `delete_collection(collection)` -- DELETE FROM embeddings WHERE collection = $1.

**`search_knowledge_base()` refactored flow:**

1. Embed the query text using `OllamaEmbeddingFunction`.
2. Call `search_similar()` with the embedding vector, collection filter, and ticker metadata filter.
3. Format results into the same `list[dict]` shape as before.

**`remember_knowledge()` refactored flow:**

1. Validate collection name against `COLLECTIONS`.
2. Generate document ID (same `{collection}::{timestamp}::{hash}` pattern).
3. Embed the text using `OllamaEmbeddingFunction`.
4. Call `store_embedding()`.
5. Return the document ID.

**Key difference from ChromaDB:** ChromaDB handled embedding internally (you passed text, it called the embedding function). With pgvector, the application must call the embedding function explicitly before storing or querying. This is actually cleaner -- no hidden side effects.

**Connection management:** Use `psycopg_pool.AsyncConnectionPool` as a context manager in each function. The pool is initialized once and shared across calls. Ensure `pool.close()` is called on shutdown (wire into `GracefulShutdown`).

**Synchronous vs async:** The current `search_knowledge_base()` and `remember_knowledge()` are synchronous. They can remain synchronous for now (psycopg3 supports both sync and async). If the callers are async graph nodes (section-07), consider making async variants. A pragmatic approach: keep the sync API, add `async_search_knowledge_base()` and `async_remember_knowledge()` that use the async pool. The sync versions can use `psycopg.Connection` (not the async pool) for backward compatibility.

### 4. Refactor `src/quantstack/rag/ingest.py`

`chunk_markdown()` and `file_to_collection()` are pure functions with no ChromaDB dependency -- they stay unchanged.

`ingest_memory_files()` needs to change its storage backend:

- Replace `chromadb_client` parameter with a database connection (or use the module-level pool).
- Replace `collection.upsert()` calls with `store_embedding()` calls.
- The idempotency check changes from `col.count() > 0` to `SELECT COUNT(*) FROM embeddings WHERE collection = $1`.

**Signature change:**

```python
def ingest_memory_files(
    memory_dir: str,
    embedding_fn: OllamaEmbeddingFunction | None = None,
) -> dict[str, int]:
    """Ingest markdown memory files into pgvector embeddings table.

    Idempotent: skips if collections already have documents.
    Uses module-level connection pool.
    """
```

### 5. Refactor `src/quantstack/rag/migrate_memory.py`

Same changes as `ingest.py` -- replace ChromaDB client usage with pgvector storage. The `route_file()` and `_chunk_text()` functions are pure and unchanged.

The `migrate_memory()` function signature changes:

```python
def migrate_memory(
    memory_dir: Path,
    embedding_fn,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Ingest .claude/memory/ markdown files into pgvector embeddings table."""
```

The `__main__` block at the bottom should be updated to use the pgvector connection instead of `chromadb.HttpClient`.

### 6. Migration Script: `scripts/migrate_chromadb_to_pgvector.py`

One-time script for existing deployments that have data in ChromaDB. New deployments skip this.

**Steps:**

1. Connect to ChromaDB (HttpClient, same config as current code).
2. Connect to PostgreSQL (psycopg, read `TRADER_PG_URL`).
3. Ensure pgvector extension and embeddings table exist.
4. For each collection in `COLLECTIONS`:
   a. Fetch all documents, embeddings, and metadata from ChromaDB via `collection.get(include=["embeddings", "metadatas", "documents"])`.
   b. Verify all embedding vectors have the same dimension. Log a warning if not.
   c. Batch-insert into pgvector using `executemany` or `COPY` for performance.
   d. Verify row count matches.
5. Run sample similarity comparison:
   a. Pick 5 random documents as queries.
   b. Run similarity search on both ChromaDB and pgvector.
   c. Compare top-5 result IDs. Log agreement percentage.
6. Print migration report: per-collection counts, dimensions, similarity agreement.

**Important:** This script should be idempotent. Use `INSERT ... ON CONFLICT (id) DO NOTHING` to handle re-runs.

### 7. Update `src/quantstack/rag/__init__.py`

Current exports:

```python
from quantstack.rag.query import search_knowledge_base, remember_knowledge, COLLECTIONS
from quantstack.rag.ingest import ingest_memory_files, chunk_markdown
```

These stay the same. The public API does not change. Internal implementation details (ChromaDB vs pgvector) are encapsulated.

### 8. Update `src/quantstack/tools/knowledge_tools.py`

This file currently imports `from quantstack.crewai_compat import BaseTool`. That import is part of the section-05 tool layer migration, not this section. However, the RAG-related tools in this file (`search_knowledge_base`, `remember_knowledge` if called from here) will transparently pick up the pgvector backend once `query.py` is refactored -- no changes needed in the tool layer for the backend swap.

### 9. Cleanup

After migration is verified:

- Remove `chromadb` from `pyproject.toml` dependencies (section-01).
- Remove `chromadb` Docker service from `docker-compose.yml` (section-12).
- Remove the `CHROMADB_HOST` and `CHROMADB_PORT` environment variable references from `query.py`.
- Delete or archive the old `migrate_memory.py` `__main__` block that uses `chromadb.HttpClient`.

## File Summary

| File | Action |
|------|--------|
| `src/quantstack/rag/embeddings.py` | Simplify: remove `name()` method, update error message |
| `src/quantstack/rag/query.py` | Rewrite: replace ChromaDB with psycopg + pgvector. Preserve public API. |
| `src/quantstack/rag/ingest.py` | Refactor: replace ChromaDB upsert with pgvector insert. Pure functions unchanged. |
| `src/quantstack/rag/migrate_memory.py` | Refactor: replace ChromaDB client with pgvector connection. |
| `src/quantstack/rag/__init__.py` | No change (exports stay the same). |
| `scripts/migrate_chromadb_to_pgvector.py` | New file: one-time migration script. |
| `tests/unit/test_rag_pipeline.py` | Rewrite: replace ChromaDB fixtures with pgvector test database fixtures. |

## Risks and Mitigations

**Risk: Ollama unavailable at embedding time.** The current code already handles this (catches `ConnectionError`). The pgvector backend inherits the same risk. Mitigation: the `search_knowledge_base()` function should return empty results (not crash) when embedding fails, same as the current ChromaDB fallback behavior.

**Risk: pgvector extension not installed.** If the PostgreSQL image does not include pgvector, `CREATE EXTENSION vector` fails. Mitigation: section-12 switches the Docker image to `pgvector/pgvector:pg16`. For local dev, developers must install pgvector manually or use the Docker image.

**Risk: Embedding dimension mismatch.** If the embedding model changes (different dimension), existing vectors become incompatible. Mitigation: store the model name in metadata. The migration script verifies dimension consistency. Consider adding a dimension check on insert.

**Risk: Connection pool exhaustion.** The RAG module shares the PostgreSQL database with the rest of the application. Mitigation: use a small bounded pool (max=5) dedicated to RAG operations, separate from the main application pool.
