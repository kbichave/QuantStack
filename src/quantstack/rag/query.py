"""RAG retrieval and write functions backed by pgvector."""

import hashlib
import logging
import os
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

COLLECTIONS = ("trade_outcomes", "strategy_knowledge", "market_research")

# SQL for table and index creation (idempotent)
_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_collection
    ON embeddings (collection);

CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_ticker
    ON embeddings USING GIN ((metadata -> 'ticker'));
"""

_conn = None


def _get_connection():
    """Return a psycopg3 connection. Lazy-initialized singleton."""
    global _conn
    if _conn is not None and not _conn.closed:
        return _conn

    pg_url = os.environ.get(
        "TRADER_PG_URL", "postgresql://quantstack:quantstack@localhost:5432/quantstack"
    )
    _conn = psycopg.connect(pg_url, autocommit=True)
    return _conn


def reset_connection() -> None:
    """Reset the singleton connection (for testing)."""
    global _conn
    if _conn is not None and not _conn.closed:
        _conn.close()
    _conn = None


def ensure_schema(conn=None) -> None:
    """Create the embeddings table and indexes if they don't exist."""
    if conn is None:
        conn = _get_connection()
    cur = conn.cursor()
    cur.execute(_INIT_SQL)
    cur.close()


def store_embedding(
    doc_id: str,
    collection: str,
    content: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
    conn=None,
) -> None:
    """Insert or update an embedding record in pgvector."""
    if conn is None:
        conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO embeddings (id, collection, content, embedding, metadata)
           VALUES (%s, %s, %s, %s::vector, %s::jsonb)
           ON CONFLICT (id) DO UPDATE
           SET content = EXCLUDED.content,
               embedding = EXCLUDED.embedding,
               metadata = EXCLUDED.metadata""",
        [doc_id, collection, content, str(embedding), Jsonb(metadata or {})],
    )
    cur.close()


def search_similar(
    query_embedding: list[float],
    collection: str | None = None,
    n_results: int = 5,
    metadata_filter: dict[str, str] | None = None,
    conn=None,
) -> list[dict[str, Any]]:
    """Find the most similar embeddings using cosine distance."""
    if conn is None:
        conn = _get_connection()

    conditions = []
    params: list[Any] = [str(query_embedding)]

    if collection:
        conditions.append("collection = %s")
        params.append(collection)

    if metadata_filter:
        for key, value in metadata_filter.items():
            conditions.append(f"metadata->>'{key}' = %s")
            params.append(value)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(n_results)

    sql = f"""
        SELECT id, collection, content, metadata,
               embedding <=> %s::vector AS distance
        FROM embeddings
        {where_clause}
        ORDER BY distance
        LIMIT %s
    """

    cur = conn.cursor(row_factory=dict_row)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()

    return [
        {
            "text": row["content"],
            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else {},
            "distance": float(row["distance"]),
            "collection": row["collection"],
        }
        for row in rows
    ]


def delete_collection(collection: str, conn=None) -> int:
    """Delete all embeddings for a collection. Returns count deleted."""
    if conn is None:
        conn = _get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM embeddings WHERE collection = %s", [collection])
    count = cur.rowcount
    cur.close()
    return count


def search_knowledge_base(
    query: str,
    collection: str | None = None,
    ticker: str | None = None,
    n_results: int = 5,
    conn=None,
    embedding_fn=None,
) -> list[dict[str, Any]]:
    """Search pgvector for relevant knowledge.

    Args:
        query: Natural language search query.
        collection: Restrict to one collection. None searches all.
        ticker: Optional metadata filter.
        n_results: Maximum results to return.
        conn: Optional psycopg connection (for testing).
        embedding_fn: Optional embedding function (for testing).

    Returns:
        List of dicts with 'text', 'metadata', 'distance', 'collection' keys.
    """
    if embedding_fn is None:
        from quantstack.rag.embeddings import OllamaEmbeddingFunction
        embedding_fn = OllamaEmbeddingFunction()

    try:
        query_vector = embedding_fn([query])[0]
    except Exception:
        logger.warning("Embedding failed, returning empty results", exc_info=True)
        return []

    metadata_filter = {"ticker": ticker} if ticker else None
    collections_to_search = [collection] if collection else list(COLLECTIONS)

    all_results: list[dict[str, Any]] = []
    for col_name in collections_to_search:
        try:
            results = search_similar(
                query_vector, collection=col_name,
                n_results=n_results, metadata_filter=metadata_filter,
                conn=conn,
            )
            all_results.extend(results)
        except Exception:
            logger.warning("Failed to query collection %s", col_name, exc_info=True)

    all_results.sort(key=lambda x: x["distance"])
    return all_results[:n_results]


def remember_knowledge(
    text: str,
    collection: str,
    metadata: dict[str, Any] | None = None,
    conn=None,
    embedding_fn=None,
) -> str:
    """Write a new document to the knowledge base. Returns the document ID."""
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection '{collection}'. Must be one of {COLLECTIONS}")

    if embedding_fn is None:
        from quantstack.rag.embeddings import OllamaEmbeddingFunction
        embedding_fn = OllamaEmbeddingFunction()

    try:
        embedding = embedding_fn([text])[0]
    except Exception:
        logger.warning("Embedding failed, knowledge not persisted", exc_info=True)
        return ""

    text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    doc_id = f"{collection}::{int(time.time())}::{text_hash}"

    try:
        store_embedding(doc_id, collection, text, embedding, metadata, conn=conn)
    except Exception:
        logger.warning("Failed to store embedding", exc_info=True)
        return ""

    return doc_id
