"""Bigtool pgvector store for dynamic tool retrieval.

Uses the existing PostgreSQL + pgvector + Ollama (mxbai-embed-large) stack
to provide semantic tool search for agents on non-Anthropic providers
(Bedrock, OpenAI, etc.) that don't support native defer_loading.

The store is populated once at startup with tool descriptions from TOOL_REGISTRY.
Agents query it via langgraph-bigtool's retrieve_tools mechanism.
"""

import atexit
import logging
import os

from langgraph.store.postgres import PostgresStore

from quantstack.rag.embeddings import EMBEDDING_DIMENSION, OllamaEmbeddingFunction

logger = logging.getLogger(__name__)

TOOL_NAMESPACE = ("tools",)

_store: PostgresStore | None = None
_conn = None
_store_initialized = False


def get_tool_store() -> PostgresStore | None:
    """Get or create the singleton tool store.

    Returns None if PostgreSQL or Ollama is unavailable (graceful degradation).
    """
    global _store, _conn, _store_initialized
    if _store_initialized:
        return _store
    _store_initialized = True

    # Prefer a dedicated vector-capable postgres (Docker ankane/pgvector image)
    # over the operational DB, which may not have the vector extension.
    pg_url = os.environ.get(
        "BIGTOOL_PG_URL",
        os.environ.get("TRADER_PG_URL", "postgresql://localhost/quantstack"),
    )

    try:
        embed_fn = OllamaEmbeddingFunction()
        # Verify ollama is reachable
        embed_fn(["test"])
    except Exception as exc:
        logger.warning("Ollama embeddings unavailable — bigtool store disabled: %s", exc)
        return None

    try:
        import psycopg
        from psycopg.rows import dict_row

        _conn = psycopg.connect(
            pg_url, autocommit=True, prepare_threshold=0, row_factory=dict_row,
        )
        atexit.register(_cleanup)

        _store = PostgresStore(
            conn=_conn,
            index={
                "embed": embed_fn,
                "dims": EMBEDDING_DIMENSION,
                "fields": ["description"],
            },
        )
        _store.setup()
        logger.info("Bigtool pgvector store initialized (dims=%d)", EMBEDDING_DIMENSION)
        return _store
    except Exception as exc:
        logger.warning("Failed to initialize bigtool store: %s", exc)
        _store = None
        return None


def _cleanup():
    """Close the persistent connection at process exit."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception as exc:
            logger.debug("bigtool_store connection close failed: %s", exc)
        _conn = None


def populate_tool_store(store: PostgresStore) -> int:
    """Populate the tool store with descriptions from TOOL_REGISTRY.

    Idempotent — uses batch() with PutOps so all embeddings are computed
    in a single ollama call instead of one-per-tool.
    Returns the number of tools stored.
    """
    from langgraph.store.base import PutOp

    from quantstack.tools.registry import TOOL_REGISTRY

    ops = [
        PutOp(
            TOOL_NAMESPACE,
            name,
            {"description": f"{tool.name}: {tool.description}"},
        )
        for name, tool in TOOL_REGISTRY.items()
    ]

    if ops:
        store.batch(ops)

    logger.info("Populated bigtool store with %d tool descriptions", len(ops))
    return len(ops)


def ensure_tool_store_populated() -> PostgresStore | None:
    """Get the store, populating it if needed. Returns None on failure."""
    store = get_tool_store()
    if store is None:
        return None

    # Check if already populated by looking for any tool
    try:
        results = store.search(TOOL_NAMESPACE, query="market data", limit=1)
        if not results:
            populate_tool_store(store)
    except Exception:
        try:
            populate_tool_store(store)
        except Exception as exc:
            logger.warning("Failed to populate bigtool store: %s", exc)
            return None

    return store
