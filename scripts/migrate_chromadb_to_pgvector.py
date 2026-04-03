#!/usr/bin/env python3
"""One-time migration: ChromaDB → pgvector.

Reads all documents + embeddings from ChromaDB collections and inserts them
into the pgvector embeddings table. Idempotent via ON CONFLICT DO NOTHING.

Usage:
    python scripts/migrate_chromadb_to_pgvector.py

Env vars:
    CHROMADB_HOST  (default: chromadb)
    CHROMADB_PORT  (default: 8000)
    TRADER_PG_URL  (default: postgresql://quantstack:quantstack@localhost:5432/quantstack)
"""

import logging
import os
import sys

import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTIONS = ("trade_outcomes", "strategy_knowledge", "market_research")


def main():
    # --- Connect to ChromaDB ---
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb package not installed — cannot migrate. pip install chromadb")
        sys.exit(1)

    host = os.environ.get("CHROMADB_HOST", "chromadb")
    port = int(os.environ.get("CHROMADB_PORT", "8000"))
    logger.info("Connecting to ChromaDB at %s:%d", host, port)
    chroma_client = chromadb.HttpClient(host=host, port=port)

    # --- Connect to PostgreSQL ---
    pg_url = os.environ.get(
        "TRADER_PG_URL", "postgresql://quantstack:quantstack@localhost:5432/quantstack"
    )
    logger.info("Connecting to PostgreSQL")
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True

    # Ensure schema
    from quantstack.rag.query import ensure_schema
    ensure_schema(conn)

    # --- Migrate each collection ---
    report: dict[str, dict] = {}

    for col_name in COLLECTIONS:
        logger.info("Migrating collection: %s", col_name)
        try:
            col = chroma_client.get_collection(name=col_name)
        except Exception:
            logger.warning("Collection %s not found in ChromaDB, skipping", col_name)
            report[col_name] = {"chromadb_count": 0, "migrated": 0, "skipped": 0}
            continue

        result = col.get(include=["embeddings", "metadatas", "documents"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        embeddings = result.get("embeddings", [])
        metadatas = result.get("metadatas", [])

        if not ids:
            logger.info("  %s: empty, nothing to migrate", col_name)
            report[col_name] = {"chromadb_count": 0, "migrated": 0, "skipped": 0}
            continue

        # Check dimension consistency
        dims = set(len(e) for e in embeddings if e)
        if len(dims) > 1:
            logger.warning("  %s: MIXED DIMENSIONS detected: %s", col_name, dims)

        migrated = 0
        skipped = 0
        cur = conn.cursor()

        for i, doc_id in enumerate(ids):
            doc = documents[i] if documents else ""
            emb = embeddings[i] if embeddings else None
            meta = metadatas[i] if metadatas else {}

            if not emb:
                skipped += 1
                continue

            try:
                cur.execute(
                    """INSERT INTO embeddings (id, collection, content, embedding, metadata)
                       VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                       ON CONFLICT (id) DO NOTHING""",
                    [doc_id, col_name, doc, str(emb), psycopg2.extras.Json(meta or {})],
                )
                migrated += 1
            except Exception as e:
                logger.warning("  Failed to insert %s: %s", doc_id, e)
                skipped += 1

        cur.close()
        report[col_name] = {
            "chromadb_count": len(ids),
            "dimensions": dims,
            "migrated": migrated,
            "skipped": skipped,
        }
        logger.info("  %s: %d/%d migrated, %d skipped", col_name, migrated, len(ids), skipped)

    # --- Print report ---
    print("\n=== Migration Report ===")
    for col_name, stats in report.items():
        print(f"  {col_name}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    # Verify counts
    cur = conn.cursor()
    print("\n=== pgvector Verification ===")
    for col_name in COLLECTIONS:
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE collection = %s", [col_name])
        count = cur.fetchone()[0]
        print(f"  {col_name}: {count} rows")
    cur.close()

    conn.close()
    logger.info("Migration complete")


if __name__ == "__main__":
    main()
