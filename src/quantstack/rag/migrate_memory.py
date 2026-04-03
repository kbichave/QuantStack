"""Migration of .claude/memory/ markdown files into pgvector embeddings table.

Routes each file to the correct collection with appropriate metadata,
chunks the content, and upserts into pgvector with deterministic IDs
for idempotent operation.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

COLLECTION_ROUTING = {
    "strategy_registry": ("strategy_knowledge", "strategy_definition"),
    "workshop_lessons": ("strategy_knowledge", "negative_result"),
    "ml_experiment_log": ("strategy_knowledge", "ml_experiment"),
    "ml_model_registry": ("strategy_knowledge", "ml_model"),
    "ml_research_program": ("strategy_knowledge", "ml_research"),
    "lit_review_findings": ("strategy_knowledge", "literature_review"),
    "strategy_C_investment": ("strategy_knowledge", "strategy_definition"),
    "trade_journal": ("trade_outcomes", "trade_outcome"),
    "session_handoffs": ("market_research", "session_handoff"),
    "regime_history": ("market_research", "regime_history"),
    "agent_performance": ("market_research", "agent_performance"),
    "daily_plan": ("market_research", "daily_plan"),
}

PATTERN_ROUTING = [
    (re.compile(r"^risk_desk_report"), "market_research", "risk_report"),
    (re.compile(r"^research_log"), "market_research", "research_log"),
    (re.compile(r"^swing_research"), "market_research", "research_log"),
]


def route_file(file_path: Path, memory_dir: Path) -> tuple[str, dict[str, str]] | None:
    """Determine target collection and metadata for a memory file.

    Returns:
        Tuple of (collection_name, metadata_dict), or None if skipped.
    """
    relative = file_path.relative_to(memory_dir)
    parts = relative.parts

    if "templates" in parts:
        return None

    stem = file_path.stem
    if stem.endswith(".archive"):
        stem = stem.replace(".archive", "")

    # Ticker files
    if len(parts) >= 2 and parts[0] == "tickers":
        ticker = file_path.stem.upper()
        return "market_research", {
            "content_type": "ticker_research",
            "ticker": ticker,
            "source_file": str(relative),
        }

    # Session handoff subdirectory
    if len(parts) >= 2 and parts[0] == "session_handoffs":
        return "market_research", {
            "content_type": "session_handoff",
            "source_file": str(relative),
        }

    # Exact stem match
    if stem in COLLECTION_ROUTING:
        collection, content_type = COLLECTION_ROUTING[stem]
        return collection, {
            "content_type": content_type,
            "source_file": str(relative),
        }

    # Pattern match
    for pattern, collection, content_type in PATTERN_ROUTING:
        if pattern.match(stem):
            return collection, {
                "content_type": content_type,
                "source_file": str(relative),
            }

    # Default: market_research
    return "market_research", {
        "content_type": "general",
        "source_file": str(relative),
    }


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring markdown header boundaries."""
    if not text or not text.strip():
        return []

    separators = ["\n## ", "\n### ", "\n\n", "\n", " "]
    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining.strip())
            break

        split_pos = chunk_size
        for sep in separators:
            pos = remaining.rfind(sep, 0, chunk_size)
            if pos > chunk_size // 4:
                split_pos = pos + len(sep)
                break

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[max(split_pos - overlap, 0):]

    return [c for c in chunks if c]


def migrate_memory(
    memory_dir: Path,
    embedding_fn,
    *,
    force: bool = False,
    conn=None,
) -> dict[str, int]:
    """Ingest .claude/memory/ markdown files into pgvector embeddings table.

    Args:
        memory_dir: Path to .claude/memory/ directory.
        embedding_fn: Embedding function — callable accepting list[str], returning list[list[float]].
        force: If True, ingest even if collections are non-empty.
        conn: Optional psycopg2 connection. Uses module-level singleton if not provided.

    Returns:
        Dict mapping collection name to number of documents ingested.
    """
    from quantstack.rag.query import store_embedding, _get_connection

    if conn is None:
        conn = _get_connection()

    if not memory_dir.is_dir():
        raise FileNotFoundError(f"Memory directory not found: {memory_dir}")

    counts: dict[str, int] = {
        "strategy_knowledge": 0,
        "trade_outcomes": 0,
        "market_research": 0,
    }

    # Skip if all collections are non-empty (unless force)
    if not force:
        cur = conn.cursor()
        all_populated = True
        for name in counts:
            cur.execute("SELECT COUNT(*) FROM embeddings WHERE collection = %s", [name])
            if cur.fetchone()[0] == 0:
                all_populated = False
                break
        cur.close()
        if all_populated:
            logger.info("All collections already populated, skipping migration")
            return counts

    now = datetime.now(timezone.utc).isoformat()

    for md_file in sorted(memory_dir.rglob("*.md")):
        if not md_file.is_file():
            continue

        routing = route_file(md_file, memory_dir)
        if routing is None:
            continue

        collection_name, metadata = routing

        try:
            text = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", md_file, e)
            continue

        chunks = _chunk_text(text)
        if not chunks:
            continue

        stem = md_file.stem
        if stem.endswith(".archive"):
            stem = stem.replace(".archive", "")

        for i, chunk in enumerate(chunks):
            doc_id = f"{stem}_{i}"
            chunk_metadata = {**metadata, "ingested_at": now}

            try:
                embedding = embedding_fn([chunk])[0]
            except Exception:
                logger.warning("Failed to embed chunk %d of %s", i, md_file.name)
                continue

            store_embedding(doc_id, collection_name, chunk, embedding, chunk_metadata, conn=conn)
            counts[collection_name] += 1

    return counts


if __name__ == "__main__":
    import os
    from quantstack.rag.embeddings import OllamaEmbeddingFunction
    from quantstack.rag.query import ensure_schema

    ensure_schema()
    embedding_fn = OllamaEmbeddingFunction()

    project_root = Path(__file__).resolve().parents[3]
    memory_dir = project_root / ".claude" / "memory"

    if not memory_dir.exists():
        print(f"Memory directory not found: {memory_dir}")
        raise SystemExit(1)

    result = migrate_memory(memory_dir, embedding_fn)
    for collection, count in result.items():
        print(f"Ingested {count} documents into {collection}")
