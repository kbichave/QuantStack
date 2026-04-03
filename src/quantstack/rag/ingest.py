"""Document ingestion into pgvector embeddings table."""

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

COLLECTION_MAP: dict[str, str] = {
    "strategy_registry": "strategy_knowledge",
    "workshop_lessons": "strategy_knowledge",
    "ml_experiment": "strategy_knowledge",
    "ml_model_registry": "strategy_knowledge",
    "trade_journal": "trade_outcomes",
    "session_handoff": "market_research",
    "community_intel": "market_research",
}


def file_to_collection(filename: str) -> str:
    """Map a memory filename to its target collection."""
    name_lower = filename.lower()
    for prefix, collection in COLLECTION_MAP.items():
        if prefix in name_lower:
            return collection
    return "market_research"


def chunk_markdown(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split markdown text into overlapping chunks with metadata.

    Returns list of dicts with 'text' and 'section' keys.
    """
    if not text.strip():
        return []

    chunks: list[dict] = []
    current_section = ""

    paragraphs = re.split(r"\n\n+", text)
    current_chunk = ""

    for para in paragraphs:
        heading_match = re.match(r"^(#{1,4})\s+(.+)", para.strip())
        if heading_match:
            current_section = heading_match.group(2).strip()

        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "section": current_section,
            })
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
        })

    return chunks


def ingest_memory_files(
    memory_dir: str,
    embedding_fn=None,
    conn=None,
) -> dict[str, int]:
    """Ingest markdown memory files into pgvector embeddings table.

    Idempotent: skips if collections already have documents.
    Uses module-level connection if conn not provided.

    Returns dict of {collection_name: documents_ingested_count}.
    """
    from quantstack.rag.query import COLLECTIONS, store_embedding, _get_connection

    if conn is None:
        conn = _get_connection()

    if embedding_fn is None:
        from quantstack.rag.embeddings import OllamaEmbeddingFunction
        embedding_fn = OllamaEmbeddingFunction()

    # Check if collections are already populated
    counts: dict[str, int] = {}
    cur = conn.cursor()
    all_empty = True
    for col_name in COLLECTIONS:
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE collection = %s", [col_name])
        existing = cur.fetchone()[0]
        if existing > 0:
            all_empty = False
            counts[col_name] = 0
    cur.close()

    if not all_empty:
        logger.info("Collections already populated, skipping ingestion")
        return counts

    memory_path = Path(memory_dir)
    if not memory_path.is_dir():
        logger.warning("Memory directory %s does not exist", memory_dir)
        return counts

    md_files = sorted(memory_path.glob("**/*.md"))
    if not md_files:
        logger.info("No .md files found in %s", memory_dir)
        return counts

    for col_name in COLLECTIONS:
        counts[col_name] = 0

    for md_file in md_files:
        text = md_file.read_text(errors="replace")
        if not text.strip():
            continue

        col_name = file_to_collection(md_file.name)

        chunks = chunk_markdown(text)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            doc_id = f"{md_file.name}::{i}"
            try:
                embedding = embedding_fn([chunk["text"]])[0]
            except Exception:
                logger.warning("Failed to embed chunk %d of %s", i, md_file.name)
                continue

            metadata = {
                "source": md_file.name,
                "section": chunk["section"],
                "chunk_index": i,
            }
            store_embedding(doc_id, col_name, chunk["text"], embedding, metadata, conn=conn)
            counts[col_name] += 1

        logger.info(
            "Ingested %d chunks from %s into %s",
            len(chunks), md_file.name, col_name,
        )

    return counts
