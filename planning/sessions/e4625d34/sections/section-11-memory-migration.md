# Section 11: Memory Migration

## Purpose

One-time ingestion of all existing `.claude/memory/` markdown files into ChromaDB collections, so that CrewAI agents have access to the accumulated knowledge (strategy registry, trade journal, workshop lessons, ML experiments, ticker research, session handoffs) via RAG retrieval. After migration, agents read/write knowledge through ChromaDB exclusively; the `.claude/memory/` directory is no longer written to by the new system.

## Dependencies

- **Section 06 (RAG Pipeline)** must be complete. This section consumes:
  - `src/quantstack/rag/ingest.py` (chunking, embedding, upsert)
  - `src/quantstack/rag/query.py` (retrieval functions)
  - `src/quantstack/rag/embeddings.py` (Ollama embedding wrapper)
  - ChromaDB collections: `trade_outcomes`, `strategy_knowledge`, `market_research`

## Tests (Write First)

All tests go in `tests/unit/test_memory_migration.py`. Use an in-memory ChromaDB client (no Docker required).

```python
# tests/unit/test_memory_migration.py

# Test: ingestion reads strategy_registry.md and chunks correctly
#   - Create a temp dir with a sample strategy_registry.md
#   - Call the migration function
#   - Assert documents were written to the "strategy_knowledge" collection
#   - Assert chunk count is > 0 and each chunk is <= 1000 chars

# Test: ingestion reads workshop_lessons.md and tags as negative results
#   - Create a temp dir with a sample workshop_lessons.md
#   - Call the migration function
#   - Assert documents in "strategy_knowledge" have metadata key
#     "content_type" == "negative_result"

# Test: ingestion reads ticker-specific files and tags with ticker metadata
#   - Create a temp dir with tickers/AAPL.md and tickers/SPY.md
#   - Call the migration function
#   - Assert documents in "market_research" have metadata "ticker" == "AAPL" / "SPY"

# Test: ingestion skips if collections are already non-empty
#   - Pre-populate the strategy_knowledge collection with one document
#   - Call the migration function
#   - Assert collection count did not change (no new documents added)

# Test: ingested content is retrievable via search query
#   - Ingest a sample strategy_registry.md containing "momentum crossover"
#   - Query the strategy_knowledge collection with "momentum"
#   - Assert at least one result contains "momentum crossover"

# Test: ingested strategy knowledge returns relevant results for
#   strategy-related queries
#   - Ingest strategy_registry.md and workshop_lessons.md
#   - Query with a strategy-related question
#   - Assert results come from the correct source files (check metadata)
```

## File-to-Collection Routing

The migration script must route each source file to the correct ChromaDB collection with appropriate metadata. Here is the complete routing table:

| Source file pattern | Target collection | Metadata tags |
|---|---|---|
| `strategy_registry.md` | `strategy_knowledge` | `content_type: strategy_definition` |
| `workshop_lessons.md` (+ `.archive.md`) | `strategy_knowledge` | `content_type: negative_result` |
| `ml_experiment_log.md` (+ `.archive.md`) | `strategy_knowledge` | `content_type: ml_experiment` |
| `ml_model_registry.md` | `strategy_knowledge` | `content_type: ml_model` |
| `ml_research_program.md` (+ `.archive.md`) | `strategy_knowledge` | `content_type: ml_research` |
| `lit_review_findings.md` | `strategy_knowledge` | `content_type: literature_review` |
| `trade_journal.md` (+ `.archive.md`) | `trade_outcomes` | `content_type: trade_outcome` |
| `session_handoffs.md` | `market_research` | `content_type: session_handoff` |
| `session_handoffs/*.md` | `market_research` | `content_type: session_handoff` |
| `tickers/*.md` | `market_research` | `content_type: ticker_research`, `ticker: <SYMBOL>` |
| `regime_history.md` | `market_research` | `content_type: regime_history` |
| `agent_performance.md` | `market_research` | `content_type: agent_performance` |
| `daily_plan.md` | `market_research` | `content_type: daily_plan` |
| `risk_desk_report_*.md` | `market_research` | `content_type: risk_report` |
| `research_log_*.md` | `market_research` | `content_type: research_log` |
| `strategy_C_investment.md` | `strategy_knowledge` | `content_type: strategy_definition` |
| `swing_research_*.md` | `market_research` | `content_type: research_log` |

Files in `templates/` are skipped entirely (they are empty structural templates, not knowledge).

## Implementation

### Module: `src/quantstack/rag/migrate_memory.py`

This is a standalone script and importable module. It contains two public functions:

```python
def migrate_memory(
    memory_dir: Path,
    chroma_client: chromadb.ClientAPI,
    embedding_fn: Callable,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Ingest .claude/memory/ markdown files into ChromaDB collections.

    Reads all .md files from memory_dir, routes each to the correct
    collection based on the file-to-collection routing table, chunks
    the content, embeds via embedding_fn, and upserts into ChromaDB.

    Args:
        memory_dir: Path to .claude/memory/ directory.
        chroma_client: ChromaDB client (HttpClient or in-memory for tests).
        embedding_fn: Callable that takes list[str] and returns list[list[float]].
        force: If True, ingest even if collections are non-empty.

    Returns:
        Dict mapping collection name to number of documents ingested.
        Example: {"strategy_knowledge": 45, "trade_outcomes": 12, "market_research": 89}
    """


def route_file(file_path: Path, memory_dir: Path) -> tuple[str, dict[str, str]] | None:
    """Determine target collection and metadata for a memory file.

    Args:
        file_path: Absolute path to the .md file.
        memory_dir: Root memory directory (for computing relative paths).

    Returns:
        Tuple of (collection_name, metadata_dict), or None if the file
        should be skipped (e.g., templates/).
    """
```

### Chunking

Use `RecursiveCharacterTextSplitter` from langchain-text-splitters (or a lightweight equivalent if that dependency is undesirable). Parameters:

- `chunk_size=1000` characters
- `chunk_overlap=200` characters
- Separators: `["\n## ", "\n### ", "\n\n", "\n", " "]` (split on markdown headers first, then paragraphs, then lines)

Each chunk gets a deterministic document ID: `{source_file_stem}_{chunk_index}` (e.g., `strategy_registry_0`, `strategy_registry_1`). This makes upserts idempotent -- re-running migration with the same files overwrites the same IDs rather than creating duplicates.

### Metadata Extraction

Every chunk carries metadata:

- `source_file`: relative path from memory_dir (e.g., `tickers/AAPL.md`)
- `content_type`: from the routing table above
- `ticker`: extracted from filename for `tickers/*.md` files (stem of filename, uppercased)
- `ingested_at`: ISO timestamp of ingestion

For ticker files, the ticker symbol is extracted from the filename: `tickers/AAPL.md` yields `ticker: AAPL`.

### Idempotency

The migration is idempotent by two mechanisms:

1. **Skip-if-non-empty**: By default, if all three collections already contain documents, the migration is a no-op. The `force` flag overrides this.
2. **Deterministic IDs**: Document IDs are derived from filename + chunk index, so re-running with `force=True` overwrites existing documents rather than duplicating.

### Invocation

The migration runs in two contexts:

1. **`start.sh` step 9**: After infrastructure is healthy and before crew services start. Runs as a one-shot Docker command:
   ```bash
   docker compose run --rm trading-crew python -m quantstack.rag.migrate_memory
   ```

2. **Programmatic**: Any runner can call `migrate_memory()` during initialization if it detects empty collections (defensive fallback in case start.sh was skipped).

The module's `__main__` block:

```python
if __name__ == "__main__":
    # Connect to ChromaDB (HttpClient, default port 8000)
    # Initialize Ollama embedding function
    # Resolve memory_dir to .claude/memory/ relative to project root
    # Call migrate_memory()
    # Print summary: "Ingested {n} documents into {collection}"
```

### Source File Inventory

The actual `.claude/memory/` directory currently contains:

**Root-level files (19):**
- `strategy_registry.md`, `trade_journal.md`, `workshop_lessons.md`, `ml_experiment_log.md`, `ml_model_registry.md`, `ml_research_program.md`, `session_handoffs.md`, `lit_review_findings.md`, `regime_history.md`, `agent_performance.md`, `daily_plan.md`, `risk_desk_report_2026_04_02.md`, `research_log_QQQ_investment.md`, `strategy_C_investment.md`, `swing_research_iter2_2026_04_01.md`
- Archive files (`.archive.md` suffix): `workshop_lessons.md.archive.md`, `ml_experiment_log.md.archive.md`, `trade_journal.md.archive.md`, `ml_research_program.md.archive.md`

**Subdirectories:**
- `tickers/` — 56 ticker-specific files (AAPL.md through XRT.md)
- `session_handoffs/` — 2 files
- `templates/` — 8 template files (skipped during migration)

**Total files to ingest: ~77** (19 root + 56 tickers + 2 session_handoffs subdirectory files). Templates are excluded.

## Ongoing Memory (Post-Migration)

After the one-time migration completes, agents interact with knowledge exclusively through CrewAI tools:

- **Read knowledge**: `search_knowledge_base_tool` (defined in section 06, wraps `rag/query.py`)
- **Write knowledge**: `remember_knowledge_tool` (defined in section 06, wraps `rag/ingest.py`)
- **Structured state**: PostgreSQL for positions, strategies, fills (unchanged)
- **Short-term context**: CrewAI Memory (auto-managed per crew cycle)

The `.claude/memory/` directory becomes read-only archival. No agent writes to it.

## Edge Cases

- **Empty files**: Some memory files may be empty or contain only template headers. The chunker should produce zero chunks for these; `migrate_memory` should not fail.
- **Large files**: Archive files (`.archive.md`) can be large. The chunker handles this naturally via the 1000-char window. No special treatment needed.
- **Non-UTF-8 content**: All memory files are markdown (UTF-8). If a file fails to decode, log a warning and skip it rather than aborting the entire migration.
- **ChromaDB unavailable at startup**: If ChromaDB is not reachable, the migration should raise a clear error and exit non-zero so that `start.sh` does not proceed to start crew services.
- **Ollama unavailable (no embeddings)**: If Ollama is down, embedding calls fail. The migration should fail fast with a clear message rather than ingesting documents without embeddings.
