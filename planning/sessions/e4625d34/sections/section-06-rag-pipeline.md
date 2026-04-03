# Section 6: RAG Pipeline (ChromaDB + Ollama)

## Overview

This section implements the Retrieval-Augmented Generation (RAG) pipeline that gives all CrewAI agents access to long-term knowledge: historical trade outcomes, strategy lessons, research discoveries, and workshop findings. The pipeline uses ChromaDB (running as a Docker service in client-server mode) for vector storage and Ollama's `mxbai-embed-large` model for embeddings.

The RAG pipeline is distinct from CrewAI's built-in `Memory` system. CrewAI Memory handles short-term, in-session context (what happened this cycle). The RAG pipeline handles long-term institutional knowledge that persists across restarts and is queryable by any agent via dedicated tools.

**Files to create:**

- `src/quantstack/rag/__init__.py`
- `src/quantstack/rag/embeddings.py` -- Ollama embedding wrapper
- `src/quantstack/rag/ingest.py` -- Document ingestion into ChromaDB
- `src/quantstack/rag/query.py` -- Retrieval functions
- `tests/unit/test_rag_pipeline.py` -- All unit tests

**Dependencies on other sections:**

- **Section 1 (Scaffolding):** Docker Compose must define the `chromadb` and `ollama` services with health checks, named volumes, and port exposure. The `pyproject.toml` must include `chromadb` and `ollama` in the `[crewai]` extras group.
- This section **blocks** Section 4 (Agent Definitions -- agents reference RAG tools) and Section 11 (Memory Migration -- uses the ingestion pipeline).

---

## Tests First

All tests use an **in-memory ChromaDB client** (no Docker dependency for unit tests). The Ollama embedding function is mocked to return deterministic float vectors.

```python
# tests/unit/test_rag_pipeline.py

import pytest

# ---------------------------------------------------------------------------
# Embedding wrapper tests
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingFunction:
    """Tests for src/quantstack/rag/embeddings.py"""

    def test_calls_ollama_embed_with_correct_model(self):
        """OllamaEmbeddingFunction.__call__ invokes ollama.embed
        with model='mxbai-embed-large' and the input texts."""

    def test_returns_list_of_float_lists(self):
        """Return type is list[list[float]], one vector per input text."""

    def test_handles_ollama_connection_error_gracefully(self):
        """When Ollama is unreachable, raises a clear error (not a
        generic socket exception) so callers can degrade gracefully."""

# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestMemoryFiles:
    """Tests for src/quantstack/rag/ingest.py"""

    def test_reads_all_md_files_from_directory(self):
        """Given a temp directory with 3 .md files, ingestion processes
        all three."""

    def test_chunks_text_with_correct_size_and_overlap(self):
        """Text is split into chunks of ~1000 chars with 200 char overlap.
        Verify chunk boundaries and that no content is lost."""

    def test_writes_to_correct_collection_per_file_type(self):
        """strategy_registry.md -> strategy_knowledge collection.
        trade_journal.md -> trade_outcomes collection.
        session_handoff_*.md -> market_research collection."""

    def test_idempotent_second_call_no_duplicates(self):
        """Running ingestion twice on the same files does not create
        duplicate documents. Verify document count is stable."""

    def test_skips_ingestion_when_collections_nonempty(self):
        """If collections already have documents (from a previous run),
        the startup ingestion is a no-op."""

    def test_metadata_includes_source_file_and_section(self):
        """Each ingested chunk carries metadata: source filename and
        the nearest markdown heading as 'section'."""

# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

class TestSearchKnowledgeBase:
    """Tests for src/quantstack/rag/query.py"""

    def test_returns_top_n_results_with_metadata(self):
        """Query returns up to n_results documents, each with text content
        and metadata dict."""

    def test_filters_by_ticker_metadata(self):
        """When ticker='AAPL' is provided, only documents with
        metadata ticker='AAPL' are returned."""

    def test_filters_by_collection(self):
        """When collection='trade_outcomes' is specified, only that
        collection is searched."""

    def test_handles_empty_collection(self):
        """Returns empty list (not error) when collection has no documents."""

    def test_searches_all_collections_when_none_specified(self):
        """When collection is None, searches across all three collections
        and merges results by relevance score."""

# ---------------------------------------------------------------------------
# Remember (write) tests
# ---------------------------------------------------------------------------

class TestRememberKnowledge:
    """Tests for src/quantstack/rag/query.py :: remember_knowledge"""

    def test_writes_document_with_correct_metadata(self):
        """remember_knowledge('lesson text', collection='trade_outcomes',
        metadata={'ticker': 'SPY', 'outcome': 'win'}) persists correctly."""

    def test_written_document_is_retrievable(self):
        """After writing, a search query matching the content returns
        the document."""

# ---------------------------------------------------------------------------
# ChromaDB client configuration tests
# ---------------------------------------------------------------------------

class TestChromaDBClient:
    """Tests for client setup in src/quantstack/rag/query.py"""

    def test_uses_http_client_with_correct_host_and_port(self):
        """In production mode, uses chromadb.HttpClient pointing to
        CHROMADB_HOST:CHROMADB_PORT (defaults: chromadb:8000)."""

    def test_creates_three_collections_on_init(self):
        """get_or_create_collection is called for trade_outcomes,
        strategy_knowledge, and market_research."""
```

---

## Implementation Details

### 6.1 ChromaDB Collections

Three collections, each with distinct metadata schemas:

| Collection | Content | Key Metadata Fields |
|---|---|---|
| `trade_outcomes` | Historical trades, reflexion episodes, P&L lessons | `ticker`, `strategy_id`, `domain`, `date`, `pnl`, `outcome` (win/loss/scratch) |
| `strategy_knowledge` | Strategy definitions, workshop lessons (negative results), ML experiment results | `strategy_name`, `domain`, `status`, `date` |
| `market_research` | Community-intel discoveries, arXiv findings, session handoffs | `source`, `topic`, `date`, `relevance_score` |

### 6.2 Embedding Wrapper

**File:** `src/quantstack/rag/embeddings.py`

This module wraps the Ollama Python client to provide a ChromaDB-compatible embedding function. ChromaDB expects a callable that takes a list of strings and returns a list of float vectors.

```python
# src/quantstack/rag/embeddings.py

class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by Ollama.

    Usage:
        ef = OllamaEmbeddingFunction(
            model_name="mxbai-embed-large",
            base_url="http://ollama:11434",
        )
        vectors = ef(["some text", "another text"])
        # vectors: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    """

    def __init__(self, model_name: str = "mxbai-embed-large", base_url: str | None = None):
        """Initialize with model name and optional Ollama base URL.

        base_url defaults to OLLAMA_BASE_URL env var, then http://localhost:11434.
        """

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Calls ollama.embed() under the hood."""
```

The `base_url` reads from `OLLAMA_BASE_URL` environment variable (set to `http://ollama:11434` in Docker Compose, falls back to `http://localhost:11434` for local dev).

### 6.3 Ingestion Pipeline

**File:** `src/quantstack/rag/ingest.py`

Two ingestion modes:

**Startup ingestion** -- runs once when ChromaDB collections are empty (first boot). Reads `.claude/memory/*.md` files, chunks them, embeds, and upserts.

**Continuous ingestion** -- called by agents at runtime via `remember_knowledge` to add new knowledge (trade reflections, strategy registrations, research discoveries).

```python
# src/quantstack/rag/ingest.py

def ingest_memory_files(memory_dir: str, chromadb_client, embedding_fn) -> dict[str, int]:
    """One-time startup ingestion of markdown memory files into ChromaDB.

    Reads all .md files from memory_dir, chunks them, and upserts into
    the appropriate collection based on filename pattern:
      - strategy_registry.md, workshop_lessons.md, ml_experiment_log.md
            -> strategy_knowledge
      - trade_journal.md
            -> trade_outcomes
      - session_handoff_*.md, tickers/*.md
            -> market_research

    Returns dict of {collection_name: documents_ingested_count}.

    Idempotent: skips if collections are already non-empty.
    """

def chunk_markdown(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split markdown text into overlapping chunks with metadata.

    Each chunk is a dict with keys:
      - 'text': the chunk content
      - 'section': nearest preceding markdown heading (## or ###)

    Uses character-based splitting with overlap to preserve context
    across chunk boundaries. Splits preferentially at paragraph
    boundaries (double newline) within the size window.
    """

def file_to_collection(filename: str) -> str:
    """Map a memory filename to its target ChromaDB collection.

    Returns one of: 'trade_outcomes', 'strategy_knowledge', 'market_research'.
    """
```

**Chunking strategy:** `RecursiveCharacterTextSplitter`-style splitting -- 1000 character chunks with 200 character overlap. Prefer splitting at paragraph boundaries (double newlines), then single newlines, then sentence boundaries. Each chunk retains the nearest markdown heading as `section` metadata.

**Document IDs:** Use a deterministic ID scheme: `{source_filename}::{chunk_index}`. This makes upserts idempotent -- re-ingesting the same file overwrites existing chunks rather than creating duplicates.

### 6.4 Retrieval Functions

**File:** `src/quantstack/rag/query.py`

```python
# src/quantstack/rag/query.py

COLLECTIONS = ("trade_outcomes", "strategy_knowledge", "market_research")

def get_chromadb_client():
    """Return a ChromaDB HttpClient connected to the chromadb service.

    Reads CHROMADB_HOST (default: 'chromadb') and CHROMADB_PORT
    (default: 8000) from environment.

    Calls get_or_create_collection for each of the three collections,
    passing the OllamaEmbeddingFunction as the embedding function.
    """

def search_knowledge_base(
    query: str,
    collection: str | None = None,
    ticker: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Search ChromaDB for relevant knowledge.

    Args:
        query: Natural language search query.
        collection: Restrict to one collection. If None, searches all three
                    and merges results by relevance score.
        ticker: Optional metadata filter -- only return docs tagged with
                this ticker symbol.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts, each with keys: 'text', 'metadata', 'distance'.
        Sorted by relevance (lowest distance first).
    """

def remember_knowledge(
    text: str,
    collection: str,
    metadata: dict | None = None,
) -> str:
    """Write a new document to the knowledge base.

    Args:
        text: The knowledge content to store.
        collection: Target collection name (must be one of COLLECTIONS).
        metadata: Optional metadata dict (ticker, date, outcome, etc.).

    Returns:
        The document ID of the stored document.

    Generates a unique document ID based on collection, timestamp, and
    a hash of the text content.
    """
```

**Cross-collection search:** When `collection=None`, the function queries all three collections independently, then merges results by ChromaDB distance score and returns the top N across all collections. This lets agents ask broad questions like "What do we know about momentum strategies?" without knowing which collection holds the answer.

**Metadata filtering:** ChromaDB supports `where` clauses on metadata. When `ticker` is provided, the query includes `where={"ticker": ticker}`. Additional metadata filters can be added as the system evolves.

### 6.5 CrewAI Memory Integration

CrewAI's built-in `Memory` class handles short-term agent context and should be configured to use the same Ollama embeddings. This is configured at crew instantiation time (covered in Section 5 and Section 9), not in this section. The RAG pipeline here is the **long-term** knowledge store that Memory does not replace.

The two systems serve different purposes:

| System | Scope | Persistence | Access Pattern |
|---|---|---|---|
| CrewAI Memory | Current cycle + recent cycles | Ephemeral (per crew instance) | Automatic (CrewAI injects into prompts) |
| ChromaDB RAG | All historical knowledge | Persistent (Docker volume) | Explicit (agent calls `search_knowledge_base_tool`) |

### 6.6 Environment Variables

```
CHROMADB_HOST=chromadb          # Docker service name (default)
CHROMADB_PORT=8000              # ChromaDB HTTP port (default)
OLLAMA_BASE_URL=http://ollama:11434  # Ollama API endpoint
```

These are set in `docker-compose.yml` for containerized runs. For local development, defaults point to `localhost`.

### 6.7 Graceful Degradation

When ChromaDB or Ollama is unreachable:

- `search_knowledge_base` catches connection errors and returns an empty list with a warning log. Agents continue operating without RAG context -- they still have DB state and CrewAI's in-cycle memory.
- `remember_knowledge` catches connection errors and logs the failure. The knowledge is lost for this write but the agent's cycle is not interrupted.
- The supervisor crew detects the outage (via its health check task) and restarts the affected container.

This degradation logic lives in the retrieval/write functions themselves, not in the tool wrappers. The tool wrappers in `src/quantstack/crewai_tools/rag_tools.py` (Section 3) simply call these functions and return the result.

### 6.8 Docker Infrastructure (Reference)

The following Docker Compose services must be in place (defined in Section 1):

- **chromadb:** `chromadb/chroma` image, named volume `chromadb-data`, health check `curl localhost:8000/api/v1/heartbeat`, port 8000 exposed.
- **ollama:** `ollama/ollama` image, named volume `ollama-models`, health check `curl localhost:11434/api/tags`, `mxbai-embed-large` model pulled at startup.

---

## Implementation Sequence

1. Create `src/quantstack/rag/__init__.py` (empty or with public API re-exports).
2. Implement `embeddings.py` -- the `OllamaEmbeddingFunction` class.
3. Implement `query.py` -- `get_chromadb_client`, `search_knowledge_base`, `remember_knowledge`. This establishes the three collections.
4. Implement `ingest.py` -- `chunk_markdown`, `file_to_collection`, `ingest_memory_files`.
5. Write all tests in `tests/unit/test_rag_pipeline.py` using in-memory ChromaDB and mocked Ollama.
6. Verify tests pass: `uv run pytest tests/unit/test_rag_pipeline.py`.
