# Section 05: Fix `search_knowledge_base` to Use RAG (Item 0.3)

## Dependencies

- **section-04-hnsw-index** must be completed first. That section adds an HNSW vector index to the `embeddings` table, which this section's semantic search relies on for acceptable query latency.

---

## Background

The `search_knowledge_base` tool is used by 15+ agents across the Research, Trading, and Supervisor graphs. It is defined in `src/quantstack/tools/langchain/learning_tools.py`. Currently, it runs a **recency-only SQL query** against the old `knowledge_base` table, completely ignoring the user's `query` text:

```python
rows = db.execute(
    """SELECT id, category, content, metadata, created_at
       FROM knowledge_base
       ORDER BY created_at DESC
       LIMIT %s""",
    (top_k,),
).fetchall()
```

A proper semantic search function already exists at `src/quantstack/rag/query.py` lines 156-203 (`search_knowledge_base()`). It accepts a natural language query, embeds it via Ollama, and searches the `embeddings` pgvector table using cosine distance. It is never called from the tool.

The RAG `search_knowledge_base()` function returns dicts with keys `{"text", "metadata", "distance", "collection"}`. The tool's current output contract is `{"id", "category", "content", "metadata", "created_at"}`. A schema mapping is required to maintain backward compatibility.

---

## Pre-Implementation Investigation (MANDATORY)

Before writing any code, investigate the data state. The tool currently reads from the `knowledge_base` table while the RAG module reads from the `embeddings` table. These may have different data.

Run the following queries against the QuantStack PostgreSQL database:

1. **Row counts:** `SELECT COUNT(*) FROM knowledge_base;` and `SELECT COUNT(*) FROM embeddings;`
2. **Data freshness:** `SELECT MAX(created_at) FROM knowledge_base;` and `SELECT MAX(created_at) FROM embeddings;`
3. **Overlap assessment:** Determine whether `knowledge_base` has entries not represented in `embeddings`.

**Decision tree based on findings:**

| Condition | Action |
|-----------|--------|
| `embeddings` has good coverage (comparable row count, recent data) | Proceed with the fix as described below |
| `knowledge_base` has unique data not in `embeddings` | Add a one-time migration step to embed and insert that data before switching the tool |
| `embeddings` table is empty | **STOP.** This item is deferred. Fix the embedding pipeline first. Exclude from this PR. |

---

## Tests (Write First)

**Test file:** `tests/unit/test_knowledge_base_tool.py`

### Test 1: Tool calls RAG search function with query

Mock `rag.query.search_knowledge_base` and invoke the tool. Assert the mock was called with the user's query string and `n_results=top_k`. This confirms the tool delegates to the RAG module rather than running raw SQL.

### Test 2: Return schema maps RAG output to tool contract

Mock the RAG function to return:
```python
[{"text": "momentum lesson", "metadata": {"id": "abc", "created_at": "2026-01-01"}, "distance": 0.12, "collection": "strategy_knowledge"}]
```

Assert the tool output (parsed from JSON) contains:
- `content` equal to `"momentum lesson"` (mapped from `text`)
- `category` equal to `"strategy_knowledge"` (mapped from `collection`)
- `metadata` passed through
- `id` extracted from metadata (or generated from hash if absent)
- `created_at` extracted from metadata (or `None` if absent)
- `distance` field present (new non-breaking addition)

### Test 3: Empty RAG results return empty list

Mock RAG to return `[]`. Assert the tool returns a valid JSON structure with `"results": []` and `"count": 0`, with no errors.

### Test 4: Tool handles Ollama unavailability gracefully

Mock RAG function to raise `ConnectionError` (simulating Ollama being down). Assert the tool returns a JSON response with an informative error message containing "embedding" or "unavailable" (not a raw traceback), and does not crash.

**Integration tests** (separate file `tests/integration/test_knowledge_base_rag.py`, require live DB):

### Integration Test 1: Semantic search returns relevant results

Insert known embeddings for "momentum strategy backtesting" and "value investing fundamentals" into the `embeddings` table. Query with "momentum". Assert the momentum entry is returned with a lower distance than the value entry.

### Integration Test 2: Query parameter affects result ranking (not recency-only)

Insert an old but semantically relevant embedding and a new but semantically irrelevant embedding. Assert the old relevant embedding ranks higher, proving the tool uses semantic similarity not recency.

---

## Implementation

### File: `src/quantstack/tools/langchain/learning_tools.py`

**Changes to `search_knowledge_base()` function (lines 14-46):**

1. **Remove the raw SQL query.** Delete the `db.execute(...)` call and the row-processing loop that reads from the `knowledge_base` table.

2. **Import and call the RAG function.** Replace the SQL with a call to `rag.query.search_knowledge_base()`. The import should be at the module level:

   ```python
   from quantstack.rag.query import search_knowledge_base as rag_search
   ```

   Inside the tool body, call:
   ```python
   rag_results = rag_search(query=query, n_results=top_k)
   ```

3. **Map the return schema.** The RAG function returns `{"text", "metadata", "distance", "collection"}`. Map to the existing tool contract:

   - `text` maps to `content` (truncate to 500 chars as the current code does)
   - `collection` maps to `category`
   - `metadata` passes through as-is
   - `id`: extract from `metadata.get("id")`, or generate a deterministic ID from a hash of the content if absent
   - `created_at`: extract from `metadata.get("created_at", None)`
   - `distance`: include as a new field (non-breaking addition for relevance ranking)

4. **Handle Ollama dependency.** The RAG function uses `OllamaEmbeddingFunction()` internally. If Ollama is down, the RAG function returns `[]` silently (it catches the exception in `search_knowledge_base()` at query.py line 183). However, distinguish between "no results" and "embedding service unavailable":

   - Wrap the RAG call in a try/except that catches `ConnectionError` and `OSError` (Ollama connection failures)
   - On connection failure, return a JSON response with a clear error message: `"Embedding service unavailable — knowledge base search requires Ollama to be running"`
   - On successful call returning empty results, return the normal empty results structure

5. **Connection management.** The RAG function accepts an optional `conn` parameter. The tool currently gets `ctx.db` from `require_ctx()`. Inspect whether `ctx.db` is a raw psycopg2 connection or a wrapper. If it is a raw psycopg2 connection, pass it to the RAG function to avoid opening a second connection. If it is a wrapper (e.g., has an `.execute()` method but is not a psycopg2 connection), let the RAG function manage its own connection by not passing `conn`.

   The `require_ctx()` call can be removed entirely if the RAG function manages its own connection, since the tool no longer needs direct DB access.

### Resulting function signature (stub)

```python
@tool
async def search_knowledge_base(
    query: Annotated[str, Field(description="Natural language search query...")],
    top_k: Annotated[int, Field(description="Maximum number of knowledge entries...")] = 5,
) -> str:
    """Retrieves past lessons, trade outcomes, and strategy notes from the knowledge base
    using semantic search. ..."""
    # 1. Call rag_search(query=query, n_results=top_k)
    # 2. Map results to tool contract schema
    # 3. Handle Ollama unavailability with clear error message
    # 4. Return JSON with {query, results, count}
```

---

## Edge Cases to Handle

| Edge Case | Expected Behavior |
|-----------|-------------------|
| Empty `embeddings` table | RAG returns `[]`; tool returns `{"results": [], "count": 0}` — no error |
| Query with no semantic matches (all distances above threshold) | Same as empty results; the RAG function has no hard distance threshold — it returns the top N regardless |
| Ollama service not running | Tool returns JSON with informative error, does not crash |
| Very long query string | Ollama embedding function handles truncation internally |
| `top_k=0` | RAG function passes `n_results=0` to SQL `LIMIT 0`; returns `[]` |
| Agent prompts parsing specific fields from output | The `id`, `category`, `content`, `metadata`, `created_at` fields are preserved. The new `distance` field is additive and non-breaking. |

---

## Validation

After implementation:

1. Run unit tests: `uv run pytest tests/unit/test_knowledge_base_tool.py -v`
2. Manually invoke the tool with a test query and verify the output schema matches the contract
3. Verify that the `query` parameter actually influences results (not just recency ordering)
4. If Ollama is stopped, verify the tool returns a clear error message rather than silently returning nothing
5. Check LangFuse traces to confirm agents using `search_knowledge_base` receive properly formatted responses
