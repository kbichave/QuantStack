# Phase 0 Quick Wins — Research Findings

## 1. Tool Registry & Binding (Item 0.1)

**Key files:** `src/quantstack/tools/registry.py`, `src/quantstack/graphs/agent_executor.py`

**Current state:**
- `TOOL_REGISTRY` is a flat dict (lines 198-276 in registry.py). Iteration order = insertion order (Python 3.7+), but tools are added from multiple modules (`_system_tools`, `_alert_tools`, etc.), making order non-obvious and potentially non-deterministic across restarts.
- `get_tools_for_agent(tool_names)` resolves tool name strings to BaseTool objects (line 320).
- `get_tools_for_agent_with_search()` partitions tools into deferred/always-loaded (line 338).
- `tool_to_anthropic_dict()` (line 369) processes tools in whatever order provided — no sorting.
- `build_system_message()` in agent_executor.py (line 152) injects tool categories as text guidance (lines 173-176). Tool categories are organized by `_TOOL_CATEGORIES` dict (lines 55-92).
- **No sorting anywhere in the pipeline.**

**Web research confirms:** LangChain's `bind_tools()` preserves input sequence order with zero sorting. Anthropic's cache key computation includes tool definitions as the FIRST level in the hierarchy (`tools -> system -> messages`). Different tool order = cache miss on ALL levels.

**Fix approach:** Sort tools by name in `get_tools_for_agent()` and/or `get_tools_for_agent_with_search()` before returning. One-line change: `return sorted(tools, key=lambda t: t.name)`.

---

## 2. LLM Routing & Prompt Caching (Item 0.2)

**Key files:** `src/quantstack/llm/provider.py`, `src/quantstack/llm/config.py`, `src/quantstack/graphs/agent_executor.py`

**Current state:**
- `get_model_with_fallback(tier)` (line 72 in provider.py) tries primary provider, then FALLBACK_ORDER: `["bedrock", "anthropic", "openai", "ollama"]`.
- `_instantiate_chat_model()` (line 116) creates provider-specific models. Bedrock (line 130-142): uses `langchain_aws.ChatBedrock` with `model_kwargs={"temperature": ..., "max_tokens": ...}`.
- `build_system_message()` (line 152 in agent_executor.py) assembles base role/goal/backstory + optional tool categories. **Plain strings, no cache_control.**
- **Zero references to `cache_control`, `CacheControl`, or `ephemeral` in entire codebase.**

**Web research — prompt caching options:**

| Approach | How | When to use |
|----------|-----|-------------|
| Automatic (top-level) | `cache_control={"type": "ephemeral"}` on `invoke()` | Multi-turn conversations |
| Explicit breakpoints | `cache_control` on individual content blocks | Different sections change at different frequencies |
| Tool-level | `cache_control` on last tool in sorted list | Stable tool definitions |

**Key constraints:**
- Minimum token thresholds: Sonnet requires 1,024-2,048 tokens, Opus requires 4,096 tokens. Short system prompts won't cache.
- Cache TTL: 5 min default (resets on each hit). QuantStack's 5-min trading cycle aligns well.
- Bedrock: No `anthropic_beta` header needed — caching is GA. Use same `cache_control` syntax via InvokeModel API.
- Cache hits don't count against Bedrock rate limits.

**LangChain integration (langchain-anthropic 1.4.0):**
1. `SystemMessage(content=[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}])` — structured content blocks pass through directly.
2. `convert_to_anthropic_tool(tool)` returns `AnthropicTool` TypedDict with explicit `cache_control` field.
3. `llm.invoke(messages, cache_control={"type": "ephemeral"})` — automatic, adds to last content block.

**Recommended approach:** Use explicit breakpoints on (1) last sorted tool definition and (2) system message. This maximizes cache reuse since tools and system prompts are stable across cycles.

---

## 3. Knowledge Base Tool & RAG (Item 0.3)

**Key files:** `src/quantstack/tools/langchain/learning_tools.py`, `src/quantstack/rag/query.py`

**Current state — the tool (learning_tools.py line 14):**
- `search_knowledge_base()` takes `query` and `top_k` parameters.
- **Implementation (lines 25-30):** `SELECT id, category, content, metadata, created_at FROM knowledge_base ORDER BY created_at DESC LIMIT %s`
- Query parameter is completely ignored. Pure recency-based retrieval.
- Queries the `knowledge_base` table (old schema).

**Semantic search that exists but is unused (rag/query.py):**
- `search_knowledge_base()` function at line 156-203 — performs proper semantic search via pgvector.
- `search_similar()` at line 94 — cosine distance search on `embeddings` table.
- Uses `embedding <=> %s::vector` operator (line 122).
- `remember_knowledge()` at line 206 — writes embeddings to pgvector.

**Two separate knowledge stores:**
- Tool queries `knowledge_base` table (old, no embeddings)
- RAG module uses `embeddings` table (pgvector-backed)

**Fix approach:** Replace the SQL query in learning_tools.py with a call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. One import, one function call change.

**Risk noted in spec:** Verify embedding coverage — if the embedding pipeline hasn't been running, results may be stale or empty.

---

## 4. Database & Embeddings (Item 0.4)

**Key files:** `src/quantstack/rag/query.py` (schema at lines 17-34), `src/quantstack/db.py`

**Current schema:**
```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
-- Existing indexes:
CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings (collection);
CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_ticker ON embeddings USING GIN ((metadata -> 'ticker'));
-- NO vector index exists
```

**Web research — HNSW recommendations for 500-10k rows:**
- `m=16` (default), `ef_construction=100` (slightly above default 64 — cheap insurance since build is instant at this scale)
- Use `vector_cosine_ops` to match the existing `<=>` operator in queries
- Set `hnsw.ef_search=100` at query time
- Build time: sub-second. No maintenance concerns at this scale.
- HNSW preferred over IVFFlat: works on empty tables, no reindex needed for incremental inserts, better recall.

**Recommended SQL:**
```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
    ON embeddings USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 100);
```

**Migration approach:** Add to `run_migrations()` in db.py as a new `_migrate_*()` function, or add to the `_INIT_SQL` in rag/query.py alongside existing index definitions.

---

## 5. Sentiment Collector (Item 0.5)

**Key files:** `src/quantstack/signal_engine/collectors/sentiment.py`, `src/quantstack/signal_engine/synthesis.py`

**Current state:**
- `_safe_defaults()` (line 156-162): Returns `{"sentiment_score": 0.5, "dominant_sentiment": "neutral", "n_headlines": 0, "source": "default"}`.
- `collect_sentiment()` (line 44): 8-second timeout, catches all exceptions, returns `_safe_defaults()`.

**IMPORTANT FINDING — Synthesis already handles this correctly:**
- `RuleBasedSynthesizer.synthesize()` checks `n_headlines > 0` before using sentiment_score (lines 308-319).
- When `n_headlines == 0` (the default fallback case), sentiment contributes 0.0 to score.
- The fake 0.5 neutral is never actually used in scoring when data is unavailable.

**Assessment:** The synthesis layer already redistributes weight correctly. The 0.5 return is cosmetically misleading but functionally inert. Changing to `{}` would be cleaner semantically, but the spec's concern about "fake neutral carrying same weight" may not reflect the actual behavior. **Verify this carefully before changing** — returning `{}` may break downstream consumers that expect specific keys.

---

## 6. Docker Configuration (Items 0.6, 0.7)

**Key files:** `docker-compose.yml`, `start.sh`

**Port bindings (all on 0.0.0.0):**
- PostgreSQL: `5434:5432`
- Ollama: `11434:11434`
- Langfuse: `3100:3000`
- Dashboard: `8421:8421`

**Default passwords:**
- PostgreSQL: `${POSTGRES_PASSWORD:-quantstack}` (line 40)
- Langfuse DB: `${LANGFUSE_DB_PASSWORD:-langfuse}` (line 66)
- Langfuse web: `LANGFUSE_INIT_USER_PASSWORD: quantstack123` (line 131) — hardcoded

**start.sh validation (lines 48-58):** Checks TRADER_PG_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPHA_VANTAGE_API_KEY. **Does NOT check POSTGRES_PASSWORD, LANGFUSE_DB_PASSWORD.**

**Fix approach:**
1. Bind PostgreSQL to `127.0.0.1:5434:5432`
2. Remove default password fallbacks from docker-compose.yml
3. Add password validation to start.sh — fail if POSTGRES_PASSWORD not set or equals "quantstack"
4. Consider also binding Ollama and Langfuse to localhost

---

## 7. Testing Setup

**Framework:** pytest

**Structure:**
```
tests/
├── conftest.py              (2400+ lines — shared fixtures)
├── unit/                    (180+ test files)
│   ├── conftest.py          (mock_settings, OHLCV generators)
│   └── fixtures/            (test data)
├── integration/             (24 subdirectories)
├── graphs/                  (graph runner tests)
├── core/                    (core feature tests)
├── coordination/            (loop coordination tests)
├── shared/                  (shared utilities)
├── quant_pod/               (signal collector tests)
└── regression/              (regression tests)
```

**Key fixtures:** `make_ohlcv_df()`, `mock_settings`, trend/pattern generators, `add_atr_column()`.

**Running tests:** `pytest tests/unit/` or `pytest tests/` for all. Specific: `pytest tests/unit/test_file.py::test_name`.

**Existing coverage for affected subsystems:**
- Tool registry: No dedicated ordering tests found
- RAG/knowledge base: Likely in integration/ or core/
- Signal collectors: In quant_pod/
- Docker/start.sh: Not unit-tested (infrastructure)

**Testing approach for Phase 0:** Each fix needs at least one regression test. Most can be unit tests with mocks. The RAG fix and HNSW index need integration tests against a real PostgreSQL+pgvector instance.
