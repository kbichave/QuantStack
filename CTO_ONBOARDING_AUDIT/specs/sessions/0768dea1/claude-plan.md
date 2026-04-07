# Phase 0: Quick Wins — Implementation Plan

## Overview

QuantStack is an autonomous trading system built on three LangGraph StateGraphs (Research, Trading, Supervisor), backed by PostgreSQL+pgvector, with LLM routing across Bedrock/Anthropic/OpenAI/Ollama. A CTO onboarding audit identified 164 findings. This plan covers the 7 lowest-effort, highest-ROI items — collectively saving $32,000-$45,000/year, primarily through prompt caching cost reduction.

All 7 items are independent. They ship as a single PR with regression tests for each. **Exception:** If the pre-implementation investigation for Item 0.3 reveals the `embeddings` table is empty, that item is excluded from this PR and tracked separately.

---

## Section 1: Deterministic Tool Ordering (Item 0.1)

### What and Why

The `TOOL_REGISTRY` in `src/quantstack/tools/registry.py` is a flat Python dict populated from multiple modules. Dict iteration order is insertion-order (Python 3.7+), but the insertion sequence depends on which tool modules are imported and in what order. This means the tool list sent to the LLM API may differ between process restarts.

Anthropic's prompt cache key is computed as a hash of the full request prefix up to each breakpoint. The `tools` field is the first level in the cache hierarchy (`tools → system → messages`). If tool definitions differ in order, the cache key changes, and the entire prefix becomes a cache miss. With 21 agents cycling every 5-10 minutes, this wastes 30-50% of prompt token spend.

### How

**File: `src/quantstack/tools/registry.py`**

Modify the two tool-resolution functions to sort by tool name before returning:

- `get_tools_for_agent(tool_names)` (around line 320): After resolving tool name strings to BaseTool objects, sort the result list by `tool.name`.
- `get_tools_for_agent_with_search()` (around line 338): **Sort `tool_names` before the iteration loop**, not just the final output lists. The `tools_for_api` dict list is built by iterating `tool_names` — if the iteration order is unsorted, the API dicts will also be unsorted even if the execution tools are sorted afterward. Sort the input, and both outputs inherit the order.

The sort key is `lambda t: t.name` for BaseTool objects. For the Anthropic dict format returned by `tool_to_anthropic_dict()` (line 369), sort the input list before conversion, not after.

**TOOL_SEARCH_TOOL handling:** The special `TOOL_SEARCH_TOOL` dict (appended at line ~371) should remain as the last item after sorting. It's a meta-tool for deferred tool search, not a regular tool. Sort all regular tools first, then append TOOL_SEARCH_TOOL.

**Scope boundary:** This change only affects the order tools are returned to callers. It does not change tool registration, tool definitions, or the TOOL_REGISTRY dict itself.

### Edge Cases

- Tools with identical names: Not possible — dict keys are unique.
- Dynamic tools added at runtime: Will be sorted on next call; insertion order doesn't matter.
- Agent configs listing tools in a specific intentional order: There's no semantic meaning to tool order in agent YAML configs. The LLM receives tool schemas, not ordered instructions.

---

## Section 2: Prompt Caching with Explicit Breakpoints (Item 0.2)

### What and Why

Zero `cache_control` references exist in the codebase. Every LLM call pays full input token price for system prompts and tool definitions. At ~$126/day, this is the single largest cost waste. Claude's prompt caching reduces cached input token cost by 90% (cache reads cost 0.1x base price).

### How

Three changes are needed, in three files:

#### 2a. Tool-Level Cache Breakpoint

**File: `src/quantstack/tools/registry.py`**

After sorting tools (from Section 1) and converting to Anthropic dict format, add `cache_control: {"type": "ephemeral"}` to the **last** tool definition in the list. This creates a cache boundary: all tool definitions before and including this breakpoint are cached as one block.

The function `tool_to_anthropic_dict()` (or its caller) should add the cache_control key to the final dict in the list. This must happen after sorting to ensure the breakpoint is always on the same tool.

#### 2b. System Message Cache Breakpoint (Provider-Aware)

**File: `src/quantstack/graphs/agent_executor.py`**

Modify `build_system_message()` (line 152) to return a structured `SystemMessage` with `cache_control` — **but only when the active provider supports it**.

The system has a fallback chain: `bedrock → anthropic → openai → ollama`. Structured content blocks with `cache_control` work on Anthropic and Bedrock but will cause errors or silent failures on OpenAI and Ollama.

**Approach:** `build_system_message()` must accept a `provider` parameter (or check the active provider from config). For Anthropic/Bedrock providers:
- `SystemMessage(content=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}])`

For OpenAI/Ollama (or unknown) providers:
- `SystemMessage(content=system_text)` (plain string, no cache_control)

The provider is known at agent executor construction time from `get_model_with_fallback()`. Thread it through to `build_system_message()`.

**Minimum token constraint:** Sonnet requires 1,024-2,048 tokens minimum for caching. Verify that system prompts meet this threshold. If an agent's system prompt is very short, caching will be silently skipped (no error). This is acceptable — short prompts are cheap anyway.

#### 2c. Cache Hit Rate Observability

**File: `src/quantstack/observability/tracing.py` (or wherever LangFuse callbacks are configured)**

After each LLM call, extract `cache_read_input_tokens` and `cache_creation_input_tokens` from the response `usage` object. Log these as custom metrics via the existing LangFuse trace. This enables:
- Verifying the 50%+ cost reduction target
- Detecting cache regressions (e.g., if someone adds non-deterministic content to prompts)
- Trending cache hit rates per agent over time

The Anthropic API response includes these fields in `response.usage`. For Bedrock, check the equivalent fields in the InvokeModel response metadata.

### Bedrock Considerations

Prompt caching is GA on Bedrock — no `anthropic_beta` header needed. The `cache_control` syntax works identically via the InvokeModel API.

**Verification required:** Confirm that `langchain_aws.ChatBedrock` passes `cache_control` content blocks through to the API rather than stripping them. If `ChatBedrock` uses the Converse API internally, `cache_control` may be silently dropped (Converse uses `cachePoint` syntax instead). After implementing, verify that Bedrock-routed calls show `cache_read_input_tokens > 0`, not just Anthropic-direct calls.

### Verification

After deployment, run 3+ consecutive trading cycles and check:
1. `cache_read_input_tokens > 0` in LangFuse traces for the second cycle onward
2. Cache hit rate logs show >80% cache reads for stable agents
3. Total prompt token cost drops 50%+ vs. pre-deployment baseline

---

## Section 3: Fix `search_knowledge_base` to Use RAG (Item 0.3)

### What and Why

The `search_knowledge_base` tool (used by 15+ agents) does not use semantic search. It runs a recency-only SQL query against the old `knowledge_base` table, completely ignoring the user's query text. A proper semantic search function exists in `rag/query.py` at lines 156-203 but is never called from the tool.

### Pre-Implementation Investigation

Before changing the tool, investigate the data state. This is critical because the tool currently queries `knowledge_base` (old table) while the RAG module uses `embeddings` (pgvector table). We need to know:

1. **Row count in each table:** `SELECT COUNT(*) FROM knowledge_base;` and `SELECT COUNT(*) FROM embeddings;`
2. **Data freshness:** `SELECT MAX(created_at) FROM knowledge_base;` and `SELECT MAX(created_at) FROM embeddings;`
3. **Overlap:** Are there entries in `knowledge_base` that don't have corresponding embeddings?

**Decision tree:**
- If `embeddings` has good coverage → proceed with the fix as-is
- If `knowledge_base` has unique data not in `embeddings` → add a migration step to embed and insert that data before switching the tool
- If `embeddings` is empty → fix the embedding pipeline first; this item gets deferred

### How

**File: `src/quantstack/tools/langchain/learning_tools.py`**

Replace the body of `search_knowledge_base()` (lines ~25-30):
- Remove the raw SQL query against `knowledge_base`
- Import and call `rag.query.search_knowledge_base(query=query, n_results=top_k)` instead

**Return schema mapping:** The RAG function returns `{"text", "metadata", "distance", "collection"}` but the tool contract is `{"id", "category", "content", "metadata", "created_at"}`. Map as follows:
- `text` → `content`
- `collection` → `category`
- `metadata` → `metadata` (pass through)
- `metadata.get("id", <generate_from_hash>)` → `id`
- `metadata.get("created_at", None)` → `created_at`
- Add `distance` as a new field for relevance ranking (non-breaking addition)

**Connection management:** The RAG function `search_knowledge_base()` in `query.py` accepts an optional `conn` parameter. Check whether the tool's context provides a raw psycopg2 connection (via `ctx.db`). If so, pass it to avoid opening a second connection. If `ctx.db` is a wrapper, let the RAG function manage its own connection.

**New Ollama dependency:** The RAG function instantiates `OllamaEmbeddingFunction()` for query embedding. This means the tool now requires Ollama to be running. If Ollama is down, the RAG function returns `[]` silently. Add error handling: if the result is empty AND Ollama health check fails, return a clear message ("Embedding service unavailable — results may be incomplete") rather than silently returning nothing.

### Edge Cases

- **Empty embeddings table:** The RAG function returns an empty list. Verify this doesn't error.
- **Query with no semantic matches:** The RAG function likely has a distance threshold — verify and tune if needed.
- **Tool consumers expecting specific fields:** Check if any agent prompt templates parse specific fields from the tool's JSON output. The return schema must remain compatible.
- **Ollama unavailable:** New failure mode. Handle gracefully with informative error message.

---

## Section 4: Add HNSW Index on Embeddings (Item 0.4)

### What and Why

The `embeddings` table has no vector index. Every `search_similar()` call in `rag/query.py` does a full sequential scan using `embedding <=> %s::vector`. At 500+ entries, latency degrades from <10ms to 100ms+. HNSW provides logarithmic-time approximate nearest neighbor search.

### How

**File: `src/quantstack/db.py`**

Add a new migration function (e.g., `_migrate_add_hnsw_index()`) to the `run_migrations()` chain:

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
    ON embeddings USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 100);
```

**Parameter rationale:**
- `m=16` (default): Sufficient for <10k rows. Higher values waste memory with negligible recall improvement at this scale.
- `ef_construction=100` (above default 64): Cheap insurance — build is sub-second at this table size.
- `vector_cosine_ops`: Matches the `<=>` cosine distance operator used in `rag/query.py` line 122.

**Why `db.py` instead of `rag/query.py`:** The existing embedding indexes (`idx_embeddings_collection`, `idx_embeddings_metadata_ticker`) are defined in `rag/query.py`'s `_INIT_SQL`. However, adding the HNSW index there would rebuild it on every process start (even though `IF NOT EXISTS` makes it idempotent, the check still runs). Adding to `db.py`'s migration chain is more explicit: it runs once on upgrade, is consistent with how all other schema changes are managed, and the user specifically requested this location.

**Migration pattern:** Follow the existing convention in `db.py` — each `_migrate_*()` function is idempotent (uses `IF NOT EXISTS`), called from `run_migrations()`, and runs once per database upgrade.

### Considerations

- **Build time at current scale:** Sub-second for <10k rows. No need for `maintenance_work_mem` tuning.
- **`ef_search` tuning:** Consider setting `SET hnsw.ef_search = 100` in the connection initialization for search quality. Default is 40. At this table size, the latency difference is negligible.
- **Future scaling:** If embeddings grow to 100k+, the current parameters remain adequate. Only at 1M+ would we consider increasing `m` to 24.

---

## Section 5: Sentiment Fallback Cleanup (Item 0.5)

### What and Why

When Groq is unavailable (timeout, API error, no headlines), the sentiment collector returns `{"sentiment_score": 0.5, "dominant_sentiment": "neutral", "n_headlines": 0, "source": "default"}`. While the `RuleBasedSynthesizer` in `synthesis.py` correctly checks `n_headlines > 0` before using the score (making the 0.5 functionally inert in that path), the return value is semantically misleading. Future consumers of sentiment data may treat 0.5 as a real signal. Cleaning this up now prevents a class of bugs.

### Pre-Implementation Audit

**Critical:** The signal engine may use `collect_sentiment_alphavantage` (from `sentiment_alphavantage.py`) at runtime, NOT `collect_sentiment` (from `sentiment.py`). The engine config at `engine.py` line ~224 determines which collector runs. Before making changes:

1. **Identify the active collector:** Check `engine.py` to confirm which sentiment collector function is actually called at runtime. It may be `collect_sentiment_alphavantage`, not `collect_sentiment`.
2. **Audit both collectors:** Both `sentiment.py` and `sentiment_alphavantage.py` likely have their own `_safe_defaults()` patterns. Fix whichever is in the runtime path (or both if both are active).
3. **Search for all consumers:** Search for all imports/calls to `collect_sentiment`, `collect_sentiment_alphavantage`, and `_safe_defaults` across the codebase.
4. **Catalog breakage:** For each consumer, check if it accesses `sentiment_score` or other keys directly without checking for their presence.

### How

**Files: `src/quantstack/signal_engine/collectors/sentiment.py` AND `src/quantstack/signal_engine/collectors/sentiment_alphavantage.py`**

Fix the active collector's `_safe_defaults()` to return `{}`. If both collectors are active in different contexts, fix both.

For any consumer found in the audit that directly accesses keys:
- Add a guard: `if sentiment_data:` or `if "sentiment_score" in sentiment_data:` before accessing.
- The synthesis path (`synthesis.py` lines 308-319) uses `sent = sentiment or {}` and `sent.get("n_headlines", 0)`, which safely handles both `{}` and missing keys via `.get()` with defaults. This path is safe.

**Document the contract:** Add a docstring to the active collector's fallback function stating: "Returns `{}` when sentiment data is unavailable. Callers must check for empty dict before accessing fields."

---

## Section 6: Bind All Docker Services to Localhost (Item 0.6)

### What and Why

All Docker services in `docker-compose.yml` expose ports to all interfaces (0.0.0.0). PostgreSQL on port 5434 is the most critical — it contains all trading state, strategy parameters, and knowledge bases. But Ollama (LLM inference), Langfuse (traces with API keys), and the dashboard also have no reason to be network-accessible.

### How

**File: `docker-compose.yml`**

Change all port mappings from `HOST:CONTAINER` to `127.0.0.1:HOST:CONTAINER`:

- PostgreSQL: `"5434:5432"` → `"127.0.0.1:5434:5432"`
- Ollama: `"11434:11434"` → `"127.0.0.1:11434:11434"`
- Langfuse: `"3100:3000"` → `"127.0.0.1:3100:3000"`
- Dashboard: `"8421:8421"` → `"127.0.0.1:8421:8421"`

### Considerations

- **Inter-container communication:** Docker Compose services communicate via the internal Docker network, not via published ports. Binding to localhost affects only host-level access. This change does not affect service-to-service communication.
- **Remote debugging:** If you ever need to access services from another machine (e.g., SSH tunnel from a laptop), use an SSH tunnel: `ssh -L 5434:localhost:5434 server`. Do not revert the binding.

---

## Section 7: Remove All Default Passwords (Item 0.7)

### What and Why

Three hardcoded passwords in `docker-compose.yml`:
1. `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quantstack}` — falls back to "quantstack"
2. `LANGFUSE_DB_PASSWORD: ${LANGFUSE_DB_PASSWORD:-langfuse}` — falls back to "langfuse"
3. `LANGFUSE_INIT_USER_PASSWORD: quantstack123` — fully hardcoded, no env var

If `.env` is missing or incomplete, services start with known passwords. Combined with the port binding issue (Section 6), this is a direct path to data compromise.

### How

#### 7a. Remove Defaults from docker-compose.yml

**File: `docker-compose.yml`**

- Change `${POSTGRES_PASSWORD:-quantstack}` → `${POSTGRES_PASSWORD}` (line 40)
- Change `${LANGFUSE_DB_PASSWORD:-langfuse}` → `${LANGFUSE_DB_PASSWORD}` (line 66)
- Change `LANGFUSE_INIT_USER_PASSWORD: quantstack123` → `LANGFUSE_INIT_USER_PASSWORD: ${LANGFUSE_INIT_USER_PASSWORD}` (line 131)
- Audit the file for any other `:-default` patterns and remove them

Without env vars set, Docker Compose will warn on `docker compose up` that variables are not set, and the services will fail to start (empty password is rejected by PostgreSQL).

#### 7b. Add Validation to start.sh

**File: `start.sh`**

Add a password validation block (near the existing env var checks at lines 48-58):

Required variables to validate:
- `POSTGRES_PASSWORD` — must be set, must NOT equal "quantstack", must be 12+ characters
- `LANGFUSE_DB_PASSWORD` — must be set, must NOT equal "langfuse", must be 12+ characters
- `LANGFUSE_INIT_USER_PASSWORD` — must be set, must NOT equal "quantstack123", must be 12+ characters

If any validation fails, print a clear error message naming the offending variable and exit 1 before starting any services. Example: `"ERROR: POSTGRES_PASSWORD is not set or uses the default value. Set a strong password (12+ characters) in .env"`

#### 7c. Update .env.example

If a `.env.example` or `.env.template` file exists, update it to include the new required variables with placeholder values (e.g., `POSTGRES_PASSWORD=CHANGE_ME`). If no template exists, create one as part of this PR.

### Considerations

- **Existing deployments:** Anyone running QuantStack without these env vars will be broken by this change. The clear error messages from start.sh are the migration path.
- **NEXTAUTH_SECRET and SALT:** Out of scope for this PR. These also have defaults in docker-compose.yml but are lower priority (Langfuse internal auth, not database access). Track for Phase 1.

---

## Testing Strategy

Each item gets at least one regression test. Test file organization follows the existing `tests/unit/` convention.

### Tests by Item

**Item 0.1 (Tool Ordering):**
- Test that `get_tools_for_agent()` returns tools sorted alphabetically by name regardless of registration order.
- Test that `get_tools_for_agent_with_search()` returns both deferred and always-loaded lists sorted, with TOOL_SEARCH_TOOL remaining last.
- Test that TOOL_SEARCH_TOOL is not accidentally sorted into the middle of the list.

**Item 0.2 (Prompt Caching):**
- Test that `build_system_message()` returns a `SystemMessage` with structured content blocks containing `cache_control` when provider is Anthropic/Bedrock.
- Test that `build_system_message()` returns a plain string `SystemMessage` when provider is OpenAI/Ollama.
- Test that the last tool in a converted Anthropic tool list has `cache_control` set.
- Negative test: verify structured content blocks don't break non-Anthropic provider paths.
- Integration test: verify cache_read_input_tokens > 0 after two consecutive calls with identical prompts (requires live API, may be integration-only).

**Item 0.3 (RAG Fix):**
- Test that `search_knowledge_base` tool calls `rag.query.search_knowledge_base()` instead of running raw SQL.
- Test that the query parameter affects results (mock the RAG function to verify it receives the query).
- Test return schema mapping: verify output contains expected fields (id, category, content, metadata, created_at).
- Integration test: insert known embeddings, query by semantic content, verify relevance ranking.

**Item 0.4 (HNSW Index):**
- Test that the migration function executes without error on a fresh database.
- Test idempotency: running the migration twice doesn't error.
- Integration test: verify index exists with `SELECT indexname FROM pg_indexes WHERE tablename = 'embeddings'`.

**Item 0.5 (Sentiment Fallback):**
- Test the active collector's `_safe_defaults()` returns `{}` (may be `sentiment_alphavantage.py`, not `sentiment.py` — audit first).
- Test that the active collector returns `{}` when data source is unavailable (mock client to raise timeout).
- Test that synthesis handles `{}` sentiment input without error.

**Items 0.6-0.7 (Docker Security):**
- These are infrastructure changes. Validate manually per the validation plan (no automated unit tests for docker-compose.yml port bindings).
- start.sh validation can be tested by running it with bad env vars and checking exit code + error message.

---

## Execution Order

All items are independent, but the following order minimizes risk:

1. **Items 0.6 + 0.7 (Docker security)** — Pure config, lowest blast radius
2. **Item 0.1 (Tool ordering)** — Prerequisite for item 0.2's cache effectiveness
3. **Item 0.4 (HNSW index)** — Database migration, run early to be available for item 0.3
4. **Item 0.3 (RAG fix)** — Depends on understanding data state; investigation first
5. **Item 0.2 (Prompt caching)** — Largest impact, depends on item 0.1 being done
6. **Item 0.5 (Sentiment fallback)** — Lowest priority, verify no downstream breakage

---

## File Change Summary

| File | Items | Changes |
|------|-------|---------|
| `src/quantstack/tools/registry.py` | 0.1, 0.2 | Sort tools by name; add cache_control to last tool |
| `src/quantstack/graphs/agent_executor.py` | 0.2 | Structured SystemMessage with cache_control |
| `src/quantstack/observability/tracing.py` | 0.2 | Log cache hit/miss tokens |
| `src/quantstack/tools/langchain/learning_tools.py` | 0.3 | Replace SQL with RAG call |
| `src/quantstack/db.py` | 0.4 | New HNSW index migration |
| `src/quantstack/signal_engine/collectors/sentiment.py` | 0.5 | Return {} instead of fake neutral (if active) |
| `src/quantstack/signal_engine/collectors/sentiment_alphavantage.py` | 0.5 | Return {} instead of fake neutral (if active) |
| `src/quantstack/signal_engine/synthesis.py` | 0.5 | Guard against missing keys (if needed) |
| `docker-compose.yml` | 0.6, 0.7 | Localhost bindings, remove default passwords |
| `start.sh` | 0.7 | Password validation |
| `tests/unit/test_tool_ordering.py` | 0.1 | New: tool sorting tests |
| `tests/unit/test_prompt_caching.py` | 0.2 | New: cache breakpoint tests |
| `tests/unit/test_knowledge_base_tool.py` | 0.3 | New: RAG integration tests |
| `tests/unit/test_hnsw_migration.py` | 0.4 | New: migration idempotency tests |
| `tests/unit/test_sentiment_fallback.py` | 0.5 | New: empty dict fallback tests |
