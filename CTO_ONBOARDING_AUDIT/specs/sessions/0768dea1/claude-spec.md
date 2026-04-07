# Phase 0: Quick Wins — Synthesized Specification

**Timeline:** Day 1-2 | **Effort:** 1 engineering day | **Rollout:** Single PR

---

## Context

QuantStack CTO Onboarding Audit identified 164 findings (38 CRITICAL, 60 HIGH, 66 MEDIUM). Phase 0 addresses 7 items that each take <1 day individually but collectively deliver $32,000-$45,000/year in savings (primarily from prompt caching cost reduction).

All items are independent and can be developed in parallel. Ship as a single PR.

---

## Item 0.1: Deterministic Tool Ordering

**Severity:** CRITICAL | **Effort:** 30 min

**Problem:** `TOOL_REGISTRY` in `registry.py` is a flat dict with insertion-order iteration. Tools are added from multiple modules (`_system_tools`, `_alert_tools`, etc.) making order non-deterministic across restarts. Since Anthropic's cache key computation starts with the `tools` field, different tool order between calls means cache misses on ALL levels (tools, system, messages). 21 agents × 5-10 min cycles = 30-50% prompt cost waste.

**Fix:** Sort tools by name in `get_tools_for_agent()` and `get_tools_for_agent_with_search()` before returning.

**Key files:** `src/quantstack/tools/registry.py` (lines 320, 338)

**Acceptance:**
- Tool definitions always injected in deterministic alphabetical order
- Identical prompts produce identical cache keys across cycles

---

## Item 0.2: Enable Prompt Caching with Explicit Breakpoints

**Severity:** CRITICAL | **Effort:** 1 hour

**Problem:** Zero `cache_control` references in codebase. Every call pays full input token price. ~$126/day in system prompt tokens.

**Fix:** Add explicit `cache_control` breakpoints at two levels:
1. **Tool level:** Add `cache_control: {"type": "ephemeral"}` to the last tool in the sorted tool list via `convert_to_anthropic_tool()`.
2. **System message level:** Convert `build_system_message()` output to structured content blocks with `cache_control`.

**Key constraints from research:**
- Minimum token thresholds: Sonnet needs 1,024-2,048 tokens, Opus needs 4,096 tokens.
- Cache TTL: 5 min default — aligns with QuantStack's 5-min trading cycle.
- Bedrock: No `anthropic_beta` header needed (GA). Use same `cache_control` syntax.
- LangChain: `SystemMessage(content=[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}])` passes through directly.

**Observability:** Add explicit logging of `cache_read_input_tokens` and `cache_creation_input_tokens` per agent per cycle via the existing LangFuse callback. Enables trending without manual trace inspection.

**Key files:** `src/quantstack/graphs/agent_executor.py` (build_system_message, line 152), `src/quantstack/tools/registry.py` (tool_to_anthropic_dict, line 369), `src/quantstack/llm/provider.py`

**Acceptance:**
- Prompt caching enabled for all Anthropic API and Bedrock calls
- System prompt tokens show cache hits in Langfuse traces after first call per cycle
- Cache hit rate logged per agent per cycle
- 50%+ reduction in system prompt token cost verified

---

## Item 0.3: Fix `search_knowledge_base` to Use RAG

**Severity:** CRITICAL | **Effort:** 1 hour

**Problem:** Tool at `learning_tools.py:14` runs `SELECT ... ORDER BY created_at DESC LIMIT 5` against the old `knowledge_base` table. Ignores query parameter entirely. Semantic search exists in `rag/query.py:156-203` but is never called.

**Pre-fix investigation required:** Check both `knowledge_base` and `embeddings` tables for:
- Row counts
- Data freshness (most recent `created_at`)
- Whether `knowledge_base` has unique data not mirrored in `embeddings`

If `knowledge_base` has unique data, we need a data migration plan before switching the tool.

**Fix:** Replace SQL query in `learning_tools.py` with call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`.

**Key files:** `src/quantstack/tools/langchain/learning_tools.py` (lines 14-30), `src/quantstack/rag/query.py` (lines 156-203)

**Acceptance:**
- `search_knowledge_base` tool uses semantic search via pgvector
- Query parameter actually filters results by relevance
- "momentum strategies AAPL" returns momentum-related entries, not most recent

---

## Item 0.4: Add HNSW Index on Embeddings

**Severity:** HIGH | **Effort:** 30 min

**Problem:** Embeddings table has no vector index. Every `search_similar()` call does full sequential scan.

**Fix:** Add migration to `run_migrations()` in `db.py`:
```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
    ON embeddings USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 100);
```

Parameters chosen per research: `m=16` (default, sufficient for <10k rows), `ef_construction=100` (slightly above default 64, instant build at this scale).

**Key files:** `src/quantstack/db.py` (run_migrations function)

**Acceptance:**
- HNSW index created on embeddings table
- Semantic search queries <10ms regardless of table size

---

## Item 0.5: Sentiment Fallback Cleanup

**Severity:** HIGH | **Effort:** 30 min

**Problem:** When Groq is unavailable, sentiment collector returns `{"sentiment_score": 0.5, ...}`. While synthesis already handles this correctly (checks `n_headlines > 0` before using the score), the 0.5 return is semantically misleading and could confuse future consumers.

**Fix:**
1. Audit all consumers of sentiment collector output (not just synthesis)
2. If safe, change `_safe_defaults()` to return `{}` instead of the fake neutral dict
3. Add defensive handling in any consumer that expects specific keys
4. Document the "empty dict = unavailable" contract

**Key files:** `src/quantstack/signal_engine/collectors/sentiment.py` (lines 156-162), `src/quantstack/signal_engine/synthesis.py` (lines 308-319)

**Acceptance:**
- Unavailable sentiment returns `{}`, not 0.5
- Synthesis correctly redistributes weight when sentiment absent
- No other consumers break

---

## Item 0.6: Bind All Docker Services to Localhost

**Severity:** CRITICAL | **Effort:** 15 min

**Problem:** All Docker services expose ports to all interfaces (0.0.0.0).

**Fix:** Change all port bindings in `docker-compose.yml`:
- PostgreSQL: `127.0.0.1:5434:5432`
- Ollama: `127.0.0.1:11434:11434`
- Langfuse: `127.0.0.1:3100:3000`
- Dashboard: `127.0.0.1:8421:8421`

**Key files:** `docker-compose.yml`

**Acceptance:**
- All services bound to 127.0.0.1 only
- `docker compose up` works correctly with localhost bindings

---

## Item 0.7: Remove All Default Passwords

**Severity:** CRITICAL | **Effort:** 15 min

**Problem:** Three hardcoded default passwords:
- `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quantstack}`
- `LANGFUSE_DB_PASSWORD: ${LANGFUSE_DB_PASSWORD:-langfuse}`
- `LANGFUSE_INIT_USER_PASSWORD: quantstack123`

**Fix:**
1. Remove all `:-default` fallbacks from docker-compose.yml
2. Move `LANGFUSE_INIT_USER_PASSWORD` to env var reference
3. Add validation to `start.sh`:
   - Fail if POSTGRES_PASSWORD not set or equals known defaults ("quantstack")
   - Fail if LANGFUSE_DB_PASSWORD not set or equals "langfuse"
   - Fail if LANGFUSE_INIT_USER_PASSWORD not set or equals "quantstack123"

**Key files:** `docker-compose.yml`, `start.sh`

**Acceptance:**
- No hardcoded passwords in docker-compose.yml
- `start.sh` validates all passwords are set and non-default
- `docker compose up` without proper `.env` fails with clear error message

---

## Dependencies

None between items. All are independent. Ship as single PR.

---

## Risks

1. **Item 0.2 (prompt caching):** Cache TTL is 5 min. If trading cycle cadence varies, cache hit rate drops. Monitor via the new logging.
2. **Item 0.3 (RAG fix):** If `embeddings` table has no data or stale data, the fix makes the tool worse. Investigate table state first.
3. **Item 0.5 (sentiment):** Changing return type from dict-with-keys to `{}` may break consumers. Audit first.
4. **Item 0.7 (passwords):** Existing `.env` files on deployed systems may lack the new required variables. Add clear error messages.

---

## Validation Plan

1. After 0.1+0.2: Compare Langfuse cost traces for 10 cycles before/after. Expect 50%+ reduction in system prompt token cost. Check new cache hit rate logs.
2. After 0.3+0.4: Query "momentum strategies AAPL" and verify semantic relevance.
3. After 0.5: Run signal engine with Groq intentionally down. Verify `{}` return and correct synthesis.
4. After 0.6+0.7: `docker compose up` without `.env` should fail. With `.env`, verify all services only on localhost.
