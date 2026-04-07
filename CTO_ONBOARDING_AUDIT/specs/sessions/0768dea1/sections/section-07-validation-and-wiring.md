# Section 07: Validation and Wiring

## Purpose

This section is the final gate before merging the Phase 0 Quick Wins PR. It validates that all six preceding sections (Docker security, tool ordering, prompt caching, HNSW index, RAG fix, sentiment fallback) work correctly in isolation AND together. No new production code is written here — the work is running tests, performing manual checks, hunting for regressions, and confirming the claimed cost/performance improvements.

This section cannot begin until sections 01 through 06 are complete.

---

## Dependencies

| Section | What It Delivered | What to Validate |
|---------|-------------------|------------------|
| section-01-docker-security | `docker-compose.yml` bound to 127.0.0.1, default passwords removed, `start.sh` password validation | Ports are localhost-only, password rejection works, services start with valid env |
| section-02-tool-ordering | `get_tools_for_agent()` and `get_tools_for_agent_with_search()` return sorted tools | Tool order is deterministic across restarts, TOOL_SEARCH_TOOL stays last |
| section-03-prompt-caching | `cache_control` breakpoints on tools and system messages, cache hit rate logging | Cache tokens appear in LLM responses, cost reduction measurable via Langfuse |
| section-04-hnsw-index | HNSW migration in `db.py`'s `run_migrations()` | Index exists after migration, query latency improved, idempotent re-run |
| section-05-rag-fix | `search_knowledge_base` tool calls RAG instead of raw SQL | Semantic results returned, schema matches tool contract, Ollama-down handled |
| section-06-sentiment-fallback | `_safe_defaults()` returns `{}`, synthesis handles empty dict | No KeyError anywhere in the signal pipeline, synthesis completes cleanly |

---

## Tests to Run

### Step 1: Full Unit Test Suite

Run the entire unit test suite. Every test from sections 01-06 must pass, plus all pre-existing tests must not regress.

```bash
uv run pytest tests/unit/ -v --tb=short
```

**Expected outcome:** Zero failures, zero errors. If any pre-existing test fails, investigate whether the Phase 0 changes caused the regression before proceeding.

**Specific test files to confirm are present and passing:**

- `tests/unit/test_tool_ordering.py` — tool sorting, TOOL_SEARCH_TOOL position, stability
- `tests/unit/test_prompt_caching.py` — provider-aware SystemMessage construction, tool cache_control placement
- `tests/unit/test_knowledge_base_tool.py` — RAG integration, schema mapping, Ollama failure handling
- `tests/unit/test_hnsw_migration.py` — index creation, idempotency, correct operator class and parameters
- `tests/unit/test_sentiment_fallback.py` — empty dict fallback, synthesis handles `{}` and `None`
- `tests/unit/test_startup_validation.py` — start.sh password rejection for missing, default, and short passwords

### Step 2: Integration Tests (If Applicable)

If a local PostgreSQL instance is available (or Docker services are running), run integration tests:

```bash
uv run pytest tests/integration/ -v --tb=short
```

Key integration tests to watch:

- `tests/integration/test_knowledge_base_rag.py` — semantic search returns relevant results, query parameter affects ranking
- HNSW index verification — confirm `idx_embeddings_hnsw` exists in `pg_indexes`

If no database is available, these tests should be skipped gracefully (not fail). Verify skip markers are in place.

### Step 3: Cross-Section Interaction Tests

These are not separate test files — they are scenarios to verify manually or via ad-hoc test scripts. The concern is that changes in one section may interfere with another.

**Tool ordering + prompt caching interaction:**
- Verify that after sorting tools (section 02), the `cache_control` key is placed on the correct (last) tool in the Anthropic dict list (section 03).
- If tools are not sorted before cache_control is applied, the breakpoint lands on a different tool each restart, defeating caching.
- Validation: Call `get_tools_for_agent_with_search()` twice in the same process and across two process starts. The tool list must be byte-identical both times, and `cache_control` must be on the last regular tool (before TOOL_SEARCH_TOOL).

**HNSW index + RAG fix interaction:**
- The RAG function in `rag/query.py` uses `embedding <=> %s::vector` (cosine distance). The HNSW index (section 04) was created with `vector_cosine_ops`. Confirm the index is actually used by the query planner.
- Validation: Run `EXPLAIN ANALYZE` on a representative vector search query and confirm the plan includes an index scan on `idx_embeddings_hnsw`, not a sequential scan.

```sql
EXPLAIN ANALYZE 
SELECT id, text, embedding <=> '[0.1, 0.2, ...]'::vector AS distance 
FROM embeddings 
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector 
LIMIT 5;
```

**Sentiment fallback + synthesis interaction:**
- With the collector now returning `{}` on failure, confirm the full signal pipeline completes without error. The synthesis path in `synthesis.py` uses `.get()` with defaults, but verify no other consumer in the pipeline accesses sentiment fields directly.
- Validation: Simulate a sentiment failure (mock Groq timeout) and run a full signal collection cycle. The `SignalBrief` should be produced with sentiment contribution = 0.0, no errors.

---

## Manual Validation Checklist

Each item below must be verified by a human (or scripted check) before the PR is approved.

### Docker Security (Section 01)

1. **Port binding verification:**
   ```bash
   docker compose up -d
   docker compose ps
   ```
   Confirm all published ports show `127.0.0.1:PORT->PORT` format, not `0.0.0.0:PORT->PORT`.

2. **Password rejection:**
   ```bash
   # Unset passwords — should fail
   unset POSTGRES_PASSWORD && ./start.sh
   # Expected: exit code 1, error names POSTGRES_PASSWORD

   # Default password — should fail
   POSTGRES_PASSWORD=quantstack ./start.sh
   # Expected: exit code 1, error mentions default value

   # Short password — should fail
   POSTGRES_PASSWORD=short ./start.sh
   # Expected: exit code 1, error mentions minimum length

   # Valid password — should pass validation
   POSTGRES_PASSWORD=mysecurepassword123 LANGFUSE_DB_PASSWORD=anothersecure123 LANGFUSE_INIT_USER_PASSWORD=thirdsecure12345 ./start.sh
   # Expected: passes password validation (may fail on Docker if not running, but no password error)
   ```

3. **Inter-container communication:**
   After `docker compose up`, verify that the trading graph container can reach PostgreSQL, Ollama, and Langfuse over the Docker internal network (these use container names, not published ports).

### Prompt Caching Cost Verification (Sections 02 + 03)

This is the highest-value validation — the implementation plan claims $32,000-$45,000/year savings.

1. **Baseline measurement:** Before deploying the caching changes, record the current prompt token cost per trading cycle from Langfuse traces. Note the date/time range and total input tokens.

2. **Post-deployment measurement:** After deploying, let 3+ consecutive trading cycles complete. Then:
   - Open Langfuse and filter traces to the post-deployment window.
   - For each LLM call trace, check for `cache_read_input_tokens` and `cache_creation_input_tokens` in the usage metadata.
   - On the first cycle, expect `cache_creation_input_tokens > 0` and `cache_read_input_tokens = 0` (cache is being built).
   - On the second cycle onward, expect `cache_read_input_tokens > 0` (cache hits).
   - Target: >80% cache read rate for stable agents (those whose prompts don't change between cycles).

3. **Bedrock-specific check:** If the primary LLM route is Bedrock (which it is for this system), verify that `ChatBedrock` actually passes `cache_control` through to the API. Check whether `cache_read_input_tokens` appears in Bedrock-routed traces. If it does not, the Converse API may be stripping cache_control — this would require switching to the InvokeModel API path or using `cachePoint` syntax instead. Document the finding either way.

4. **Cost comparison:** Compare total input token cost for 24 hours pre-deployment vs. 24 hours post-deployment. Expected reduction: 50%+ on prompt token spend.

### Semantic Search Verification (Sections 04 + 05)

1. **Index existence:**
   ```sql
   SELECT indexname, indexdef 
   FROM pg_indexes 
   WHERE tablename = 'embeddings' AND indexname = 'idx_embeddings_hnsw';
   ```
   Confirm the index exists with `vector_cosine_ops`, `m=16`, `ef_construction=100`.

2. **Semantic relevance test:**
   - Insert two known entries into the knowledge base via the RAG pipeline: one about "momentum crossover strategy" and one about "dividend yield screening."
   - Call the `search_knowledge_base` tool with query "momentum."
   - Verify the momentum entry ranks higher (lower distance) than the dividend entry.
   - This confirms the tool is using semantic search, not recency-only SQL.

3. **Ollama-down scenario:**
   - Stop the Ollama container: `docker compose stop ollama`
   - Call `search_knowledge_base` tool.
   - Verify the tool returns an informative error message ("Embedding service unavailable") rather than crashing or returning silent empty results.
   - Restart Ollama: `docker compose start ollama`

### Sentiment Pipeline (Section 06)

1. **Simulated failure:**
   - Set `GROQ_API_KEY` to an invalid value (or unset it).
   - Trigger a signal collection cycle that includes sentiment.
   - Verify: the collector returns `{}`, synthesis completes, `SignalBrief` is produced with sentiment contribution = 0.0.
   - No `KeyError`, no crash, no fake 0.5 neutral score in the output.

2. **Normal operation:**
   - Restore valid `GROQ_API_KEY`.
   - Trigger a signal collection cycle.
   - Verify: sentiment data returns normally, synthesis uses it, `SignalBrief` reflects real sentiment scores.

---

## Regression Checklist

These are specific regressions to watch for, based on the nature of the changes:

| Risk | How It Breaks | How to Detect |
|------|---------------|---------------|
| Tool sort breaks agent that relies on tool position | No agent should depend on tool order, but verify by running one full trading cycle | Trading graph completes without tool-not-found errors |
| cache_control breaks non-Anthropic providers | OpenAI/Ollama provider paths receive structured content they cannot parse | Run a cycle with Ollama as primary LLM; verify no serialization errors |
| RAG schema change breaks agent prompt parsing | An agent's prompt template expects `content` field but receives `text` (or vice versa) | Search for all consumers of `search_knowledge_base` output in agent prompts; verify field names match |
| HNSW index slows writes | Index maintenance on INSERT adds latency to embedding writes | Benchmark: insert 100 embeddings, confirm wall time < 5 seconds |
| Empty sentiment breaks downstream aggregation | A module computes `mean([])` or indexes into empty dict | Run full signal pipeline with sentiment disabled; no exceptions |
| Password validation blocks CI/CD | CI environment does not set the new required env vars | Check CI config for `POSTGRES_PASSWORD`, `LANGFUSE_DB_PASSWORD`, `LANGFUSE_INIT_USER_PASSWORD` |

---

## Final Sign-Off Criteria

The PR is ready to merge when ALL of the following are true:

1. `uv run pytest tests/unit/ -v` passes with zero failures
2. `uv run pytest tests/integration/ -v` passes (or all DB-dependent tests are cleanly skipped)
3. All manual validation checklist items above are verified
4. No regressions in existing functionality (full trading cycle completes in paper mode)
5. Langfuse traces confirm `cache_read_input_tokens > 0` on second+ cycle (prompt caching is working)
6. `docker-compose.yml` shows `127.0.0.1` prefix on all port bindings
7. `start.sh` rejects missing, default, and short passwords with clear error messages
8. `search_knowledge_base` returns semantically relevant results, not recency-only
9. Sentiment fallback returns `{}` (not fake neutral), and synthesis handles it cleanly

Once all criteria are met, squash-merge the PR with commit message:

```
fix: Phase 0 quick wins — prompt caching, tool ordering, RAG, HNSW, Docker security, sentiment cleanup

Addresses 7 findings from CTO onboarding audit (items 0.1-0.7).
Expected impact: 50%+ prompt token cost reduction via caching,
semantic search for knowledge base, hardened Docker config.
```
