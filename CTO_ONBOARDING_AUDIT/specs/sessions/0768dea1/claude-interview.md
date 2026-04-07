# Phase 0 Quick Wins — Interview Transcript

## Q1: Sentiment fallback (item 0.5) — change to {} or skip?

**Context:** Research found synthesis already checks `n_headlines > 0` before using `sentiment_score`, making the 0.5 fallback functionally inert.

**Answer:** Deferred to staff-level judgment. 

**Decision:** Keep the item but scope it correctly. The 0.5 return is semantically misleading even though synthesis handles it. Change the fallback to return `{}` with clear documentation. However, audit any other consumers of the sentiment collector output before changing the return schema. If other code depends on `sentiment_score` always being present, add defensive handling there first.

## Q2: Prompt caching strategy (item 0.2)

**Question:** Explicit breakpoints on tools + system, or automatic top-level caching?

**Answer:** Explicit breakpoints on tools + system (Recommended). Maximum cache reuse — tools and system cached separately. More code but optimal savings.

## Q3: Docker security scope (items 0.6, 0.7)

**Question:** Just PostgreSQL, or all services to localhost + all passwords to env vars?

**Answer:** All services to localhost + all passwords to env vars. Comprehensive security fix — bind all ports to 127.0.0.1, no hardcoded secrets.

## Q4: Knowledge base data state (item 0.3)

**Question:** Are both `knowledge_base` and `embeddings` tables actively populated?

**Answer:** Not sure — investigate. Check both tables for row counts and freshness before deciding.

**Action:** The plan must include a data investigation step before implementing the RAG fix.

## Q5: HNSW index deployment (item 0.4)

**Question:** Add to `run_migrations()` in db.py, or `_INIT_SQL` in rag/query.py?

**Answer:** Add to `run_migrations()` in db.py. Consistent with existing migration pattern, runs once on upgrade.

## Q6: Cache hit rate observability

**Question:** Langfuse traces sufficient, or add explicit logging?

**Answer:** Deferred to staff-level judgment.

**Decision:** Add explicit cache hit rate logging. Langfuse traces are good for debugging but not for trending. Log `cache_read_input_tokens` and `cache_creation_input_tokens` per agent per cycle via the existing LangFuse callback handler. This lets us verify the 50%+ cost reduction target without manually inspecting traces. One-time setup cost, permanent observability benefit.

## Q7: Rollout strategy

**Question:** One PR per item, grouped PRs, or single PR?

**Answer:** Single PR for all. Fastest to ship, all items are independent.
