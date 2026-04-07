# Phase 0: Quick Wins — Deep Plan Spec

**Timeline:** Day 1-2
**Effort:** 1 engineering day
**Annual savings from this phase alone:** $32,000-$45,000

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan. The full audit comprises 164 findings (38 CRITICAL, 60 HIGH, 66 MEDIUM) across three independent audits — CTO Architecture Audit, Principal Quant Scientist Deep Audit, and Deep Operational Audit. The system's overall grade is C-: architecture B+, quant substance D+.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md) — read `00_EXECUTIVE_BRIEFING.md` for the unified scorecard and top-10 dangers.

---

## Objective

Execute 7 items that each take <1 day and have outsized impact. These are the lowest-effort, highest-ROI changes in the entire 164-finding audit.

---

## Items

### 0.1 Deterministic Tool Ordering

- **Finding:** CTO MC1 | **Severity:** CRITICAL | **Effort:** 30 min
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.9](../01_SAFETY_HARDENING.md), [`08_COST_OPTIMIZATION.md` §7.1](../08_COST_OPTIMIZATION.md)
- **Problem:** Tool definitions injected into system prompts in non-deterministic order → prompt cache hash changes between invocations → full-price input tokens on every call. 21 agents × 5-10 min cycles = 30-50% prompt cost waste.
- **Fix:** Sort tool definitions alphabetically by name before injection in `tool_binding.py`. One-line change: `tools = sorted(tools, key=lambda t: t.name)`
- **Key files:** `src/quantstack/tools/registry.py`, tool binding logic
- **Acceptance criteria:**
  - [ ] Tool definitions always injected in deterministic alphabetical order
  - [ ] Identical prompts produce identical cache keys across cycles

### 0.2 Enable Prompt Caching

- **Finding:** CTO MC0c | **Severity:** CRITICAL | **Effort:** 1 hour
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.10](../01_SAFETY_HARDENING.md), [`08_COST_OPTIMIZATION.md` §7.2](../08_COST_OPTIMIZATION.md)
- **Problem:** Claude's prompt caching (90% cost reduction on cached input tokens) not enabled. Zero `cache_control`, `CacheControl`, or `ephemeral` references in codebase. Every call pays full input token price. ~$126/day in system prompt tokens → should be ~$12.60/day.
- **Fix:** Add `cache_control` breakpoints to system message construction: `SystemMessage(content=base, additional_kwargs={"cache_control": {"type": "ephemeral"}})`. For Bedrock: use `anthropic_beta: ["prompt-caching-2024-07-31"]` header.
- **Key files:** Agent executor system message construction, LLM routing layer
- **Acceptance criteria:**
  - [ ] Prompt caching enabled for all Anthropic API and Bedrock calls
  - [ ] System prompt tokens show cache hits in Langfuse traces after first call per cycle
  - [ ] Cost reduction of 50%+ on system prompt tokens verified

### 0.3 Fix `search_knowledge_base` to Use RAG

- **Finding:** CTO MC0 | **Severity:** CRITICAL | **Effort:** 1 hour
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.3](../08_COST_OPTIMIZATION.md)
- **Problem:** `search_knowledge_base` tool (used by 15+ agents) does NOT use pgvector. Runs `SELECT ... ORDER BY created_at DESC LIMIT 5` — recency-only, ignores query parameter entirely. A proper semantic search exists in `rag/query.py:156-203` but is never called.
- **Fix:** Replace the SQL recency query with a call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. One import, one function call change.
- **Key files:** Knowledge base tool implementation, `src/quantstack/rag/query.py`
- **Acceptance criteria:**
  - [ ] `search_knowledge_base` tool uses semantic search via pgvector
  - [ ] Query parameter actually filters results by relevance
  - [ ] Verified: "momentum strategies AAPL" returns momentum-related entries, not most recent

### 0.4 Add HNSW Index on Embeddings

- **Finding:** CTO MC0b | **Severity:** HIGH | **Effort:** 30 min
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.4](../08_COST_OPTIMIZATION.md)
- **Problem:** Embeddings table has no vector index. Every `search_similar()` call does full sequential scan. At 500+ entries, latency degrades from <10ms to 100ms+.
- **Fix:** `CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`
- **Key files:** Database migrations in `src/quantstack/db.py`
- **Acceptance criteria:**
  - [ ] HNSW index created on embeddings table
  - [ ] Semantic search queries <10ms regardless of table size

### 0.5 Sentiment Fallback: `{}` Not 0.5

- **Finding:** CTO DH3 | **Severity:** HIGH | **Effort:** 30 min
- **Audit section:** [`09_DATA_SIGNALS.md` §8.5](../09_DATA_SIGNALS.md)
- **Problem:** When no headlines available or Groq times out, sentiment returns 0.5 (neutral). Fake neutral carries same weight as real signal in synthesis.
- **Fix:** Return `{}` instead of fake neutral when data unavailable. Let synthesis redistribute weight to other active collectors.
- **Key files:** Sentiment collector in signal engine
- **Acceptance criteria:**
  - [ ] Unavailable sentiment returns `{}`, not 0.5
  - [ ] Synthesis correctly redistributes weight when sentiment absent

### 0.6 Bind PostgreSQL to Localhost

- **Finding:** QS-I2 | **Severity:** CRITICAL | **Effort:** 15 min
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.5](../01_SAFETY_HARDENING.md)
- **Problem:** `docker-compose.yml` exposes port `5434:5432` to all interfaces. DB accessible from network.
- **Fix:** Change to `127.0.0.1:5434:5432` in `docker-compose.yml`
- **Key files:** `docker-compose.yml`
- **Acceptance criteria:**
  - [ ] PostgreSQL port bound to 127.0.0.1 only

### 0.7 Remove Default DB Password

- **Finding:** QS-I2 | **Severity:** CRITICAL | **Effort:** 15 min
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.5](../01_SAFETY_HARDENING.md)
- **Problem:** `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quantstack}` — if `.env` not set, password is "quantstack".
- **Fix:** Remove default fallback. Add `.env` validation to `start.sh` — fail if passwords are defaults.
- **Key files:** `docker-compose.yml`, `start.sh`
- **Acceptance criteria:**
  - [ ] Default password "quantstack" rejected at startup
  - [ ] `.env` validation prevents startup with insecure defaults

---

## Dependencies

None — this is the first phase. All items are independent of each other and can be executed in parallel.

---

## Risks

- **0.2 (prompt caching):** Verify that the 5-min trading cycle aligns with Claude's 5-min cache TTL. If cycle cadence varies, cache hit rate may be lower than expected.
- **0.3 (RAG fix):** The semantic search function in `rag/query.py` may have stale embeddings if the embedding pipeline hasn't been running. Verify embedding coverage before declaring done.

---

## Validation Plan

1. After 0.1+0.2: Compare Langfuse cost traces for 10 cycles before/after. Expect 50%+ reduction in system prompt token cost.
2. After 0.3+0.4: Query "momentum strategies AAPL" and verify semantic relevance of results.
3. After 0.5: Run signal engine with Groq intentionally down. Verify sentiment collector returns `{}`.
4. After 0.6+0.7: `docker compose up` without `.env` should fail. With `.env`, verify DB only listens on localhost.
