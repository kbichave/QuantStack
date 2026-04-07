# 07 — Cost Optimization: Stop Paying 10x What You Should

**Priority:** P1-P2 (some items are in Section 01 as P0)
**Timeline:** Week 3-5
**Gate:** Prompt cache hit rate > 80%. Per-agent cost tracking live. Context compaction at merge points.

---

## Why This Section Matters

QuantStack runs 21 agents across 3 graphs, every 5-10 minutes, 24/7. At current (uncached) rates, system prompt tokens alone cost ~$126/day. With prompt caching enabled and tool ordering fixed, that drops to ~$12.60/day. The cost optimizations in this section pay for themselves in the first week.

Additionally, the `search_knowledge_base` tool bypasses the actual RAG pipeline, and the embeddings table has no vector index. These aren't just performance issues — they cause agents to make decisions on irrelevant context.

---

## 7.1 Deterministic Tool Ordering (Already in Section 01)

**Finding ID:** CTO MC1
**Cross-reference:** Section 01, item 1.9
**Annual savings:** $2,000-$5,000

Sort tool definitions alphabetically by name before injection. One-line fix. Prevents prompt cache breakage from ordering variance.

---

## 7.2 Enable Prompt Caching (Already in Section 01)

**Finding ID:** CTO MC0c
**Cross-reference:** Section 01, item 1.10
**Annual savings:** $30,000-$40,000

Add `cache_control` breakpoints to system messages. Our 5-min trading cycle perfectly matches Claude's 5-min cache TTL.

---

## 7.3 Fix `search_knowledge_base` — Actually Use RAG

**Finding ID:** CTO MC0
**Severity:** CRITICAL
**Effort:** 1 hour

### The Problem

The `search_knowledge_base` tool — used by 15+ agents — does NOT use pgvector. It runs:

```sql
SELECT id, category, content, metadata, created_at
FROM knowledge_base
ORDER BY created_at DESC
LIMIT 5
```

This is recency-only retrieval. The `query` parameter is accepted but never used. An agent asking "What momentum strategies failed on AAPL?" gets the 5 most recent entries regardless of topic.

A proper semantic search function exists in `rag/query.py:156-203` (`search_knowledge_base()`) but the LangChain tool never calls it.

### The Fix

Replace the SQL recency query with a call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. One import, one function call change.

### Acceptance Criteria

- [ ] `search_knowledge_base` tool uses semantic search via pgvector
- [ ] Query parameter actually filters results by relevance
- [ ] Verified: "momentum strategies AAPL" returns momentum-related entries, not most recent

---

## 7.4 Add HNSW Index on Embeddings Table

**Finding ID:** CTO MC0b
**Severity:** HIGH
**Effort:** 30 minutes

### The Problem

The embeddings table has no vector index. Every `search_similar()` call does a full sequential scan. At 500+ entries, query latency degrades from <10ms to 100ms+.

### The Fix

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

### Acceptance Criteria

- [ ] HNSW index created on embeddings table
- [ ] Semantic search queries < 10ms regardless of table size

---

## 7.5 Context Compaction at Merge Points

**Finding IDs:** CTO MC2, CTO OC-3
**Severity:** HIGH
**Effort:** 3 days

### The Problem

Each agent accumulates context. By the time the 10th agent in Trading Graph runs, the context window contains all prior agents' outputs. The only guard is reactive 150K char pruning that drops oldest tool rounds — potentially losing critical data.

### The Fix

Add compaction nodes after parallel merge points (`merge_parallel`, `merge_pre_execution`):

| Merge Point | Input | Compaction | Output |
|-------------|-------|------------|--------|
| `merge_parallel` | position_review (full) + entry_scan (full) | Haiku summarizes both | Structured brief: exits[], entries[], risks[] |
| `merge_pre_execution` | portfolio_review (full) + options_analysis (full) | Haiku summarizes both | Structured brief: portfolio_state, options_opportunities[] |

**Cost:** ~$0.08/MTok (haiku cache read) for compaction vs. carrying 50K+ tokens of raw output at $3/MTok (sonnet input) for downstream agents.

**Savings:** 40-60% context size reduction at merge points.

### Acceptance Criteria

- [ ] Compaction nodes added at both merge points
- [ ] Downstream agents receive structured briefs, not raw outputs
- [ ] Context size at `execute_entries` reduced by 40%+

---

## 7.6 Per-Agent Cost Tracking

**Finding ID:** CTO LH4, CTO (MEDIUM)
**Severity:** MEDIUM
**Effort:** 2 days

### The Problem

No agent has a token budget. No cost tracking per agent per cycle. Can't answer: "Which agent costs the most? Is that agent worth it?" A runaway agent could consume $50+ in a single cycle.

### The Fix

| Step | Action |
|------|--------|
| 1 | Log `tokens_in`, `tokens_out`, `model_id`, `cost_estimate` per LLM call |
| 2 | Aggregate by agent/day/graph in a `llm_costs` table |
| 3 | Add `max_tokens_budget` to `AgentConfig` in `agents.yaml` |
| 4 | When budget exhausted: graceful exit at next node boundary |
| 5 | Alert when any agent exceeds daily cost threshold |

### Acceptance Criteria

- [ ] Per-agent per-cycle cost logged
- [ ] Daily cost dashboard or query available
- [ ] Token budget per agent configured and enforced

---

## 7.7 Agent Tier Classification Fix

**Finding ID:** CTO LH1
**Severity:** HIGH
**Effort:** 1 day

### The Problem

Agents not matching naming conventions default to the most expensive "assistant" tier (Sonnet). 10+ agents fall to this default, costing more than necessary. `daily_planner`, `position_monitor`, `exit_evaluator` don't need Sonnet — Haiku is sufficient for their structured output tasks.

### The Fix

| Step | Action |
|------|--------|
| 1 | Explicit tier assignment in `agents.yaml` per agent |
| 2 | Remove naming-convention-based classification |
| 3 | Log a warning if any agent falls to default tier |

**Recommended tiers:**

| Agent | Current (default) | Recommended | Savings |
|-------|------------------|-------------|---------|
| hypothesis_generation | Sonnet | Sonnet (keep — needs creativity) | — |
| trade_debater | Sonnet | Sonnet (keep — complex reasoning) | — |
| fund_manager | Sonnet | Sonnet (keep — high-stakes decisions) | — |
| daily_planner | Sonnet (default) | Haiku | 80% per call |
| position_monitor | Sonnet (default) | Haiku | 80% per call |
| exit_evaluator | Sonnet (default) | Haiku | 80% per call |
| market_intel | Sonnet (default) | Haiku | 80% per call |
| health_monitor | Sonnet (default) | Haiku | 80% per call |
| executor | Sonnet (default) | Haiku | 80% per call |

### Acceptance Criteria

- [ ] Every agent has explicit tier in `agents.yaml`
- [ ] No agent falls to default tier
- [ ] Cost reduction measured after migration

---

## 7.8 Per-Agent Temperature Configuration

**Finding ID:** CTO LC3
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

All 21 agents use `temperature=0.0`. This kills diversity in research (where novel hypotheses are valuable) and provides no benefit for tasks that are already deterministic.

### The Fix

| Agent Type | Temperature | Rationale |
|-----------|-------------|-----------|
| Hypothesis generation | 0.7 | Needs novel ideas |
| Trade debate | 0.3-0.5 | Needs some creativity for counterarguments |
| Community intel | 0.3 | Needs varied search queries |
| Validation/parsing | 0.0 | Deterministic output |
| Execution | 0.0 | Must be predictable |

### Acceptance Criteria

- [ ] Per-agent temperature configured in `agents.yaml`
- [ ] Research agents use higher temperature than execution agents

---

## 7.9 EWF Analysis Deduplication

**Finding ID:** CTO LH5
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

Three agents (daily_planner, position_monitor, trade_debater) each independently call `get_ewf_analysis` for the same symbol in the same cycle. 3x redundant API calls.

### The Fix

Fetch EWF once per symbol per cycle, store in graph state, share across agents.

### Acceptance Criteria

- [ ] EWF fetched once per symbol per cycle
- [ ] All agents read from shared state, not individual API calls

---

## 7.10 Remove Hardcoded Model Strings

**Finding ID:** CTO LH3
**Severity:** HIGH
**Effort:** 1 day

### The Problem

6+ locations hardcode model names instead of using `get_chat_model()`:

| Location | Hardcoded | Should Be |
|----------|----------|-----------|
| `tool_search_compat.py:21` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `trading/nodes.py:843` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `trade_evaluator.py` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `mem0_client.py` | `gpt-4o-mini` | `get_chat_model("memory")` |
| `hypothesis_agent.py` | `groq/llama-3.3-70b-versatile` | `get_chat_model("bulk")` |
| `opro_loop.py` | `groq/llama-3.3-70b-versatile` | `get_chat_model("bulk")` |

### The Fix

Replace all hardcoded strings with `get_chat_model()` calls using appropriate tier names.

### Acceptance Criteria

- [ ] Zero hardcoded model strings in codebase
- [ ] All model references go through `get_chat_model()`
- [ ] Changing `LLM_PROVIDER` env var switches ALL model calls

---

## 7.11 LLM Provider Runtime Fallback

**Finding IDs:** CTO LH2, CTO OC-4
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Provider availability checked at startup only. If Bedrock returns 429 mid-execution, no fallback. The Anthropic API credit exhaustion on 2026-04-05 blocked the EWF analyzer for hours.

### The Fix

Wrap `get_chat_model()` with retry-failover chain:

| Step | Action |
|------|--------|
| 1 | On 429/500/timeout: retry same provider 2x |
| 2 | If still failing: switch to next provider (Anthropic → Bedrock → Groq) |
| 3 | Cooldown failed provider for 5 minutes |
| 4 | Log provider switches as operational events |

### Acceptance Criteria

- [ ] Mid-session provider failure triggers automatic fallback
- [ ] Provider cooldown prevents hammering a failing service
- [ ] No single-provider outage can halt the system

---

## 7.12 Memory Temporal Decay

**Finding ID:** CTO MC3
**Severity:** MEDIUM
**Effort:** 1 day

### The Problem

Memory entries about market reads, strategy states, and session handoffs have no expiry. The 2026-04-04 EWF market reads (already flagged as "low-trust") persist indefinitely, wasting tokens on outdated information.

### The Fix

| Memory Type | TTL | Action on Expiry |
|------------|-----|-----------------|
| Market reads (EWF, sentiment) | 7 days | Archive |
| Session handoffs | 30 days | Archive |
| Strategy states | Monthly validation against DB | Update or remove |
| Workshop lessons | Permanent | — |
| Validated principles | Permanent | — |

### Acceptance Criteria

- [ ] Memory entries have date metadata
- [ ] Weekly pruning job removes expired entries
- [ ] Stale entries archived, not deleted (recoverable)

---

## 7.13 Session-Type-Aware Memory Loading

**Finding ID:** CTO MC4
**Severity:** MEDIUM
**Effort:** 1 day

### The Problem

`MEMORY.md` is loaded in full at session start (first 200 lines). All entries load regardless of relevance. A trading session doesn't need community intel scan details; a research session doesn't need trade journal entries.

### The Fix

Structure MEMORY.md as a lean index with category headers. Implement session-type-aware loading: Trading sessions load strategy + execution memories; Research sessions load hypothesis + intel memories; Supervisor loads health + lifecycle memories.

### Acceptance Criteria

- [ ] Memory loading filtered by session type
- [ ] Trading sessions skip research-only memories
- [ ] Token savings from reduced irrelevant context measured

---

## 7.14 Memory Promotion / Dreaming Pattern

**Finding ID:** CTO OC-2 (OpenClaw benchmark)
**Severity:** HIGH
**Effort:** 2 days

### The Problem

OpenClaw's three-phase Dreaming system (Light/Deep/REM) with `recencyHalfLifeDays` and `maxAgeDays` handles memory aging automatically. QuantStack has manual pruning only — stale entries persist indefinitely.

### The Fix

Implement tiered memory promotion aligned to QuantStack's workflow:

| Phase | Trigger | TTL | Example |
|-------|---------|-----|---------|
| **Light** | Research discovers signal | 7 days | Add to `research_findings.md` |
| **Deep** | Signal passes backtest | 30 days | Promote to `validated_signals.md` |
| **REM** | Strategy produces live profit | Permanent | Extract principle to `strategy_registry.md` |

### Acceptance Criteria

- [ ] Memory entries have TTL metadata
- [ ] `compact-memory` skill applies TTL rules weekly
- [ ] Stale entries auto-archived, not blocking context

---

## 7.15 Kill Switch Cancel Propagation

**Finding ID:** CTO OC-6 (OpenClaw TaskFlow pattern)
**Severity:** MEDIUM
**Effort:** 2 days

### The Problem

When kill switch fires, cancellation propagates via poll-based `require_ctx()` check. Each node completes its current execution before checking — delay of up to one full node execution time (potentially minutes).

### The Fix

Replace poll-based checking with sticky cancel intent via LangGraph's interrupt mechanism or shared `threading.Event`. All nodes check before execution, not after.

### Acceptance Criteria

- [ ] Kill switch propagates to all in-flight nodes within seconds
- [ ] No node starts execution after kill switch fires
- [ ] Kill-switch-to-halt time reduced from minutes to seconds

---

## Annual Cost Impact Summary

| Optimization | Annual Savings | Effort |
|-------------|---------------|--------|
| Prompt caching (7.2) | $30,000-$40,000 | 1 hour |
| Deterministic tool ordering (7.1) | $2,000-$5,000 | 30 min |
| Agent tier reclassification (7.7) | $5,000-$10,000 | 1 day |
| Context compaction (7.5) | $1,000-$3,000 | 3 days |
| EWF deduplication (7.9) | $500-$1,000 | 0.5 day |
| Per-agent budgets (7.6) | $2,000-$5,000 (prevents runaway) | 2 days |
| **TOTAL** | **$40,000-$64,000** | **~8 days** |

**ROI: 5-8x engineering investment in the first year.**
