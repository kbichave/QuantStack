# Phase 5: Cost Optimization — Deep Plan Spec

**Timeline:** Week 3-5 (parallel with Phase 3)
**Effort:** 11 days
**Annual savings:** $40,000-$64,000

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 5 addresses the LLM cost structure. QuantStack runs 21 agents across 3 graphs every 5-10 minutes. At current uncached rates, system prompt tokens alone cost ~$126/day. Most items here pay for themselves within the first week.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`08_COST_OPTIMIZATION.md`](../08_COST_OPTIMIZATION.md)
**Note:** Items 5.1 (tool ordering) and prompt caching are already covered in Phase 0.

---

## Objective

Cut LLM costs by 60-80% through context compaction, agent tier reclassification, per-agent cost tracking/budgets, provider fallback, and memory lifecycle management.

---

## Items

### 5.1 Context Compaction at Merge Points

- **Findings:** CTO MC2, OC-3 | **Severity:** HIGH | **Effort:** 3 days
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.5](../08_COST_OPTIMIZATION.md)
- **Problem:** By the 10th agent, context window contains all prior outputs. Only guard is reactive 150K char pruning that drops oldest tool rounds — potentially losing critical data.
- **Fix:** Add compaction nodes after merge points:
  - `merge_parallel`: Haiku summarizes position_review + entry_scan → structured brief (exits[], entries[], risks[])
  - `merge_pre_execution`: Haiku summarizes portfolio_review + options_analysis → structured brief
  - Cost: ~$0.08/MTok (Haiku) vs. carrying 50K+ at $3/MTok (Sonnet)
- **Key files:** Trading graph merge nodes, new compaction node implementations
- **Acceptance criteria:**
  - [ ] Compaction nodes added at both merge points
  - [ ] Downstream agents receive structured briefs, not raw outputs
  - [ ] Context size at `execute_entries` reduced by 40%+

### 5.2 Per-Agent Cost Tracking

- **Findings:** CTO LH4 | **Severity:** MEDIUM | **Effort:** 2 days
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.6](../08_COST_OPTIMIZATION.md)
- **Problem:** No agent has a token budget. No cost tracking per agent per cycle. Can't answer "which agent costs the most?" A runaway agent could consume $50+ in a single cycle.
- **Fix:**
  1. Log `tokens_in`, `tokens_out`, `model_id`, `cost_estimate` per LLM call
  2. Aggregate by agent/day/graph in `llm_costs` table
  3. Add `max_tokens_budget` to `AgentConfig` in `agents.yaml`
  4. When budget exhausted: graceful exit at next node boundary
  5. Alert when any agent exceeds daily cost threshold
- **Key files:** Agent executor, `agents.yaml`, new `llm_costs` table
- **Acceptance criteria:**
  - [ ] Per-agent per-cycle cost logged
  - [ ] Daily cost dashboard or query available
  - [ ] Token budget per agent configured and enforced

### 5.3 Agent Tier Reclassification

- **Finding:** CTO LH1 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.7](../08_COST_OPTIMIZATION.md)
- **Problem:** Agents not matching naming conventions default to Sonnet ("assistant" tier). 10+ agents fall to this default. `daily_planner`, `position_monitor`, `exit_evaluator` don't need Sonnet.
- **Fix:** Explicit tier assignment in `agents.yaml`:
  - Keep Sonnet: `hypothesis_generation`, `trade_debater`, `fund_manager`
  - Downgrade to Haiku: `daily_planner`, `position_monitor`, `exit_evaluator`, `market_intel`, `health_monitor`, `executor`
  - Remove naming-convention-based classification; log warning if any agent falls to default
- **Key files:** `src/quantstack/graphs/*/config/agents.yaml`, LLM routing
- **Acceptance criteria:**
  - [ ] Every agent has explicit tier in `agents.yaml`
  - [ ] No agent falls to default tier
  - [ ] Cost reduction measured after migration

### 5.4 Per-Agent Temperature Config

- **Finding:** CTO LC3 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.8](../08_COST_OPTIMIZATION.md)
- **Problem:** All 21 agents use `temperature=0.0`. Kills diversity in research; provides no benefit for already-deterministic tasks.
- **Fix:** Hypothesis generation → 0.7; trade debate → 0.3-0.5; community intel → 0.3; validation/execution → 0.0.
- **Key files:** `agents.yaml`, agent executor temperature handling
- **Acceptance criteria:**
  - [ ] Per-agent temperature configured in `agents.yaml`
  - [ ] Research agents use higher temperature than execution agents

### 5.5 EWF Deduplication

- **Finding:** CTO LH5 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.9](../08_COST_OPTIMIZATION.md)
- **Problem:** Three agents (`daily_planner`, `position_monitor`, `trade_debater`) each independently call `get_ewf_analysis` for same symbol in same cycle. 3x redundant API calls.
- **Fix:** Fetch EWF once per symbol per cycle, store in graph state, share across agents.
- **Key files:** EWF analysis tool, graph state management
- **Acceptance criteria:**
  - [ ] EWF fetched once per symbol per cycle
  - [ ] All agents read from shared state

### 5.6 Remove Hardcoded Model Strings

- **Finding:** CTO LH3 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.10](../08_COST_OPTIMIZATION.md)
- **Problem:** 6+ locations hardcode model names instead of using `get_chat_model()`:
  - `tool_search_compat.py:21` → `anthropic/claude-sonnet-4-20250514`
  - `trading/nodes.py:843` → same
  - `trade_evaluator.py` → same
  - `mem0_client.py` → `gpt-4o-mini`
  - `hypothesis_agent.py` → `groq/llama-3.3-70b-versatile`
  - `opro_loop.py` → same
- **Fix:** Replace all with `get_chat_model()` calls using appropriate tier names.
- **Key files:** All files with hardcoded model strings
- **Acceptance criteria:**
  - [ ] Zero hardcoded model strings in codebase
  - [ ] All model references go through `get_chat_model()`
  - [ ] Changing `LLM_PROVIDER` env var switches ALL model calls

### 5.7 LLM Provider Runtime Fallback

- **Findings:** CTO LH2, OC-4 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.11](../08_COST_OPTIMIZATION.md)
- **Problem:** Provider availability checked at startup only. Mid-execution 429 → no fallback. Anthropic API credit exhaustion on 2026-04-05 blocked EWF analyzer for hours.
- **Fix:**
  1. On 429/500/timeout: retry same provider 2x
  2. If still failing: switch to next provider (Anthropic → Bedrock → Groq)
  3. Cooldown failed provider for 5 minutes
  4. Log provider switches as operational events
- **Key files:** `get_chat_model()`, LLM routing layer
- **Acceptance criteria:**
  - [ ] Mid-session provider failure triggers automatic fallback
  - [ ] Provider cooldown prevents hammering failing service
  - [ ] No single-provider outage can halt the system

### 5.8 Memory Temporal Decay

- **Finding:** CTO MC3 | **Severity:** MEDIUM | **Effort:** 1 day
- **Audit section:** [`08_COST_OPTIMIZATION.md` §7.12](../08_COST_OPTIMIZATION.md)
- **Problem:** Memory entries have no expiry. 2026-04-04 EWF market reads (flagged "low-trust") persist indefinitely, wasting tokens.
- **Fix:** TTLs: market reads = 7 days; session handoffs = 30 days; strategy states = monthly validation; workshop lessons + validated principles = permanent. Weekly pruning job archives expired entries.
- **Key files:** `.claude/memory/`, compact-memory skill
- **Acceptance criteria:**
  - [ ] Memory entries have date metadata
  - [ ] Weekly pruning job removes expired entries
  - [ ] Stale entries archived, not deleted (recoverable)

---

## Dependencies

- **Runs parallel with:** Phase 3 (Operational Resilience)
- **5.6 feeds into Phase 9** item 9.8 (unify LLM provider systems)
- **Phase 0 items 0.1+0.2 are prerequisites** (tool ordering + caching already done)

---

## Annual Cost Impact

| Optimization | Annual Savings | Effort |
|-------------|---------------|--------|
| Agent tier reclassification (5.3) | $5,000-$10,000 | 1 day |
| Context compaction (5.1) | $1,000-$3,000 | 3 days |
| EWF deduplication (5.5) | $500-$1,000 | 0.5 day |
| Per-agent budgets (5.2) | $2,000-$5,000 (prevents runaway) | 2 days |
| Provider fallback (5.7) | Prevents outage cost (unquantified) | 2 days |
| **TOTAL (Phase 5 only)** | **$8,500-$19,000** | **~9 days** |
| **Combined with Phase 0** | **$40,000-$64,000** | **~10 days** |

ROI: 5-8x engineering investment in the first year.

---

## Validation Plan

1. **Compaction (5.1):** Measure context window size at `execute_entries` before/after. Expect 40%+ reduction.
2. **Cost tracking (5.2):** Run 10 trading cycles → query `llm_costs` table → verify per-agent breakdown.
3. **Tier reclassification (5.3):** Verify Langfuse traces show correct model per agent. Compare daily cost before/after.
4. **Hardcoded strings (5.6):** `grep -r "claude-sonnet\|gpt-4o\|llama-3" src/` → expect zero hits.
5. **Provider fallback (5.7):** Block Anthropic API → verify automatic switch to Bedrock → verify cooldown period.
