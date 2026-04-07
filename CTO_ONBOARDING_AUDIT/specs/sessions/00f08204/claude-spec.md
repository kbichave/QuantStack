# Phase 5: Cost Optimization — Complete Specification

---

## Overview

Cut QuantStack's LLM costs by 60-80% ($40K-$64K annually, combined with Phase 0) through 8 items implemented in spec order over ~11 days. This runs parallel with Phase 3 (Operational Resilience).

**System context:** 21 agents across 3 LangGraph StateGraphs (Research, Trading, Supervisor) running every 5-10 minutes. Current uncached system prompt tokens cost ~$126/day. Most items pay for themselves within the first week.

---

## Item 5.1: Context Compaction at Merge Points

**Severity:** HIGH | **Effort:** 3 days | **Annual savings:** $1,000-$3,000

### Problem
Two no-op merge points in the Trading graph (`merge_parallel`, `merge_pre_execution`) pass all prior agent outputs uncompacted to downstream agents. By the time `execute_entries` runs, context contains 65-120KB of raw data per cycle. The only guard is reactive 150K-char pruning in `agent_executor.py` that drops oldest tool rounds — potentially losing critical data.

### Current State (from codebase research)
- `merge_parallel`: Joins `execute_exits` + `earnings_analysis` outputs → no transformation
- `merge_pre_execution`: Joins `portfolio_review` + `analyze_options` outputs → no transformation
- Data volume: candidate context (10-40KB) + portfolio state (30-50KB) + risk matrices (20KB) + decision log (2.5KB)
- File: `src/quantstack/graphs/trading/graph.py`

### Solution
Replace no-op merges with Haiku-powered compaction nodes that produce **strict typed schemas** (Pydantic models):

**Post-parallel brief** (merge_parallel → risk_sizing):
- `ParallelMergeBrief`: exits[] (symbol, action, reason), entries[] (symbol, signal_strength, thesis), risks[] (type, severity, detail), market_context (regime, vix, breadth)

**Pre-execution brief** (merge_pre_execution → execute_entries):
- `PreExecutionBrief`: approved_entries[] (symbol, size, structure, rationale), rejected_entries[] (symbol, reason), options_specs[] (symbol, legs, greeks, max_loss), risk_checks (correlation_ok, capital_ok, sector_ok)

**Implementation approach:**
- Use `langmem.short_term.SummarizationNode` pattern with haiku-tier model
- Define Pydantic output models for structured extraction
- Add `trim_messages` as belt-and-suspenders at every LLM-calling node
- Cost: ~$0.08/MTok (Haiku) vs. carrying 50K+ at $3/MTok (Sonnet)

### Acceptance Criteria
- [ ] Compaction nodes at both merge points
- [ ] Downstream agents receive typed briefs, not raw outputs
- [ ] Context size at `execute_entries` reduced by 40%+
- [ ] Brief schemas are Pydantic models with validation

---

## Item 5.2: Per-Agent Cost Tracking

**Severity:** MEDIUM | **Effort:** 2 days | **Annual savings:** $2,000-$5,000 (prevents runaway)

### Problem
No agent has a token budget. No per-agent cost aggregation. Can't answer "which agent costs the most?" A runaway agent could consume $50+ in a single cycle.

### Current State
- Langfuse captures token counts per LLM call via `log_llm_call()` in `src/quantstack/observability/instrumentation.py`
- No `llm_costs` table in PostgreSQL
- No agent-level rollup or budget enforcement
- No alerting on cost spikes

### Solution
**Langfuse as source of truth** with aggregation queries for agent-level rollup:

1. Ensure every LLM call logs `agent_name`, `graph_name`, `cycle_id` as Langfuse metadata tags
2. Build Langfuse aggregation queries/views for:
   - Per-agent per-day cost
   - Per-graph per-cycle cost
   - Top-N most expensive agents
3. Add `max_tokens_budget` field to `AgentConfig` in `agents.yaml`
4. Enforce at node boundary in `agent_executor.py` tool-calling loop — when budget exhausted, graceful exit at next node boundary (produce partial result with "budget exceeded" flag)
5. Alert via Supervisor graph when any agent exceeds daily cost threshold

### Acceptance Criteria
- [ ] Per-agent per-cycle cost visible in Langfuse
- [ ] Daily cost rollup query available
- [ ] Token budget per agent configured in `agents.yaml` and enforced
- [ ] Graceful degradation when budget exhausted (not hard kill)

---

## Item 5.3: Agent Tier Reclassification

**Severity:** HIGH | **Effort:** 1 day | **Annual savings:** $5,000-$10,000

### Problem
Despite YAML configs, some agents still land on incorrect tiers. Naming-convention-based classification causes agents not matching patterns to default to Sonnet.

### Current State
- Tiers configured in `agents.yaml` for all 3 graphs
- `get_chat_model()` in `src/quantstack/llm/provider.py` resolves tier → model
- Some agents may still fall to default due to naming convention logic

### Solution
1. Audit every agent's actual model usage via Langfuse traces
2. Ensure explicit `llm_tier` in every agent config — no fallback-to-default path
3. Recommended reclassification:
   - **Keep heavy:** hypothesis_generation, trade_debater, fund_manager, quant_researcher, ml_scientist, strategy_rd, options_analyst
   - **Downgrade to medium:** (already correct for most)
   - **Downgrade to light:** health_monitor (already light), executor, daily_planner if analysis shows it doesn't need medium
4. Remove naming-convention-based classification; log warning if any agent falls to default
5. Measure cost reduction after migration via Langfuse

### Acceptance Criteria
- [ ] Every agent has explicit tier in `agents.yaml`
- [ ] No agent falls to default tier (warning logged if it would)
- [ ] Cost reduction measured via Langfuse before/after comparison

---

## Item 5.4: Per-Agent Temperature Config

**Severity:** HIGH | **Effort:** 0.5 day

### Problem
All 21 agents use `temperature=0.0` globally. Kills diversity in research; provides no benefit for already-deterministic tasks.

### Current State
- Global default `temperature=0.0` in `src/quantstack/llm/config.py:16`
- Overrides only in OPRO (0.7), TextGrad (0.3), sentiment (0/0.1)
- `AgentConfig` dataclass has no `temperature` field

### Solution
1. Add `temperature: float | None = None` to `AgentConfig` dataclass
2. Add `temperature` field to all `agents.yaml` configs
3. Pass temperature through to `get_chat_model()` / LiteLLM
4. Recommended values:
   - **0.7:** quant_researcher, hypothesis_critic (diversity in ideation)
   - **0.3-0.5:** trade_debater, community_intel (moderate diversity)
   - **0.0:** executor, fund_manager, risk nodes, exit_evaluator (determinism required)
   - **0.1:** position_monitor, daily_planner (slight variation, mostly deterministic)

### Acceptance Criteria
- [ ] Per-agent temperature in `agents.yaml`
- [ ] Research agents use higher temperature than execution agents
- [ ] Temperature passed through LLM instantiation path

---

## Item 5.5: EWF Deduplication

**Severity:** HIGH | **Effort:** 0.5 day

### Problem
5 agents independently call `get_ewf_analysis` for the same symbol in the same cycle. 3-5x redundant DB queries.

### Current State
- `get_ewf_analysis` in `src/quantstack/tools/langchain/ewf_tools.py`
- Called by: quant_researcher, domain_researcher, position_monitor, exit_evaluator, trade_debater
- `get_ewf_blue_box_setups` called by: daily_planner, trade_debater
- No caching — each call hits DB directly
- DB: `ewf_chart_analyses` table with existing freshness TTLs

### Solution
**Graph state per-invocation cache:**
1. Add `ewf_cache: Dict[str, Dict[str, Any]]` to graph state (keyed by `{symbol}:{timeframe}`)
2. At cycle start (e.g., `data_refresh` node), pre-fetch EWF data for all watchlist symbols
3. Store results in graph state
4. Modify `get_ewf_analysis` tool to check graph state first, fall back to DB only if cache miss
5. Same for `get_ewf_blue_box_setups` — fetch once, share via state

### Acceptance Criteria
- [ ] EWF fetched once per symbol per cycle
- [ ] All agents read from shared graph state
- [ ] DB query count reduced by 3-5x per cycle

---

## Item 5.6: Remove Hardcoded Model Strings

**Severity:** HIGH | **Effort:** 1 day

### Problem
Multiple locations hardcode model names instead of using `get_chat_model()`.

### Current State (from audit, needs verification)
- `tool_search_compat.py:21` → `anthropic/claude-sonnet-4-20250514`
- `trading/nodes.py:843` → same
- `trade_evaluator.py` → same
- `mem0_client.py` → `gpt-4o-mini`
- `hypothesis_agent.py` → `groq/llama-3.3-70b-versatile`
- `opro_loop.py` → same

### Solution
1. Grep for all hardcoded model strings: `claude-sonnet`, `gpt-4o`, `llama-3`, `anthropic/`, `bedrock/`, `openai/`, `groq/`, `gemini/` outside of `config.py`
2. Replace each with appropriate `get_chat_model(tier)` call
3. For cases where a specific model is intentionally needed (e.g., embedding), add a named tier or use LiteLLM model alias
4. After LiteLLM deployment (5.7), changing `LLM_PROVIDER` should switch ALL model calls

### Acceptance Criteria
- [ ] Zero hardcoded model strings in `src/` outside config
- [ ] All model references go through `get_chat_model()` → LiteLLM
- [ ] `grep -r "claude-sonnet\|gpt-4o\|llama-3" src/` returns zero hits (excluding config)

---

## Item 5.7: LLM Provider Runtime Fallback (LiteLLM Proxy)

**Severity:** HIGH | **Effort:** 2 days

### Problem
Provider availability checked at startup only. Mid-execution 429/timeout → no fallback. Anthropic API credit exhaustion on 2026-04-05 blocked EWF analyzer for hours.

### Current State
- `get_chat_model()` in `src/quantstack/llm/provider.py` resolves tier → provider → model
- Fallback chain exists (`FALLBACK_ORDER`) but only checked at startup
- No retry logic, no circuit breaker, no runtime provider switching
- Docker Compose runs: postgres, pgvector, langfuse, ollama, 3 graph services

### Solution
**Deploy LiteLLM proxy as Docker service — full replacement of existing routing:**

1. Add `litellm` service to `docker-compose.yml`
   - Mount Zscaler cert bundle (`~/.zscaler_certifi_bundle.pem`) into container
   - Set `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` env vars for Zscaler compatibility
   - Expose on internal Docker network (not public)

2. Configure LiteLLM `config.yaml`:
   - Define logical model names mapping to tiers: `trading-heavy`, `trading-medium`, `research-heavy`, etc.
   - Provider priority chains via `order` parameter:
     - Heavy: Bedrock (order=1) → Anthropic (order=2) → Groq/llama (order=3, degraded)
     - Medium/Light: Bedrock (order=1) → Anthropic (order=2)
   - Circuit breaker: `allowed_fails=3, cooldown_time=60`
   - Retry policy: 429/timeout → 2 retries with backoff, auth errors → 0 retries
   - Context window fallbacks for research agents with long contexts

3. Refactor `get_chat_model()` to thin wrapper:
   - Point all LLM calls at LiteLLM proxy URL
   - Map tier names to LiteLLM model aliases
   - Preserve existing API for callers — no changes needed in graph/agent code

4. Operational:
   - Log provider switches as events in Langfuse
   - Health check endpoint for Supervisor graph
   - Dashboard for provider availability/latency

### Acceptance Criteria
- [ ] LiteLLM running in Docker Compose with Zscaler cert
- [ ] Mid-session provider failure triggers automatic fallback
- [ ] Provider cooldown (60s) prevents hammering failing service
- [ ] No single-provider outage halts the system
- [ ] `get_chat_model()` API unchanged for callers

---

## Item 5.8: Memory Temporal Decay

**Severity:** MEDIUM | **Effort:** 1 day

### Problem
Memory entries in both `.claude/memory/` files and PostgreSQL `agent_memory` table have no expiry. Stale entries persist indefinitely, wasting tokens.

### Current State
- **PostgreSQL:** `agent_memory` table via `src/quantstack/memory/blackboard.py` — no TTL, no `last_accessed_at`
- **Markdown:** `.claude/memory/` files — strategy_registry.md (53KB), session_handoffs.md (16KB), etc. Manual archival at ~100KB

### Solution
**Full lifecycle management across both storage layers:**

1. **PostgreSQL `agent_memory`:**
   - Add `last_accessed_at TIMESTAMP` column, update on every read
   - Add `archived_at TIMESTAMP` column (null = active)
   - Weekly pruning job (Supervisor graph scheduled task):
     - Trade outcomes: archive after 14 days
     - Strategy parameters: archive after 30 days
     - Market regime observations: archive after 7 days
     - Research findings: archive after 90 days
   - Archive = move to `agent_memory_archive` table (not delete)
   - Temporal decay at retrieval: weight by `0.5^(age_days / half_life_days)` when building context

2. **`.claude/memory/` files:**
   - Add date metadata to all memory entries (frontmatter `created:` field)
   - Enhance compact-memory skill to enforce TTLs:
     - Market reads: 7 days
     - Session handoffs: 30 days
     - Strategy states: monthly validation
     - Workshop lessons + validated principles: permanent
   - Archive expired entries to `*.archive.md` files (recoverable)

### Acceptance Criteria
- [ ] Memory entries have date metadata (both systems)
- [ ] Weekly pruning job archives expired entries
- [ ] Stale entries archived, not deleted (recoverable)
- [ ] Retrieval applies temporal decay weighting
- [ ] `last_accessed_at` tracked in `agent_memory`

---

## Implementation Order & Dependencies

| Order | Item | Effort | Dependencies | Notes |
|-------|------|--------|--------------|-------|
| 1 | 5.1 Context Compaction | 3 days | None | Biggest context reduction |
| 2 | 5.2 Cost Tracking | 2 days | None (Langfuse already running) | Enables measurement of later items |
| 3 | 5.3 Tier Reclassification | 1 day | 5.2 (to measure impact) | Quick win with 5.2 in place |
| 4 | 5.4 Temperature Config | 0.5 day | 5.3 (AgentConfig changes) | Piggybacks on config schema |
| 5 | 5.5 EWF Dedup | 0.5 day | None | Independent quick win |
| 6 | 5.6 Remove Hardcoded Strings | 1 day | None, but ideally before 5.7 | Clean prerequisite for LiteLLM |
| 7 | 5.7 LiteLLM Proxy | 2 days | 5.6 (no hardcoded strings) | Infrastructure change |
| 8 | 5.8 Memory Decay | 1 day | None | Independent |

**Total: ~11 days**

---

## Validation Plan

1. **5.1:** Measure context window size at `execute_entries` before/after → expect 40%+ reduction
2. **5.2:** Run 10 trading cycles → query Langfuse → verify per-agent cost breakdown
3. **5.3:** Verify Langfuse traces show correct model per agent; compare daily cost before/after
4. **5.4:** Verify Langfuse traces show correct temperature per agent
5. **5.5:** Count `ewf_chart_analyses` queries per cycle before/after → expect 3-5x reduction
6. **5.6:** `grep -r "claude-sonnet\|gpt-4o\|llama-3" src/` → zero hits (excluding config)
7. **5.7:** Block Anthropic API → verify automatic switch to Bedrock → verify cooldown period
8. **5.8:** Insert old entries → run pruning → verify archived, not deleted; verify decay weighting

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Compaction loses critical data | Missed trades or wrong exits | Validate briefs contain all fields downstream agents need; keep raw data in checkpoint |
| LiteLLM adds latency | Slower trading cycles | Deploy on same Docker network; benchmark p50/p99 before/after |
| Zscaler blocks LiteLLM → provider calls | All LLM calls fail | Mount cert bundle, test connectivity before go-live |
| Budget enforcement kills mid-analysis | Incomplete agent output | Graceful exit at node boundary with partial result flag |
| Tier downgrade degrades quality | Worse trading decisions | A/B test on paper trading before live; measure signal quality |
| Memory archival removes useful context | Agents miss relevant history | Archive (don't delete), temporal decay at retrieval preserves access |
