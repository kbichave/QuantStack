# Phase 5 Cost Optimization — Research Findings

---

## Part 1: Codebase Analysis

### 1. LLM Routing & Model Selection

**`get_chat_model(tier, thinking)`** — `src/quantstack/llm/provider.py:208`
- Returns a configured LangChain ChatModel for the given tier
- Tier-to-model mapping via `ModelConfig(provider, model_id, tier, max_tokens, temperature, thinking)`
- Fallback chain: primary provider → FALLBACK_ORDER

**Tier system** — `src/quantstack/llm/config.py`:
- **Heavy (reasoning):** fund_manager, quant_researcher, trade_debater, risk, ml_scientist, strategy_rd, options_analyst
- **Medium (synthesis):** earnings_analyst, position_monitor, daily_planner, market_intel, trade_reflector
- **Light (coordination):** community_intel, execution_researcher, supervisor
- **Embedding:** memory, RAG

**Default provider models (Bedrock):**
- heavy: `bedrock/us.anthropic.claude-sonnet-4-6`
- medium: `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`
- light: `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`
- embedding: `ollama/mxbai-embed-large`

**Temperature:** Global default `temperature=0.0`. Overrides exist only in OPRO loop (0.7), TextGrad (0.3), sentiment collectors (0/0.1).

### 2. Agent Configurations (All 3 Graphs)

**Research Graph** — `src/quantstack/graphs/research/config/agents.yaml` (7 agents):

| Agent | Tier | EWF? |
|-------|------|------|
| quant_researcher | heavy | yes |
| ml_scientist | heavy | no |
| strategy_rd | heavy | no |
| hypothesis_critic | medium | no |
| community_intel | medium | no |
| domain_researcher | heavy | yes (swing only) |
| execution_researcher | heavy | no |

**Trading Graph** — `src/quantstack/graphs/trading/config/agents.yaml` (10 agents):

| Agent | Tier | EWF? |
|-------|------|------|
| daily_planner | medium | yes (blue_box) |
| position_monitor | medium | yes |
| exit_evaluator | medium | yes |
| trade_debater | heavy | yes + blue_box |
| fund_manager | heavy | no |
| options_analyst | heavy | no |
| earnings_analyst | medium | no |
| market_intel | medium | no |
| trade_reflector | medium | no |
| executor | medium | no |

**Supervisor Graph** — `src/quantstack/graphs/supervisor/config/agents.yaml` (4 agents):

| Agent | Tier |
|-------|------|
| health_monitor | light |
| self_healer | medium |
| portfolio_risk_monitor | medium |
| strategy_promoter | medium |

### 3. Trading Graph Merge Points

**File:** `src/quantstack/graphs/trading/graph.py`

**Graph flow (16 nodes + 2 merges):**
```
START → data_refresh → safety_check → market_intel → plan_day
  ├→ position_review → execute_exits ──┐
  └→ entry_scan → earnings_analysis? ──┘
                                    merge_parallel (no-op join)
                                        ↓
                                    risk_sizing (deterministic)
                                        ↓
  ┌─ portfolio_review ──────────────────┐
  └─ analyze_options ───────────────────┘
                                    merge_pre_execution (no-op join)
                                        ↓
                                    execute_entries → reflect → END
```

**Both merges are no-op convergence points** — no data transformation, no compaction. All prior agent outputs flow through unmodified.

**Estimated context volume at merge points:**
- Candidate context: 5-20 symbols × ~2KB = 10-40KB
- Portfolio state: positions, Greeks, equity curve = ~30-50KB
- Risk matrices: correlation, factor tilts = ~20KB
- Decision log: ~2.5KB
- **Total: ~65-120KB per cycle** entering execute_entries uncompacted

### 4. EWF Analysis Tool

**File:** `src/quantstack/tools/langchain/ewf_tools.py`

**`get_ewf_analysis(symbol, timeframe)`** — Returns JSON: bias, wave_position, key_levels, blue_box_active/zone, confidence, staleness.

**TTL by timeframe:** 1h=4h, 4h=6h, daily=26h, weekly=8d

**DB:** `ewf_chart_analyses` table with indices on (symbol, timeframe, fetched_at DESC)

**Agents calling it (5 total):**
1. quant_researcher (research) — hypothesis formation
2. domain_researcher (research, swing) — entry/exit zones
3. position_monitor (trading) — thesis invalidation
4. exit_evaluator (trading) — wave completion exits
5. trade_debater (trading) — bull/bear weighting
6. daily_planner (trading) — via `get_ewf_blue_box_setups`

**No caching/deduplication exists.** Each tool call queries DB directly.

### 5. Cost Tracking

**Current state:** No `llm_costs` table. No per-agent cost aggregation.

**Langfuse integration** — `src/quantstack/observability/instrumentation.py` (v2) and `tracing.py` (v4):
- `log_llm_call()` instruments provider, model, tokens, latency
- `log_tool_call()` instruments tool invocations
- Cost computed server-side by Langfuse
- No agent-level rollup, no per-symbol attribution

### 6. Agent Executor

**File:** `src/quantstack/graphs/agent_executor.py`

```
MAX_TOOL_ROUNDS = 10
MAX_TOOL_RESULT_CHARS = 4,000
MAX_MESSAGE_CHARS = 150,000 (~37.5k tokens)
```

**Tool-calling loop:** LLM → tool calls → execute → feed results back → repeat until final answer or MAX_TOOL_ROUNDS.

**Message pruning:** Reactive at 150K chars — drops oldest tool rounds (may lose critical data).

**AgentConfig** — `src/quantstack/graphs/config.py`:
- Loaded from YAML: name, role, goal, backstory, llm_tier, thinking, tools, always_loaded_tools, max_iterations, timeout_seconds
- **No `temperature` field currently** — would need adding
- **No `max_tokens_budget` field** — would need adding

### 7. Memory System

**PostgreSQL-backed:** `agent_memory` table via `src/quantstack/memory/blackboard.py`
- Schema: id, session_id, sim_date, agent, symbol, category, content_json, created_at
- **No TTL/expiry logic.** Entries accumulate indefinitely.
- **No `last_accessed_at` tracking.**

**Legacy markdown files** in `.claude/memory/`:
- strategy_registry.md (53KB), session_handoffs.md (16KB), ml_experiment_log.md (6KB), trade_journal.md (3KB)
- Manual archival at ~100KB threshold

### 8. Testing Setup

**Framework:** pytest with custom markers

**Structure:**
```
tests/
├── conftest.py
├── unit/
│   ├── test_llm_provider.py      — tier resolution, provider selection, fallback
│   ├── test_agent_executor.py    — tool filtering, pruning, loop execution
│   ├── test_agent_configs.py     — YAML validation, tool constraints
│   └── ~50+ other unit tests
├── regression/                    — strategy backtests
└── _fixtures/ohlcv_generators.py — synthetic data generators
```

**Markers:** `@slow`, `@integration`, `@benchmark`, `@requires_api`, `@requires_gpu`, `@regression`

**Run:** `pytest tests/unit/` (default), `pytest tests/unit/ -m slow`, `pytest tests/regression/ -m regression`

**Existing relevant tests:**
- `test_llm_provider.py` — model tier resolution, provider selection, fallback chain
- `test_agent_executor.py` — tool filtering, system message gen, pruning
- `test_agent_configs.py` — YAML structure, tool constraints per agent

---

## Part 2: External Best Practices Research (2026)

### Topic 1: LangGraph Context Compaction Patterns

**Sources:** LangGraph docs, Anthropic "Building Effective Agents"

**Three built-in strategies:**

1. **`trim_messages`** — sliding window with token counting. Operates on messages passed to LLM, not persisted state. Use as belt-and-suspenders at every LLM-calling node.

2. **`RemoveMessage`** — permanent deletion from checkpointed state. For bulk cleanup, `REMOVE_ALL_MESSAGES` sentinel available.

3. **`SummarizationNode` (from `langmem`)** — model-based compression. Fires when token count exceeds `max_tokens_before_summary`, replacing older messages with compressed summary. **Recommended for QuantStack merge points** using haiku-tier models.

**Inter-agent patterns:**
- **Private state channels:** Agent-private state that doesn't flow downstream — prevents context leakage
- **`Command` API:** Explicit state update specification at handoffs — only specified fields transfer
- **`remaining_steps` guard:** Prevents runaway loops by checking step budget

**Recommendations for QuantStack:**
1. Add `SummarizationNode` at merge_parallel and merge_pre_execution with haiku-tier models
2. Define typed brief schemas (SignalBrief, RiskAssessment) for each handoff instead of raw outputs
3. Use private state channels for agent-internal scratchpad data
4. Place `trim_messages` at every LLM-calling node as safety net

### Topic 2: LLM Token Budget Enforcement

**Source:** LiteLLM docs

**LiteLLM provides multi-level budget hierarchy:**
- Global proxy, team, team member, API key, model-specific, agent session, end user
- Rolling windows with auto-reset: 30s, 30m, 30h, 30d
- Per-response cost in `x-litellm-response-cost` header
- Aggregation via `/global/spend/report` endpoint

**Agent-specific controls:**
- `tpm_limit`, `rpm_limit`, `session_tpm_limit`, `max_iterations`, `max_budget_per_session`
- Overages return 429 (graceful rejection)

**LangGraph-native:** `remaining_steps` in state decrements each step — secondary guard.

**Recommendations for QuantStack:**
1. Consider LiteLLM proxy for unified cost tracking + budget enforcement
2. OR implement lightweight per-agent tracking in the existing `agent_executor.py` (log tokens per call, aggregate by agent/day)
3. Add `max_tokens_budget` to AgentConfig, enforce at node boundary in tool-calling loop
4. Dual enforcement: application-level graceful exit + hard budget cutoff

### Topic 3: LLM Provider Fallback / Circuit Breaker

**Source:** LiteLLM routing docs

**LiteLLM Router capabilities:**
- **Priority chains via `order` parameter:** Lower order = higher priority, automatic escalation on failure
- **Three fallback types:** Standard (429/500/timeout), content policy, context window
- **Circuit breaker:** `allowed_fails=3, cooldown_time=30` — deployment enters cooldown after threshold failures
- **Retry with error-specific policies:** Auth errors = 0 retries, rate limits = 5 retries with exponential backoff
- **Routing strategies:** simple-shuffle, latency-based, usage-based, least-busy, cost-based
- **Pre-call validation:** Filters deployments that can't handle request before sending

**Recommendations for QuantStack:**
1. **Option A (full):** Deploy LiteLLM proxy as Docker service — handles routing, fallback, circuit breaker
2. **Option B (lightweight):** Enhance existing `get_chat_model()` with retry + fallback logic:
   - On 429/500/timeout: retry same provider 2x with exponential backoff
   - If still failing: try next provider in FALLBACK_ORDER
   - Track failed providers with cooldown timestamps
   - Log all provider switches as operational events
3. Priority chain: Bedrock (order=1, most reliable) → Anthropic direct (order=2) → Groq (order=3, degraded capability)
4. Set `cooldown_time=60` for production (30s too aggressive for sustained outages)

### Topic 4: Memory Lifecycle TTL Patterns

**Sources:** CrewAI, Mem0, LangMem docs

**Memory tier taxonomy (cross-validated):**

| Tier | Lifetime | Strategy |
|------|----------|----------|
| Conversation | Single turn | Auto-expire |
| Session | Minutes-hours | Clear on task completion |
| User/Agent | Weeks-forever | Temporal decay + review |
| Organizational | Configured | Owner-maintained |

**CrewAI temporal decay model (most sophisticated):**
- Formula: `decay = 0.5^(age_days / half_life_days)`
- Default half_life = 30 days
- Applied at retrieval time as weighting factor, not deletion
- Automatic consolidation: similarity > 0.85 → LLM decides merge/update/delete

**Recommendations for QuantStack:**
1. Add `created_at` and `last_accessed_at` to `agent_memory` table
2. Implement temporal decay at retrieval (not deletion) — weight by `0.5^(age/half_life)`
3. Different half-lives: trade outcomes=14d, strategy params=30d, market regime=7d, research findings=90d
4. Weekly consolidation job in Supervisor graph: scan similar memories, LLM-assisted merge
5. Archive (don't delete) entries not accessed in 60+ days
6. Active "forget" on strategy decommission — scoped deletion of associated memories

---

## Key Findings Summary

| Area | Current State | Gap | Priority |
|------|---------------|-----|----------|
| Merge points | No-op joins, full context flows through | 65-120KB uncompacted per cycle | HIGH — biggest cost driver |
| EWF dedup | No caching, 5 agents call independently | 3-5x redundant DB queries per cycle | HIGH — easy win |
| Agent tiers | Properly configured in YAML already | Spec says 10+ agents default to Sonnet — **this may be outdated** (audit was done before tier config was added) | VERIFY — may already be fixed |
| Temperature | Global 0.0 except OPRO/TextGrad | No per-agent config in AgentConfig | MEDIUM |
| Hardcoded models | Config.py defines all providers cleanly | Need to grep for any remaining hardcoded strings outside config | VERIFY |
| Cost tracking | Langfuse only, no agent-level aggregation | No budget enforcement, no per-agent cost rollup | HIGH |
| Provider fallback | Startup-only check, no runtime fallback | 429 mid-execution = failure | HIGH |
| Memory TTL | No expiry, no decay, no consolidation | Unbounded growth, stale entries waste tokens | MEDIUM |
| AgentConfig | No temperature or budget fields | Need schema extension | Prerequisite for 5.2, 5.4 |
