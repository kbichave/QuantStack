# Complete Specification — Phase 4: Agent Architecture & Coordination

## Context & Scope

QuantStack is an autonomous trading system built on 3 LangGraph StateGraphs (Research, Trading, Supervisor) with 21 agents. The architecture earned a B+ for role specialization but agents don't coordinate safely. Phase 4 fixes multi-agent coordination bugs that produce unpredictable behavior at scale.

**Timeline:** Week 4-6 (13-15 days, parallelizable to ~7 days)
**Deployment:** Atomic release — all items ship together
**Dependencies:** Phase 1 (safety), Phase 2 partial (2.1 IC tracking)

---

## Items with Implementation Decisions

### 4.1 Race Condition Fix (Parallel Branches)

**Problem:** `position_review` and `entry_scan` run in parallel. Both write to accumulating lists (`exit_orders` and `entry_candidates` via `Annotated[list, operator.add]`). Same symbol can appear in both exit and entry lists — no semantic conflict resolution.

**Decision:** Add a new `resolve_symbol_conflicts` node between `merge_parallel` and `risk_sizing`. This node:
- Scans `exit_orders` and `entry_candidates` for overlapping symbols
- Exits always take priority (risk-off bias)
- Removes conflicting entries from `entry_candidates`
- Logs all conflict events with both sides' reasoning for analysis
- Safe default on failure: drop all conflicted entries

### 4.2 Errors Block Execution

**Problem:** Every node catches exceptions and appends to `errors` list. Even after 5 errors, pipeline reaches `execute_entries`. No gating.

**Decision:** Node classification based on principle: **"If this node fails and pipeline continues, can we lose money unintentionally?"**

**Blocking (failure halts pipeline):**
- `data_refresh` — stale data corrupts all downstream decisions
- `safety_check` — non-negotiable gate
- `position_review` — failure means risk_sizing doesn't know current exposure
- `risk_sizing` — can't execute without valid sizes
- `execute_exits` — failure to close leaves unintended exposure open

**Non-blocking (safe defaults):**
- `plan_day` — defaults to neutral bias
- `entry_scan` — defaults to empty candidate list
- `market_intel` — informational only
- `portfolio_construction` — empty universe is safe
- `execute_entries` — no new positions
- `resolve_symbol_conflicts` — drop all conflicted entries
- `reflect` / `trade_reflector` — post-execution, no risk impact

**Implementation:** Error count > 2 from blocking nodes prevents execution for that cycle. Non-blocking failures use safe defaults and continue.

### 4.3 State Schema Validation (Pydantic)

**Problem:** State is `TypedDict`. Typos like `{"daly_plan": "..."}` silently accepted. No type checking at boundaries.

**Decision:** Migrate all 3 graphs simultaneously (TradingState, ResearchState, SupervisorState) to Pydantic `BaseModel` with:
- `ConfigDict(extra="forbid")` — rejects unknown keys at merge
- `field_validator` for domain invariants (e.g., position size bounds)
- `model_validator(mode="after")` for cross-field invariants
- **Typed output models per node** — each node gets its own Pydantic output model (not plain dicts)
- Separate input/output schemas per graph to constrain boundaries

### 4.4 Node Circuit Breaker

**Problem:** If `daily_planner` fails 5 consecutive cycles, graph still calls it. No backoff, no circuit breaker.

**Decision:** Decorator pattern — `@circuit_breaker(threshold=3)` wrapping node functions:
- State stored in PostgreSQL (`circuit_breaker_state` table), persists across cycles
- 3 consecutive failures → skip node, use typed safe default (empty/neutral response)
- 5 consecutive → alert and halt graph
- On success → reset counter
- Separate from strategy-level `StrategyBreaker` (different concerns: node health vs strategy P&L)
- LLM-specific failure type discrimination: rate limits trip immediately, token limits route to pruning, parse failures count separately

### 4.5 Tool Access Control per Graph

**Problem:** No access control. Any agent can call any tool in `TOOL_REGISTRY`. Configuration-only, not enforced.

**Decision:** Add `blocked_tools` per graph in `agents.yaml`:
- Research blocks: `execute_order`, `cancel_order`, `activate_kill_switch`
- Trading blocks: `register_strategy`, `train_model`
- Supervisor blocks: all execution tools (read-only)
- 5-line guard in `agent_executor.py` at tool invocation point
- **Hard reject** on violation — return error to agent, log as security event
- No circuit-breaking on violations (that would mask the config bug)

### 4.6 Event Bus Cursor Atomicity

**Problem:** Cursor updated via DELETE + INSERT. Crash between operations loses cursor → duplicate event processing.

**Decision:** PostgreSQL upsert: `INSERT ... ON CONFLICT (consumer_id) DO UPDATE SET last_event_id = EXCLUDED.last_event_id, last_polled_at = EXCLUDED.last_polled_at`

### 4.7 Dead Letter Queue

**Problem:** `parse_json_response()` fails → output replaced with `{}`. No record of failure.

**Decision:** Add `agent_dlq` table in PostgreSQL:
- Store: agent_name, graph_name, run_id, raw_output, error_type, error_detail, prompt_hash, model_used
- DLQ rate tracked as Langfuse metric with 24h rolling window
- Alert thresholds: warn at 5%, critical at 10%
- Critical alert fires outbound notification (Slack webhook or email)
- DLQ metrics surfaced on system dashboard
- **No self-healing integration** — prompt auto-patching is unsafe for capital allocation decisions. Automate only when cause is unambiguous and fix is reversible.
- Human reviews and deploys fixes manually
- Revisit auto-remediation after 60+ days of paper trading data with correlation patterns

### 4.8 Priority-Based Message Pruning

**Problem:** When conversation exceeds 150K chars, oldest tool rounds dropped (FIFO). `fund_manager` may lose `position_review` results.

**Decision:** Hybrid priority system:
- **Config defaults** in agents.yaml: each agent gets a `priority_tier` (P0-P3)
- **Type overrides**: risk/execution messages always P0 regardless of source agent
- Priority tiers:
  - P0 (never prune): risk gate output, kill switch, position state
  - P1 (summarize): signal briefs, trade decisions, regime assessment
  - P2 (prune first): raw analysis, verbose tool outputs
  - P3 (ephemeral): debug logs, trace metadata — never in LLM context
- Compaction at merge points using Haiku summarization
- Use LangGraph `RemoveMessage` + `add_messages` reducer for rolling-window compaction

### 4.9 Pre-Trade Correlation Check

**Problem:** Correlation only in post-hoc monitoring loop. Not checked pre-trade.

**Decision:** Add to `risk_gate.check()`: reject if new position correlation > 0.7 with any existing position (30-day rolling correlation of daily returns).
- **Fail closed**: if correlation data unavailable, check blocks (does not pass)
- CLAUDE.md updated first: "Never modify" → "Never weaken or bypass"

### 4.10 Portfolio Heat Budget

**Problem:** No cap on daily notional deployed.

**Decision:** Add to `risk_gate.check()`: max daily notional deployed, configurable, default 30% of equity/day.
- Fail closed on data unavailability

### 4.11 Sector Concentration Pre-Trade

**Problem:** Sector concentration only post-hoc via Herfindahl.

**Decision:** Add to `risk_gate.check()`: reject if sector would exceed 40% concentration (configurable).
- Requires sector mapping per symbol (new data requirement)
- Fail closed on missing sector data

### 4.12 Regime Flip Forced Review

**Problem:** Regime flips logged but no action taken. Positions stay open in hostile regimes.

**Decision:**
- Store `regime_at_entry` in both DB (source of truth) and `MonitoredPosition` (runtime cache)
- Reconstruct regime cache from DB on restart
- Moderate mismatch (trending → ranging): tighten stops 50%
- Severe mismatch (trending_up → trending_down): auto-exit within 1 cycle
- Regime comparison logic moved from monitor-only to also fire at position review time

---

## New Database Objects

1. `circuit_breaker_state` — per-node breaker state across cycles
2. `agent_dlq` — dead letter queue for parse/validation failures
3. `regime_at_entry` column on positions table (or new table)
4. `loop_cursors` upsert migration (schema unchanged, query changed)

## Files Modified

- `CLAUDE.md` — "Never modify" → "Never weaken or bypass" on risk gate
- `src/quantstack/graphs/state.py` — TypedDict → Pydantic BaseModel (all 3 states)
- `src/quantstack/graphs/trading/graph.py` — new resolve_symbol_conflicts node, error gating
- `src/quantstack/graphs/trading/nodes.py` — typed output models per node
- `src/quantstack/graphs/agent_executor.py` — tool access guard, DLQ routing, message priority pruning
- `src/quantstack/graphs/*/config/agents.yaml` — blocked_tools, priority_tier per agent
- `src/quantstack/coordination/event_bus.py` — upsert cursor
- `src/quantstack/execution/risk_gate.py` — correlation, heat budget, sector concentration checks
- `src/quantstack/execution/execution_monitor.py` — regime_at_entry field + comparison logic
- New: node output models, circuit breaker decorator, DLQ table + monitoring
