# Phase 4: Agent Architecture & Coordination — Deep Plan Spec

**Timeline:** Week 4-6
**Effort:** 13-15 days (parallelizable to ~7 days with 2 engineers)
**Gate:** Race conditions fixed. Errors block execution. Tool access controlled.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 4 fixes the multi-agent coordination bugs that produce unpredictable behavior at scale. The architecture is a B+ with excellent role specialization across 21 agents — but agents don't coordinate safely.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`06_AGENT_ARCHITECTURE.md`](../06_AGENT_ARCHITECTURE.md)
**Supporting sections:** [`05_GRAPH_RESTRUCTURING.md`](../05_GRAPH_RESTRUCTURING.md) (architectural context), [`03_EXECUTION_LAYER.md`](../03_EXECUTION_LAYER.md) (pre-trade checks)

---

## Objective

Fix multi-agent coordination: prevent parallel branch conflicts on the same symbol, make errors block execution, validate state schemas, add circuit breakers for failing nodes, enforce tool access control per graph, and add pre-trade portfolio-level checks.

---

## Items

### 4.1 Race Condition Fix (Parallel Branches)

- **Finding:** QS-A2 | **Severity:** CRITICAL | **Effort:** 1-2 days
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.1](../06_AGENT_ARCHITECTURE.md)
- **Problem:** `position_review` and `entry_scan` run in parallel. If review decides to exit XYZ and scan simultaneously decides to enter XYZ, merged state has conflicting orders. No transactional guarantee prevents buy+sell for same symbol in same cycle.
- **Fix:** Conflict resolution at `merge_parallel` — if same symbol in exits and entries, exits take priority (risk-off bias). Log conflict events.
- **Key files:** Trading graph merge nodes, `src/quantstack/graphs/trading/graph.py`
- **Acceptance criteria:**
  - [ ] Same symbol cannot appear in both exit and entry lists within one cycle
  - [ ] Exits take priority over entries when conflict detected
  - [ ] Conflict events logged for analysis

### 4.2 Errors Block Execution

- **Finding:** QS-A4 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.2](../06_AGENT_ARCHITECTURE.md)
- **Problem:** Every node: catch exception → append to `errors` list → continue. Even critical nodes (`data_refresh`, `daily_planner`). A cycle can have 5 errors and still reach `execute_entries`.
- **Fix:**
  1. Add error-count check before `execute_entries`: `if len(errors) > 2: skip execution`
  2. Critical nodes (`data_refresh`, `safety_check`, `risk_sizing`) are blocking — failure halts pipeline
  3. Non-critical nodes (`market_intel`, `trade_reflector`) can fail without halting
  4. Classify each node as `blocking` or `non_blocking` in graph definition
- **Key files:** Graph node definitions, execution gate logic
- **Acceptance criteria:**
  - [ ] Critical node failures halt the pipeline
  - [ ] Error count > 2 prevents execution for that cycle
  - [ ] Node classification (blocking/non-blocking) documented

### 4.3 State Schema Validation (Pydantic)

- **Finding:** QS-A3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.3](../06_AGENT_ARCHITECTURE.md)
- **Problem:** Node returns merged via dict update. Typo `{"daly_plan": "..."}` silently added; `daily_plan` remains stale. No schema validation, no type checking at state boundaries.
- **Fix:**
  1. Add Pydantic model for `TradingState` with typed fields
  2. Each node declares its output schema
  3. State merge validates against schema — rejects unknown keys
  4. Type mismatches logged and rejected
- **Key files:** State definitions for all 3 graphs, node return types
- **Acceptance criteria:**
  - [ ] `TradingState` is a Pydantic model with typed fields
  - [ ] Typos in state keys rejected at merge point
  - [ ] Type mismatches detected and logged

### 4.4 Node Circuit Breaker

- **Finding:** QS-A5 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.4](../06_AGENT_ARCHITECTURE.md)
- **Problem:** If `daily_planner` fails 5 consecutive cycles, graph still calls it on cycle 6. No backoff, no circuit breaker, no fallback.
- **Fix:** 3 consecutive failures → skip node, use safe defaults (empty plan = no new entries); 5 consecutive → alert and halt graph; on success → reset counter.
- **Key files:** Graph runner, per-node failure tracking
- **Acceptance criteria:**
  - [ ] Per-node failure tracking across cycles
  - [ ] Auto-skip after 3 consecutive failures
  - [ ] Alert after 5 consecutive failures

### 4.5 Tool Access Control per Graph

- **Findings:** QS-A9, CTO OC-5 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.5](../06_AGENT_ARCHITECTURE.md)
- **Problem:** No negative access control. Research agents could call execution tools. Trading agents could call strategy registration tools. Misconfigured `agents.yaml` could allow `hypothesis_critic` to call `execute_order`.
- **Fix:** Add `blocked_tools` field per graph in `agents.yaml`: Research blocks `execute_order, cancel_order, activate_kill_switch`; Trading blocks `register_strategy, train_model`; Supervisor blocks all execution tools (read-only). 5-line guard in `agent_executor.py`.
- **Key files:** `src/quantstack/graphs/*/config/agents.yaml`, agent executor
- **Acceptance criteria:**
  - [ ] Research agents cannot call execution tools
  - [ ] Trading agents cannot call strategy registration tools
  - [ ] Blocked tool invocation logged as security event

### 4.6 Event Bus Cursor Atomicity

- **Finding:** QS-A7 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.7](../06_AGENT_ARCHITECTURE.md)
- **Problem:** After polling events, cursor updated via DELETE + INSERT. Process crash between DELETE and INSERT → cursor lost → duplicate event processing.
- **Fix:** Use single `INSERT ... ON CONFLICT DO UPDATE` (upsert) for cursor management.
- **Key files:** EventBus implementation
- **Acceptance criteria:**
  - [ ] Cursor updates are atomic (single upsert statement)
  - [ ] Process crash doesn't cause event re-processing

### 4.7 Dead Letter Queue

- **Finding:** QS-A8 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.8](../06_AGENT_ARCHITECTURE.md)
- **Problem:** `parse_json_response()` fails → output silently replaced with `{}`. No record of what agent said, why it failed, or frequency.
- **Fix:** Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)`. Monitor frequency per agent — high DLQ rate = prompt quality issue.
- **Key files:** JSON parsing utilities, database schema
- **Acceptance criteria:**
  - [ ] All parse failures stored in dead letter queue
  - [ ] DLQ frequency per agent queryable
  - [ ] Alert when any agent DLQ rate > 10% over 24 hours

### 4.8 Priority-Based Message Pruning

- **Finding:** QS-A6 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.9](../06_AGENT_ARCHITECTURE.md)
- **Problem:** When conversation exceeds 150K chars, oldest tool rounds dropped (FIFO). `fund_manager` (10th agent) may lose `position_review` results — the data it needs for allocation.
- **Fix:**
  1. Tag messages with `priority=critical|normal|verbose`
  2. Prune verbose first (market intel), then normal. Keep critical (risk data, position state) longest.
  3. Compaction at merge points (Haiku summarization)
- **Key files:** Agent executor message management, merge nodes
- **Acceptance criteria:**
  - [ ] Messages tagged with priority
  - [ ] Pruning respects priority — critical data retained longest
  - [ ] Compaction step summarizes branch outputs at merge points

### 4.9 Pre-Trade Correlation Check

- **Finding:** CTO H1 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.8](../03_EXECUTION_LAYER.md)
- **Problem:** Pairwise correlation only in post-hoc monitoring loop. Not checked pre-trade.
- **Fix:** Add to `risk_gate.check()`: reject if new position correlation > 0.7 with any existing position.
- **Key files:** `src/quantstack/execution/risk_gate.py`
- **Acceptance criteria:**
  - [ ] New position rejected if correlation > 0.7 with any existing position

### 4.10 Portfolio Heat Budget

- **Finding:** CTO H3 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.8](../03_EXECUTION_LAYER.md)
- **Problem:** No cap on daily notional deployed. Could deploy 100% of equity in a single day.
- **Fix:** Max daily notional deployed (configurable, default 30% of equity/day) in risk gate.
- **Key files:** `src/quantstack/execution/risk_gate.py`
- **Acceptance criteria:**
  - [ ] Daily notional deployment capped (configurable, default 30%)

### 4.11 Sector Concentration Pre-Trade

- **Finding:** CTO H4 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.8](../03_EXECUTION_LAYER.md)
- **Problem:** Sector concentration only checked post-hoc via Herfindahl.
- **Fix:** Pre-trade: reject if sector would exceed 40% concentration.
- **Key files:** `src/quantstack/execution/risk_gate.py`
- **Acceptance criteria:**
  - [ ] Sector concentration checked pre-trade (configurable, default 40% max)

### 4.12 Regime Flip Forced Review

- **Finding:** CTO H5 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.12](../03_EXECUTION_LAYER.md)
- **Problem:** If momentum trade entered in `trending_up` and regime flips to `ranging`, system logs alert but takes no action. Position stays open in hostile regime.
- **Fix:** Moderate mismatch (trending → ranging) → tighten stops 50%. Severe (trending_up → trending_down) → auto-exit within 1 cycle.
- **Key files:** Execution monitor, position management
- **Acceptance criteria:**
  - [ ] Regime flip triggers position review for affected strategies
  - [ ] Severe regime mismatches trigger automatic exit
  - [ ] Regime-at-entry stored per position for comparison

---

## Dependencies

- **Depends on:** Phase 1 (safety), Phase 2 partial (2.1 IC tracking for regime-related items)
- **4.9-4.11 inform Phase 6** (execution layer pre-trade checks)

---

## Validation Plan

1. **Race condition (4.1):** Inject conflicting exit+entry for same symbol → verify exit wins.
2. **Error blocking (4.2):** Crash `data_refresh` → verify `execute_entries` skipped.
3. **Schema validation (4.3):** Return `{"daly_plan": "..."}` from node → verify rejection.
4. **Circuit breaker (4.4):** Fail `daily_planner` 3x → verify skip with safe defaults.
5. **Tool access (4.5):** Attempt `execute_order` from research agent → verify blocked + logged.
6. **Correlation (4.9):** Add AAPL position → attempt MSFT entry with 0.8 correlation → verify rejected.
