# 05 — Agent Architecture: Fix the Multi-Agent Coordination

**Priority:** P2
**Timeline:** Week 4-6
**Gate:** Race conditions resolved, errors block execution, state validated, tool access controlled.

---

## Why This Section Matters

21 agents across 3 graphs with excellent role specialization. The architecture is a B+. But the agents don't coordinate safely — parallel branches can conflict on the same symbol, errors accumulate without blocking execution, state merges accept typos silently, and any agent can call any tool. These are the coordination bugs that produce unpredictable behavior at scale.

---

## 5.1 Race Condition: Parallel Branches Conflict on Same Symbol

**Finding ID:** QS-A2
**Severity:** CRITICAL
**Effort:** 1-2 days

### The Problem

Position review and entry scan run in parallel:
```
plan_day → position_review → execute_exits → merge_parallel
        → entry_scan → earnings_router → merge_parallel
```

If `position_review` decides to exit XYZ and `entry_scan` simultaneously decides to enter XYZ, the merged state has conflicting orders. No transactional guarantee prevents buy and sell orders for the same symbol in the same cycle.

### The Fix

Two options (recommend Option A):
- **Option A:** Conflict resolution at `merge_parallel` — if same symbol in exits and entries, exits take priority (risk-off bias)
- **Option B:** Sequential execution — exits complete before entries begin

### Acceptance Criteria

- [ ] Same symbol cannot appear in both exit and entry lists within one cycle
- [ ] Exits take priority over entries when conflict detected
- [ ] Conflict events logged for analysis

---

## 5.2 Errors Must Block Execution

**Finding ID:** QS-A4
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

Every node has the same pattern: catch exception → append to `errors` list → continue. Even critical nodes like `data_refresh` and `daily_planner` — if they crash, the graph continues. The `errors` list accumulates but no downstream node checks it. A cycle can have 5 errors from 5 different nodes and still reach `execute_entries`.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add error-count check before `execute_entries`: `if len(errors) > 2: skip execution` |
| 2 | Critical nodes (`data_refresh`, `safety_check`, `risk_sizing`) are blocking — failure halts pipeline |
| 3 | Non-critical nodes (`market_intel`, `trade_reflector`) can fail without halting |
| 4 | Classify each node as `blocking` or `non_blocking` in graph definition |

### Acceptance Criteria

- [ ] Critical node failures halt the pipeline — no trades executed on corrupt data
- [ ] Error count > 2 prevents execution for that cycle
- [ ] Node classification (blocking/non-blocking) documented

---

## 5.3 State Schema Validation

**Finding ID:** QS-A3
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Node returns are merged into `TradingState` via dict update. If a node returns `{"daly_plan": "..."}` (typo) instead of `{"daily_plan": "..."}`, the typo key is silently added and `daily_plan` remains stale from the prior cycle. No schema validation, no type checking at state boundaries.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add Pydantic model for `TradingState` with typed fields |
| 2 | Each node declares its output schema |
| 3 | State merge validates against schema — rejects unknown keys |
| 4 | Type mismatches logged and rejected |

### Acceptance Criteria

- [ ] `TradingState` is a Pydantic model with typed fields
- [ ] Typos in state keys rejected at merge point
- [ ] Type mismatches detected and logged

---

## 5.4 Circuit Breaker for Failing Nodes

**Finding ID:** QS-A5
**Severity:** HIGH
**Effort:** 1 day

### The Problem

If `daily_planner` fails 5 consecutive cycles, the graph still calls it on cycle 6. No backoff, no circuit breaker, no fallback.

### The Fix

| Failures | Action |
|----------|--------|
| 3 consecutive | Skip node, use safe defaults (empty plan = no new entries) |
| 5 consecutive | Alert and halt graph for investigation |
| On success | Reset counter |

### Acceptance Criteria

- [ ] Per-node failure tracking across cycles
- [ ] Auto-skip after 3 consecutive failures
- [ ] Alert after 5 consecutive failures

---

## 5.5 Tool Access Control (Blocked Tools per Graph)

**Finding ID:** QS-A9, CTO OC-5
**Severity:** HIGH
**Effort:** 1 day

### The Problem

All tools in an agent's YAML config are bound. No negative access control. Research agents have no block on execution tools. Trading agents have no block on strategy registration tools. A misconfigured `agents.yaml` could allow `hypothesis_critic` to call `execute_order`.

### The Fix

| Graph | Blocked Tools | Rationale |
|-------|--------------|-----------|
| Research | `execute_order`, `cancel_order`, `activate_kill_switch` | Research cannot trade |
| Trading | `register_strategy`, `train_model` | Trading cannot modify strategies |
| Supervisor | All execution tools (read-only mode) | Supervisor monitors, doesn't act |

Implement as `blocked_tools` field in `agents.yaml` per graph, enforced in `agent_executor.py` (5-line guard).

### Acceptance Criteria

- [ ] Research agents cannot call execution tools
- [ ] Trading agents cannot call strategy registration tools
- [ ] Blocked tool invocation logged as security event

---

## 5.6 Missing Agent Roles

**Finding ID:** QS-A1
**Severity:** CRITICAL (Compliance), HIGH (others)
**Effort:** 5-10 days total

### The Problem

A real trading desk staffs roles that QuantStack lacks entirely. Not all need to be LLM agents — most can be deterministic functions.

| Missing Role | Implementation | Priority | Effort |
|-------------|---------------|----------|--------|
| **Compliance Officer** | Pre-trade regulatory validation node (deterministic). Wash sale, PDT, margin checks. | CRITICAL | Covered in Section 03 |
| **Corporate Actions Monitor** | Daily check for dividends, splits, mergers on holdings. Adjust cost basis, flag thesis changes. | HIGH | 2 days |
| **Factor Exposure Monitor** | Track portfolio beta, sector tilts, style exposure. Alert on drift. Tool exists but no agent. | HIGH | 1 day |
| **Performance Attribution** | Decompose P&L by factor, timing, selection, cost. Currently runs nightly in supervisor. Move to per-cycle. | HIGH | 2 days |
| **Market Microstructure** | Analyze bid-ask dynamics, trade flow, volume clocks. Informs execution algo selection. | MEDIUM | 3 days |
| **Counterparty Risk** | Monitor broker health, clearing risk, concentration limits. | MEDIUM | 1 day |

**Note:** Trade Reconciliation was initially listed as missing but was found to exist in `guardrails/agent_hardening.py:463-550` and `execution_monitor._reconcile_loop()`. RETRACTED.

### Acceptance Criteria

- [ ] Corporate actions checked daily for all holdings
- [ ] Factor exposure computed and limit-checked per cycle
- [ ] Performance attribution available per-cycle (not just nightly)

---

## 5.7 Event Bus Cursor Atomicity

**Finding ID:** QS-A7
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

After polling events, the cursor is updated via DELETE + INSERT. If process crashes between DELETE and INSERT, cursor is lost. Next poll re-reads all events — causing duplicate processing.

### The Fix

Use single `INSERT ... ON CONFLICT DO UPDATE` (upsert) for cursor management. Atomic operation.

### Acceptance Criteria

- [ ] Cursor updates are atomic (single upsert statement)
- [ ] Process crash doesn't cause event re-processing

---

## 5.8 Dead Letter Queue for Agent Outputs

**Finding ID:** QS-A8
**Severity:** HIGH
**Effort:** 1 day

### The Problem

When `parse_json_response()` fails, the output is silently replaced with `{}` or `[]`. No record of what the agent actually said, why it failed, or how often this happens.

### The Fix

Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)`. Monitor frequency per agent — high DLQ rate = prompt quality issue that needs investigation.

### Acceptance Criteria

- [ ] All parse failures stored in dead letter queue
- [ ] DLQ frequency per agent queryable
- [ ] Alert when any agent DLQ rate > 10% over 24 hours

---

## 5.9 Message Pruning: Priority-Based Retention

**Finding ID:** QS-A6
**Severity:** HIGH
**Effort:** 2 days

### The Problem

When conversation exceeds 150K chars, oldest tool rounds are dropped (FIFO). By the time `fund_manager` (10th agent) runs, it may have lost `position_review` results — the very data it needs for allocation decisions. No summary of pruned content.

### The Fix

Two approaches (implement both):
1. **Priority-based pruning:** Tag messages with `priority=critical|normal|verbose`. Prune verbose first (market intel), then normal. Keep critical (risk data, position state) longest.
2. **Compaction at merge points:** Haiku-tier summarization at `merge_parallel` and `merge_pre_execution` (detailed in Section 07).

### Acceptance Criteria

- [ ] Messages tagged with priority
- [ ] Pruning respects priority — critical data retained longest
- [ ] Compaction step summarizes branch outputs at merge points

---

## 5.10 Inter-Graph Urgency Channel

**Finding ID:** CTO GC2
**Severity:** CRITICAL
**Effort:** 2-3 days

### The Problem

Supervisor discovers critical issues (strategy IC decay, kill switch condition) but communicates via Postgres EventBus. The trading graph only polls at the start of each 5-min cycle. Worst case: 5 minutes of trading on a decayed strategy before the trading graph sees the event.

### The Fix

Add a "priority interrupt" mechanism — either a shared Redis pub/sub channel or file sentinel that trading checks before each trade execution, not just at cycle start. Or: direct process signal (SIGUSR1) from supervisor to trading container.

### Acceptance Criteria

- [ ] Critical supervisor events reach trading graph within seconds, not minutes
- [ ] Pre-trade check for emergency events before every `execute_entries`
- [ ] Latency reduced from 5 minutes to <30 seconds for critical events

---

## 5.11 Serialization Bottleneck in Trading Pipeline

**Finding ID:** CTO GC3
**Severity:** CRITICAL
**Effort:** 3-5 days

### The Problem

The `trade_debater` (600s timeout) must complete before `fund_manager` (300s timeout) can evaluate. Critical path from entry_scan to execute_entries is 900s minimum — 15 minutes for LLM reasoning in a 5-minute cycle. The watchdog timeout is 600s, but the full pipeline can take 900s+. This causes missed entries due to stale signals by the time execution happens.

### The Fix

Two options:
- **Option A:** Stream debater outputs incrementally to fund_manager (process candidates as debated, not as batch)
- **Option B:** Tighter per-candidate timeout (120s) and parallelize across candidates via `Send()`

### Acceptance Criteria

- [ ] Trading pipeline critical path < 300s (fits within 5-min cycle)
- [ ] No stale signals at execution time due to LLM reasoning delay

---

## 5.12 Research Queue Row Lock

**Finding ID:** CTO GH1
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

When claiming tasks from `research_queue`, there's no `SELECT FOR UPDATE` lock. If two research cycles overlap (possible on restart), both could claim the same task.

### The Fix

Add `FOR UPDATE SKIP LOCKED` to the claim query.

### Acceptance Criteria

- [ ] Research queue claim uses `SELECT FOR UPDATE SKIP LOCKED`
- [ ] Concurrent claims cannot grab same task

---

## 5.13 Covariance Matrix Identity Fallback

**Finding ID:** CTO GH2
**Severity:** HIGH
**Effort:** 1 day

### The Problem

If OHLCV data is stale (>2 trading days), portfolio optimizer falls back to `last_covariance` from state. If empty (first cycle), falls back to identity matrix scaled by 0.02 — assuming zero correlation between all assets. This destroys diversification logic.

### The Fix

Never fall back to identity. Use EWMA covariance estimator that degrades gracefully, or block portfolio construction until fresh data is available.

### Acceptance Criteria

- [ ] Identity matrix fallback removed
- [ ] EWMA covariance estimator used when data is stale
- [ ] Portfolio construction blocked if no valid covariance available

---

## 5.14 Consecutive Failures Must Escalate

**Finding ID:** CTO GH3
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

When a graph fails 3+ consecutive cycles, it logs CRITICAL but takes no action. No kill switch trigger, no supervisor notification, no trading pause.

### The Fix

After 3 consecutive failures: (1) publish `GRAPH_FAILURE` event to EventBus, (2) if trading graph, auto-trigger kill switch, (3) supervisor handles with dedicated recovery playbook.

### Acceptance Criteria

- [ ] 3 consecutive failures → EventBus notification
- [ ] Trading graph failures → automatic kill switch
- [ ] Supervisor monitors and attempts recovery

---

## 5.15 Supervisor Conditional Routing

**Finding ID:** CTO GH4
**Severity:** HIGH
**Effort:** 1-2 days

### The Problem

Supervisor runs all 7 nodes sequentially every cycle, even when healthy. Strategy pipeline backtests every draft every 5 minutes. EOD sync runs at 2 PM.

### The Fix

Add conditional routing: skip `strategy_pipeline` if no new drafts. Skip `eod_data_sync` unless within EOD window. Skip `diagnose_issues` + `execute_recovery` if health_check returns healthy.

### Acceptance Criteria

- [ ] Supervisor skips unnecessary nodes based on conditions
- [ ] Cycle time reduced when system is healthy
- [ ] All conditions documented

---

## Summary: Agent Architecture Delivery

| # | Item | Effort | Priority |
|---|------|--------|----------|
| 5.1 | Race condition fix | 1-2 days | CRITICAL |
| 5.2 | Errors block execution | 1 day | CRITICAL |
| 5.3 | State schema validation | 2 days | CRITICAL |
| 5.4 | Node circuit breaker | 1 day | HIGH |
| 5.5 | Tool access control | 1 day | HIGH |
| 5.6 | Missing agent roles | 5-10 days | HIGH |
| 5.7 | Event bus cursor atomicity | 0.5 day | HIGH |
| 5.8 | Dead letter queue | 1 day | HIGH |
| 5.9 | Priority-based pruning | 2 days | HIGH |
| 5.10 | Inter-graph urgency channel | 2-3 days | CRITICAL |
| 5.11 | Serialization bottleneck fix | 3-5 days | CRITICAL |
| 5.12 | Research queue row lock | 0.5 day | HIGH |
| 5.13 | Covariance identity fallback fix | 1 day | HIGH |
| 5.14 | Consecutive failure escalation | 0.5 day | HIGH |
| 5.15 | Supervisor conditional routing | 1-2 days | HIGH |

**Total estimated effort: 23-32 engineering days.**
