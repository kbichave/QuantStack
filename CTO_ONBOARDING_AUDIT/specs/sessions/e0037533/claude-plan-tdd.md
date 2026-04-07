# TDD Plan — Phase 4: Agent Architecture & Coordination

Testing framework: **pytest** with **AsyncMock/MagicMock** from unittest.mock. Tests in `tests/unit/`. Existing patterns: ConfigWatcher fixtures with tmp_path, mock ChatModel factories, graph structure validation.

---

## Section 1: Pydantic State Schema Migration

### Pre-Migration Audit Tests

```python
# Test: state_key_audit — log all keys returned by every node, assert all exist in TradingState fields
# Test: state_key_audit — same for ResearchState, SupervisorState, SymbolValidationState
# Test: alpha_signals ghost field — verify _risk_gate_router's alpha_signals reference is resolved
```

### Pydantic Model Tests

```python
# Test: TradingState rejects unknown key (extra="forbid") — {"daly_plan": "..."} raises ValidationError
# Test: TradingState rejects wrong type — {"cycle_number": "not_an_int"} raises ValidationError
# Test: TradingState accepts valid state dict with all required fields
# Test: Annotated[list, operator.add] reducer still works with Pydantic BaseModel
# Test: field_validator catches invalid vol_state value (e.g., "invalid_state")
# Test: model_validator(mode="after") catches cross-field invariant violation
# Test: ResearchState, SupervisorState, SymbolValidationState same rejection/acceptance patterns
```

### Node Output Model Tests

```python
# Test: DataRefreshOutput only allows fields that data_refresh writes to state
# Test: DataRefreshOutput rejects fields not in its schema
# Test: PlanDayOutput.safe_default() returns valid neutral response
# Test: Every node output model's safe_default() passes parent state validation
```

### Input/Output Schema Tests

```python
# Test: TradingInput accepts valid input schema
# Test: TradingOutput constrains graph output to expected shape
```

---

## Section 2: Error Blocking & Node Classification

```python
# Test: blocking node (data_refresh) error → execution gate halts pipeline
# Test: blocking node (position_review) error → execution gate halts pipeline
# Test: blocking node (execute_exits) error → execution gate halts pipeline (entry pipeline also blocked)
# Test: non-blocking node (plan_day) error → safe default used, pipeline continues
# Test: non-blocking node (entry_scan) error → empty candidate list, pipeline continues
# Test: error count > 2 from any source → pipeline halted as safety net
# Test: error count = 2 (boundary) → pipeline continues
# Test: error count = 3 (boundary) → pipeline halted
# Test: mixed errors (1 blocking + 1 non-blocking) → blocking error halts regardless of count
# Test: no errors → pipeline proceeds normally
```

---

## Section 3: Race Condition Fix

```python
# Test: overlapping symbol in exit_orders and entry_candidates → entry removed, exit preserved
# Test: multiple overlapping symbols → all conflicts resolved
# Test: no overlapping symbols → both lists unchanged
# Test: conflict event logged with symbol, exit reasoning, entry reasoning
# Test: resolve_symbol_conflicts failure → safe default drops ALL conflicted entries
# Test: empty exit_orders or empty entry_candidates → no-op
```

---

## Section 4: Node Circuit Breaker

### State Machine Tests

```python
# Test: closed state — failure increments count, stays closed until threshold
# Test: closed → open — 3 consecutive failures trips breaker
# Test: open state — node skipped, safe default returned
# Test: open → half_open — after cooldown_seconds (300s) expires
# Test: half_open success → closed, counter reset
# Test: half_open failure → back to open
# Test: success in closed state → counter reset to 0
```

### DB Persistence Tests

```python
# Test: breaker state persists across test invocations (write, read back, verify)
# Test: concurrent increment — two overlapping cycles both incrementing → correct count (atomic)
# Test: initial state for new breaker_key → default closed/0
```

### LLM Failure Type Tests

```python
# Test: rate limit (429) → trips immediately regardless of threshold
# Test: token limit exceeded → does NOT trip breaker, routes to pruning
# Test: parse failure → counted separately from execution failures
# Test: provider outage (5xx) → trips immediately
```

### Safe Default Tests

```python
# Test: blocking node circuit-broken → safe default includes error flag → execution gate catches it
# Test: non-blocking node circuit-broken → safe default is neutral, pipeline continues
```

### Alert Tests

```python
# Test: 5 consecutive failures → Langfuse event emitted + outbound notification triggered
# Test: 5 consecutive on blocking node → graph halted for cycle
```

---

## Section 5: Tool Access Control

```python
# Test: research agent calls execute_order → blocked, error returned to agent
# Test: research agent calls execute_order → security event logged (agent name, tool, graph, timestamp)
# Test: trading agent calls register_strategy → blocked
# Test: supervisor agent calls any execution tool → blocked
# Test: trading agent calls fetch_portfolio (allowed) → execution proceeds
# Test: blocked tool call does NOT circuit-break the agent
# Test: tool access bypass — call blocked tool through TOOL_REGISTRY directly → still blocked at invocation layer
# Test: ConfigWatcher hot-reload of blocked_tools → new blocks take effect
```

---

## Section 6: Event Bus Cursor Atomicity

```python
# Test: single upsert replaces DELETE+INSERT — verify one SQL statement
# Test: new consumer_id → cursor created via INSERT path of upsert
# Test: existing consumer_id → cursor updated via UPDATE path of upsert
# Test: concurrent cursor updates from multiple consumers → no lost cursors
# Test: UNIQUE constraint exists on consumer_id (or is added by migration)
# Test: PgConnection wrapper supports ON CONFLICT syntax
```

---

## Section 7: Dead Letter Queue

```python
# Test: unparseable LLM output → DLQ row written with agent_name, raw_output, error_type
# Test: DLQ row includes prompt_hash for clustering
# Test: DLQ row includes model_used for debugging
# Test: parse_json_response still returns fallback after DLQ write (behavior unchanged for caller)
# Test: DLQ rate calculation — 5 failures out of 100 calls = 5% rate
# Test: DLQ rate > 5% → Langfuse warn event emitted
# Test: DLQ rate > 10% → Langfuse critical event + outbound notification
# Test: DLQ rate query per agent over 24h rolling window
```

---

## Section 8: Priority-Based Message Pruning

```python
# Test: P2 messages pruned before P1 when over budget
# Test: P1 messages summarized (not pruned) when P2 exhausted and still over budget
# Test: P0 messages never pruned or summarized regardless of budget
# Test: P3 messages never added to LLM context
# Test: type override — risk gate output is P0 even if source agent defaults to P1
# Test: type override — error from blocking node is P0
# Test: Haiku summarization timeout (>2s) → falls back to truncation
# Test: Haiku unavailable → falls back to truncation
# Test: message priority tag correctly set in metadata during construction
# Test: compaction at merge point — verbose branch outputs replaced with summary
```

---

## Section 9: Pre-Trade Risk Gate Additions

### 4.9 Correlation Check

```python
# Test: new position with 0.8 correlation to existing → rejected with RiskViolation
# Test: new position with 0.6 correlation → approved
# Test: boundary — 0.7 exactly → rejected (>=)
# Test: correlation data unavailable (insufficient history) → fail closed, rejected
# Test: data feed error → fail closed, rejected
# Test: new symbol with <20 days history → sector proxy correlation used
# Test: sector proxy requires sector mapping from 4.11 — verify dependency
```

### 4.10 Heat Budget

```python
# Test: daily notional at 31% of equity → rejected
# Test: daily notional at 29% → approved
# Test: boundary — 30% exactly → rejected (>=)
# Test: cumulative — two entries totaling 31% in same day → second rejected
# Test: system-wide — deployment from another graph service counted in budget
# Test: day rollover → budget resets
# Test: configurable threshold — set to 50%, verify 40% approved
```

### 4.11 Sector Concentration

```python
# Test: adding position would push sector to 41% → rejected
# Test: adding position keeps sector at 39% → approved
# Test: unknown sector → treated as own sector (conservative, no concentration trigger)
# Test: configurable threshold — set to 50%, verify 45% approved
# Test: sector mapping data missing → fail closed
```

---

## Section 10: Regime Flip Forced Review

```python
# Test: regime_at_entry stored in DB on position creation
# Test: regime_at_entry loaded into MonitoredPosition from DB on restart
# Test: trending_up → trending_down (severe) → auto-exit order generated with reason
# Test: trending_up → ranging (moderate) → stop tightened by 50%
# Test: stop tightening math — $100 current, $90 stop → new stop $95
# Test: minimum stop floor — after tightening, stop distance >= max(2x ATR, 1% price)
# Test: repeated tightening capped at floor — two flips don't push stop into noise
# Test: stop_price = None → stop SET at floor distance (not tightened)
# Test: auto-exit flows through normal execution pipeline (not a bypass)
# Test: existing positions backfilled with regime_at_entry = 'unknown'
```

---

## Section 11: Database Migration

```python
# Test: circuit_breaker_state table created with correct schema
# Test: agent_dlq table created with correct schema
# Test: regime_at_entry column added to positions table (nullable)
# Test: existing positions have regime_at_entry = 'unknown' after backfill
# Test: loop_cursors has UNIQUE constraint on consumer_id
# Test: migration is idempotent (running twice doesn't error)
```

---

## Integration Tests

```python
# Test: full trading graph cycle — parallel branches produce conflicting symbols → conflict resolved → execution proceeds
# Test: full cycle — blocking node failure → execution gate halts → clean cycle termination
# Test: circuit breaker trips on blocking node → safe default sets error → execution gate halts → graceful end
# Test: circuit breaker state persists across graph invocations → node stays open in next cycle
```
