# Section 02: Mandatory Stop-Loss Enforcement

## Background

QuantStack is an autonomous trading system that executes bracket orders (entry + stop-loss + optional take-profit) through broker adapters. The current execution flow is: `trade_service.execute_trade()` -> `risk_gate.check()` -> `alpaca_broker.execute_bracket()`.

**The problem**: The bracket order path in `trade_service.py` (lines 213-223) only fires when `broker.supports_bracket` and both `stop_price` and `target_price` are set. If bracket submission fails, `alpaca_broker.py` line 223 falls back to a **plain market order** -- the position opens with zero stop protection. This is an existential risk when deploying real capital.

**The goal**: Make it physically impossible for a position to exist without a corresponding stop-loss order. Six defense layers, each catching what the previous one missed.

## Dependencies

- **Depends on**: Section 01 (psycopg3 migration) -- the `bracket_legs` persistence table and startup reconciliation queries use the psycopg3 connection pool.
- **No downstream blockers**: Other sections do not depend on this one.

---

## Tests (Write First)

All test stubs go in `tests/` following the existing directory structure. Use pytest markers as appropriate (`@pytest.mark.integration`, etc.).

### Layer 1-2: Validation (unit tests)

```python
# File: tests/unit/test_stop_loss_enforcement.py

# Test: execute_trade() rejects when stop_price is None -- raises ValueError
# Test: execute_trade() accepts when stop_price is provided -- proceeds to risk gate
# Test: OMS submit() rejects entry orders without stop_price
# Test: OMS submit() allows exit/close orders without stop_price (exits don't need stops)
```

### Layer 3: Bracket-or-contingent pattern (unit + integration tests)

```python
# File: tests/unit/test_bracket_intent.py

# Test: BracketIntent model validates required fields (stop_price mandatory)
# Test: client_order_id format includes millisecond precision and random suffix
# Test: client_order_id is deterministic given same inputs (idempotent retry)

# File: tests/unit/test_broker_bracket.py

# Test: AlpacaBroker.submit_bracket() uses native bracket API when available
# Test: AlpacaBroker.submit_bracket() falls back to entry+contingent SL when bracket fails
# Test: AlpacaBroker.submit_bracket() NEVER falls back to plain order (verify old behavior removed)
# Test: PaperBroker.submit_bracket() tracks SL/TP internally
# Test: EtradeBroker.submit_bracket() implements same interface
```

### Layer 3 continued: Contingent SL path

```python
# File: tests/unit/test_contingent_sl.py

# Test: fill detected via WebSocket within 2s of broker fill
# Test: SL submitted immediately after fill detection
# Test: SL submission retried 3x with exponential backoff on failure
# Test: all 3 SL retries fail -> kill switch triggered for that symbol
# Test: partial fill + SL rejection -> remaining entry qty cancelled + standalone SL for filled qty
```

### Layer 4: Post-submission verification

```python
# File: tests/integration/test_bracket_verification.py

# Test: bracket leg verification runs 5s after submission
# Test: missing leg detected -> SL submitted immediately
# Test: rejected leg detected -> remaining legs cancelled + SL submitted
# Test: verification re-runs every 30s while position open
```

### Layer 5: Startup reconciliation

```python
# File: tests/integration/test_startup_reconciliation.py

# Test: position with active stop order -> no action taken
# Test: position without stop order -> ATR-based SL submitted automatically
# Test: reconciliation logs warning for each auto-fixed position
# Test: trading proceeds after auto-fix (not halted)
# Test: reconciliation runs before first graph cycle
```

### Layer 6: Bracket leg persistence

```python
# File: tests/integration/test_bracket_legs.py

# Test: bracket_legs table created with correct schema
# Test: bracket leg IDs persisted on bracket order submission
# Test: bracket legs queryable after process restart (DB persistence)
```

### Circuit breaker integration

```python
# File: tests/integration/test_circuit_breaker_sl.py

# Test: broker API degraded -> new entries blocked
# Test: broker API degraded -> SL retries continue with exponential backoff
# Test: broker API recovers -> entries resume
```

### Chaos test

```python
# File: tests/regression/test_sl_chaos.py

# Test: Broker API returns HTTP 500 three times during contingent SL fallback
#       -> verify SL eventually placed or kill switch triggered
```

---

## Implementation

### Layer 1: Reject orders without stop_price

**File**: `src/quantstack/execution/trade_service.py`

In `execute_trade()`, add validation before the risk gate check. `stop_price` is a function parameter (not a field on `OrderRequest`), so the validation goes on the function's input:

```python
def execute_trade(self, ..., stop_price: float | None = None, ...):
    if stop_price is None:
        raise ValueError(
            f"stop_price is required for all entry orders. "
            f"Symbol: {symbol}, strategy: {strategy_id}"
        )
    # ... existing risk gate check follows
```

This is the first line of defense. No order reaches the risk gate without a stop.

### Layer 2: OMS enforcement

**File**: `src/quantstack/execution/order_lifecycle.py`

In `submit()` (lines 484-521, compliance checks section), add a check: if the order is an entry (not an exit/close), `stop_price` must be set. This catches any code path that bypasses `trade_service`:

```python
# In submit() compliance checks:
if order.side == "buy" and order.intent != "close":  # entry order
    if not order.stop_price:
        raise ComplianceError("Entry orders require stop_price")
```

### Layer 3: Bracket-or-contingent pattern

**New file**: Define `BracketIntent` in the execution module (e.g., `src/quantstack/execution/models.py` or extend existing models):

```python
class BracketIntent(BaseModel):
    symbol: str
    side: str
    quantity: int
    entry_type: str           # "market" or "limit"
    entry_price: float | None
    stop_price: float         # REQUIRED, not Optional
    target_price: float | None
    client_order_id: str      # format: {strategy_id}_{symbol}_{ts_ms}_{leg}_{random4}
```

Each broker adapter implements `submit_bracket(intent: BracketIntent)` with three strategies attempted in order:

1. **Native bracket** (if supported) -- one API call, broker manages OCO.
2. **Entry + separate contingent SL** -- submit entry, on fill submit SL as separate order.
3. **Reject** -- if neither works, reject the order. **Never fall back to plain market order.**

**Alpaca-specific details** (`src/quantstack/execution/alpaca_broker.py`):
- Use native bracket API (`order_class: "bracket"`). On failure, submit entry, then on fill immediately submit a stop-loss order as a separate `stop` order type.
- Fill detection uses Alpaca's trade update WebSocket stream (already consumed by execution monitor).
- Maximum unprotected window: approximately 2 seconds (WebSocket latency + SL submission).
- If SL submission fails: retry 3 times with exponential backoff (1s, 2s, 4s).
- If all retries fail: trigger kill switch for that symbol -- the position is unprotectable.

**Partial fill handling**: If the entry leg partially fills and the SL leg is rejected, cancel remaining entry quantity immediately, then submit a standalone SL for the filled quantity only. The position must never have exposure without a corresponding stop.

**PaperBroker** (`src/quantstack/execution/paper_broker.py`): Simulate bracket behavior -- track SL/TP internally, evaluate on price ticks.

**E*Trade**: Implement the same interface using E*Trade's conditional order API.

**Critical change**: Remove the existing fallback-to-plain-order code path in `alpaca_broker.py` line 223. This line is the root cause of the vulnerability.

### Layer 4: Post-submission verification

After bracket submission, verify all legs exist. This runs in the execution monitor's tick loop:

1. Query broker for the order and its legs after 5 seconds.
2. If any leg is missing or rejected: cancel entry (if unfilled) or submit missing SL immediately.
3. Re-verify every 30 seconds while the position is open.

Add this verification logic to `src/quantstack/execution/execution_monitor.py` as part of the existing tick loop.

### Layer 5: Startup reconciliation

**File**: `src/quantstack/runners/trading_runner.py` (or a new `src/quantstack/execution/reconciliation.py`)

On trading runner startup, before entering the graph loop:

1. Fetch all open positions from broker.
2. Fetch all open orders from broker.
3. For each position, check for an active stop order on that symbol.
4. If missing: compute ATR-based stop price, submit SL order, log warning.
5. Continue trading (auto-fix, don't halt).

This runs once at startup, not on every cycle.

### Layer 6: Bracket leg persistence

**New table**: `bracket_legs`

```python
class BracketLeg(BaseModel):
    parent_order_id: str
    leg_type: str              # "entry", "stop_loss", "take_profit"
    broker_order_id: str
    status: str
    price: float
    created_at: datetime
```

This replaces the current in-memory `Fill` tracking that is lost on crash. Bracket legs are written to the database on submission and updated on status changes. On restart, the reconciliation step (Layer 5) can query this table to understand the state of in-flight brackets.

Add the table creation SQL to the database migration/setup path.

### Circuit breaker integration

When the broker API is degraded (consecutive failures detected by `AutoTriggerMonitor`):

- **Stop submitting new entries** -- a missed entry is a missed opportunity.
- **Continue retrying stop-loss submissions** with exponential backoff -- a missed stop is unbounded loss.
- Resume entries when the broker API recovers.

This asymmetric behavior (block entries, keep retrying stops) must be explicit in the circuit breaker logic.

---

## Key Invariants

These must remain true at all times, verified by tests and runtime checks:

1. No order can exist with `stop_price=None`.
2. Bracket failure NEVER degrades to a plain order.
3. Every open position has a broker-side stop order (verified on startup and continuously).
4. Bracket leg state survives process restarts (persisted in DB, not held in memory).

---

## Files to Create or Modify

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/execution/trade_service.py` | Modify | Add `stop_price` is-not-None validation before risk gate |
| `src/quantstack/execution/order_lifecycle.py` | Modify | Add entry-order compliance check for `stop_price` |
| `src/quantstack/execution/models.py` (or equivalent) | Create/Modify | Add `BracketIntent` and `BracketLeg` models |
| `src/quantstack/execution/alpaca_broker.py` | Modify | Implement `submit_bracket()`, remove plain-order fallback |
| `src/quantstack/execution/paper_broker.py` | Modify | Implement `submit_bracket()` with internal SL/TP tracking |
| `src/quantstack/execution/execution_monitor.py` | Modify | Add bracket leg verification to tick loop (5s initial, 30s periodic) |
| `src/quantstack/runners/trading_runner.py` | Modify | Add startup reconciliation before first graph cycle |
| Database migration | Create | `bracket_legs` table DDL |
| `tests/unit/test_stop_loss_enforcement.py` | Create | Validation layer tests |
| `tests/unit/test_bracket_intent.py` | Create | BracketIntent model tests |
| `tests/unit/test_broker_bracket.py` | Create | Broker adapter bracket tests |
| `tests/unit/test_contingent_sl.py` | Create | Contingent SL path tests |
| `tests/integration/test_bracket_verification.py` | Create | Post-submission verification tests |
| `tests/integration/test_startup_reconciliation.py` | Create | Startup reconciliation tests |
| `tests/integration/test_bracket_legs.py` | Create | Bracket leg persistence tests |
| `tests/integration/test_circuit_breaker_sl.py` | Create | Circuit breaker asymmetry tests |
| `tests/regression/test_sl_chaos.py` | Create | Chaos/failure injection tests |
