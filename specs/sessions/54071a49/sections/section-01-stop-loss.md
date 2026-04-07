# Section 1: Stop-Loss Enforcement & Bracket Orders

## Background

QuantStack is an autonomous trading platform where positions can exist with no downside protection. Two critical findings drive this section:

- **C1**: `trade_service.py` allows `stop_price=None` on entry orders. A position can be opened with zero stop-loss protection.
- **C2**: Bracket orders silently degrade to plain market orders when the Alpaca bracket API call fails. The fallback path in `alpaca_broker.py` calls `self.execute(req)` on any exception, which submits a naked entry with no attached stop or take-profit.

This is the single most important safety invariant in the system: **no position may exist without a corresponding stop-loss order.**

## Current State

Before implementing, verify what already exists. As of the last audit:

- `trade_service.py` (line 121) already has a stop-loss guard for `buy`/`long` actions that raises `ValueError` when `stop_price is None`. This was added recently. **Verify it covers all entry action strings** (e.g., `"buy"`, `"long"`, `"buy_to_open"` for options). Exit/close orders are correctly exempt.
- `alpaca_broker.py` has `execute_bracket()` (lines 158-227) that uses Alpaca's native bracket API (`OrderClass.BRACKET` with `StopLossRequest` and `TakeProfitRequest`). However, its fallback on line 227 degrades to a plain `self.execute(req)` with no stop-loss attached.
- `paper_broker.py` returns `False` from `supports_bracket_orders()` and has no bracket simulation.

## Dependencies

- **No upstream dependencies.** This section can be implemented immediately.
- **Downstream**: Section 05 (Database Backups & Durable Checkpoints) depends on the idempotency guards (`client_order_id` deduplication) introduced here.

## Tests

Write tests first in `tests/execution/test_stop_loss_enforcement.py`. The test framework is pytest with `asyncio_mode = "auto"`.

```python
# tests/execution/test_stop_loss_enforcement.py

# Test: submit_order rejects OrderRequest with stop_price=None -> RiskViolation
# Test: submit_order accepts OrderRequest with valid stop_price
# Test: execute_bracket uses Alpaca bracket/OTO API when available
# Test: execute_bracket falls back to separate SL/TP when bracket API fails
# Test: partial fill places stop for filled quantity, not original quantity
# Test: stop price too close to market -> widens to minimum distance + logs warning
# Test: startup reconciliation detects position without stop order -> logs error
# Test: paper_broker mirrors bracket simulation and tracks linked stops
# Test: extended hours uses stop-limit as fallback when stop-market unavailable
```

Guidelines for test implementation:

- Mock the Alpaca `TradingClient` for all broker tests. Do not make real API calls.
- For `trade_service.py` tests, mock the portfolio, risk_gate, broker, audit, and kill_switch dependencies (they are all injected explicitly via `execute_trade()`).
- The "partial fill places stop for filled quantity" test should verify that when an order for 100 shares only fills 60, the stop order is placed for 60 shares, not 100.
- The "startup reconciliation" test should create a portfolio state with a position that has no corresponding open stop order, then verify the reconciliation check logs an error-level message identifying the unprotected position.

## Implementation

### 1. Harden stop-loss validation in `trade_service.py`

**File:** `src/quantstack/execution/trade_service.py`

The existing guard on line 121 checks `action in ("buy", "long")`. Expand this to cover all entry action variants and raise a more specific exception type.

Tasks:
- Expand the action check to include `"buy_to_open"` (options entry) in addition to `"buy"` and `"long"`.
- Change the raised exception from `ValueError` to a dedicated `RiskViolation` exception (or import the existing one if the codebase already has it). This distinguishes "programmer error" from "risk policy rejection" in error handling.
- Ensure the error message includes the symbol and strategy_id for debugging.

### 2. Fix bracket order fallback in `alpaca_broker.py`

**File:** `src/quantstack/execution/alpaca_broker.py` (lines 158-227)

The current fallback on bracket failure (line 224-227) calls `self.execute(req)`, which submits a naked entry order with no stop-loss. This violates the core safety invariant.

Tasks:
- When the bracket API call fails, instead of falling back to a plain `execute()`, fall back to placing the entry order followed by a **separate** stop-loss order linked by `client_order_id`. The sequence is:
  1. Submit the entry order via `execute()`.
  2. If the entry fills (even partially), immediately submit a stop order for the **filled quantity** using `StopOrderRequest` with the original `stop_price`.
  3. If the stop order submission also fails, **cancel the entry order** (or close the position if already filled) and return a rejected Fill. Never allow a filled entry to exist without a stop.
- Add a `client_order_id` parameter to bracket orders for idempotency. Generate it from a deterministic hash of `(symbol, strategy_id, timestamp_bucket)` or use `uuid4()` and persist it. This is the prerequisite that Section 05 depends on.
- Log a warning when falling back from bracket to separate orders, including the original bracket error.

### 3. Handle partial fills correctly

**File:** `src/quantstack/execution/alpaca_broker.py`

When an entry order partially fills (e.g., 60 of 100 shares), the stop order must protect the filled quantity, not the requested quantity.

Tasks:
- After polling for fill, check `fill.filled_quantity`. If it differs from `fill.requested_quantity`, place the stop for `filled_quantity`.
- If the partial fill later completes (Alpaca sends additional fills), the system should detect the increased position and adjust the stop quantity. This can be deferred to a reconciliation pass rather than implemented inline.

### 4. Handle broker rejection of stop price

**File:** `src/quantstack/execution/alpaca_broker.py`

Some brokers reject stop prices that are too close to the current market price (e.g., within the bid-ask spread).

Tasks:
- If the stop order submission returns a rejection with a "price too close" or similar error, widen the stop to the minimum allowed distance. A reasonable default minimum distance is 0.5% from the current price.
- Log a warning when the stop price is widened, including the original and adjusted prices.

### 5. Extended hours stop-limit fallback

**File:** `src/quantstack/execution/alpaca_broker.py`

Stop-market orders are not available during extended hours on some exchanges. The system must use stop-limit orders as a fallback.

Tasks:
- When submitting a stop order during extended hours (check via Alpaca's `get_clock()` or a local time check), use a stop-limit order with the limit price set to `stop_price * 0.99` (1% below stop for sells) to ensure execution even with a small gap.
- This applies to both bracket legs and standalone stop orders.

### 6. Implement bracket simulation in `paper_broker.py`

**File:** `src/quantstack/execution/paper_broker.py`

The paper broker currently does not support brackets. For testing the full trading loop in paper mode, it needs basic bracket simulation.

Tasks:
- Change `supports_bracket_orders()` to return `True`.
- Add an `execute_bracket()` method that:
  1. Calls `execute()` for the entry leg.
  2. Records the stop and take-profit as "pending" orders in an in-memory dict (`_pending_stops: dict[str, PendingStop]`).
  3. Returns the Fill with `bracket_stop_order_id` and `bracket_tp_order_id` populated.
- The pending stops are checked during position price updates. When the current price crosses the stop price, the position is closed automatically. This can be a simple check method called from the trading graph's position monitoring node.
- Define a `PendingStop` dataclass with fields: `symbol`, `stop_price`, `take_profit_price`, `quantity`, `created_at`, `order_id`.

### 7. Startup reconciliation check

**File:** `src/quantstack/execution/alpaca_broker.py` (in `_reconcile_on_startup`)

The existing reconciliation syncs positions from Alpaca but does not verify that each position has a corresponding stop order.

Tasks:
- After syncing positions, query Alpaca for open orders (`get_orders(status=QueryOrderStatus.OPEN)`).
- For each position, check if there is at least one open stop order for that symbol.
- If a position has no stop order, log an `error`-level message: `"UNPROTECTED POSITION: {symbol} has {qty} shares with no stop order. Manual intervention required."`
- Do not auto-fix (placing a stop without knowing the strategy's intended stop level could be wrong). The error log ensures the operator is alerted.

### 8. Add `client_order_id` deduplication to `trade_service.py`

**File:** `src/quantstack/execution/trade_service.py`

This is the idempotency prerequisite for Section 05 (PostgresSaver crash recovery). When LangGraph replays a node after a crash, duplicate order submissions must be detected and rejected.

Tasks:
- Add a `client_order_id` parameter to `execute_trade()`. If not provided, generate one from `uuid4()`.
- Before submitting to the broker, check if a fill already exists for this `client_order_id` in the `fills` table.
- If a fill exists and is not rejected, return the existing fill result (idempotent replay).
- If a fill exists and is rejected, allow resubmission (the previous attempt failed).
- Pass `client_order_id` through to the broker's `execute()` and `execute_bracket()` calls so Alpaca also deduplicates at the broker level.

## Edge Cases to Handle

| Scenario | Expected Behavior |
|----------|-------------------|
| `stop_price=None` on buy order | Raise `RiskViolation`, order never reaches broker |
| `stop_price=None` on sell/close order | Allowed (exit orders are the risk reduction) |
| Bracket API fails, entry fills, stop fails | Cancel/close the entry position immediately |
| Partial fill (60 of 100 shares) | Stop placed for 60 shares, not 100 |
| Stop price too close to market | Widen to minimum 0.5% distance, log warning |
| Extended hours order | Use stop-limit instead of stop-market |
| Duplicate `client_order_id` | Return existing fill (idempotent) |
| Position exists with no stop at startup | Log error, do not auto-fix |

## Verification

After implementation, verify the core invariant: **it must be impossible for `portfolio_state` to contain a position without a corresponding stop order in `open_orders`.** Specifically:

1. Write an integration test that runs the full `execute_trade()` path with a mocked broker and confirms a stop order exists after every successful entry.
2. Run the startup reconciliation against a test portfolio with one protected and one unprotected position. Confirm the unprotected one triggers an error log.
3. Verify the bracket fallback path by mocking the Alpaca bracket API to raise an exception, then confirming separate entry + stop orders are placed.

## Rollback

If this section causes issues:
- The `client_order_id` deduplication and stop-loss validation are purely additive. Reverting means removing the validation check, which is a one-line change.
- The bracket fallback changes are contained within `execute_bracket()`. Reverting to the old `self.execute(req)` fallback restores previous behavior (though this is unsafe).
- Paper broker bracket simulation is new code with no impact on existing paths if `supports_bracket_orders()` is reverted to `False`.
