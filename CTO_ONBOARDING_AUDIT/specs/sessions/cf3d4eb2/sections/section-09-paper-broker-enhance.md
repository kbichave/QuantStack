# Section 09: Paper Broker Enhancement for TWAP/VWAP

## Overview

The paper broker (`src/quantstack/execution/paper_broker.py`) currently fills every order as a single instant fill: half-spread (2 bps) plus square-root impact (k=5). This is acceptable for IMMEDIATE orders but makes TWAP/VWAP simulation meaningless — the algo scheduler (section-07, section-08) generates time-sliced child orders, and the paper broker must fill each child against a realistic bar-level simulation rather than collapsing them all into one instant fill.

This section adds a bar-aware fill path for TWAP/VWAP child orders while leaving the existing IMMEDIATE fill model completely untouched.

**Depends on:** section-08-twap-vwap (algo scheduler must exist and produce child orders with scheduled times and parent algo type metadata).

**Blocks:** section-14-integration-tests.

---

## Tests (Write First)

All tests go in `tests/unit/execution/test_paper_broker_algo.py`.

```python
# --- TWAP/VWAP child fills against historical bars ---

# Test: TWAP child order filled against historical bar at scheduled_time
#   Given a child order for 100 shares of SPY scheduled at 10:15,
#   and a historical 5-min bar at 10:15 with OHLCV (450, 451, 449.5, 450.5, 120000),
#   the fill price should be derived from the bar VWAP plus directional noise,
#   NOT from current_price with the flat spread+impact model.

# Test: child_qty > bar_volume * participation_rate returns partial fill
#   Given a child order for 5000 shares, bar volume = 120000, participation_rate = 0.02,
#   max_fillable = 2400. Fill should return filled_quantity=2400, partial=True.

# Test: fill price falls within the bar's [low, high] range
#   For any fill, assert: bar_low <= fill_price <= bar_high.
#   Run 50 iterations with random seeds to verify noise stays bounded.

# Test: buy order fills above bar VWAP, sell order fills below bar VWAP (directional noise)
#   A buy child should get fill_price >= bar_vwap (adverse fill).
#   A sell child should get fill_price <= bar_vwap (adverse fill).

# Test: IMMEDIATE orders still use existing instant-fill model
#   An OrderRequest with no algo metadata should route through the original
#   _calc_fill_price path. Verify fill_price matches the spread+impact formula.

# Test: missing historical bar falls back to instant-fill model with warning
#   If no bar data exists for the child's scheduled_time, the broker should
#   log a warning and fill using the existing model rather than crashing.

# Test: participation cap respects configurable rate
#   Default participation_rate=0.02 (2%). If overridden to 0.05 (5%),
#   max_fillable should be bar_volume * 0.05.

# Test: zero bar volume rejects child order
#   If the historical bar has volume=0 (e.g., halt), the child should be
#   rejected with reason "zero volume in target bar".

# Test: fill records a fill_leg via the fill_legs dual-write (section-02)
#   After a TWAP child fill, assert a row exists in fill_legs with the
#   correct parent order_id and leg_sequence.

# Test: partial fill from participation cap leaves remainder for redistribution
#   The returned Fill should have filled_quantity < requested_quantity and
#   partial=True, so the algo scheduler can redistribute the remainder.
```

---

## Background: Current Fill Model

The existing `PaperBroker.execute()` method (lines 167-241 of `paper_broker.py`) does this for every order regardless of algo type:

1. Validates price and quantity
2. Caps fill at 2% of daily volume (partial fill modeling)
3. Calls `_calc_fill_price()` which applies half-spread + sqrt-impact for market orders
4. Records fill to `fills` table
5. Updates portfolio state

This model is correct for IMMEDIATE orders. The problem is that TWAP/VWAP child orders also hit this path, so a 30-minute TWAP execution with 6 children all get the same static slippage model applied 6 times against `current_price`, with no reference to what the market actually did during each time slice.

---

## Implementation Design

### New Method: `execute_algo_child()`

Add a new public method on `PaperBroker` specifically for algo child order fills. The existing `execute()` method remains unchanged for IMMEDIATE orders. The algo scheduler (section-07/08) calls `execute_algo_child()` for TWAP/VWAP children and `execute()` for IMMEDIATE orders.

**File:** `src/quantstack/execution/paper_broker.py`

**Signature:**

```python
def execute_algo_child(
    self,
    req: OrderRequest,
    scheduled_time: datetime,
    participation_rate: float = 0.02,
) -> Fill:
    """
    Fill a TWAP/VWAP child order against the historical bar
    covering scheduled_time.

    Unlike execute(), this method:
    - Looks up the historical intraday bar for req.symbol at scheduled_time
    - Applies a participation cap based on bar volume (not daily volume)
    - Derives fill price from bar VWAP + directional noise (not flat spread model)
    - Rejects if bar volume is zero (halted/no data)
    - Falls back to execute() if no bar data available (with logged warning)
    """
```

### Bar Lookup

The method needs to fetch the historical intraday bar (5-minute granularity) covering the child's `scheduled_time`. This data comes from the same source used by the VWAP volume profile builder in section-08. The lookup function should:

1. Accept symbol and timestamp
2. Return the bar (open, high, low, close, volume) covering that timestamp
3. Return `None` if no bar data is available for that period

A helper function or method is needed:

```python
def _get_historical_bar(
    self, symbol: str, timestamp: datetime
) -> BarData | None:
    """
    Fetch the intraday bar covering the given timestamp.
    Returns None if no data available.
    """
```

`BarData` is a simple dataclass or named tuple: `(open, high, low, close, volume, vwap)`. If the data source provides VWAP directly, use it. Otherwise compute `vwap = (high + low + close) / 3` as a standard approximation (typical price), or use `sum(price * volume) / sum(volume)` from tick data if available.

### Fill Price Calculation for Algo Children

For each child order submitted to the paper broker:

1. **Look up bar:** Find the historical bar covering `scheduled_time`
2. **Participation cap:** `max_fillable = bar.volume * participation_rate` (default 2%)
3. **Fill quantity:** `min(child_qty, max_fillable)`
4. **Fill price derivation:**
   - Start from `bar.vwap`
   - Add directional noise: `direction * uniform(0, (bar.high - bar.low) * 0.3)`
   - Where `direction = +1` for buys (adverse = higher), `-1` for sells (adverse = lower)
   - Clamp result to `[bar.low, bar.high]` to prevent impossible fills
5. **If `child_qty > max_fillable`:** return a partial fill; the algo scheduler will handle redistribution

### Fallback Behavior

If `_get_historical_bar()` returns `None` (no bar data for the scheduled time):
- Log a warning: `"No bar data for {symbol} at {scheduled_time}, falling back to instant fill"`
- Delegate to the existing `execute()` method
- This prevents crashes when historical data has gaps (halts, extended hours, missing data)

### Handling Zero-Volume Bars

If the bar exists but `volume == 0` (trading halt, pre-market with no activity):
- Reject the child order with reason `"zero volume in target bar"`
- The algo scheduler's failure handling (section-08) will redistribute the quantity to a later child

---

## Integration Points

### Algo Scheduler (section-07/08) Calls

The algo scheduler's async execution loop calls the broker for each child order. The dispatch logic should be:

- If `parent.algo_type` is `"twap"` or `"vwap"`: call `broker.execute_algo_child(req, scheduled_time=child.scheduled_time, participation_rate=parent.max_participation_rate)`
- If `parent.algo_type` is `"immediate"` (or no parent): call `broker.execute(req)` as today

This dispatch logic lives in the algo scheduler, not in the paper broker. The paper broker just exposes both methods.

### Fill Legs (section-02)

After `execute_algo_child()` produces a fill, the same `_record_fill()` method writes to the `fills` summary table. The fill_legs dual-write (inserting a row into `fill_legs` with the parent `order_id` and incrementing `leg_sequence`) should be triggered identically to how it works for any other fill — the section-02 enhancement to `_record_fill()` handles this transparently.

### TCA EWMA (section-06)

Each child fill triggers the post-fill TCA EWMA update hook. The arrival price for computing implementation shortfall is the `arrival_price` on the parent order (the mid at signal time), NOT the bar VWAP. This is important: implementation shortfall measures the cost of the entire decision, including timing delay and market drift during execution.

### Audit Trail (section-05)

Each child fill gets its own `execution_audit` row. The NBBO at fill time is approximated from the bar's bid/ask (or `bar.close` as midpoint if no quote data). The `fill_leg_id` field links to the `fill_legs` row for that child.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `participation_rate` | 0.02 (2%) | Max fraction of bar volume fillable per child |
| Noise range | 30% of bar range | `(bar.high - bar.low) * 0.3` — controls fill price dispersion around bar VWAP |
| Fallback on missing data | `True` | Whether to delegate to `execute()` or reject when no bar data |

These can be constructor parameters on `PaperBroker` or config constants at module level. Constructor parameters are preferred for testability — tests can inject different values without monkeypatching.

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/execution/paper_broker.py` | Add `execute_algo_child()` method, `_get_historical_bar()` helper, `BarData` type |
| `tests/unit/execution/test_paper_broker_algo.py` | New test file for all algo child fill tests |

No changes to `execute()`, `_calc_fill_price()`, `_record_fill()`, or `_update_portfolio()`. The existing IMMEDIATE fill path is untouched.

---

## Edge Cases

**Extended hours children:** If the algo scheduler produces a child scheduled outside regular trading hours (9:30-16:00 ET), bar data may not exist. The fallback-to-instant-fill path handles this gracefully.

**Very large child vs. small bar:** If a child order is 10,000 shares but the bar volume is only 5,000 and participation_rate is 2%, `max_fillable = 100`. The partial fill is tiny. This is correct behavior — it signals to the algo scheduler that this time slice has insufficient liquidity, and the scheduler should redistribute.

**Bar data granularity mismatch:** If historical data is 1-minute bars but the scheduler uses 5-minute buckets, the lookup should find the bar whose time range contains `scheduled_time`. If multiple bars cover the window, aggregate them (sum volume, take VWAP across bars, use min low / max high for range).

**Concurrency:** The `_lock` (RLock) on `PaperBroker` serializes all fill recording. Since algo children arrive sequentially from the async scheduler (one at a time per `await`), contention is not a concern. If multiple parents execute concurrently in the future, the existing lock provides safety.
