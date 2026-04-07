# Section 06: TCA EWMA Feedback Loop

## Problem

The TCA engine (`src/quantstack/core/execution/tca_engine.py`) computes pre-trade cost forecasts using static Almgren-Chriss coefficients (eta=0.142, gamma=0.314, beta=0.60). Post-trade, it measures implementation shortfall in bps (lines 139-153 of `order_lifecycle.py`) and persists to `TCAStore`. But the realized costs never feed back into future forecasts. If slippage is consistently 2x forecast, the next trade still uses the same stale parameters.

The Almgren-Chriss module (`src/quantstack/core/execution/almgren_chriss.py`) has a calibration-from-fills path (lines 219-267) that requires 10+ fills via least-squares, but it is not wired into the fill lifecycle. This section implements a faster, per-fill EWMA update layer that runs alongside the existing A-C coefficients.

## Dependencies

- **Requires section-02 (fill-legs):** EWMA updates consume fill leg data to compute realized cost components accurately. The `fill_legs` table and `compute_fill_vwap()` helper must exist before this section can compute realized spread and impact from granular fill data.
- **Blocks section-08 (TWAP/VWAP):** The algo scheduler's child fills will trigger EWMA updates through the same hook.
- **Blocks section-11 (slippage-enhance):** The paper broker's enhanced slippage model reads EWMA-calibrated parameters from the `tca_parameters` table created here.

## Tests

Write tests in `tests/unit/execution/test_tca_ewma.py`.

### EWMA Update Logic

```python
# Test: first fill for a symbol creates a tca_parameters row with realized values as initial EWMA
# Test: EWMA update with alpha=0.1: new_forecast = 0.9 * old_forecast + 0.1 * realized_cost
# Test: sample_count increments by 1 on each fill
# Test: parameters stored per (symbol, time_bucket) — two fills for same symbol in different buckets produce two rows
# Test: morning fill at 10:00 updates "morning" bucket; midday fill at 12:00 updates "midday" bucket
```

### Conservative Multiplier

```python
# Test: at sample_count=1, multiplier is approximately 2.0
# Test: at sample_count=25, multiplier is approximately 1.5
# Test: at sample_count=50, multiplier is exactly 1.0
# Test: at sample_count=100, multiplier is still 1.0 (no further decay past 50)
```

### Pre-Trade Forecast Integration

```python
# Test: pre_trade_forecast uses EWMA values directly when sample_count >= 50
# Test: pre_trade_forecast applies conservative multiplier when sample_count < 50
# Test: pre_trade_forecast falls back to default A-C coefficients when no EWMA row exists for symbol+bucket
# Test: higher EWMA cost estimate results in smaller position size recommendation
```

## Implementation

### 1. Schema: `tca_parameters` Table

**File:** `src/quantstack/db.py` (add to schema initialization alongside existing tables)

Create a `tca_parameters` table with this structure:

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | `TEXT NOT NULL` | e.g., "SPY" |
| `time_bucket` | `TEXT NOT NULL` | One of: "morning", "midday", "afternoon", "close" |
| `ewma_spread_bps` | `DOUBLE PRECISION NOT NULL` | EWMA of realized spread cost in basis points |
| `ewma_impact_bps` | `DOUBLE PRECISION NOT NULL` | EWMA of realized market impact in basis points |
| `ewma_total_bps` | `DOUBLE PRECISION NOT NULL` | EWMA of total realized execution cost in basis points |
| `sample_count` | `INTEGER NOT NULL DEFAULT 0` | Number of fills contributing to this EWMA |
| `last_updated` | `TIMESTAMP NOT NULL` | Timestamp of most recent EWMA update |

Primary key: `(symbol, time_bucket)`. Each row is upserted (INSERT ON CONFLICT UPDATE) after every fill.

Time bucket definitions:
- **morning:** 09:30 - 11:00 ET
- **midday:** 11:00 - 14:00 ET
- **afternoon:** 14:00 - 15:30 ET
- **close:** 15:30 - 16:00 ET

### 2. EWMA Update Function

**File:** `src/quantstack/execution/tca_ewma.py` (new file)

Core function signature:

```python
def update_ewma_after_fill(
    order_id: str,
    symbol: str,
    fill_timestamp: datetime,
    arrival_price: float,
    fill_price: float,
    fill_quantity: int,
    adv: float,
) -> None:
    """Update EWMA cost parameters after a completed fill.

    Computes realized cost components (spread, impact, total) from fill data,
    then applies exponential weighted moving average update to the stored
    parameters for this symbol and time bucket.

    Alpha = 0.1 (smoothing factor).
    """
```

This function:

1. Determines the time bucket from `fill_timestamp` using the ET time ranges above.
2. Computes realized cost components from the fill:
   - `realized_total_bps = abs(fill_price - arrival_price) / arrival_price * 10_000`
   - Spread and impact decomposition uses fill legs from section-02 if multiple legs exist; for single-leg fills, the total is attributed proportionally using the existing A-C ratio defaults.
3. Loads current EWMA parameters for `(symbol, time_bucket)` from DB.
4. If no existing row: inserts with realized values as initial EWMA and `sample_count = 1`.
5. If existing row: applies EWMA update formula with alpha = 0.1:
   ```
   new_value = alpha * realized + (1 - alpha) * old_value
   ```
   Increments `sample_count` by 1. Upserts.

### 3. Conservative Multiplier

**Same file:** `src/quantstack/execution/tca_ewma.py`

```python
def conservative_multiplier(sample_count: int) -> float:
    """Return cost multiplier that decays linearly from 2.0 to 1.0 over 50 fills.

    Prevents undertrained models from underestimating costs. Once 50 fills
    are accumulated, the multiplier is 1.0 (no inflation).
    """
```

Formula: `multiplier = max(1.0, 2.0 - (sample_count / 50))`

At 0 fills: 2.0. At 25 fills: 1.5. At 50 fills: 1.0. At 100 fills: still 1.0.

### 4. Time Bucket Resolution

**Same file:** `src/quantstack/execution/tca_ewma.py`

```python
def resolve_time_bucket(timestamp: datetime) -> str:
    """Map a timestamp to one of four intraday time buckets.

    Returns one of: 'morning', 'midday', 'afternoon', 'close'.
    Timestamp is interpreted in US/Eastern timezone.
    """
```

Buckets:
- 09:30 <= t < 11:00 → "morning"
- 11:00 <= t < 14:00 → "midday"
- 14:00 <= t < 15:30 → "afternoon"
- 15:30 <= t < 16:00 → "close"
- Outside market hours → "close" (conservative: uses highest-cost bucket as default)

### 5. Pre-Trade Forecast Integration

**File:** `src/quantstack/core/execution/tca_engine.py` (modify existing `pre_trade_forecast()`)

Before returning the forecast, add a lookup step:

1. Query `tca_parameters` for `(symbol, current_time_bucket)`.
2. If row exists and `sample_count >= 50`: override the default A-C forecast components with EWMA values. These are calibrated from real fills and more accurate than the static coefficients.
3. If row exists and `sample_count < 50`: use EWMA values but multiply by `conservative_multiplier(sample_count)`. This inflates the estimate to account for uncertainty from limited data.
4. If no row exists: use default A-C coefficients unchanged (existing behavior).

The resulting `total_expected_bps` feeds into position sizing logic downstream. Higher estimated cost means smaller recommended position size. This is the automatic feedback loop: poor execution quality self-corrects by reducing future position sizes.

### 6. Fill Hook Integration

**File:** `src/quantstack/execution/order_lifecycle.py` (modify fill completion path)

After the existing fill recording logic (which writes to `fills` table and now also `fill_legs` from section-02), add a call to `update_ewma_after_fill()`. This runs on every completed fill, including:

- IMMEDIATE order fills (single leg)
- Future TWAP/VWAP child fills (once section-08 lands, each child fill triggers an independent EWMA update)

The hook should be fail-safe: if the EWMA update raises an exception, log the error but do not fail the fill. The EWMA is an optimization layer, not a correctness requirement. A failed update means one data point is lost, not that the fill is invalid.

### 7. Relationship to Existing A-C Calibration

The EWMA layer and the existing least-squares calibration in `almgren_chriss.py` (lines 219-267) are separate systems:

- **EWMA (this section):** Fast, per-fill updates. Smooths realized costs with exponential decay. Provides immediate feedback after every trade.
- **A-C least-squares (existing):** Periodic batch recalibration from accumulated fill data. Requires 10+ fills. Could run weekly or monthly for deeper coefficient tuning.

This plan implements only the EWMA layer. The A-C least-squares calibration remains available for future periodic recalibration but is not wired into the fill lifecycle by this section.

## File Summary

| File | Action |
|------|--------|
| `src/quantstack/db.py` | Add `tca_parameters` table to schema init |
| `src/quantstack/execution/tca_ewma.py` | New — EWMA update, conservative multiplier, time bucket resolution |
| `src/quantstack/core/execution/tca_engine.py` | Modify `pre_trade_forecast()` to use EWMA parameters |
| `src/quantstack/execution/order_lifecycle.py` | Add post-fill hook calling `update_ewma_after_fill()` |
| `tests/unit/execution/test_tca_ewma.py` | New — all tests listed above |
