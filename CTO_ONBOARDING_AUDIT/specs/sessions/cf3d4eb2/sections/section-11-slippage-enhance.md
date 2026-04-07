# Section 11: Slippage Model Enhancement

## Overview

The paper broker (`src/quantstack/execution/paper_broker.py`) currently uses hardcoded constants for slippage simulation: a fixed 2 bps half-spread and a `k=5` square-root impact model. Every symbol, at every time of day, gets the same slippage treatment. This means paper trading results do not reflect reality -- a liquid large-cap at midday and a small-cap at the open both incur identical friction.

This section replaces those fixed constants with EWMA-calibrated per-symbol, per-time-bucket parameters from the `tca_parameters` table (built in section-06-tca-ewma), adds time-of-day slippage profiles, and introduces accuracy tracking so the system detects when its slippage model drifts from realized execution costs.

**Depends on:** section-06-tca-ewma (provides `tca_parameters` table and EWMA update logic)

**Blocks:** section-14-integration-tests

---

## Tests (Write First)

All tests go in `tests/unit/execution/test_slippage_enhance.py`.

```python
# --- EWMA-calibrated slippage in paper broker ---

# Test: paper broker uses EWMA-calibrated spread when tca_parameters exist
#   Setup: Insert a tca_parameters row for symbol="SPY", time_bucket="midday"
#          with ewma_spread_bps=3.5 and ewma_impact_bps=8.0 and sample_count >= 50
#   Action: Submit a market order for SPY during the midday window
#   Assert: The spread component of the fill price uses 3.5 bps (not the hardcoded 2 bps)
#   Assert: The impact component uses 8.0 as the coefficient (not hardcoded k=5)

# Test: paper broker falls back to fixed constants when no EWMA data
#   Setup: No tca_parameters rows for symbol "NEWSTOCK"
#   Action: Submit a market order for NEWSTOCK
#   Assert: Fill price uses HALF_SPREAD_BPS=2 and k=5 (existing behavior, unchanged)

# Test: conservative multiplier applied when sample_count < 50
#   Setup: tca_parameters for symbol="AAPL", time_bucket="morning",
#          ewma_spread_bps=4.0, sample_count=10
#   Action: Submit market order for AAPL during morning window
#   Assert: Spread component uses 4.0 * multiplier (multiplier = 2.0 - (10/50) * 1.0 = 1.8)

# --- Time-of-day slippage profiles ---

# Test: time-of-day multiplier applied to slippage estimate
#   Setup: tca_parameters with baseline ewma_spread_bps=3.0 for all buckets
#   Action: Submit orders at 9:45 (morning), 12:00 (midday), 14:30 (afternoon), 15:45 (close)
#   Assert: Morning fill has spread * 1.3x multiplier
#   Assert: Midday fill has spread * 1.0x multiplier (baseline)
#   Assert: Afternoon fill has spread * 1.1x multiplier
#   Assert: Close fill has spread * 1.2x multiplier

# Test: time bucket classification is correct at boundaries
#   Action: Classify 9:30 -> "morning", 11:00 -> "midday", 14:00 -> "afternoon", 15:30 -> "close"
#   Assert: Each boundary falls into the expected bucket

# --- Slippage accuracy tracking ---

# Test: slippage accuracy tracked: predicted vs realized ratio stored
#   Setup: Pre-trade forecast predicts 5.0 bps total cost
#   Action: Fill completes with 7.5 bps realized slippage
#   Assert: slippage_accuracy row created with predicted=5.0, realized=7.5, ratio=0.667

# Test: alert triggered when accuracy ratio drifts beyond 0.5x or 2.0x
#   Setup: Insert 10 slippage_accuracy rows with ratio averaging 0.4 (predicting 2.5x too high)
#   Action: Run drift detection check
#   Assert: Alert logged indicating model over-prediction drift

# Test: no alert when accuracy ratio is within bounds
#   Setup: Insert 10 slippage_accuracy rows with ratio averaging 1.1
#   Action: Run drift detection check
#   Assert: No alert logged

# Test: accuracy tracking handles zero predicted slippage gracefully
#   Setup: Pre-trade forecast predicts 0.0 bps (e.g., limit order)
#   Action: Fill completes with 1.0 bps realized
#   Assert: Row stored with ratio = None or sentinel, no division-by-zero error
```

---

## Schema: `slippage_accuracy` Table

New table for tracking model prediction quality. Created as part of the schema migration (section-01-schema-foundation defines the migration infrastructure; this table is added here).

```python
@dataclass
class SlippageAccuracy:
    id: int                # auto-increment PK
    order_id: str          # FK to orders
    symbol: str
    time_bucket: str       # "morning" | "midday" | "afternoon" | "close"
    predicted_bps: float   # from pre_trade_forecast() at order time
    realized_bps: float    # from fill vs arrival price
    ratio: float | None    # predicted / realized (None if predicted == 0)
    fill_timestamp: datetime
    created_at: datetime
```

File: `src/quantstack/execution/schema.py` (or wherever the section-01 migration infrastructure places DDL). The table is additive -- no existing tables are modified.

---

## Implementation Details

### 1. Paper Broker: Replace Fixed Constants with EWMA Lookup

**File to modify:** `src/quantstack/execution/paper_broker.py`

The current `_compute_fill_price` method (around line 264) uses:
- `HALF_SPREAD_BPS = 2` (class constant, line 131)
- `k = 5` hardcoded in the square-root impact formula (line 272)

The change introduces a helper method that looks up calibrated parameters before computing the fill price.

**New method signature on `PaperBroker`:**

```python
def _get_slippage_params(self, symbol: str, timestamp: datetime) -> tuple[float, float]:
    """Return (spread_bps, impact_coefficient) for the given symbol and time.

    Lookup order:
    1. tca_parameters row for (symbol, time_bucket) with sample_count >= 50
       -> use ewma_spread_bps and ewma_impact_bps directly
    2. tca_parameters row with sample_count < 50
       -> apply conservative multiplier: value * (2.0 - sample_count / 50)
    3. No row found
       -> return (HALF_SPREAD_BPS, 5.0) -- existing defaults
    """
    ...
```

**Modification to `_compute_fill_price`:**

Replace the hardcoded constants with a call to `_get_slippage_params`. The rest of the computation (direction, spread cost, square-root impact, fill price) stays the same -- only the input parameters change.

```python
# Before (current):
spread_cost = ref_price * self.HALF_SPREAD_BPS / 10_000
impact_bps = 5 * math.sqrt(quantity / max(1, daily_volume * 0.01))

# After:
spread_bps, impact_k = self._get_slippage_params(symbol, timestamp)
spread_cost = ref_price * spread_bps / 10_000
impact_bps = impact_k * math.sqrt(quantity / max(1, daily_volume * 0.01))
```

The `timestamp` parameter is already available in the fill context (it is the fill time). The time bucket classification reuses the same bucket definitions from section-06-tca-ewma:
- morning: 09:30-11:00 ET
- midday: 11:00-14:00 ET
- afternoon: 14:00-15:30 ET
- close: 15:30-16:00 ET

### 2. Time-of-Day Slippage Multipliers

Even with EWMA calibration, the `tca_parameters` table stores per-bucket values that already reflect time-of-day variation (since fills are bucketed by time). However, when falling back to defaults (no EWMA data), the fixed constants need time-of-day adjustment.

**Time-of-day multipliers (applied to spread component only):**

| Bucket | Spread Multiplier | Rationale |
|--------|-------------------|-----------|
| morning | 1.3x | Wider spreads at open, price discovery period |
| midday | 1.0x | Baseline liquidity |
| afternoon | 1.1x | Slight widening as volume tapers |
| close | 1.2x | Wider spreads but offset by volume spike |

These multipliers are applied in two scenarios:
- When using default constants (no EWMA data): `spread_bps = HALF_SPREAD_BPS * multiplier`
- When EWMA data has low sample count (< 10): blend the EWMA value with the default-with-multiplier to avoid over-reliance on sparse data

When EWMA data has 10+ samples, the EWMA values already encode time-of-day effects through the per-bucket storage, so no additional multiplier is applied.

**Helper function:**

```python
def get_time_of_day_multiplier(time_bucket: str) -> float:
    """Return spread multiplier for the given time bucket."""
    ...
```

Place this in a shared location accessible to both the paper broker and the slippage accuracy tracker. A reasonable location is `src/quantstack/execution/slippage.py` -- a new module dedicated to slippage model logic. This avoids bloating the paper broker with model calibration concerns.

### 3. Slippage Accuracy Tracking

**File to create:** `src/quantstack/execution/slippage.py`

This module contains:
- `get_time_of_day_multiplier()` -- time bucket spread multipliers
- `classify_time_bucket(timestamp) -> str` -- maps a datetime to a bucket name (shared with section-06 TCA EWMA; if that section already defines this, import from there instead of duplicating)
- `record_slippage_accuracy(order_id, symbol, time_bucket, predicted_bps, realized_bps)` -- writes to `slippage_accuracy` table
- `check_slippage_drift(symbol, lookback_count=20) -> str | None` -- returns an alert message if the rolling average ratio of the last N fills is outside [0.5, 2.0], or None if within bounds

**Integration point -- after every fill:**

In the fill completion hook (the same hook chain used by TCA EWMA update in section-06 and audit trail in section-05), add a call to `record_slippage_accuracy`. The predicted value comes from the pre-trade forecast that was computed when the order was submitted (stored on the Order object or passed through the fill context). The realized value is `abs(fill_price - arrival_price) / arrival_price * 10_000`.

**Integration point -- drift alerting:**

After recording accuracy, call `check_slippage_drift`. If it returns an alert string, log it at WARNING level and (optionally) write to the system alerts mechanism. This is a diagnostic signal for the Supervisor graph -- the model is miscalibrated and may need investigation.

**Drift detection logic:**

```python
def check_slippage_drift(symbol: str, lookback_count: int = 20) -> str | None:
    """Check if predicted/realized ratio has drifted outside acceptable bounds.

    Query the last `lookback_count` slippage_accuracy rows for the symbol.
    Compute the mean ratio (excluding None values).
    If mean < 0.5: model is over-predicting slippage (real fills are 2x+ better than forecast).
    If mean > 2.0: model is under-predicting slippage (real fills are 2x+ worse than forecast).
    Return an alert message string, or None if within bounds.
    """
    ...
```

### 4. Backward Compatibility

- IMMEDIATE orders: unchanged behavior when no EWMA data exists (same fixed constants)
- TWAP/VWAP child fills (from section-09): each child fill goes through `_compute_fill_price`, so they automatically pick up the calibrated parameters
- Existing tests: any test that asserts exact fill prices based on the 2 bps / k=5 constants will need the test fixture to ensure no `tca_parameters` rows exist for the test symbol, which is the default state in a clean test DB
- The `HALF_SPREAD_BPS` class constant remains as a documented default, referenced by `_get_slippage_params` when no EWMA data is available

### 5. Module Structure Summary

```
src/quantstack/execution/
    slippage.py          # NEW: slippage model logic
        get_time_of_day_multiplier(time_bucket) -> float
        classify_time_bucket(timestamp) -> str
        record_slippage_accuracy(order_id, symbol, time_bucket, predicted_bps, realized_bps)
        check_slippage_drift(symbol, lookback_count) -> str | None
    paper_broker.py      # MODIFIED: _get_slippage_params, _compute_fill_price
```

---

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| `tca_parameters` table missing or query fails | Paper broker cannot look up EWMA data | Fall back to fixed constants; log warning. Never let a DB read failure block a fill. |
| EWMA values are stale (last_updated weeks ago) | Calibrated params may not reflect current market | Accept staleness -- stale EWMA is still better than uncalibrated defaults. The conservative multiplier handles low-sample cases. |
| Drift alert fires frequently | Alert fatigue | Only alert on sustained drift (20-fill rolling window, not per-fill). Consider suppressing repeated alerts for the same symbol within a cooldown period. |
| Division by zero in ratio calculation | Predicted slippage = 0 for limit orders or tiny orders | Store ratio as None when predicted_bps == 0. Exclude from drift calculation. |
| Time bucket classification wrong for pre/post market fills | Fills outside 9:30-16:00 have no matching bucket | Default to "midday" multiplier for out-of-hours fills (most conservative baseline). |

---

## Acceptance Criteria

1. Paper broker fill prices for a symbol with 50+ fills in `tca_parameters` use EWMA-calibrated spread and impact values (not hardcoded 2 bps / k=5).
2. Paper broker fill prices for a symbol with no `tca_parameters` data use the existing fixed constants (backward compatible).
3. Time-of-day multipliers are applied when using default constants (no EWMA data).
4. Every fill writes a `slippage_accuracy` row comparing predicted vs realized slippage.
5. A WARNING-level log is emitted when the rolling accuracy ratio drifts outside [0.5, 2.0] for a symbol.
6. All tests in `tests/unit/execution/test_slippage_enhance.py` pass.
