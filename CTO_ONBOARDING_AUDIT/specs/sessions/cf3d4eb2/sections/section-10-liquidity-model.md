# Section 10: Liquidity Model

## Overview

This section implements a `LiquidityModel` class that provides spread estimation, depth estimation, time-of-day adjustment, and stressed exit scenario analysis. The model integrates into the risk gate as a pre-trade check (returning PASS, SCALE_DOWN, or REJECT) and into the continuous risk monitor for portfolio-level stressed exit alerts.

**Dependencies:** section-01-schema-foundation (tables must exist). This section is parallelizable with sections 08, 09, 11, 12, and 13.

**Files to create/modify:**

- **New:** `src/quantstack/execution/liquidity_model.py` -- the `LiquidityModel` class
- **Modify:** `src/quantstack/execution/risk_gate.py` -- integrate `LiquidityModel.pre_trade_check()` between the existing ADV check and participation cap
- **New:** `tests/unit/execution/test_liquidity_model.py` -- all tests for this section

---

## Tests (Write These First)

All tests go in `tests/unit/execution/test_liquidity_model.py`. The test structure follows the existing project pattern using pytest.

### Spread Estimation

```python
# Test: spread_estimation returns bps value for a given symbol from historical data
def test_spread_estimation_returns_bps_for_symbol():
    """
    Given historical quote data for symbol 'SPY',
    LiquidityModel.estimate_spread('SPY') returns a positive float
    representing the typical bid-ask spread in basis points.
    """
    ...

# Test: spread_estimation returns per-time-bucket values when time_bucket is specified
def test_spread_estimation_per_time_bucket():
    """
    estimate_spread('SPY', time_bucket='morning') returns a different
    (typically wider) spread than estimate_spread('SPY', time_bucket='midday').
    """
    ...
```

### Depth Estimation

```python
# Test: depth_estimation returns shares available per time bucket
def test_depth_estimation_returns_shares_per_bucket():
    """
    Given a symbol with known daily volume and a volume profile,
    estimate_depth('SPY', time_bucket='midday') returns the estimated
    number of shares executable in that bucket without excessive impact.
    Depth = daily_volume * bucket_volume_pct.
    """
    ...
```

### Pre-Trade Check (PASS / SCALE_DOWN / REJECT)

```python
# Test: order well within depth limit returns PASS
def test_order_within_depth_returns_pass():
    """
    Order of 100 shares where estimated depth is 50,000 shares.
    100 / 50000 = 0.2% of depth, well under 10% threshold.
    Result: PASS.
    """
    ...

# Test: order exceeding 10% of depth returns SCALE_DOWN or REJECT
def test_order_exceeding_depth_threshold_returns_scale_down_or_reject():
    """
    Order of 6,000 shares where estimated depth is 50,000 shares.
    6000 / 50000 = 12%, exceeds 10% threshold.
    Result: SCALE_DOWN (with recommended reduced quantity) or REJECT
    depending on how far over the threshold the order is.
    """
    ...

# Test: order for illiquid symbol with very low depth returns REJECT
def test_illiquid_symbol_returns_reject():
    """
    Order of 500 shares where estimated depth is 1,000 shares.
    500 / 1000 = 50%, far exceeds threshold.
    Result: REJECT with reason describing insufficient liquidity.
    """
    ...
```

### Time-of-Day Adjustment

```python
# Test: open = 1.5x spread multiplier, midday = 1.0x, close = 1.3x
def test_time_of_day_multiplier_open():
    """
    At 9:45 (market open window), spread estimate is 1.5x the baseline.
    """
    ...

def test_time_of_day_multiplier_midday():
    """
    At 12:00 (midday), spread estimate uses baseline (1.0x multiplier).
    """
    ...

def test_time_of_day_multiplier_close():
    """
    At 15:45 (close window), spread estimate is 1.3x the baseline.
    """
    ...
```

### Stressed Exit Scenario

```python
# Test: stressed exit computes portfolio-level slippage for simultaneous exit
def test_stressed_exit_portfolio_slippage():
    """
    Given a portfolio with 3 positions of known sizes and per-symbol
    slippage estimates, stressed_exit_slippage() returns the sum of
    (position_size_i * estimated_slippage_bps_i) across all positions.
    """
    ...

# Test: stressed exit alerts when slippage exceeds threshold
def test_stressed_exit_alert_above_threshold():
    """
    When total stressed slippage exceeds the configurable threshold
    (e.g., 100 bps of portfolio value), the check returns an alert.
    """
    ...

# Test: stressed exit below threshold does not alert
def test_stressed_exit_no_alert_below_threshold():
    """
    When total stressed slippage is within threshold, no alert is raised.
    """
    ...
```

### Risk Gate Integration

```python
# Test: liquidity check integrates into risk gate between ADV check and participation cap
def test_liquidity_check_in_risk_gate_ordering():
    """
    The risk gate calls LiquidityModel.pre_trade_check() after the
    existing ADV/volume validation check and before the participation cap.
    A REJECT from the liquidity model prevents the order from reaching
    subsequent checks.
    """
    ...
```

---

## Implementation Details

### LiquidityModel Class

Create `src/quantstack/execution/liquidity_model.py` with a `LiquidityModel` class.

```python
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class LiquidityVerdict(Enum):
    PASS = "pass"
    SCALE_DOWN = "scale_down"
    REJECT = "reject"


@dataclass
class LiquidityCheckResult:
    verdict: LiquidityVerdict
    reason: str
    recommended_quantity: int | None = None  # populated when SCALE_DOWN
    estimated_spread_bps: float = 0.0
    estimated_depth_shares: int = 0


@dataclass
class StressedExitResult:
    total_slippage_bps: float
    portfolio_value: float
    slippage_dollar_estimate: float
    alert: bool
    per_symbol_breakdown: dict  # symbol -> slippage_bps


class LiquidityModel:
    """
    Estimates spread, depth, and time-of-day liquidity for pre-trade gating
    and portfolio-level stressed exit analysis.
    """

    # Time-of-day spread multipliers
    TOD_MULTIPLIERS: dict  # "open" -> 1.5, "midday" -> 1.0, "afternoon" -> 1.1, "close" -> 1.3

    # Depth threshold: reject/scale if order > this fraction of estimated depth
    DEPTH_THRESHOLD: float  # 0.10 (10%)

    # Stressed exit alert threshold in bps of portfolio value
    STRESS_THRESHOLD_BPS: float  # configurable, default 100

    def estimate_spread(self, symbol: str, time_bucket: str | None = None) -> float:
        """
        Return estimated bid-ask spread in bps for the symbol.

        Uses historical quote data. If time_bucket is provided, applies
        the time-of-day multiplier. Falls back to a conservative default
        if no historical data is available for the symbol.
        """
        ...

    def estimate_depth(self, symbol: str, time_bucket: str | None = None) -> int:
        """
        Return estimated executable shares for the symbol in the given time bucket.

        Computed as: daily_volume * bucket_volume_pct.
        Uses the same intraday volume profile that VWAP scheduling uses
        (section-08). Falls back to daily_volume / num_buckets if no
        intraday profile is available.
        """
        ...

    def pre_trade_check(
        self, symbol: str, order_quantity: int, current_time: datetime | None = None
    ) -> LiquidityCheckResult:
        """
        Pre-trade liquidity gate. Called by risk_gate.py between the ADV
        volume validation and the participation cap check.

        Logic:
        1. Determine time bucket from current_time (or now).
        2. Estimate depth for symbol + time bucket.
        3. Compute order_quantity / estimated_depth.
        4. If ratio <= DEPTH_THRESHOLD: PASS.
        5. If ratio > DEPTH_THRESHOLD but <= 2 * DEPTH_THRESHOLD: SCALE_DOWN,
           recommended_quantity = estimated_depth * DEPTH_THRESHOLD.
        6. If ratio > 2 * DEPTH_THRESHOLD: REJECT.
        """
        ...

    def stressed_exit_slippage(
        self, positions: list, current_time: datetime | None = None
    ) -> StressedExitResult:
        """
        Portfolio-level stressed exit analysis. Assumes all positions are
        liquidated simultaneously.

        For each position:
        - Estimate spread + impact slippage for full position size
        - Apply time-of-day multiplier
        - Sum across portfolio

        Called by the continuous risk monitor (every 60s alongside existing
        position drift checks). If total slippage exceeds STRESS_THRESHOLD_BPS
        of portfolio value, returns alert=True.
        """
        ...
```

### Spread Estimation Data Source

The spread estimation needs historical bid-ask data. The system has two available sources:

1. **Alpaca IEX quotes** (15-min delayed, free/unlimited) -- best for real-time spread snapshots
2. **Alpha Vantage** -- has historical intraday data that can be used to derive spreads from bar data (proxy: `(high - low) / midpoint` as a rough spread estimate from OHLCV when quote data is unavailable)

For the initial implementation, use a hybrid approach:
- If `tca_parameters` has EWMA spread data for the symbol (populated by section-06), use that as the primary source since it reflects actual trading experience.
- Otherwise, estimate from recent intraday bar data: average `(high - low) / ((high + low) / 2)` across bars as a proxy for spread, then scale down (bar range overstates spread; use a configurable scaling factor, default 0.2).
- If no data at all, use a conservative default (e.g., 10 bps for large-caps, 25 bps for mid-caps, 50 bps for small-caps based on ADV thresholds).

### Depth Estimation Data Source

Depth is derived from intraday volume profiles. If section-08 (TWAP/VWAP) has already built the volume profile builder with daily caching, reuse that function. Otherwise, implement a simplified version:

- Fetch daily volume from the orders/positions data or Alpha Vantage
- Divide into time buckets using either historical intraday bars or the synthetic U-curve fallback (same approach as section-08's VWAP volume profile)

### Time Bucket Classification

Reuse the same time buckets defined in TCA (section-06):

| Bucket | Time Range (ET) |
|--------|----------------|
| morning | 9:30 -- 11:00 |
| midday | 11:00 -- 14:00 |
| afternoon | 14:00 -- 15:30 |
| close | 15:30 -- 16:00 |

Add a helper function `classify_time_bucket(dt: datetime) -> str` that returns the bucket name. If the time falls outside market hours, return the nearest bucket.

The "open" multiplier (1.5x) applies during the first 30 minutes of the "morning" bucket (9:30-10:00). The rest of the "morning" bucket uses a 1.2x multiplier. This granularity is important because spreads at 9:35 are meaningfully wider than at 10:45.

### Risk Gate Integration

In `src/quantstack/execution/risk_gate.py`, the existing pre-trade check sequence (lines 293-688) runs checks in order. The liquidity model check should be inserted **after** the existing volume validation (ADV < 500K warn) and **before** the participation cap (> 1% ADV scale down).

The existing flow:

1. Daily halt sentinel
2. Restricted symbol check
3. Volume validation (ADV < 500K warn)
4. **NEW: LiquidityModel.pre_trade_check()** -- insert here
5. Daily loss limit
6. Participation cap (> 1% ADV scale down)
7. ... remaining checks

If the liquidity model returns REJECT, the risk gate returns rejection immediately (same pattern as other hard blocks). If SCALE_DOWN, the recommended quantity is passed forward to the participation cap check, which may further reduce it.

### Continuous Risk Monitor Integration

The stressed exit check runs in the execution monitor's periodic loop (or a dedicated risk monitor loop). The pattern:

1. Every 60 seconds, call `LiquidityModel.stressed_exit_slippage()` with all current positions
2. If `alert == True`, log a warning with full breakdown and optionally trigger a supervisor notification
3. Do NOT auto-exit on stressed liquidity alone -- this is informational. The trading graph decides whether to reduce exposure.

### Configuration

All thresholds should be configurable via environment variables or a config dict, with sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LIQUIDITY_DEPTH_THRESHOLD` | 0.10 | Max fraction of estimated depth per order |
| `LIQUIDITY_STRESS_THRESHOLD_BPS` | 100 | Portfolio stress slippage alert threshold |
| `LIQUIDITY_DEFAULT_SPREAD_BPS` | 10 | Default spread for unknown symbols |
| `LIQUIDITY_BAR_SPREAD_SCALE` | 0.2 | Scale factor for bar-range-to-spread proxy |

---

## Key Design Decisions

**Why a separate class instead of inline in risk_gate.py:** The liquidity model has two consumers (risk gate pre-trade check and continuous risk monitor stressed exit). Keeping it as a standalone class with a clear interface avoids duplicating logic and makes it testable in isolation.

**Why SCALE_DOWN instead of only PASS/REJECT:** Binary pass/reject is too coarse. A 1,000-share order in a stock with 8,000 shares of estimated bucket depth (12.5% of depth) is suboptimal but not catastrophic. Scaling down to 800 shares (10% of depth) lets the trade proceed at a safer size. This matches the pattern already used by the participation cap in the risk gate.

**Why bar-range proxy for spread estimation:** Direct bid-ask quote history is not stored in the current schema. The bar high-low range is a rough but available proxy. The 0.2 scaling factor is deliberately conservative (overstates spread), which is the safe direction for a risk check. As EWMA spread data from section-06 accumulates, it progressively replaces the proxy with actual trading experience.

**Why not block on stressed exit:** A stressed exit alert means "if everything goes wrong simultaneously, slippage would be painful." This is a tail scenario. Auto-exiting positions based on a liquidity stress test would cause unnecessary turnover and realized losses. The alert feeds into the supervisor graph for strategic position reduction decisions.

---

## Dependencies on Other Sections

- **section-01-schema-foundation:** Any new tables referenced here must be created in the schema migration.
- **section-06-tca-ewma (optional enhancement):** If `tca_parameters` table is populated, the liquidity model uses EWMA spread data for better estimates. The model works without it (falls back to bar-range proxy or defaults).
- **section-08-twap-vwap (optional enhancement):** If the volume profile builder exists, depth estimation reuses it. Otherwise, a simplified version is implemented locally.

These are soft dependencies -- the liquidity model works with graceful fallbacks when these sections are not yet implemented.
