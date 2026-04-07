# Section 08: TWAP/VWAP Scheduling and Execution

## Overview

This section implements the actual time-slicing logic for TWAP and VWAP execution algorithms. The algo scheduler core (section-07) provides the parent/child state machine, async execution loop, crash recovery, and cancellation infrastructure. This section builds on top of that foundation to implement the two scheduling strategies that turn a single large order into a sequence of smaller child orders spread across time.

The current system selects TWAP/VWAP/POV based on ADV thresholds (order_lifecycle.py lines 455-478) but executes everything as a single immediate fill. This means paper trading shows zero market impact, while live trading would incur 10-50 bps impact per trade. The scheduling logic here closes that gap.

**Dependencies:**
- section-06-tca-ewma: EWMA parameters provide per-symbol cost calibration; each child fill triggers a TCA update
- section-07-algo-scheduler-core: AlgoParentOrder, ChildOrder types, async execution loop, parent/child state machine, cancellation triggers, crash recovery

**Blocks:**
- section-09-paper-broker-enhance: Paper broker needs to understand child orders from TWAP/VWAP to simulate realistic fills

---

## Key Design Decisions

**TWAP vs VWAP distinction:** TWAP distributes shares equally across time buckets with random jitter. VWAP weights child sizes by historical intraday volume, concentrating execution during high-volume periods (open/close) where per-share market impact is lower. VWAP is strictly better when volume profile data is available; TWAP is the fallback when it is not.

**POV fallback:** The algo selection logic has four outcomes (IMMEDIATE, TWAP, VWAP, POV). This section implements TWAP and VWAP. POV orders dispatch as VWAP with `max_participation_rate` capped at 5%. This is documented explicitly in the scheduler's dispatch logic so future implementers know POV is not a separate algorithm.

**Volume profile caching:** Historical intraday volume profiles are expensive to compute (aggregate 10-20 days of minute bars). The volume profile builder caches per symbol per day. If no historical data is available, it falls back to a synthetic U-shaped curve that approximates the well-known intraday volume pattern (high at open, low midday, high at close).

**Child quantity invariant:** At all times, `sum(child.filled_qty) == parent.filled_qty`. This is enforced after every child fill update and is the fundamental correctness check for the entire scheduling system.

---

## Tests

All tests live in `tests/unit/execution/test_twap_vwap.py`. These tests assume the algo scheduler core (section-07) types and infrastructure are available.

### TWAP Scheduling

```python
# Test: 1000 shares over 30 min with 5-min buckets produces 6 children, ~167 each
# Test: child quantities sum to parent total (accounting for rounding)
# Test: child scheduled_times span the execution window evenly
# Test: child times have jitter (+/-20% of bucket width)
# Test: child quantities have variation (+/-10%)
```

### VWAP Scheduling

```python
# Test: child sizes proportional to volume profile
# Test: larger children at open/close (high volume), smaller at midday
# Test: falls back to synthetic U-curve when no historical data available
# Test: volume profile cached and not refetched within same day
```

### Parent/Child State Machine (scheduling-specific transitions)

```python
# Test: parent starts PENDING, transitions to ACTIVE on first child submit
# Test: parent transitions to COMPLETING when end_time reached
# Test: parent transitions to COMPLETED when filled >= 99.5% total
# Test: invariant: sum(child.filled_qty) == parent.filled_qty at all times
# Test: child REJECTED -> retry with 50% size up to max_attempts (3)
# Test: child timeout -> cancelled, qty redistributed to next slice
# Test: 3 consecutive child failures -> all active parents paused
```

### POV Fallback

```python
# Test: POV order dispatched as VWAP with max_participation_rate = 5%
```

### Cancellation (scheduling-specific)

```python
# Test: kill switch cancels all active parents and their open children
# Test: risk gate halt cancels all active parents
# Test: execution monitor exit for symbol cancels that symbol's parent only
```

### Crash Recovery

```python
# Test: startup_recovery finds ACTIVE parents and cancels them
# Test: startup_recovery cancels open child broker orders
# Test: startup_recovery logs full context for orphaned state
```

### Algo Performance

```python
# Test: algo_performance row created after parent completion
# Test: implementation_shortfall_bps computed from arrival price vs avg fill price
# Test: vwap_slippage_bps computed from execution VWAP vs benchmark VWAP
# Test: child counts (filled, failed) are accurate
```

---

## Implementation

### File: `src/quantstack/execution/algo_scheduler.py`

This file is created in section-07 with the core infrastructure. This section adds the TWAP and VWAP scheduling functions and the volume profile builder. All additions go into the same file.

### TWAP Scheduling Logic

When an order is submitted with `exec_algo == TWAP`:

1. Create an `AlgoParentOrder` with `start_time = now`, `end_time = now + duration`
2. Divide total quantity into N equal slices where N = `duration_minutes / bucket_size_minutes` (default bucket = 5 minutes)
3. For each slice, create a `ChildOrder` with:
   - `scheduled_time` = bucket_start + random jitter in range `[-0.2 * bucket_width, +0.2 * bucket_width]`
   - `target_quantity` = base_qty + random variation in range `[-0.10 * base_qty, +0.10 * base_qty]`
4. Adjust the last child's quantity so that `sum(all child quantities) == parent total_quantity` exactly (handles rounding)
5. Children are enqueued into the scheduler's priority queue (from section-07) sorted by `scheduled_time`

Function signature:

```python
def schedule_twap(
    parent: AlgoParentOrder,
    bucket_minutes: int = 5,
    jitter_pct: float = 0.20,
    size_variation_pct: float = 0.10,
) -> list[ChildOrder]:
    """Generate child orders for TWAP execution.

    Divides total quantity into equal time slices with randomized
    timing jitter and size variation to reduce detectability.

    The last child's quantity is adjusted so that
    sum(child.target_quantity) == parent.total_quantity exactly.
    """
    ...
```

### VWAP Scheduling Logic

Same structure as TWAP but child sizes are weighted by historical intraday volume:

1. Load the volume profile for the symbol via `build_volume_profile(symbol)`
2. For each time bucket in the execution window, compute the fraction of total daily volume that trades in that bucket
3. For each bucket: `child_qty = int(total_qty * bucket_volume_pct / sum_remaining_volume_pct)`
4. The denominator uses `sum_remaining_volume_pct` (volume fraction from execution start to end), not total daily volume -- this ensures the full parent quantity is allocated even if execution starts mid-day
5. Apply the same jitter to scheduled times as TWAP (but not size variation, since sizes are already weighted)
6. Adjust the last child for rounding

Function signature:

```python
def schedule_vwap(
    parent: AlgoParentOrder,
    volume_profile: dict[str, float],
    bucket_minutes: int = 5,
    jitter_pct: float = 0.20,
) -> list[ChildOrder]:
    """Generate child orders for VWAP execution.

    Child sizes are proportional to the historical intraday volume
    profile. Concentrates execution during high-volume periods
    (open/close) where per-share market impact is lower.

    Falls back to TWAP-style equal slicing if volume_profile is empty.
    """
    ...
```

### Volume Profile Builder

A function that takes a symbol and returns a mapping from time bucket labels to volume percentages. This is the data source for VWAP scheduling.

```python
def build_volume_profile(
    symbol: str,
    lookback_days: int = 20,
    bucket_minutes: int = 5,
) -> dict[str, float]:
    """Build normalized intraday volume profile from historical bars.

    Averages volume distribution across the last `lookback_days` trading
    days. Each bucket's value is the fraction of total daily volume that
    trades during that bucket (values sum to ~1.0).

    Returns synthetic U-curve fallback if fewer than 5 days of
    historical bar data are available.

    Results are cached per (symbol, date) -- calling twice on the same
    day returns the cached result without re-querying.
    """
    ...
```

**Data source:** Historical 1-minute or 5-minute intraday bars. These are available from Alpha Vantage (`TIME_SERIES_INTRADAY`) or from the existing bar data already acquired during data updates.

**Synthetic U-curve fallback:** When historical data is insufficient (fewer than 5 days of bars), return a hardcoded volume profile based on the well-documented intraday volume pattern:

| Time | Volume % (approx) |
|------|-------------------|
| 9:30-10:00 | 12% |
| 10:00-10:30 | 8% |
| 10:30-11:00 | 6% |
| 11:00-11:30 | 5% |
| 11:30-12:00 | 4% |
| 12:00-12:30 | 4% |
| 12:30-13:00 | 4% |
| 13:00-13:30 | 4% |
| 13:30-14:00 | 5% |
| 14:00-14:30 | 6% |
| 14:30-15:00 | 8% |
| 15:00-15:30 | 12% |
| 15:30-16:00 | 22% |

This approximates the "U-shape" or "bathtub curve" observed in US equity markets.

**Caching:** Use a simple dict keyed by `(symbol, date.today())`. The cache is module-level and resets naturally each day. No TTL management needed -- the scheduler runs within a single trading day.

### POV Dispatch

POV is not a separate scheduling algorithm. When the OMS selects `exec_algo == POV` (order size > 5% of ADV), the scheduler dispatches it as VWAP with a hard participation cap:

```python
def dispatch_order(parent: AlgoParentOrder) -> list[ChildOrder]:
    """Route parent order to the appropriate scheduling algorithm.

    - TWAP: equal time slices with jitter
    - VWAP: volume-weighted slices
    - POV: dispatched as VWAP with max_participation_rate = 0.05
    """
    ...
```

### Child Failure Handling

When a child order fails, the scheduling logic must decide what to do with the unfilled quantity. These handlers are called by the async execution loop (section-07) when it observes a child state transition.

**REJECTED (buying power):** Reduce child size by 50%, resubmit. Maximum 3 attempts per child. After 3 failures, the unfilled quantity is redistributed to subsequent children.

**Timeout (no fill within 2x bucket duration):** Cancel the child via broker API. Redistribute the unfilled quantity equally across remaining pending children.

**3 consecutive failures across any children:** Pause ALL active parents (not just the one with failures). This handles network-level failures that would affect all orders. Log an alert. Resume requires manual intervention or the next scheduler loop iteration after conditions clear.

**Quantity redistribution function:**

```python
def redistribute_unfilled(
    unfilled_qty: int,
    remaining_children: list[ChildOrder],
) -> None:
    """Distribute unfilled quantity from a failed child across remaining
    pending children. Modifies children in place.

    If no remaining children, the unfilled quantity becomes parent shortfall
    (logged but not force-filled).
    """
    ...
```

### Algo Performance Recording

After a parent order reaches terminal state (COMPLETED or CANCELLED), record execution quality metrics:

```python
@dataclass
class AlgoPerformance:
    parent_order_id: str
    symbol: str
    side: str
    algo_type: str                          # "twap" | "vwap"
    total_qty: int
    filled_qty: int
    arrival_price: float                    # mid-price at decision time
    avg_fill_price: float                   # VWAP across all child fills
    benchmark_vwap: float | None            # market VWAP over execution window
    implementation_shortfall_bps: float     # (avg_fill - arrival) / arrival * 10000
    vwap_slippage_bps: float | None         # (avg_fill - benchmark_vwap) / benchmark_vwap * 10000
    delay_cost_bps: float                   # cost of waiting from decision to first fill
    market_impact_bps: float                # residual after removing delay cost
    num_children: int
    num_children_filled: int
    num_children_failed: int
    max_participation_rate: float
    actual_participation_rate: float
    decision_time: datetime
    first_fill_time: datetime | None
    last_fill_time: datetime | None
    scheduled_end_time: datetime
```

**Implementation shortfall decomposition:**
- `delay_cost_bps`: price movement between decision time and first fill, attributable to waiting
- `market_impact_bps`: `implementation_shortfall_bps - delay_cost_bps`, the residual impact of executing
- `vwap_slippage_bps`: how the execution VWAP compares to the market VWAP over the same window (requires benchmark VWAP from bar data)

This row is written to the `algo_performance` table (schema defined in section-01).

---

## Database Schema

The following tables are defined in section-01 (schema foundation) and used here:

**`algo_parent_orders`** -- stores parent order lifecycle (created in section-07, populated here during scheduling)

**`algo_child_orders`** -- stores child order lifecycle with `scheduled_time`, `target_quantity`, and link to parent

**`algo_performance`** -- stores post-completion execution quality metrics (written here after parent reaches terminal state)

No new tables are introduced in this section. All schema is defined in section-01 and the state machine types are defined in section-07.

---

## Integration Points

### With TCA EWMA (section-06)

After each child fill completes:
1. The fill is recorded as a fill leg (section-02)
2. The TCA EWMA update runs for the symbol + time bucket of the fill
3. This means a 30-minute TWAP with 6 children produces 6 incremental EWMA updates, giving faster convergence than a single fill

### With Audit Trail (section-05)

Each child fill gets its own `execution_audit` row with `fill_leg_id` linking to the specific fill leg. The `algo_rationale` field records the scheduling parameters (e.g., "TWAP: 6 children over 30 min, bucket=5min, participation_cap=2%").

### With Paper Broker (section-09)

The paper broker receives child orders from the scheduler and must simulate realistic fills against historical bars. Section-09 implements the paper broker side. The contract between this section and section-09 is: the scheduler submits child orders via the broker's `execute()` method, and the broker returns fill results (full, partial, or rejected).

### With Business Calendar (section-03)

TWAP/VWAP scheduling must only generate child orders during market hours. The business calendar provides `is_market_open(datetime)` and `next_market_open()` to ensure children are not scheduled during pre/post-market or holidays.

### With Execution Monitor

The execution monitor can trigger an exit for a specific symbol. When this happens, the scheduler must cancel the parent order for that symbol and all its pending children. The cancellation infrastructure is in section-07; this section ensures the scheduler's dispatch logic respects the cancellation signal by checking parent status before submitting each child.

---

## Edge Cases and Failure Modes

**Rounding:** Integer share quantities mean children may not divide evenly. The last child absorbs the remainder. For example, 1000 shares / 6 children = 166 * 5 + 170. This is handled in both `schedule_twap` and `schedule_vwap`.

**Execution window shorter than one bucket:** If the remaining market time is less than `bucket_minutes`, fall back to a single child (effectively IMMEDIATE). This prevents scheduling children after market close.

**Volume profile with zero-volume buckets:** Some symbols may have zero volume in certain intraday buckets (e.g., illiquid names at midday). Skip zero-volume buckets entirely rather than generating zero-quantity children.

**Stale price feed during execution:** If the price feed goes stale (no update for >30 seconds), the scheduler pauses child submission. Resumes on reconnection. If stale for >5 minutes, cancels the parent order. This protects against executing into unknown market conditions.

**Market close approaching:** If `scheduled_end_time` is within 5 minutes of market close and the parent is not yet complete, the scheduler does NOT submit a final aggressive order. The unfilled quantity becomes reported shortfall. This is the conservative default -- better to under-fill than to take excessive market impact in the close auction.
