# Phase 6: Execution Layer — TDD Plan

Testing framework: **pytest** (existing project). Tests in `tests/`. Existing fixtures: `MonitoredPosition` builder, `PaperBroker` with in-memory SQLite context, `OrderRequest` helper. DB tests use `db_conn()` context managers.

---

## 4. Partial Fill Tracking (6.2)

### Schema & Data Layer
```python
# Test: fill_legs table created by migration
# Test: inserting a fill leg with all fields succeeds
# Test: unique constraint on (order_id, leg_sequence) prevents duplicates
# Test: inserting fill leg with nonexistent order_id fails (FK constraint)
```

### Fill Recording
```python
# Test: single fill creates one fill_leg row and updates fills summary
# Test: two partial fills for same order create two legs with correct sequences
# Test: fills summary row has VWAP of all legs after multiple partials
# Test: VWAP computation: 50@100 + 50@102 = avg 101.00
# Test: fill recording works for both paper_broker and alpaca_broker paths
```

### VWAP Helper
```python
# Test: compute_fill_vwap returns correct VWAP for multiple legs
# Test: compute_fill_vwap returns single price for single-leg fills
# Test: compute_fill_vwap raises for nonexistent order_id
```

---

## 5. SEC Compliance (6.5)

### PDT Checker
```python
# Test: 0 day trades in 5-day window → order APPROVED
# Test: 2 day trades in 5-day window → order APPROVED
# Test: 3 day trades in 5-day window AND account < $25K → 4th day trade REJECTED
# Test: 3 day trades in 5-day window AND account >= $25K → order APPROVED
# Test: day trade on Monday, window counts only business days (skip weekend)
# Test: day trade counting resets after 5 business days roll forward
# Test: partial fill that closes intraday position counts as day trade
# Test: options PDT matches on full OCC contract symbol, not underlying
# Test: two different SPY option contracts closed same day count as 2 day trades
# Test: position opened yesterday and closed today is NOT a day trade
```

### Day Trade Recording
```python
# Test: buy then sell same symbol same day creates day_trade record
# Test: buy then sell different symbol same day does NOT create day_trade record
# Test: multiple round-trips same symbol same day creates multiple records
```

### Wash Sale Tracker
```python
# Test: sell at loss creates pending_wash_losses record with 30-day window
# Test: sell at gain does NOT create pending_wash_losses record
# Test: buy within 30 days of pending loss → wash sale flagged, loss disallowed
# Test: buy after 30 days of pending loss → no wash sale
# Test: wash sale adjusts cost basis of replacement shares by disallowed amount
# Test: pending_wash_losses marked resolved after buy triggers
# Test: pre-trade warning surfaces for buy with open wash window
# Test: pre-trade warning does NOT block the order
```

### Tax Lot Manager
```python
# Test: buy fill creates tax lot with correct cost basis and date
# Test: sell fill matches FIFO — oldest lot consumed first
# Test: sell of 150 shares with lots [100@$50, 100@$55] → first lot fully consumed, second partially
# Test: gain/loss computed correctly per lot
# Test: wash sale adjustment added to cost basis before gain/loss calculation
# Test: selling more shares than open lots raises error (or handles gracefully)
```

### Margin Calculator
```python
# Test: long equity order requires 50% cash margin
# Test: long option order requires premium as margin
# Test: debit spread requires net premium as margin
# Test: order rejected when margin_required exceeds available equity
# Test: reducing position does NOT require additional margin
```

### Business Day Calendar
```python
# Test: weekends excluded from business day count
# Test: market holidays excluded (e.g., July 4th)
# Test: 5 business days from Monday = next Monday
# Test: 30 calendar days correctly spans across months
```

---

## 6. Best Execution Audit Trail (6.6)

```python
# Test: IMMEDIATE order fill creates execution_audit row
# Test: audit row captures NBBO bid/ask at fill time
# Test: price_improvement_bps computed correctly (positive = favorable)
# Test: price_improvement_bps negative when fill is worse than midpoint
# Test: algo_selected and algo_rationale populated
# Test: query "fills worse than NBBO midpoint" returns correct results
# Test: audit row created even when NBBO fetch fails (with null NBBO fields)
```

---

## 7. TCA EWMA Feedback Loop (6.1)

### EWMA Update
```python
# Test: first fill creates tca_parameters row with realized values
# Test: EWMA update with alpha=0.1: forecast = 0.9 * old + 0.1 * realized
# Test: sample_count increments on each fill
# Test: parameters stored per symbol + time bucket
# Test: morning fill (10:00) updates "morning" bucket, not "midday"
```

### Conservative Multiplier
```python
# Test: at sample_count=1, multiplier ≈ 2.0
# Test: at sample_count=25, multiplier ≈ 1.5
# Test: at sample_count=50, multiplier = 1.0
# Test: at sample_count=100, multiplier = 1.0 (no further decay)
```

### Pre-Trade Integration
```python
# Test: pre_trade_forecast uses EWMA values when available (sample >= 50)
# Test: pre_trade_forecast applies conservative multiplier when sample < 50
# Test: pre_trade_forecast falls back to default A-C coefficients when no EWMA
# Test: higher EWMA cost → smaller position size recommendation
```

---

## 8. Real TWAP/VWAP Execution (6.3)

### TWAP Scheduling
```python
# Test: 1000 shares over 30 min with 5-min buckets → 6 children, ~167 each
# Test: child quantities sum to parent total (accounting for rounding)
# Test: child scheduled_times span the execution window
# Test: child times have jitter (+/-20% of bucket width)
# Test: child quantities have variation (+/-10%)
```

### VWAP Scheduling
```python
# Test: child sizes proportional to volume profile
# Test: larger children at open/close (high volume), smaller at midday
# Test: falls back to synthetic U-curve when no historical data
# Test: volume profile cached and not refetched within same day
```

### Parent/Child State Machine
```python
# Test: parent starts PENDING, transitions to ACTIVE on first child submit
# Test: parent transitions to COMPLETING when end_time reached
# Test: parent transitions to COMPLETED when filled >= 99.5% total
# Test: invariant: sum(child.filled_qty) == parent.filled_qty
# Test: child REJECTED → retry with 50% size up to max_attempts
# Test: child timeout → cancelled, qty redistributed to next slice
# Test: 3 consecutive child failures → all active parents paused
```

### POV Fallback
```python
# Test: POV order dispatched as VWAP with max_participation_rate = 5%
```

### Cancellation
```python
# Test: kill switch cancels all active parents and their open children
# Test: risk gate halt cancels all active parents
# Test: execution monitor exit for symbol cancels that symbol's parent
```

### Crash Recovery
```python
# Test: startup_recovery finds ACTIVE parents and cancels them
# Test: startup_recovery cancels open child broker orders
# Test: startup_recovery logs full context for orphaned state
```

### Paper Broker Enhancement
```python
# Test: TWAP child filled against historical bar with participation cap
# Test: child_qty > bar_volume * participation_rate → partial fill
# Test: fill price within bar's [low, high] range
# Test: IMMEDIATE orders still use existing instant-fill model
```

### Algo Performance
```python
# Test: algo_performance row created after parent completion
# Test: implementation_shortfall_bps computed from arrival vs avg_fill
# Test: vwap_slippage_bps computed from execution VWAP vs benchmark VWAP
# Test: child counts (filled, failed) accurate
```

---

## 9. Liquidity Model (6.4)

```python
# Test: spread estimation returns bps per symbol from historical data
# Test: depth estimation returns shares per time bucket
# Test: order exceeding 10% of depth → SCALE_DOWN or REJECT
# Test: time-of-day multiplier: open = 1.5x, midday = 1.0x, close = 1.3x
# Test: stressed exit: portfolio-level slippage computed for simultaneous exit
# Test: stressed exit alert when slippage > threshold
# Test: liquidity check integrates into risk gate between ADV check and participation cap
```

---

## 10. Options Monitoring Rules (6.7)

```python
# Test: theta_acceleration triggers when DTE < 7 AND theta/premium > 5%/day
# Test: theta_acceleration does NOT trigger when DTE = 8
# Test: pin_risk triggers when DTE < 3 AND price within 1% of strike
# Test: assignment_risk triggers when short call ITM AND ex-div within 2 days
# Test: iv_crush triggers when post-earnings AND IV dropped > 30%
# Test: max_theta_loss triggers when cumulative decay > 40% of entry premium
# Test: auto_exit rules call _submit_exit()
# Test: flag_only rules log alert but do NOT trigger exit
# Test: rule configuration overrides default action
# Test: equity positions skip options rule evaluation
```

---

## 11. Slippage Model Enhancement (6.8)

```python
# Test: paper broker uses EWMA-calibrated spread when tca_parameters exist
# Test: paper broker falls back to fixed constants when no EWMA data
# Test: time-of-day multiplier applied to slippage estimate
# Test: slippage accuracy tracked: predicted vs realized ratio stored
# Test: alert triggered when accuracy ratio drifts beyond 0.5x or 2.0x
```

---

## 12. Borrowing/Funding Cost Model (6.9)

```python
# Test: daily_interest = margin_used * annual_rate / 252
# Test: cumulative_funding_cost accumulates over multiple days
# Test: position with zero margin_used has zero funding cost
# Test: funding cost deducted from unrealized P&L
# Test: funding cost visible in strategy performance metrics
```

---

## Integration Tests

```python
# Test: TWAP child fill → fill_leg created → TCA EWMA updated → audit trail written
# Test: intraday round-trip via TWAP → PDT day_trade recorded
# Test: sell at loss → buy within 30 days → wash sale flagged → tax lot cost basis adjusted
# Test: PDT at 3 day trades + account < $25K → 4th order rejected by risk gate
# Test: options position with DTE < 3 near strike → pin risk exit triggered
# Test: algo scheduler crash recovery → active parents cancelled on restart
```
