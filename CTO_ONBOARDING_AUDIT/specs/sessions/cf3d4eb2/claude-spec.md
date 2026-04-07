# Phase 6: Execution Layer Completion — Synthesized Specification

## Overview

Transform QuantStack's execution layer from a prototype that simulates trading into a system capable of real algorithmic execution, regulatory compliance, and self-calibrating cost models. Nine work items, full scope, no cuts.

**Current state:** The OMS selects TWAP/VWAP/POV based on ADV thresholds but executes everything as IMMEDIATE. Paper broker fills instantly with a basic half-spread + sqrt-impact model. Zero SEC compliance. No partial fill tracking (fills table uses order_id as PK). TCA is computed but never fed back. All exit rules are equity-centric.

**Target state:** Real TWAP/VWAP execution with child orders. TCA EWMA feedback loop. Fill legs table for partial fill reconstruction. SEC compliance (PDT hard-block, wash sale tracking, tax lots). Best execution audit trail. Options-specific monitoring rules. Calibrated slippage model. Funding cost tracking.

---

## Architectural Decisions (from interview)

1. **Algo scheduler extends `order_lifecycle.py`** — keep all order state management in one file. No separate scheduler module.

2. **SEC enforcement model:**
   - **PDT:** Hard block in risk gate (account < $25K, critical path)
   - **Wash sale:** Post-trade accounting with pre-trade warning (flag, don't block)
   - **Tax lots:** Post-trade accounting on every fill (FIFO matching)
   - **Margin:** Pre-trade check for leveraged positions

3. **Schema evolution:** Add `fill_legs` table alongside existing `fills` (backward compatible). Existing code reading `fills` continues to work.

4. **TCA feedback:** New EWMA layer — updates after every fill, 2x conservative multiplier until 50 fills. Does NOT wire into existing A-C least-squares calibration (that's a separate periodic recalibration path).

5. **Paper broker data:** Historical intraday bar data is available for volume profiles. Use real data for VWAP scheduling.

6. **NBBO source:** Alpaca IEX (15-min delayed) — acceptable for paper trading audit. Plan for real-time NBBO when going live.

7. **Options exit behavior:** Configurable per rule (some auto-exit like pin risk, some flag-only like IV crush).

8. **No equity shorts currently** — funding cost model focuses on margin interest for leveraged longs, not borrow fees.

---

## Item Details

### 6.1 TCA Feedback Loop (EWMA) — 2 days

**Problem:** Pre-trade Almgren-Chriss forecasts cost. Post-trade stores realized cost. No feedback loop — stale forecasts persist regardless of realized slippage.

**Solution:**
- After every fill, compute EWMA update: `forecast_new = α * realized + (1 - α) * forecast_old` (α = 0.1)
- Store per-symbol, per-time-of-day cost model parameters in DB
- Until 50 fills per symbol: apply 2x conservative multiplier to forecast
- Position sizing consumes updated forecast — higher costs automatically reduce sizes
- This is a new EWMA layer, not a modification of the existing A-C calibration code

**Integration points:**
- Hook into `OrderLifecycle.record_fill()` to trigger EWMA update
- Feed updated parameters into `tca_engine.pre_trade_forecast()` for next trade
- Store parameters in new `tca_parameters` table (symbol, time_bucket, eta, gamma, sample_count, last_updated)

### 6.2 Partial Fill Tracking — 1 day

**Problem:** `fills` table uses `order_id` as PK. Only one fill record per order. Can't reconstruct average fill price or fill trajectory.

**Solution:**
- New `fill_legs` table: `(leg_id, order_id, leg_sequence, quantity, price, timestamp, venue)`
- Keep existing `fills` table as summary (backward compatible)
- Compute VWAP from legs: `avg_price = sum(qty_i * price_i) / sum(qty_i)`
- Update `fills.fill_price` as running VWAP of all legs

**Schema:**
```sql
CREATE TABLE fill_legs (
    leg_id SERIAL PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    leg_sequence INT NOT NULL,
    quantity INT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    venue TEXT,
    UNIQUE(order_id, leg_sequence)
);
```

### 6.3 Real TWAP/VWAP Execution — 5 days

**Problem:** OMS selects algo but paper broker executes everything as single fill. No time-slicing, no child orders, no participation constraints.

**Solution (extends `order_lifecycle.py`):**

**TWAP:**
- Split parent into N equal slices over T minutes
- Child order generation: `slice_qty = total_qty / N`
- Randomize timing (+/-20%) and sizing (+/-10%) to avoid detection
- Track `exposed_size` (open unfilled children) separately from `executed_so_far`

**VWAP:**
- Build intraday volume profile from historical bar data (available per interview)
- Weight child sizes proportional to expected volume per bucket
- Favorability check: buy only when ask < running VWAP

**Parent/child state machine:**
- Parent: PENDING → ACTIVE → COMPLETING → COMPLETED (or CANCELLED)
- Child: PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED (or CANCELLED/EXPIRED/REJECTED)
- Key invariant: `sum(child.filled_qty) == parent.filled_qty` always

**Paper broker enhancement:**
- Fill children against historical intraday bars
- Participation cap: child can't fill more than X% of bar volume (configurable, default 2%)
- Price: bar VWAP + noise within [low, high]

**Performance tracking:**
- New `algo_performance` table: parent_order_id, algo_type, arrival_price, avg_fill_price, benchmark_vwap, IS_bps, vwap_slippage_bps, child counts, participation rates

### 6.4 Liquidity Model — 3 days

**Problem:** Only check is `if daily_volume < min_daily_volume: warn`. No spread, depth, time-of-day, or stressed liquidity.

**Solution:**
- Spread estimation from historical quote data
- Depth estimation from intraday volume profiles (data available)
- Pre-trade check: `if order_size > estimated_depth * 0.1: scale down or reject`
- Time-of-day adjustment: wider spreads at open/close
- Stressed exit: estimate total slippage if all positions exit simultaneously

**Integration:** Feeds into risk gate as additional pre-trade check.

### 6.5 SEC Compliance — 5-7 days

**Problem:** Zero SEC compliance infrastructure.

**Solution (enforcement model per interview):**

**PDT (FINRA 4210) — Hard block in risk gate:**
- Count round-trips in rolling 5 business days
- If ≥3 AND account < $25K → REJECT 4th day-trade
- Account is currently below $25K — this is critical path
- Configurable by account type for future flexibility

**Wash Sale (26 USC §1091) — Post-trade accounting + pre-trade warning:**
- On sell at loss: check if same symbol bought within 30 calendar days before or after
- If wash sale: flag in DB, compute adjusted cost basis
- Pre-trade: if buying a symbol sold at loss within 30 days, surface warning (don't block)

**Tax Lots (IRS Form 8949) — Post-trade accounting:**
- On BUY: create `tax_lot(symbol, qty, price, date, lot_id)`
- On SELL: match FIFO, compute gain/loss per lot
- Adjust cost basis for wash sales

**Margin (Reg T) — Pre-trade check:**
- Long equity: 50% cash requirement
- Options: max_loss as margin
- If insufficient margin → REJECT

### 6.6 Best Execution Audit Trail — 2 days

**Problem:** No NBBO reference, venue data, or algo rationale stored per fill.

**Solution:**
- New `execution_audit` table: `(order_id, nbbo_bid, nbbo_ask, fill_price, fill_venue, algo_selected, algo_rationale, timestamp_ns)`
- NBBO source: Alpaca IEX quotes (15-min delayed, acceptable for paper)
- Populate on every fill
- Query: "show all fills worse than NBBO midpoint"

### 6.7 Options Monitoring Rules — 2 days

**Depends on:** Phase 2 item 2.12 (Greeks in risk gate)

**Problem:** All exit rules in `execution_monitor.py` are equity-centric.

**Solution:**
- Add options-specific evaluation after standard equity rules
- Per-rule configurable action (auto-exit vs flag):

| Rule | Trigger | Default Action |
|------|---------|---------------|
| Theta acceleration | DTE < 7 AND theta/premium > 5%/day | Auto-exit |
| Pin risk | DTE < 3 AND price near strike | Auto-exit |
| Assignment risk | Short call ITM + ex-div within 2 days | Flag |
| IV crush | Post-earnings + IV dropped > 30% | Flag |
| Max theta loss | Cumulative decay > 40% | Auto-exit |

- Wire to existing Greeks in `core/options/engine.py:293-364`

### 6.8 Slippage Model Enhancement — 2 days

**Problem:** Basic half-spread + sqrt impact. No time-of-day variation, no calibration from realized fills.

**Solution:**
- Integrate with TCA EWMA feedback (6.1) — use realized fills to calibrate per-symbol
- Add time-of-day slippage profiles from historical data
- Track model accuracy: predicted vs. realized slippage per trade
- Paper broker uses calibrated model instead of fixed constants

### 6.9 Borrowing/Funding Cost Model — 1 day

**Problem:** No funding cost tracking for leveraged positions.

**Solution (scoped per interview — options only, no equity shorts):**
- Add margin interest calculation to position P&L
- Approximate daily margin interest: `position_value * margin_rate / 252`
- Surface in strategy performance metrics
- No borrow fee infrastructure needed (no equity shorts currently)

---

## Dependencies

- **6.1 (TCA) feeds → 6.8 (Slippage):** EWMA parameters used by enhanced slippage model
- **6.2 (Partial Fills) feeds → 6.3 (TWAP/VWAP):** Child orders produce fill legs
- **6.3 (TWAP/VWAP) feeds → 6.6 (Audit Trail):** Child fills need NBBO recording
- **6.5 (SEC) integrates → risk gate:** PDT check in risk gate, wash sale in fill hooks
- **6.7 (Options) depends → Phase 2 item 2.12:** Greeks must be in risk gate first
- **6.1 feeds → Phase 7:** TCA feedback loop is a learning loop

## Implementation Order (recommended)

1. **6.2 Partial Fill Tracking** — schema foundation for everything else
2. **6.1 TCA Feedback Loop** — EWMA infrastructure, needed by 6.8
3. **6.5 SEC Compliance** — PDT is critical (account < $25K)
4. **6.3 TWAP/VWAP** — largest item, depends on 6.2
5. **6.4 Liquidity Model** — enhances risk gate
6. **6.6 Audit Trail** — depends on 6.3 child fills
7. **6.8 Slippage Enhancement** — depends on 6.1
8. **6.7 Options Monitoring** — depends on Phase 2
9. **6.9 Funding Costs** — simplest, lowest dependency
