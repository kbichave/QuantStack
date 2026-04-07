# Section 13: Borrowing/Funding Cost Model

## Overview

This section implements margin interest calculation for leveraged positions. The system currently tracks no funding costs, meaning strategies that hold leveraged positions for days or weeks appear more profitable than they actually are. This section corrects that by accruing daily margin interest and surfacing it in P&L and strategy performance metrics.

**Scope constraint:** No equity shorts are currently traded. Funding cost is scoped to margin interest on leveraged long positions only (no borrow fees, no short rebates).

**Severity:** HIGH | **Effort:** 1 day

**Dependencies:**
- **section-01-schema-foundation** must be complete (adds `margin_used` and `cumulative_funding_cost` columns to the `positions` table)
- No other section dependencies

**Blocks:** section-14-integration-tests

---

## Tests First

All tests use pytest. Tests live in `tests/unit/execution/` or `tests/unit/funding/` depending on where the implementation lands.

```python
# --- File: tests/unit/execution/test_funding_costs.py ---

# Test: daily_interest = margin_used * annual_rate / 252
#   Given margin_used = 10_000, annual_rate = 0.08 (8%)
#   Then daily_interest = 10_000 * 0.08 / 252 = approx 3.17

# Test: cumulative_funding_cost accumulates over multiple days
#   Given margin_used = 10_000 and annual_rate = 0.08
#   After 5 days of accrual, cumulative_funding_cost = 5 * (10_000 * 0.08 / 252) = approx 15.87

# Test: position with zero margin_used has zero funding cost
#   Given margin_used = 0.0
#   Then daily_interest = 0.0 and cumulative_funding_cost unchanged

# Test: funding cost deducted from unrealized P&L
#   Given position with unrealized_pnl = 500.0 and cumulative_funding_cost = 15.87
#   Then adjusted_pnl = 500.0 - 15.87 = 484.13

# Test: funding cost visible in strategy performance metrics
#   Given a strategy with 2 positions each accruing funding costs
#   Then strategy-level metrics include total_funding_cost as sum of position-level costs
```

---

## Funding Cost Formula

The calculation uses the standard margin interest formula with 252 trading days per year:

```
daily_interest = margin_used * annual_rate / 252
```

Where:
- `margin_used` is the dollar amount of margin borrowed for the position. For a fully cash-funded position, this is 0. For a position bought on 50% Reg T margin, `margin_used = position_notional * 0.5`.
- `annual_rate` is the broker's margin interest rate. Alpaca's current rate should be stored as a configuration value (not hardcoded). Default: `0.08` (8% APR, typical for small accounts).
- 252 is the standard trading-day convention for annualization.

---

## Schema Changes

Section 01 (schema-foundation) adds two columns to the `positions` table:

```sql
ALTER TABLE positions ADD COLUMN IF NOT EXISTS margin_used DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS cumulative_funding_cost DOUBLE PRECISION DEFAULT 0.0;
```

No new tables are needed. Funding cost state lives directly on the position record.

---

## Configuration

Add a margin rate configuration constant. This belongs in the execution config or as an environment variable:

```python
# src/quantstack/execution/funding.py (new file)

MARGIN_ANNUAL_RATE_DEFAULT = 0.08  # 8% APR — Alpaca typical for accounts < $100K
TRADING_DAYS_PER_YEAR = 252
```

The rate should be overridable via environment variable `MARGIN_ANNUAL_RATE` to handle broker rate changes without code deploys.

---

## Implementation: FundingCostCalculator

Create a new module `src/quantstack/execution/funding.py` with a single class:

```python
class FundingCostCalculator:
    """Computes daily margin interest and accumulates funding costs on positions."""

    def __init__(self, annual_rate: float | None = None):
        """
        Args:
            annual_rate: Annual margin interest rate (e.g., 0.08 for 8%).
                         Falls back to MARGIN_ANNUAL_RATE env var, then default.
        """
        ...

    def daily_interest(self, margin_used: float) -> float:
        """
        Compute one day's margin interest.

        Returns 0.0 if margin_used <= 0.
        """
        ...

    def accrue_funding_costs(self, positions: list[Position]) -> list[tuple[str, float]]:
        """
        Compute and apply daily funding cost accrual for all positions.

        For each position with margin_used > 0:
        1. Compute daily_interest
        2. Add to cumulative_funding_cost
        3. Return list of (symbol, daily_cost) tuples for logging

        Does NOT write to DB — caller is responsible for persisting.
        """
        ...
```

The class is intentionally stateless (no DB connection). It computes values; the caller persists them. This makes testing trivial (no DB fixtures needed for unit tests) and keeps the DB write path in one place.

---

## Integration Point 1: Daily P&L Update

The daily funding cost accrual should run as part of the existing price-update cycle in `PortfolioState`.

**Where:** `src/quantstack/execution/portfolio_state.py`

**Current behavior:** `PortfolioState.update_prices()` (around line 401) iterates positions, computes `unrealized_pnl = mult * (price - avg_cost) * abs(quantity)`, and writes to DB.

**New behavior:** After computing unrealized P&L, subtract `cumulative_funding_cost` to produce the funding-adjusted unrealized P&L. The raw unrealized P&L (before funding) remains available as `unrealized_pnl`. The funding cost is tracked separately in `cumulative_funding_cost` so both raw and adjusted figures are queryable.

The daily accrual itself (incrementing `cumulative_funding_cost`) should run once per trading day, not on every price tick. Options for triggering:

1. **Preferred:** A dedicated `accrue_daily_funding()` method on `PortfolioState` called by the supervisor graph's daily maintenance cycle (or the trading runner's end-of-day hook).
2. **Fallback:** A check inside `update_prices()` that runs accrual if `last_accrual_date < today`. This is simpler but mixes concerns.

The method should:
1. Instantiate `FundingCostCalculator`
2. Fetch all positions with `margin_used > 0`
3. For each, compute `daily_interest` and add to `cumulative_funding_cost`
4. Write the updated `cumulative_funding_cost` to the DB
5. Log each accrual for audit trail

---

## Integration Point 2: Margin Used Tracking

`margin_used` must be set when a position is opened or modified. This connects to the MarginCalculator from section-04 (SEC compliance):

- When a position is opened on margin: `margin_used = position_notional - cash_allocated`
- When a position is fully cash-funded: `margin_used = 0.0`
- When position size changes (partial exit, add-on): recalculate `margin_used`

For the initial implementation, `margin_used` can be computed simply as:

```
margin_used = max(0, position_notional - account_cash_available_for_this_position)
```

If the account has sufficient cash for all positions, `margin_used` will be 0 for all of them. This reflects the real-world behavior: margin interest is only charged on borrowed funds.

**Where to set it:** In `PortfolioState.upsert_position()` or in the fill-processing hook that calls `upsert_position()`. The margin calculation should use the same logic as the MarginCalculator from section-04 to stay consistent.

---

## Integration Point 3: Strategy Performance Metrics

Strategy performance reports should include funding costs. The relevant location is the strategy performance / attribution system.

**What to surface:**
- `total_funding_cost`: Sum of `cumulative_funding_cost` across all positions for a strategy
- `funding_cost_as_pct_of_pnl`: How much of gross P&L is consumed by funding costs
- `annualized_funding_drag_bps`: Funding cost expressed in basis points of capital deployed, annualized

These metrics help identify strategies where margin usage erodes returns. A swing strategy holding leveraged positions for 2 weeks at 8% APR on $10K margin costs ~$6.35 -- small but meaningful for small-account strategies where every dollar counts.

**Where:** The exact integration point depends on how strategy performance is computed. Key files to modify:
- `src/quantstack/core/attribution_engine.py` -- if it aggregates per-strategy P&L
- `src/quantstack/performance/equity_tracker.py` -- if daily equity snapshots include strategy breakdown
- `src/quantstack/tools/langchain/attribution_tools.py` -- if LLM-facing tools surface strategy metrics

The funding cost data comes directly from the `positions` table (`cumulative_funding_cost` column), so no new queries are needed beyond including that column in existing aggregation queries.

---

## Position Model Update

The `Position` Pydantic model in `src/quantstack/execution/portfolio_state.py` needs two new fields:

```python
class Position(BaseModel):
    # ... existing fields ...
    margin_used: float = 0.0
    cumulative_funding_cost: float = 0.0
```

The `_POS_COLS` tuple and `_row_to_position()` method must be updated to include these columns. The `upsert_position()` SQL must include them in INSERT and UPDATE statements.

---

## Edge Cases

- **Position with no margin:** `margin_used = 0.0` produces `daily_interest = 0.0`. No-op.
- **Position closed mid-day:** Funding cost accrues for the full day if the position was open at accrual time. This is conservative and matches broker behavior (most brokers charge interest on the opening balance).
- **Rate changes:** If the broker changes margin rates, update the `MARGIN_ANNUAL_RATE` env var. Historical accruals are not retroactively adjusted (matching real-world broker behavior).
- **Options positions:** Options bought with cash have `margin_used = 0.0`. Options bought on margin (if applicable) follow the same formula. Credit spreads that generate margin requirements would need their margin requirement tracked -- deferred until those strategies are traded.
- **Negative margin_used:** Should never occur. The calculator should clamp to 0.0 and log a warning if a negative value is encountered.

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/execution/funding.py` | **Create** | `FundingCostCalculator` class with `daily_interest()` and `accrue_funding_costs()` |
| `src/quantstack/execution/portfolio_state.py` | **Modify** | Add `margin_used` and `cumulative_funding_cost` to `Position` model, `_POS_COLS`, `_row_to_position()`, `upsert_position()`. Add `accrue_daily_funding()` method to `PortfolioState`. |
| `src/quantstack/db.py` | **Modify** | Add `ALTER TABLE positions ADD COLUMN IF NOT EXISTS` for both new columns (may already be handled by section-01) |
| `tests/unit/execution/test_funding_costs.py` | **Create** | Unit tests for `FundingCostCalculator` |
| Strategy performance files (attribution_engine, equity_tracker, or attribution_tools) | **Modify** | Include `cumulative_funding_cost` in strategy-level aggregations |

---

## Verification Checklist

1. `FundingCostCalculator.daily_interest(10_000)` returns approximately 3.17 at 8% APR
2. Accruing 5 days produces cumulative cost of approximately 15.87
3. Zero-margin positions produce zero cost
4. `Position` model round-trips `margin_used` and `cumulative_funding_cost` through DB
5. Strategy performance queries include funding costs
6. Rate is configurable via environment variable
