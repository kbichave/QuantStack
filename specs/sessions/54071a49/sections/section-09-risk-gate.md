# Section 09: Risk Gate Enhancements

## Overview

The risk gate (`src/quantstack/execution/risk_gate.py`) is the hard enforcement boundary between agent recommendations and the broker. Three specific gaps need to be addressed:

1. **Pre-trade correlation check** runs post-hoc only -- it should apply a concentration haircut instead of a hard reject when correlation exceeds 0.7.
2. **Market hours hard gating** should reject orders outside the configured trading window unless `extended_hours=True` is explicitly set.
3. **Daily notional deployment cap** (30% of equity) should limit cumulative new capital deployed each day, resetting at market open.

The risk gate already has substantial infrastructure for all three of these checks. The work here is about changing behavior (correlation haircut instead of reject), tightening enforcement (market hours as a hard gate), and adding a new tracking mechanism (daily notional cap with reset).

## Dependencies

- **Phase 1 complete** (sections 01-06). No direct code dependency, but the risk gate modifications assume the stop-loss enforcement from section 01 is in place.
- **No downstream blockers.** This section can be implemented in parallel with sections 07, 08, and 10.

## Current State of the Code

The file `src/quantstack/execution/risk_gate.py` already contains:

- **`_check_pretrade_correlation()`** (lines 790-856): Computes 30-day rolling correlation between a candidate symbol and all existing positions. Currently returns a hard `RiskViolation` reject when correlation >= 0.7. Uses sector ETF proxy fallback for symbols with < 20 days history. Fails closed on data unavailability.

- **`_check_market_hours()`** (lines 943-995): Already checks `OperatingMode` via `get_operating_mode()`. Rejects exposure-increasing orders outside `MARKET` mode. Allows closing/reducing trades in all modes.

- **`_check_heat_budget()`** (lines 858-894): Queries the `positions` table for today's opened positions and sums their notional. Rejects when cumulative deployment exceeds `max_daily_heat_pct` (default 0.30). This is the daily notional cap.

- **`RiskLimits`** dataclass (lines 77-176): Already has `max_pretrade_correlation` (0.70), `max_daily_heat_pct` (0.30), and `max_sector_concentration_pct` (0.40) fields, all configurable via environment variables.

**Key finding:** Much of this section's planned work already exists in the codebase. The implementation focus should be on the behavioral changes described below, not on building from scratch.

## Tests

All tests go in `tests/execution/test_risk_gate_enhancements.py`.

```python
# tests/execution/test_risk_gate_enhancements.py

"""
Tests for risk gate enhancements: correlation haircut, market hours gating,
and daily notional deployment cap.
"""

# --- Pre-trade correlation haircut ---

# Test: pre-trade correlation > 0.7 applies 50% haircut to position size
#   Setup: mock portfolio with 1 existing position, mock DataStore returning
#   correlated return series (corr ~0.85). Call check() with quantity=100.
#   Assert: verdict.approved is True, verdict.approved_quantity == 50.

# Test: pre-trade correlation < 0.7 passes through unchanged
#   Setup: mock portfolio with 1 existing position, mock DataStore returning
#   uncorrelated return series (corr ~0.3). Call check() with quantity=100.
#   Assert: verdict.approved is True, verdict.approved_quantity == 100.

# Test: pre-trade correlation with insufficient data fails closed
#   Setup: mock DataStore returning None for candidate symbol, no sector proxy.
#   Assert: verdict.approved is False, violation rule == "pretrade_correlation_data_missing".

# --- Market hours gating ---

# Test: order outside market hours rejected (no extended_hours flag)
#   Setup: mock get_operating_mode() to return OperatingMode.OVERNIGHT.
#   Call check() for a new buy (no existing position).
#   Assert: verdict.approved is False, violation rule == "market_hours".

# Test: order outside market hours accepted with extended_hours=True
#   Setup: mock get_operating_mode() to return OperatingMode.EXTENDED.
#   Call check() with extended_hours=True for a new buy.
#   Assert: verdict.approved is True.

# Test: order within market hours accepted normally
#   Setup: mock get_operating_mode() to return OperatingMode.MARKET.
#   Call check() for a new buy.
#   Assert: verdict.approved is True.

# Test: closing trade outside market hours is always permitted
#   Setup: mock get_operating_mode() to return OperatingMode.OVERNIGHT.
#   Mock existing long position. Call check() with side="sell".
#   Assert: verdict.approved is True (closing trade bypasses hours check).

# --- Daily notional deployment cap ---

# Test: daily notional cap exceeded -> order rejected
#   Setup: mock positions table query returning $2900 deployed today.
#   Equity = $10000, max_daily_heat_pct = 0.30 (cap = $3000).
#   Call check() with order notional = $200.
#   Assert: verdict.approved is False, violation rule == "daily_heat_budget".

# Test: daily notional cap resets at market open
#   Setup: query positions table for today's date. Verify the SQL uses
#   CURRENT_DATE (which resets at midnight). Confirm no stale state
#   carried from previous day.
#   This is implicitly tested by the SQL query using opened_at::date = CURRENT_DATE.

# Test: daily notional tracks cumulative new deployments correctly
#   Setup: mock positions table with 3 positions opened today totaling $2000.
#   Equity = $10000, cap = $3000. New order = $900.
#   Assert: verdict.approved is True (2000 + 900 = 2900 < 3000).
#   Then call again with order = $200.
#   Assert: verdict.approved is False (2900 + 200 = 3100 > 3000).

# Test: with $10K equity and 30% cap, max daily deployment is $3K
#   Setup: mock equity = $10000, max_daily_heat_pct = 0.30.
#   No existing positions today. Call check() with order notional = $3100.
#   Assert: verdict.approved is False.
#   Call check() with order notional = $2900.
#   Assert: verdict.approved is True.
```

## Implementation Details

### Change 1: Correlation Haircut Instead of Hard Reject

**File:** `src/quantstack/execution/risk_gate.py`

**Current behavior:** `_check_pretrade_correlation()` returns a `RiskViolation` (hard reject) when any existing position has correlation >= 0.7 with the candidate.

**Target behavior:** When correlation exceeds the threshold, instead of rejecting, apply a 50% haircut to the proposed position size and allow the trade. The caller receives `approved=True` with `approved_quantity` set to half the requested amount.

**Approach:** Modify `_check_pretrade_correlation()` to return a different signal type -- instead of a list of `RiskViolation` objects, return a tuple of `(violations: list[RiskViolation], haircut: float)` where `haircut` is 1.0 (no change) or 0.5 (50% reduction). The calling code in `check()` (around line 597) applies the haircut to `quantity` before continuing.

The haircut value should be configurable via `RiskLimits` (add a `correlation_haircut` field, default 0.5). Log a warning when the haircut is applied, including the correlated pair and the correlation value, so the decision is auditable.

**Edge case:** If correlation is detected with multiple existing positions, apply the haircut once (not multiplicatively). The worst correlation value determines whether the haircut triggers.

### Change 2: Market Hours Hard Gating with extended_hours Override

**File:** `src/quantstack/execution/risk_gate.py`

**Current behavior:** `_check_market_hours()` already rejects exposure-increasing orders outside market hours. However, there is no `extended_hours` parameter that allows explicit override for extended-hours trading.

**Target behavior:** Add an `extended_hours: bool = False` parameter to `check()`. When `extended_hours=True`, bypass the market hours rejection for EXTENDED mode (but still reject for OVERNIGHT and WEEKEND). This allows strategies that explicitly opt in to extended-hours trading.

**Approach:** 
1. Add `extended_hours: bool = False` parameter to `check()`.
2. Pass it through to `_check_market_hours()`.
3. In `_check_market_hours()`, if `extended_hours=True` and mode is `OperatingMode.EXTENDED`, return `None` (allow).
4. OVERNIGHT and WEEKEND modes remain hard-blocked regardless of the flag.

Default equity window: 9:30-16:00 ET. Default options window: 9:30-16:15 ET. These are already handled by the `OperatingMode` / `TradingWindow` infrastructure.

### Change 3: Daily Notional Deployment Cap

**File:** `src/quantstack/execution/risk_gate.py`

**Current behavior:** `_check_heat_budget()` already implements this check. It queries `positions` table for `opened_at::date = CURRENT_DATE`, sums notional, and rejects when cumulative deployment exceeds `max_daily_heat_pct` (30% of equity).

**Target behavior:** The existing implementation is correct. Verify it works as specified:
- Cap resets daily because the SQL uses `CURRENT_DATE`.
- The check runs in `check()` at line 600 for non-reducing equity orders.
- Default is 30% (`max_daily_heat_pct = 0.30`), configurable via `RISK_MAX_DAILY_HEAT_PCT` env var.

**Potential gap to verify:** The heat budget query uses `positions.opened_at::date` which means it counts positions that were opened today, not capital that was deployed today. If a position is opened and then partially closed, the full original notional still counts toward the budget. This is conservative (correct behavior for a risk gate -- fail safe).

**Enhancement:** Add in-memory tracking as a fast-path cache to avoid a DB round-trip on every `check()` call. Maintain a `_daily_notional_deployed: float` counter that resets when `date.today()` changes. Update it when `check()` approves a new entry. Fall back to the DB query if the in-memory state is stale or on process restart.

```python
# Sketch of the in-memory fast-path (add to RiskGate.__init__):
self._heat_date: date | None = None
self._heat_notional: float = 0.0

# In _check_heat_budget, before the DB query:
if self._heat_date != date.today():
    self._heat_date = date.today()
    self._heat_notional = 0.0  # Reset at new day
    # Fall through to DB query to recover state from other processes

# After check() approves a new entry, update:
self._heat_notional += order_notional
```

## File Manifest

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/execution/risk_gate.py` | Modify | All three changes: correlation haircut, extended_hours param, in-memory heat cache |
| `tests/execution/test_risk_gate_enhancements.py` | Create | Test file for all three enhancements |

## Verification Checklist

After implementation, verify:

- [ ] `check()` with a highly correlated candidate returns `approved=True` with `approved_quantity` at 50% of requested
- [ ] `check()` with a low-correlation candidate returns full `approved_quantity`
- [ ] `check()` outside market hours returns rejection for new entries
- [ ] `check()` outside market hours with `extended_hours=True` allows in EXTENDED mode
- [ ] `check()` in OVERNIGHT/WEEKEND mode rejects even with `extended_hours=True`
- [ ] Closing trades are always allowed regardless of market hours
- [ ] Daily heat budget correctly rejects when cumulative deployment exceeds 30% of equity
- [ ] Heat budget resets each day (no stale state from yesterday)
- [ ] All existing risk gate tests still pass (no regressions)
