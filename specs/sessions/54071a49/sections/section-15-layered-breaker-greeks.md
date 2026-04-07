# Section 15: Layered Circuit Breaker & Greeks Risk

## Overview

This section adds two independent protection systems to the execution layer:

1. **Layered circuit breaker** in `ExecutionMonitor` -- dual threshold system (daily P&L + portfolio high-water mark) that progressively restricts trading as losses mount, culminating in emergency liquidation with limit orders and a dead-man's switch.
2. **Greeks integration in RiskGate** -- portfolio-level delta/gamma/vega/theta limits enforced at order entry time for options positions, using the existing `PortfolioGreeksManager` from `core/risk/options_risk.py`.

Both systems operate independently. The circuit breaker monitors open positions continuously; the Greeks gate runs at order submission time.

## Dependencies

- **Phase 2 complete** (sections 7-11). Specifically:
  - Section 8 (email alerting) must exist for CRITICAL email alerts on circuit breaker triggers. If not yet implemented, degrade to logging only.
  - Section 11 (kill switch recovery) -- the circuit breaker's defensive exit triggers the kill switch.
- No dependency on other Phase 3 sections (12, 14, 16). Can be implemented in parallel with them.

## Current State of Files

Before implementing, verify the current state:

- `src/quantstack/execution/execution_monitor.py` already has a `_circuit_breaker_loop` that monitors feed health and DB connectivity. The new layered P&L circuit breaker is a **separate concern** -- it monitors portfolio value, not infrastructure health. Add it alongside the existing loop, not replacing it.
- `src/quantstack/execution/risk_gate.py` already has options-specific checks (DTE bounds, premium-at-risk, per-position limits) in the `check()` method. The Greeks integration adds **portfolio-level** Greek exposure limits as a new check in the options path, not replacing existing checks.
- `src/quantstack/core/risk/options_risk.py` contains `PortfolioGreeksManager` with `check_new_trade(proposed_delta, proposed_gamma, symbol)` and `GreeksLimits`. This is the existing implementation to wire into the risk gate.

---

## Tests

Write these tests first. All tests use pytest with `asyncio_mode = "auto"`.

### Circuit Breaker Tests

```python
# tests/execution/test_layered_circuit_breaker.py

# --- Daily P&L layer ---

# Test: daily_pnl_halts_new_entries
#   Setup: portfolio with equity $10,000, daily P&L = -$150 (i.e., -1.5%)
#   Assert: circuit breaker state is HALT_NEW_ENTRIES
#   Assert: existing positions are NOT closed

# Test: daily_pnl_systematic_exit
#   Setup: portfolio with equity $10,000, daily P&L = -$250 (i.e., -2.5%)
#   Assert: circuit breaker state is SYSTEMATIC_EXIT
#   Assert: weakest position (largest unrealized loss) flagged for exit first

# Test: daily_pnl_emergency_liquidation
#   Setup: portfolio with equity $10,000, daily P&L = -$500 (i.e., -5%)
#   Assert: circuit breaker state is EMERGENCY_LIQUIDATION
#   Assert: all positions flagged for exit

# Test: daily_pnl_resets_at_market_open
#   Setup: trigger -2.5% daily P&L loss, then simulate market open (new day)
#   Assert: daily layer resets, state returns to NORMAL

# --- Portfolio HWM layer ---

# Test: hwm_halts_all_trading
#   Setup: peak equity $10,000, current equity $9,700 (i.e., -3% from HWM)
#   Assert: circuit breaker state is HALT_ALL_TRADING

# Test: hwm_defensive_exit
#   Setup: peak equity $10,000, current equity $9,500 (i.e., -5% from HWM)
#   Assert: circuit breaker state is DEFENSIVE_EXIT
#   Assert: kill switch triggered
#   Assert: CRITICAL alert sent (or logged if email not available)

# --- Limit orders for emergency liquidation ---

# Test: emergency_liquidation_uses_limit_orders
#   Setup: trigger emergency liquidation
#   Assert: exit orders are limit orders with price = midpoint - 1% collar
#   Assert: exit orders are NOT market orders

# Test: dead_mans_switch_unfilled_after_60s
#   Setup: trigger emergency liquidation, simulate 60s with no fills
#   Assert: kill switch activated
#   Assert: CRITICAL alert sent

# --- Independence ---

# Test: both_layers_trigger_independently
#   Setup: daily P&L at -1.0% (below daily threshold), HWM drawdown at -3.5%
#   Assert: HWM layer triggers HALT_ALL_TRADING even though daily layer is fine
#   Setup: daily P&L at -2.5% (systematic exit), HWM drawdown at -1%
#   Assert: daily layer triggers SYSTEMATIC_EXIT even though HWM layer is fine
```

### Greeks Risk Gate Tests

```python
# tests/execution/test_greeks_risk_gate.py

# Test: options_exceeding_max_delta_rejected
#   Setup: portfolio already at 450 delta, proposed trade adds 60 delta, limit is 500
#   Assert: RiskVerdict.approved is False
#   Assert: violation rule is "greeks_delta_limit"

# Test: options_exceeding_gamma_limit_rejected
#   Setup: portfolio at 90 gamma, proposed trade adds 15 gamma, limit is 100
#   Assert: RiskVerdict.approved is False
#   Assert: violation rule is "greeks_gamma_limit"

# Test: options_exceeding_vega_limit_rejected
#   Setup: portfolio at 950 vega, proposed trade adds 100 vega, limit is 1000
#   Assert: RiskVerdict.approved is False
#   Assert: violation rule is "greeks_vega_limit"

# Test: daily_theta_budget_exceeded_rejects_new_options
#   Setup: portfolio theta = -480, proposed trade theta = -30, limit is -500
#   Assert: RiskVerdict.approved is False
#   Assert: violation rule is "greeks_theta_limit"

# Test: portfolio_greeks_aggregation_correct
#   Setup: 3 options positions with known Greeks
#   Assert: PortfolioGreeksManager.current_metrics matches expected aggregation

# Test: equity_positions_bypass_greeks_checks
#   Setup: portfolio at max delta, submit equity order
#   Assert: RiskVerdict.approved is True (Greeks checks skipped for equity)
```

---

## Implementation: Layered Circuit Breaker

### File: `src/quantstack/execution/execution_monitor.py`

Add the following to `ExecutionMonitor`. This is additive -- do not modify the existing `_circuit_breaker_loop` (which handles feed/DB health).

#### New Data Structures

Add a `CircuitBreakerState` enum and a `LayeredCircuitBreaker` class as module-level definitions:

```python
class CircuitBreakerLevel(str, Enum):
    """Progressive response levels for P&L circuit breakers."""
    NORMAL = "normal"
    HALT_NEW_ENTRIES = "halt_new_entries"
    SYSTEMATIC_EXIT = "systematic_exit"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"
    HALT_ALL_TRADING = "halt_all_trading"
    DEFENSIVE_EXIT = "defensive_exit"
```

#### `LayeredCircuitBreaker` Class

Define a class that encapsulates the dual-layer logic. Key design points:

- **Daily P&L layer**: Tracks `(realized + unrealized) P&L / start-of-day equity`. Resets when the date changes (market open). Three thresholds:
  - -1.5% unrealized+realized: `HALT_NEW_ENTRIES`
  - -2.5%: `SYSTEMATIC_EXIT` (close weakest positions first, ordered by unrealized loss descending)
  - -5.0%: `EMERGENCY_LIQUIDATION` (close all)

- **Portfolio HWM layer**: Tracks `(equity - high_water_mark) / high_water_mark`. HWM only moves up, never resets. Two thresholds:
  - -3% from HWM: `HALT_ALL_TRADING`
  - -5% from HWM: `DEFENSIVE_EXIT` (close all, trigger kill switch, send CRITICAL email)

- The class exposes `evaluate(equity: float, daily_pnl: float, daily_start_equity: float) -> CircuitBreakerLevel` that returns the most severe level across both layers.

- Store `_hwm: float` (initialized from DB or first equity reading), `_daily_date: date` (for reset detection), configurable thresholds via constructor parameters with defaults matching the values above.

#### Emergency Liquidation with Limit Orders

When the breaker triggers `EMERGENCY_LIQUIDATION` or `DEFENSIVE_EXIT`, the exit submission must use **limit orders with a 1% collar below midpoint**, not market orders. This prevents catastrophic fills during liquidity gaps.

In `_submit_exit`, when the exit reason originates from the circuit breaker:
1. Compute limit price as `current_price * 0.99` for long exits (selling), `current_price * 1.01` for short exits (covering).
2. Submit limit order.
3. Start a 60-second timer. If the order is not filled within 60 seconds, this is the **dead-man's switch**: cancel the unfilled order, trigger the kill switch, and send a CRITICAL alert. The system must not remain in an indefinite "trying to liquidate" state.

#### Integration into `ExecutionMonitor`

Add a new async loop `_pnl_circuit_breaker_loop` that runs every 5 seconds (same cadence as the existing infrastructure circuit breaker loop):

1. Read current portfolio equity and daily P&L from `self._portfolio.get_snapshot()`.
2. Call `self._layered_breaker.evaluate(equity, daily_pnl, daily_start_equity)`.
3. Based on the returned level:
   - `HALT_NEW_ENTRIES`: Set an internal flag `self._halt_new_entries = True`. Graph nodes check this before submitting new entry orders.
   - `SYSTEMATIC_EXIT`: Sort positions by unrealized loss (worst first). Submit exit for the worst position. Re-evaluate on next tick.
   - `EMERGENCY_LIQUIDATION`: Submit limit exits for all positions using the collar logic above.
   - `HALT_ALL_TRADING` / `DEFENSIVE_EXIT`: Submit limit exits for all positions, trigger kill switch, send CRITICAL email.

Start this loop in `start()` alongside the existing `_poll_task`, `_reconcile_task`, and `_cb_task`.

#### Exposing halt state to the graph

Add a public method `should_halt_new_entries() -> bool` on `ExecutionMonitor` that returns `True` when the daily P&L layer has reached -1.5% or the HWM layer has reached -3%. The Trading Graph's entry nodes should call this before submitting new orders. This is the mechanism by which the circuit breaker halts new entries without closing existing positions.

---

## Implementation: Greeks Integration in Risk Gate

### File: `src/quantstack/execution/risk_gate.py`

Wire the existing `PortfolioGreeksManager` from `src/quantstack/core/risk/options_risk.py` into the options path of `RiskGate.check()`.

#### New Fields on `RiskLimits`

Add Greeks-related limits to the `RiskLimits` dataclass. These set portfolio-level caps appropriate for a $5-10K account:

```python
# Portfolio-level Greeks limits (options only)
max_portfolio_delta: float = 200.0       # Max absolute net delta
max_portfolio_gamma: float = 50.0        # Max absolute gamma
max_portfolio_vega: float = 500.0        # Max absolute vega
max_daily_theta_budget: float = -50.0    # Max daily theta decay (negative = cost)
```

Add corresponding `from_env` loaders (`RISK_MAX_PORTFOLIO_DELTA`, etc.).

These values are conservative for a $5-10K account. They prevent options from dominating the portfolio:
- 200 delta ~ equivalent exposure to 200 shares of the underlying
- 50 gamma bounds second-order risk
- 500 vega limits sensitivity to vol moves
- -$50/day theta budget prevents slow bleed from theta decay

#### Integration Point

In the options path of `check()` (after the existing DTE bounds and premium-at-risk checks, around line 615), add a Greeks check:

1. Instantiate or retrieve a `PortfolioGreeksManager` (singleton, same pattern as `get_risk_gate()`).
2. Compute the proposed trade's delta and gamma. For a simple long call/put, delta and gamma come from the Black-Scholes model. The existing `compute_greeks_dispatch()` in `core/options/engine.py` already does this -- call it with the trade's parameters.
3. Call `greeks_manager.check_new_trade(proposed_delta, proposed_gamma, symbol)`.
4. Additionally check vega: if `abs(current_metrics.total_vega + proposed_vega) > limits.max_portfolio_vega`, reject.
5. Check theta budget: if `current_metrics.total_theta + proposed_theta < limits.max_daily_theta_budget`, reject (theta is negative for long options, so the sum goes more negative).
6. On any breach, return a `RiskVerdict` with the appropriate `RiskViolation`.

The Greeks manager's `current_metrics` must be refreshed periodically (e.g., by the `ExecutionMonitor` calling `update_from_positions()` each cycle). For the risk gate check, stale metrics (up to 60s old) are acceptable -- this is a pre-trade gate, not real-time monitoring.

#### Equity Bypass

The Greeks check must be gated by `instrument_type == "options"`. Equity positions skip this entirely. The existing code structure already branches on `instrument_type` at line 582 and 615, so the Greeks check slots into the options branch naturally.

---

## File Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/execution/execution_monitor.py` | Modify | Add `CircuitBreakerLevel` enum, `LayeredCircuitBreaker` class, `_pnl_circuit_breaker_loop`, limit-order exit logic with 60s dead-man's switch, `should_halt_new_entries()` method |
| `src/quantstack/execution/risk_gate.py` | Modify | Add Greeks limit fields to `RiskLimits`, wire `PortfolioGreeksManager.check_new_trade()` into options path of `check()`, add vega/theta checks |
| `tests/execution/test_layered_circuit_breaker.py` | Create | 9 tests covering daily layer, HWM layer, limit orders, dead-man's switch, independence |
| `tests/execution/test_greeks_risk_gate.py` | Create | 6 tests covering delta/gamma/vega/theta rejection, aggregation, equity bypass |

## Configuration

Environment variables (all optional, defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `CB_DAILY_HALT_PCT` | `0.015` | Daily P&L % to halt new entries |
| `CB_DAILY_EXIT_PCT` | `0.025` | Daily P&L % to begin systematic exit |
| `CB_DAILY_EMERGENCY_PCT` | `0.05` | Daily P&L % for emergency liquidation |
| `CB_HWM_HALT_PCT` | `0.03` | HWM drawdown % to halt all trading |
| `CB_HWM_DEFENSIVE_PCT` | `0.05` | HWM drawdown % for defensive exit |
| `CB_LIQUIDATION_COLLAR_PCT` | `0.01` | Limit order collar below/above mid |
| `CB_DEADMAN_TIMEOUT_S` | `60` | Seconds before dead-man's switch escalates |
| `RISK_MAX_PORTFOLIO_DELTA` | `200.0` | Max absolute net delta |
| `RISK_MAX_PORTFOLIO_GAMMA` | `50.0` | Max absolute gamma |
| `RISK_MAX_PORTFOLIO_VEGA` | `500.0` | Max absolute vega |
| `RISK_MAX_DAILY_THETA` | `-50.0` | Max daily theta decay budget |

## Key Design Decisions

1. **Limit orders, not market orders for emergency liquidation.** Market orders during a flash crash or liquidity gap can amplify losses (e.g., a -5% trigger becomes -10% realized). The 1% collar ensures reasonable execution while the 60-second dead-man's switch prevents indefinite stuck states.

2. **Two independent layers.** The daily P&L layer catches intraday bleeding. The HWM layer catches multi-day drawdowns that the daily layer misses (e.g., three consecutive -1.4% days are each below the daily threshold but represent -4.2% from HWM). Both must be checked independently.

3. **Greeks limits are conservative for small accounts.** A $5-10K account should not have 500+ delta of exposure. The defaults are intentionally tight. They can be loosened via env vars as the account grows.

4. **Greeks are checked at entry only, not continuously.** Continuous Greeks monitoring is the `PortfolioGreeksManager`'s job (called by `ExecutionMonitor`). The risk gate check at entry prevents adding to an already-breached portfolio. Existing positions that drift past limits are handled by the monitor's reduction order logic (`get_reduction_orders()`).

5. **The dead-man's switch is essential.** Without it, a circuit breaker that fires limit orders during a gap-down could leave the system in a "trying to exit" state indefinitely while losses continue to mount. The 60-second escalation to kill switch is the safety net.
