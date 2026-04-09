# Section 03: Gamma Scalping

**Depends on:** section-01 (vol arb strategy -- shared `vol_strategy_signals` table and VolArbConfig pattern)

## Objective

Build a gamma scalping strategy that profits from realized volatility exceeding implied volatility by maintaining a long gamma position (ATM straddle) and delta-hedging frequently. The profit mechanism is capturing the difference between realized and implied vol through repeated delta rebalancing.

## Files to Create

### `src/quantstack/core/strategy/gamma_scalping.py`

**Key components:**

1. **`GammaScalpConfig` dataclass**:
   - `hedge_interval_minutes: int = 30` -- re-hedge on schedule
   - `hedge_threshold_pct: float = 0.005` -- re-hedge on 0.5% underlying move
   - `theta_bleed_exit_ratio: float = 1.5` -- exit when cumulative theta > cumulative gamma P&L * ratio
   - `min_rv_iv_spread: float = 0.05` -- minimum RV - IV spread to enter (RV must exceed IV)
   - `max_holding_days: int = 14` -- time stop (shorter than vol arb due to theta drag)
   - `target_gamma_dollars: float = 100.0` -- target portfolio gamma exposure in dollars

2. **`GammaScalpingStrategy(Strategy)`** class:
   - `on_bar(state: MarketState) -> list[TargetPosition]`:
     - Guard: need `realized_vol_21d` and `atm_iv` in features
     - Entry signal: `rv_21d - atm_iv > min_rv_iv_spread` (realized vol exceeds implied)
     - Propose: buy ATM straddle (long call + long put at nearest ATM strike)
     - Metadata: `{"strategy_type": "gamma_scalp", "hedge_required": True, "hedge_interval_minutes": 30, "hedge_threshold_pct": 0.005}`
   - `compute_hedge_action(state: MarketState, position_delta: float) -> dict | None`:
     - If `abs(position_delta) * underlying_price > hedge_threshold_dollars`: return hedge spec
     - Hedge spec: `{"action": "buy" | "sell", "shares": int, "reason": "delta_rebalance"}`
   - `should_exit(state, entry_metadata) -> bool`:
     - Theta bleed ratio exceeded (cumulative theta cost > gamma_profit * `theta_bleed_exit_ratio`)
     - Time stop exceeded
     - RV dropped below IV (edge disappeared)
   - `get_required_data() -> DataRequirements`:
     - Timeframes: `["1D", "30min"]` (intraday for hedge timing)
     - Features: `["realized_vol_21d", "atm_iv", "iv_rank"]`

3. **`GammaScalpPnLTracker`** -- lightweight tracker for per-position gamma vs theta attribution:
   - `record_hedge(timestamp, shares, price, delta_before, delta_after)`
   - `record_theta_decay(timestamp, theta_amount)`
   - `cumulative_gamma_pnl() -> float` -- sum of hedge round-trip P&L
   - `cumulative_theta_cost() -> float` -- sum of theta decay
   - `bleed_ratio() -> float` -- theta_cost / gamma_pnl (> 1.0 means losing)

## Files to Modify

### `src/quantstack/core/strategy/__init__.py`

Add export: `from quantstack.core.strategy.gamma_scalping import GammaScalpingStrategy, GammaScalpConfig`

## Implementation Details

- Gamma scalping is the inverse bet of vol arb: you want high realized vol. Entry requires RV > IV (the opposite condition from vol arb's sell_vol signal).
- The hedge computation is the profit engine. Each delta-hedge round-trip captures a small profit when the underlying moves enough. The sum of these micro-profits must exceed theta decay.
- `compute_hedge_action` is called by the hedging engine (section 05) on schedule, not by `on_bar`. The strategy only proposes the initial entry via `on_bar`.
- The `GammaScalpPnLTracker` is critical for the exit decision -- without it, you cannot measure whether theta is overwhelming gamma profits.
- Hedge frequency is a tunable parameter. Too frequent = transaction costs erode profits. Too infrequent = miss moves and lose to theta.
- Use the `vol_strategy_signals` table (created in section 01) to persist signals.

## Test Requirements

File: `tests/unit/strategy/test_gamma_scalping.py`

1. **Entry signal**: RV > IV by threshold generates long straddle proposal; RV < IV returns empty.
2. **Hedge action**: Position delta exceeding threshold triggers hedge; within threshold returns None.
3. **PnL tracker -- gamma profit**: Two hedges at different prices produce correct round-trip P&L.
4. **PnL tracker -- theta bleed**: Accumulated theta correctly tracked and bleed ratio computed.
5. **Exit on theta bleed**: `should_exit` returns True when bleed ratio exceeds threshold.
6. **Exit on RV collapse**: `should_exit` returns True when realized vol drops below implied vol.
7. **Time stop**: Triggers after `max_holding_days`.
8. **Edge case -- zero gamma**: No hedge action when position gamma is zero (flat).

## Acceptance Criteria

- [ ] `GammaScalpingStrategy` extends `Strategy` base class
- [ ] `compute_hedge_action` returns actionable hedge specs with share count
- [ ] `GammaScalpPnLTracker` accurately tracks gamma P&L vs theta cost
- [ ] Exit triggers correctly on theta bleed ratio, RV collapse, and time stop
- [ ] Uses shared `vol_strategy_signals` table from section 01
- [ ] All 8 unit tests pass
- [ ] Hedge metadata in `TargetPosition` compatible with hedging engine (section 05)
