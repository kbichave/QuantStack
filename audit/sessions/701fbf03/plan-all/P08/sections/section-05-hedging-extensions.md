# Section 05: Hedging Engine Extensions (Gamma + Vega Hedging)

**Depends on:** section-03 (gamma scalping -- defines hedge metadata contract)

## Objective

Extend the options risk management infrastructure with gamma hedging and vega hedging strategies. These complement the delta hedging that P06 established, providing full Greek exposure management for the vol strategies built in sections 01-04.

## Files to Create

### `src/quantstack/core/risk/hedging_engine.py`

Central hedging engine that coordinates delta, gamma, and vega hedging.

**Key components:**

1. **`HedgingConfig` dataclass**:
   - `delta_threshold: float = 500.0` -- max portfolio delta before hedge (from P06 GreeksLimits)
   - `gamma_threshold: float = 100.0` -- max portfolio gamma before hedge
   - `vega_threshold: float = 1000.0` -- max portfolio vega before hedge
   - `hedge_cost_limit_pct: float = 0.005` -- max 0.5% of equity per hedge adjustment
   - `prefer_options_for_gamma: bool = True` -- use options (not shares) for gamma hedging

2. **`HedgingEngine`** class:
   - `__init__(config: HedgingConfig)`
   - `compute_hedges(portfolio_greeks: dict, positions: list[dict], market_data: dict) -> list[HedgeAction]`:
     - Evaluate all Greek exposures against thresholds
     - Prioritize: delta first (fastest-moving risk), then gamma, then vega
     - Return ordered list of hedge actions
   - `_compute_delta_hedge(delta_exposure, spot_price) -> HedgeAction | None`:
     - Hedge with underlying shares (existing P06 pattern)
   - `_compute_gamma_hedge(gamma_exposure, positions, chain_data) -> HedgeAction | None`:
     - Reduce gamma by trading options at different strikes
     - If long gamma > threshold: sell OTM options to reduce
     - If short gamma > threshold: buy ATM options to increase
     - Minimize cost via strike selection that provides most gamma per dollar
   - `_compute_vega_hedge(vega_exposure, positions, chain_data) -> HedgeAction | None`:
     - Reduce vega by trading options at different expiries (calendar-like)
     - If long vega > threshold: sell near-term options
     - If short vega > threshold: buy far-term options
     - Target: flatten vega while minimizing impact on other Greeks

3. **`HedgeAction` dataclass**:
   - `greek_target: str` -- "delta", "gamma", "vega"
   - `instrument_type: str` -- "equity", "option"
   - `symbol: str`
   - `action: str` -- "buy", "sell"
   - `quantity: int`
   - `strike: float | None` -- for options
   - `expiry: str | None` -- for options
   - `option_type: str | None` -- "call", "put"
   - `estimated_cost: float`
   - `exposure_before: float`
   - `exposure_after: float`
   - `reasoning: str`

4. **`aggregate_portfolio_greeks(positions: list[dict]) -> dict`**:
   - Sum delta, gamma, theta, vega across all positions
   - Return: `{"delta": float, "gamma": float, "theta": float, "vega": float}`
   - Handle mixed equity + options positions (equity: delta=shares, gamma/vega=0)

## Files to Modify

### `src/quantstack/core/risk/options_risk.py`

Add integration point:

- Import `HedgingEngine` and `HedgingConfig`
- Add `get_hedge_recommendations(positions, market_data) -> list[HedgeAction]` method to the risk management flow
- Update `GreeksLimits` to include `max_gamma_portfolio: float = 200` and `max_vega_portfolio: float = 5000` (the portfolio-level limits from the plan)

### `src/quantstack/execution/risk_gate.py`

Add Greek limit checks to the risk gate (strengthening, not weakening):

- Before approving a trade, compute post-trade Greek exposure
- Reject if post-trade gamma or vega would exceed portfolio limits
- Log: which Greek limit would be breached, current vs post-trade exposure

## Implementation Details

- Gamma hedging with options is more complex than delta hedging with shares. The key insight: buying/selling options at different strikes changes gamma without proportionally changing delta (because gamma peaks ATM and declines OTM).
- Vega hedging uses the term structure: near-term options have less vega than far-term. Selling near-term vs buying far-term (or vice versa) adjusts vega exposure.
- Cost minimization is important: hedge actions should be the cheapest way to reduce exposure below threshold, not necessarily zero out the Greek.
- Hedge actions are proposals, not executions. They flow through the risk gate and trade service like any other trade.
- The `hedge_cost_limit_pct` prevents expensive hedges that cost more than the risk they mitigate.
- Portfolio Greeks aggregation must handle the case where a position has no Greek data (equity-only position: delta = shares held, all other Greeks = 0).

## Test Requirements

File: `tests/unit/risk/test_hedging_engine.py`

1. **Delta hedge**: Portfolio delta > threshold produces share-based hedge action with correct quantity.
2. **Gamma hedge -- long gamma**: Excess long gamma produces sell-option action.
3. **Gamma hedge -- short gamma**: Excess short gamma produces buy-option action.
4. **Vega hedge -- long vega**: Excess long vega produces near-term option sell.
5. **Vega hedge -- short vega**: Excess short vega produces far-term option buy.
6. **Priority ordering**: Delta hedges ordered before gamma before vega.
7. **Cost limit**: Hedge action rejected when estimated cost exceeds `hedge_cost_limit_pct * equity`.
8. **Aggregation**: Mixed equity + options positions aggregate Greeks correctly.
9. **Below threshold**: No hedge actions when all Greeks within limits.
10. **Risk gate integration**: Trade rejected when post-trade gamma would breach limit.

## Acceptance Criteria

- [ ] `HedgingEngine` computes delta, gamma, and vega hedges independently
- [ ] Hedge actions are ordered by priority (delta > gamma > vega)
- [ ] Cost limit prevents excessively expensive hedges
- [ ] `aggregate_portfolio_greeks` handles mixed equity + options positions
- [ ] Risk gate strengthened with gamma and vega limit checks
- [ ] All 10 unit tests pass
- [ ] `HedgeAction` dataclass contains enough information for execution
- [ ] No weakening of existing risk gate checks
