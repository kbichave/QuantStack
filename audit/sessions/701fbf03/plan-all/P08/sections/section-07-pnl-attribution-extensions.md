# Section 07: P&L Attribution Extensions

**Depends on:** section-06 (market-making agent -- provides vol strategy positions to attribute)

## Objective

Extend the existing Greek P&L attribution (from P06) with strategy-level decomposition and a vol-specific P&L metric (realized vol vs implied vol profit). This enables the system to learn which vol strategies work in which regimes and feed that back into strategy selection.

## Files to Create

### `src/quantstack/core/analysis/vol_pnl_attribution.py`

**Key components:**

1. **`VolPnLComponents` dataclass**:
   - `delta_pnl: float` -- P&L from delta exposure (underlying price change * delta)
   - `gamma_pnl: float` -- P&L from gamma (0.5 * gamma * price_change^2)
   - `theta_pnl: float` -- P&L from time decay (theta * days)
   - `vega_pnl: float` -- P&L from IV change (vega * iv_change)
   - `vol_pnl: float` -- realized vol vs implied vol P&L (core market-making metric)
   - `unexplained_pnl: float` -- residual (total - sum of components)
   - `total_pnl: float`

2. **`attribute_vol_pnl(position: dict, market_data_t0: dict, market_data_t1: dict) -> VolPnLComponents`**:
   - Compute each Greek component using position Greeks and market changes
   - Vol P&L = gamma_pnl + theta_pnl (the core vol trade: gamma profits minus theta cost)
   - Unexplained = total observed P&L - sum of attributed components

3. **`StrategyPnLSummary` dataclass**:
   - `strategy_type: str` -- "vol_arb", "gamma_scalp", "condor", "dispersion"
   - `period: str` -- "daily", "weekly", "mtd"
   - `total_pnl: float`
   - `delta_contribution: float` -- % of P&L from delta
   - `gamma_contribution: float` -- % from gamma
   - `theta_contribution: float` -- % from theta
   - `vega_contribution: float` -- % from vega
   - `vol_contribution: float` -- % from vol (gamma + theta net)
   - `win_rate: float`
   - `avg_holding_days: float`

4. **`compute_strategy_attribution(closed_trades: list[dict], strategy_type: str, period: str) -> StrategyPnLSummary`**:
   - Aggregate P&L components across all closed trades of a given strategy type within period
   - Compute contribution percentages
   - Compute win rate and average holding period

5. **`get_regime_strategy_performance(strategy_type: str, regime: str) -> dict`**:
   - Query `strategy_outcomes` table filtered by strategy_type and regime
   - Return performance metrics per regime-strategy pair
   - Used by market-making agent to select optimal strategy for current regime

## Files to Modify

### `src/quantstack/db.py`

Add columns to `strategy_outcomes` table (via `ADD COLUMN IF NOT EXISTS`):
- `delta_pnl REAL`
- `gamma_pnl REAL`
- `theta_pnl REAL`
- `vega_pnl REAL`
- `vol_pnl REAL`

### `src/quantstack/tools/langchain/vol_strategy_tools.py` (created in section 06)

Add LLM-facing tool:
- **`@tool get_vol_strategy_performance`**: Returns `StrategyPnLSummary` for a given strategy type and period. Used by market-making agent and trade reflector to assess which vol strategies are working.

Register in `src/quantstack/tools/registry.py`.

## Implementation Details

- Greek P&L attribution is an approximation. The unexplained component captures higher-order effects (charm, vanna, volga) that are not explicitly modeled. A large unexplained residual indicates the position has significant higher-order risk.
- Vol P&L (gamma_pnl + theta_pnl) is the core metric for market-making. A positive vol P&L means realized vol exceeded implied vol (good for long gamma) or time decay exceeded vol moves (good for short gamma/condors).
- Strategy-level attribution feeds the learning loop: if condors consistently produce positive theta P&L in ranging regimes but negative in trending, the market-making agent should avoid condors in trending regimes. This closes the research-to-execution feedback loop.
- The `get_regime_strategy_performance` function queries historical data. It needs at least 10 closed trades per strategy-regime pair to produce statistically meaningful results. Below that, return a low-confidence flag.
- All P&L attribution uses EOD Greeks snapshots. Intraday attribution is not in scope (see anti-goals).

## Test Requirements

File: `tests/unit/analysis/test_vol_pnl_attribution.py`

1. **Delta P&L**: Known delta and price change produce correct delta P&L.
2. **Gamma P&L**: Known gamma and price change produce correct 0.5 * gamma * dS^2.
3. **Theta P&L**: Known theta and day count produce correct theta P&L.
4. **Vega P&L**: Known vega and IV change produce correct vega P&L.
5. **Vol P&L**: Equals gamma_pnl + theta_pnl.
6. **Unexplained residual**: Total - sum of components equals unexplained.
7. **Strategy summary**: Aggregation across multiple trades produces correct totals and percentages.
8. **Low-confidence flag**: Fewer than 10 trades returns low-confidence indicator.

## Acceptance Criteria

- [ ] `VolPnLComponents` decomposes P&L into 5 Greek components + unexplained
- [ ] Vol P&L (gamma + theta) correctly computed as core market-making metric
- [ ] Strategy-level aggregation produces correct contribution percentages
- [ ] Regime-strategy performance lookup works with sufficient and insufficient data
- [ ] New columns added to `strategy_outcomes` via idempotent migration
- [ ] LLM-facing tool registered and accessible to market-making agent
- [ ] All 8 unit tests pass
