# Section 01: Vol Arb Strategy

## Objective

Build a volatility arbitrage strategy engine that detects IV-vs-realized-vol mispricings and generates delta-neutral trade proposals. This extends the existing `alpha_discovery/vol_arb.py` signal generation into a full strategy class integrated with the Strategy base class and options infrastructure.

## Files to Create

### `src/quantstack/core/strategy/vol_arb_engine.py`

Full volatility arbitrage strategy implementing the `Strategy` base class.

**Key components:**

1. **`VolArbConfig` dataclass** -- configurable thresholds:
   - `iv_rank_min: float = 50.0` -- minimum IV rank percentile to consider
   - `rv_iv_divergence_threshold: float = 0.10` -- minimum abs(RV - IV) / IV ratio
   - `z_threshold: float = 1.0` -- z-score threshold for signal (passed to existing `generate_vol_signal`)
   - `profit_target_pct: float = 0.50` -- close at 50% of max profit
   - `max_holding_days: int = 30` -- time stop
   - `hedge_frequency_minutes: int = 30` -- delta-hedge interval

2. **`VolArbStrategy(Strategy)`** class:
   - `on_bar(state: MarketState) -> list[TargetPosition]`:
     - Guard: skip if `state.iv_rank` is None or < `iv_rank_min`
     - Compute vol spread via `alpha_discovery.vol_arb.compute_vol_spread(state.atm_iv, rv_21d)`
     - Calibrate params via `calibrate_vol_params` using feature history
     - Generate signal via `generate_vol_signal`
     - If `sell_vol`: propose short straddle/strangle (sell ATM call + put)
     - If `buy_vol`: propose long straddle (buy ATM call + put)
     - Attach `metadata={"strategy_type": "vol_arb", "hedge_required": True}`
   - `get_required_data() -> DataRequirements`:
     - Timeframes: `["1D"]`
     - Features: `["realized_vol_21d", "iv_rank", "atm_iv"]`
   - `should_exit(state: MarketState, entry_metadata: dict) -> bool`:
     - IV mean-reverted (spread z-score < 0.5)
     - Time stop exceeded
     - Profit target reached

3. **`compute_rv_from_features(features: dict) -> float | None`** -- extract realized vol from feature dict, handling multiple naming conventions (`realized_vol_21d`, `1D_realized_vol_21d`).

## Files to Modify

### `src/quantstack/core/strategy/__init__.py`

Add export: `from quantstack.core.strategy.vol_arb_engine import VolArbStrategy, VolArbConfig`

## Implementation Details

- Reuse existing `alpha_discovery/vol_arb.py` functions (`compute_vol_spread`, `calibrate_vol_params`, `generate_vol_signal`, `select_structure`) rather than duplicating logic.
- The strategy class is the decision layer; structure selection delegates to `select_structure` for the actual legs.
- `TargetPosition` metadata must include `hedge_required: True` so the hedging engine (section 05) knows to delta-hedge.
- IV surface sparseness guard: if `state.atm_iv` is None or zero, return empty list and log a warning.

## Database Schema

Add to `db.py` `ensure_schema()`:

```sql
CREATE TABLE IF NOT EXISTS vol_strategy_signals (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    signal_date DATE NOT NULL,
    strategy_type TEXT NOT NULL,  -- 'vol_arb', 'dispersion', 'gamma_scalp', 'condor'
    signal_value TEXT,            -- 'sell_vol', 'buy_vol', etc.
    iv_rank REAL,
    realized_vol REAL,
    implied_vol REAL,
    z_score REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, signal_date, strategy_type)
);
```

This table is shared across all vol strategies (sections 01-04).

## Test Requirements

File: `tests/unit/strategy/test_vol_arb_engine.py`

1. **Signal direction correctness**: When IV > RV by threshold, signal is `sell_vol`. When RV > IV, signal is `buy_vol`.
2. **IV rank gate**: Strategy returns empty list when `iv_rank` < threshold.
3. **Sparse IV guard**: Returns empty list when `atm_iv` is None or 0.
4. **Exit conditions**: Verify `should_exit` triggers on mean-reversion, time stop, and profit target independently.
5. **Config override**: Non-default thresholds produce different signal behavior.
6. **Feature name fallback**: `compute_rv_from_features` handles both prefixed and bare feature names.

## Acceptance Criteria

- [ ] `VolArbStrategy` extends `Strategy` base class and implements `on_bar` / `get_required_data`
- [ ] Reuses `alpha_discovery.vol_arb` functions -- no duplicated spread/signal logic
- [ ] Returns empty list with warning log when IV data is sparse or absent
- [ ] All 6 unit tests pass
- [ ] `vol_strategy_signals` table created idempotently in `ensure_schema()`
- [ ] No bare `except:` or `except Exception:` blocks
