# Section 02: Dispersion Trading

## Objective

Build a dispersion trading strategy that profits from the spread between index implied correlation and realized correlation. Sell index volatility (SPY straddle) and buy component volatility (individual stock straddles) when implied correlation is rich relative to realized.

## Files to Create

### `src/quantstack/core/strategy/dispersion.py`

**Key components:**

1. **`DispersionConfig` dataclass**:
   - `correlation_threshold: float = 0.10` -- minimum implied-minus-realized correlation spread
   - `min_components: int = 5` -- minimum number of components with valid IV data
   - `index_symbol: str = "SPY"` -- index proxy
   - `component_weight_method: str = "equal"` -- "equal" or "market_cap"
   - `max_holding_days: int = 21` -- expiry-aligned time stop
   - `profit_target_pct: float = 0.40` -- close at 40% max profit
   - `max_correlation_spike: float = 0.85` -- emergency exit threshold

2. **`compute_implied_correlation(index_iv: float, component_ivs: list[float], weights: list[float]) -> float`**:
   - Formula: `implied_corr = (index_var - sum(w_i^2 * sigma_i^2)) / (2 * sum_{i<j}(w_i * w_j * sigma_i * sigma_j))`
   - Where `index_var = index_iv^2`, `sigma_i = component_ivs[i]`
   - Guard: return None if fewer than `min_components` have valid IV

3. **`compute_realized_correlation(returns_matrix: pd.DataFrame, window: int = 21) -> float`**:
   - Compute pairwise correlation matrix from trailing returns
   - Return average off-diagonal correlation
   - Guard: return None if returns_matrix has < `window` rows

4. **`DispersionStrategy(Strategy)`** class:
   - `on_bar(state: MarketState) -> list[TargetPosition]`:
     - Requires augmented state with `component_ivs` and `component_returns` in features
     - Compute implied and realized correlation
     - If `implied_corr - realized_corr > threshold`: generate dispersion trade
     - Trade legs: short index straddle + long component straddles (weighted)
     - Metadata: `{"strategy_type": "dispersion", "hedge_required": True, "legs": [...]}`
   - `should_exit(state, entry_metadata) -> bool`:
     - Correlation spike above `max_correlation_spike` (left tail protection)
     - Profit target reached
     - Time stop exceeded

5. **Helper: `build_dispersion_legs(index_symbol, components, weights, direction) -> list[dict]`**:
   - Returns list of leg specifications for the trade service

## Files to Modify

### `src/quantstack/core/strategy/__init__.py`

Add export: `from quantstack.core.strategy.dispersion import DispersionStrategy, DispersionConfig`

### `src/quantstack/db.py`

Add to `ensure_schema()`:

```sql
CREATE TABLE IF NOT EXISTS dispersion_trades (
    id SERIAL PRIMARY KEY,
    index_symbol TEXT NOT NULL,
    components JSONB NOT NULL,
    implied_correlation REAL NOT NULL,
    realized_correlation REAL NOT NULL,
    correlation_spread REAL NOT NULL,
    entry_date DATE NOT NULL,
    exit_date DATE,
    exit_reason TEXT,
    pnl REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Implementation Details

- Correlation computation is the core edge -- must be numerically stable. Use `np.corrcoef` for realized correlation, not manual computation.
- Implied correlation formula assumes equal or known weights. For equal-weight: simplifies to standard dispersion formula.
- Component IVs come from the IV surface (P06). If a component has no valid IV surface, exclude it and reweight.
- The strategy produces multi-leg trade proposals. Each leg is a separate `TargetPosition` linked by a shared `trade_group_id` in metadata.
- Left tail risk (correlation spike) is the primary risk. The `max_correlation_spike` exit is a hard circuit breaker.

## Test Requirements

File: `tests/unit/strategy/test_dispersion.py`

1. **Implied correlation computation**: Known inputs produce expected implied correlation value.
2. **Realized correlation computation**: Synthetic returns matrix with known correlation structure.
3. **Signal generation**: Implied > realized by threshold triggers trade; below threshold returns empty.
4. **Insufficient components**: Fewer than `min_components` with valid IV returns empty list.
5. **Correlation spike exit**: `should_exit` triggers when realized correlation exceeds `max_correlation_spike`.
6. **Edge case -- identical IVs**: All components have same IV; verify no division-by-zero or NaN.
7. **Leg construction**: `build_dispersion_legs` returns correct number of legs with correct directions.

## Acceptance Criteria

- [ ] `DispersionStrategy` extends `Strategy` base class
- [ ] Implied correlation formula is mathematically correct (verified against known textbook example)
- [ ] Returns empty list when correlation data is unavailable or insufficient
- [ ] Hard exit on correlation spike above threshold
- [ ] `dispersion_trades` table created idempotently
- [ ] All 7 unit tests pass
- [ ] No `sys.path.insert` or deferred imports
