# Section 05: Risk Gate Multi-Asset Extension

## Objective

Extend the existing `RiskGate` in `src/quantstack/execution/risk_gate.py` to enforce per-asset-class position limits, cross-asset correlation exposure checks, margin requirements per asset class, and total portfolio notional limits. The risk gate is LAW — these additions only strengthen it.

**Depends on:** section-01-asset-class-base

## Files to Modify

### `src/quantstack/execution/risk_gate.py`
Extend `RiskGate.check()` to accept an optional `asset_class: AssetClassType` parameter (defaults to `EQUITY` for backward compatibility). Add four new checks:

1. **Per-asset-class position limits**
   - Look up `PositionLimits` from the asset class registry
   - Reject if adding this position would exceed the asset class's `max_pct_equity` or `max_positions`
   - Example: crypto max 2% per position, futures max 5% per position

2. **Cross-asset correlation exposure**
   - Before approving a new position, check the correlation between the proposed instrument and existing portfolio positions across all asset classes
   - If adding the position would increase portfolio-level correlation exposure above threshold (default 0.80 avg pairwise), reject with reason
   - Use a precomputed correlation matrix stored in DB or computed on-demand from recent returns

3. **Margin requirements per asset class**
   - For futures: check SPAN margin requirement against available margin
   - For crypto: check 100% margin (no leverage)
   - For equity/options: existing logic (no change)
   - Reject if total margin used + new order margin > 90% of account equity

4. **Total portfolio notional limit**
   - Sum notional exposure across all asset classes (using contract multipliers for futures)
   - Reject if total notional > 3x account equity (conservative multi-asset leverage cap)
   - This is a portfolio-level check that supersedes per-asset-class limits

### `src/quantstack/execution/risk_gate.py` — `RiskVerdict`
Add `asset_class: AssetClassType | None` field to `RiskVerdict` dataclass for audit trail.

### `src/quantstack/execution/portfolio_state.py`
Extend `get_portfolio_state()` to return positions grouped by asset class. Add:
- `positions_by_asset_class: dict[AssetClassType, list[Position]]`
- `notional_by_asset_class: dict[AssetClassType, float]`
- `margin_used_by_asset_class: dict[AssetClassType, float]`
- `total_notional: float`
- `total_margin_used: float`

### `src/quantstack/execution/risk_state.py`
Add tracking for per-asset-class daily P&L and loss limits. Extend existing daily loss tracking to be asset-class-aware. If any single asset class hits its daily loss limit, halt that asset class only (not the entire system).

## Files to Create

### `src/quantstack/execution/correlation_monitor.py`
Precompute and cache cross-asset correlation matrix:

- `compute_correlation_matrix(positions, lookback_days=60) -> pd.DataFrame`
- Uses returns from DataStore for all held instruments
- Caches result with 1h TTL (correlations don't change fast)
- `check_correlation_exposure(existing_positions, new_symbol, new_asset_class) -> tuple[bool, float, str]`
  - Returns (is_acceptable, avg_correlation, reason)

## Implementation Details

1. **Backward compatibility is critical.** All new parameters are optional. Existing equity-only calls to `gate.check()` must work identically to before. Add `asset_class=AssetClassType.EQUITY` as default.
2. The cross-asset correlation check is the most expensive new operation. Keep it behind a flag (`CORRELATION_CHECK_ENABLED=true`) and cache aggressively. If the check takes >500ms, log a warning.
3. For the total notional calculation, futures notional = contracts * multiplier * price. This is much larger than equity notional for the same dollar margin — make sure the 3x limit accounts for this correctly.
4. The per-asset-class daily loss limit should default to: equity=2%, futures=1.5%, crypto=1%, options=2%. These are tighter for higher-vol asset classes.
5. **Never weaken existing checks.** All new checks are additive. An order must pass ALL existing checks AND all new checks.

## Test Requirements

### `tests/unit/execution/test_risk_gate_multi_asset.py`
- Equity order with no asset class specified -> existing behavior unchanged
- Futures order at 6% equity -> rejected (5% limit)
- Futures order at 4% equity -> approved
- Crypto order at 3% equity -> rejected (2% limit)
- Crypto order at 1.5% equity -> approved
- Total notional > 3x equity -> rejected regardless of individual limits
- Margin check: futures order that would exceed 90% margin -> rejected

### `tests/unit/execution/test_correlation_monitor.py`
- Two uncorrelated positions (corr < 0.3) -> acceptable
- Portfolio of highly correlated positions (corr > 0.8) + new correlated position -> rejected
- Cache hit: second call within TTL returns cached result
- Missing price data for one instrument -> graceful degradation (exclude from correlation, don't block)

### `tests/unit/execution/test_portfolio_state_multi_asset.py`
- Portfolio with equity + futures positions -> correct grouping by asset class
- Notional calculation: 2 ES contracts at 5000 -> $500k notional
- Margin used correctly summed per asset class

### `tests/unit/execution/test_risk_state_multi_asset.py`
- Crypto daily loss > 1% -> crypto halted, equity still trading
- Futures daily loss > 1.5% -> futures halted, others unaffected
- All asset classes recover next trading day

## Acceptance Criteria

- [ ] `RiskGate.check()` backward compatible — existing equity tests still pass
- [ ] Per-asset-class position limits enforced
- [ ] Cross-asset correlation check rejects concentrated portfolios
- [ ] Margin requirement check works for futures and crypto
- [ ] Total portfolio notional limit (3x equity) enforced
- [ ] `RiskVerdict` includes `asset_class` field
- [ ] Portfolio state reports positions/notional/margin by asset class
- [ ] Per-asset-class daily loss limits halt individual asset classes
- [ ] All new checks are additive (never weaken existing gates)
- [ ] All tests pass under `uv run pytest tests/unit/execution/test_risk_gate_multi_asset.py tests/unit/execution/test_correlation_monitor.py tests/unit/execution/test_portfolio_state_multi_asset.py tests/unit/execution/test_risk_state_multi_asset.py`
