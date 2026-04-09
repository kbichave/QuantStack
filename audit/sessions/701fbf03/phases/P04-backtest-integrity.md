# P04: Backtest Integrity

**Objective:** Make backtests trustworthy by fixing transaction cost modeling, survivorship bias, look-ahead bias, walk-forward enforcement, and Monte Carlo validation.

**Scope:** core/backtesting/, core/validation/, tools/langchain/qc_*.py, universe.py

**Depends on:** None

**Enables:** P01 (validates signals with honest backtests)

**Effort estimate:** 1 week

---

## What Changes

### 4.1 Realistic Transaction Costs (QS-B1)

**Problem:** Default 10 bps (commissions only). Missing spread, impact, opportunity cost.

**Fix:**
```python
# Update default all-in cost model
DEFAULT_COST_BPS = 30  # Until TCA calibrated from live fills
# Breakdown: ~5 bps commission + ~10 bps spread + ~15 bps impact

# For options: 5% of bid-ask spread as cost floor
# Small cap: 50 bps default (wider spreads, less depth)
# Mid cap: 30 bps
# Large cap: 15 bps
```

**Files:**
- `src/quantstack/core/backtesting/backtest_engine.py` — update default costs
- `src/quantstack/finrl/config.py` — update `default_transaction_cost`

### 4.2 Survivorship Bias Adjustment (QS-B2)

**Problem:** Backtests use current universe only. Delisted companies excluded.

**Implementation:**
```python
# New function in universe.py
def universe_as_of(date: datetime) -> list[str]:
    """Return symbols active (not delisted, not pre-IPO) at given date."""
    query = """
        SELECT symbol FROM symbols 
        WHERE ipo_date <= %s 
        AND (delisted_at IS NULL OR delisted_at > %s)
    """
    # Use in all backtests as the universe filter
```

**Files:**
- `src/quantstack/universe.py` — add `universe_as_of()` function
- `src/quantstack/core/backtesting/backtest_engine.py` — use `universe_as_of()` for universe
- `src/quantstack/db.py` — ensure `delisted_at` column populated from data providers

### 4.3 Look-Ahead Bias Detection (QS-S4)

**Problem:** `check_lookahead_bias()` stubbed. No feature timestamp validation.

**Implementation:**
```python
# New: src/quantstack/core/validation/lookahead_detector.py
@dataclass
class FeatureTimestamp:
    value: float
    as_of_date: datetime       # What period the data covers
    known_since: datetime      # When it became available
    
def check_lookahead(features: list[FeatureTimestamp], signal_time: datetime) -> list[str]:
    """Flag any feature where known_since > signal_time."""
    violations = []
    for f in features:
        if f.known_since > signal_time:
            violations.append(f"Feature '{f.name}' known at {f.known_since} used for signal at {signal_time}")
    return violations
```

**Key risk areas:**
- Earnings data from Alpha Vantage: announcement time vs market close
- Options flow: live delta/gamma overlap with prediction window
- Fundamentals refreshed nightly but signals fire intraday

**Files:**
- New: `src/quantstack/core/validation/lookahead_detector.py`
- `src/quantstack/tools/langchain/qc_research_tools.py` — implement `check_lookahead_bias()`
- `src/quantstack/core/features/` — add `known_since` metadata to features

### 4.4 Walk-Forward Mandatory Gate (QS-B4)

**Problem:** Walk-forward exists in `core/research/walkforward.py` but tool wrapper stubbed.

**Fix:**
- Implement `run_walkforward()` tool (currently stubbed in `qc_backtesting_tools.py`)
- Make WFV mandatory before strategy advances past `backtested`
- Gate: OOS Sharpe must be ≥ 50% of IS Sharpe
- Overfit ratio (IS Sharpe / OOS Sharpe) must be < 2.0

**Files:**
- `src/quantstack/tools/langchain/backtest_tools.py` — implement `run_walkforward()`
- `src/quantstack/autonomous/strategy_lifecycle.py` — enforce WFV gate

### 4.5 Monte Carlo Validation (QS-B3)

**Problem:** `run_monte_carlo()` stubbed.

**Implementation:**
```python
def run_monte_carlo(returns: pd.Series, n_simulations: int = 10000) -> MonteCarloResult:
    """Bootstrap Monte Carlo for backtest confidence intervals."""
    sharpe_distribution = []
    for _ in range(n_simulations):
        resampled = np.random.choice(returns, size=len(returns), replace=True)
        sharpe_distribution.append(compute_sharpe(resampled))
    
    return MonteCarloResult(
        mean_sharpe=np.mean(sharpe_distribution),
        ci_5th=np.percentile(sharpe_distribution, 5),
        ci_95th=np.percentile(sharpe_distribution, 95),
        prob_negative=np.mean(np.array(sharpe_distribution) < 0),
    )
```

**Gate:** Reject strategies where 5th percentile Sharpe < 0.3

**Files:**
- `src/quantstack/tools/langchain/qc_research_tools.py` — implement `run_monte_carlo()`
- `src/quantstack/autonomous/strategy_lifecycle.py` — add MC validation gate

### 4.6 Point-in-Time Data Semantics (QS-B5)

**Problem:** Features don't have `as_of_date` / `known_since` fields.

**Fix:** Add `(value, as_of_date, available_date)` triple to fundamental features. Filter: `available_date < signal_timestamp`.

**Files:**
- `src/quantstack/core/features/fundamental_features.py` — add PIT metadata
- `src/quantstack/data/providers/` — populate `available_date` from provider timestamps

### 4.7 Feature Multicollinearity Audit (QS-B6)

**Problem:** 150+ features, no VIF analysis, ~100 redundant.

**Fix:**
- Weekly VIF computation. Remove features with VIF > 10.
- Correlation matrix: drop one of each pair with corr > 0.85
- Expected effective rank: ~30 (down from 150+)

**Files:**
- New: `src/quantstack/core/features/multicollinearity.py`
- `src/quantstack/ml/trainer.py` — run VIF filter before training

## Tests

| Test | What It Verifies |
|------|-----------------|
| `test_realistic_costs_applied` | 30 bps all-in cost reduces Sharpe by expected amount |
| `test_universe_as_of` | Historical universe excludes delisted stocks |
| `test_lookahead_detection` | Flag raised when feature known_since > signal_time |
| `test_walkforward_gate` | Strategy with OOS < 50% IS rejected |
| `test_monte_carlo_ci` | 5th percentile Sharpe < 0.3 → rejection |
| `test_vif_filter` | Features with VIF > 10 removed |

## Acceptance Criteria

1. Backtests use 30 bps default cost (not 10 bps)
2. `universe_as_of()` returns point-in-time universe
3. Look-ahead bias checker catches known violations
4. Walk-forward validation mandatory for `backtested` → `forward_testing` promotion
5. Monte Carlo 5th percentile Sharpe gate enforced
6. VIF analysis removes features with VIF > 10

## References

- CTO Audit: QS-B1 through QS-B6, QS-S4
