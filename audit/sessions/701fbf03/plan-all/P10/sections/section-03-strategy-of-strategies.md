# Section 03: Strategy-of-Strategies Meta-Allocator

## Objective

Build a meta-model that dynamically allocates capital weight across active strategies based on regime, per-strategy rolling IC, volatility, and correlation structure. Replaces equal-weight allocation in the fund_manager node with data-driven weights that adapt to current market conditions.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/learning/meta_allocator.py` | Create | Meta-allocation model: input features, weight computation, retraining |
| `src/quantstack/db.py` | Modify | Add `meta_allocator_weights` and `meta_allocator_training_log` tables |

## Implementation Details

### 1. Database Schema

Add to `ensure_tables()` in `db.py`:

```sql
CREATE TABLE IF NOT EXISTS meta_allocator_weights (
    id              SERIAL PRIMARY KEY,
    strategy_id     TEXT NOT NULL,
    weight          REAL NOT NULL,
    regime          TEXT NOT NULL,
    vol_bucket      TEXT NOT NULL,      -- 'low', 'normal', 'high'
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_maw_computed
    ON meta_allocator_weights (computed_at DESC);

CREATE TABLE IF NOT EXISTS meta_allocator_training_log (
    id              SERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,
    training_window_days INT NOT NULL,
    n_strategies    INT NOT NULL,
    r_squared       REAL,
    training_mse    REAL,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 2. Meta-Allocator Module (`learning/meta_allocator.py`)

**Core data class:**

```python
@dataclass
class StrategyFeatures:
    strategy_id: str
    rolling_ic_21d: float       # from ic_attribution module
    rolling_sharpe_21d: float   # realized Sharpe over trailing 21 days
    vol_contribution: float     # strategy's contribution to portfolio vol
    correlation_avg: float      # avg pairwise correlation with other strategies
    regime_fit: float           # from regime-strategy matrix (0.0-1.0)
    days_active: int            # how long strategy has been live
```

**`compute_weights(strategies: list[StrategyFeatures], regime: str, portfolio_vol: float) -> dict[str, float]`**
- Input: current feature snapshot for each active strategy, current regime, portfolio vol level
- Constraints: weights sum to 1.0, each weight in [0.0, 0.5] (no single strategy > 50%), minimum weight 0.02 for any included strategy
- Initial model: weighted score approach (not ML -- upgradeable later):
  ```
  raw_score = (0.35 * normalized_ic) + (0.25 * normalized_sharpe) + (0.20 * regime_fit) + (0.10 * (1 - correlation_avg)) + (0.10 * experience_factor)
  ```
  where `experience_factor = min(1.0, days_active / 30)` (ramp-up for new strategies)
- Apply vol-scaling: if portfolio_vol > target (e.g., 15% annualized), scale down high-vol-contribution strategies
- Normalize to sum to 1.0 after applying floor/cap constraints
- Return `{"strategy_id": weight, ...}`

**`retrain_model(window_days=90)`**
- Gather historical: per-strategy features at each monthly snapshot + realized forward returns
- Fit sklearn LinearRegression: features -> realized_return (or Sharpe)
- Store coefficients as the new model version
- Log training metrics to `meta_allocator_training_log`
- Graceful fallback: if < 3 months of data or < 3 strategies, use the default weighted-score approach

**`get_current_weights(regime: str) -> dict[str, float]`**
- Fetch the most recent weights from `meta_allocator_weights` for the given regime
- If no weights exist or weights are stale (> 7 days), recompute from current features

**`persist_weights(weights: dict[str, float], regime: str, vol_bucket: str, model_version: str)`**
- Write to `meta_allocator_weights` table

### 3. Feature Collection

The module needs to pull features from existing systems:
- **Rolling IC**: from `learning/ic_attribution.py` (`ICAttributionTracker.get_report()`)
- **Rolling Sharpe**: compute from `closed_trades` table (realized P&L per strategy)
- **Correlation**: compute from daily strategy returns (from `closed_trades` or `positions`)
- **Regime fit**: look up from the regime-strategy matrix in CLAUDE.md (hard-coded mapping, configurable)

### 4. Regime-Strategy Mapping

Encode the regime-strategy matrix from the project config as a lookup:

```python
REGIME_FIT = {
    ("trending_up", "swing_momentum"): 1.0,
    ("trending_up", "mean_reversion"): 0.1,
    ("trending_down", "short_setups"): 1.0,
    ("trending_down", "aggressive_longs"): 0.0,
    ("ranging", "mean_reversion"): 1.0,
    ("ranging", "trend_following"): 0.1,
    # ... etc.
}
```

Default to 0.5 for unknown combinations.

## Test Requirements

File: `tests/unit/learning/test_meta_allocator.py`

1. **test_weights_sum_to_one** -- compute weights for 5 strategies, verify sum == 1.0
2. **test_weight_cap_enforced** -- one dominant strategy with very high IC, verify weight <= 0.5
3. **test_weight_floor_enforced** -- strategy with low scores still gets >= 0.02 if included
4. **test_regime_fit_impacts_weight** -- trending_up regime favors momentum over mean_reversion
5. **test_high_vol_scales_down** -- when portfolio_vol is high, high-vol-contribution strategies get lower weight
6. **test_new_strategy_ramp_up** -- strategy with days_active=5 gets lower weight than days_active=60 (all else equal)
7. **test_correlation_penalty** -- highly correlated strategy gets lower weight
8. **test_retrain_insufficient_data** -- fewer than 3 months data, verify graceful fallback to default model
9. **test_persist_and_retrieve_weights** -- persist weights, retrieve via get_current_weights, verify match
10. **test_stale_weights_trigger_recompute** -- weights older than 7 days trigger recomputation

## Acceptance Criteria

- [ ] `meta_allocator_weights` and `meta_allocator_training_log` tables created
- [ ] `compute_weights` produces valid weights (sum to 1.0, within [0.02, 0.5] bounds)
- [ ] Regime fit, IC, Sharpe, correlation, and experience all influence weight allocation
- [ ] Vol-scaling reduces exposure when portfolio volatility exceeds target
- [ ] `retrain_model` gracefully falls back when insufficient data
- [ ] Monthly retraining logs training metrics
- [ ] All 10 unit tests pass
