# Section 01: Portfolio Optimization Environment

## Objective

Create a custom Gymnasium environment (`PortfolioOptEnv`) that learns optimal portfolio weight allocations across multiple assets using continuous action spaces. This environment trains PPO agents to produce risk-adjusted, turnover-penalized target weights.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/finrl/environments.py` | **Modify** | Add `PortfolioOptEnv` class alongside existing `ExecutionEnv`, `SizingEnv`, `AlphaSelectionEnv` |

## Implementation Details

### PortfolioOptEnv Class

**State space** `(n_assets, n_features)` flattened to 1D:
- Per-asset features: returns (1d, 5d, 21d), volume z-score, RSI, MACD signal, Bollinger %B
- Portfolio-level features: regime one-hot (4), current allocation vector, cash fraction
- Total observation dim: `n_assets * n_per_asset_features + 4 + n_assets + 1`

**Action space**: `Box(low=0, high=1, shape=(n_assets,))` with softmax normalization applied post-action so weights sum to 1.0. This matches the plan's "continuous, sum-to-1 target weights" requirement.

**Reward function** (from plan section 3.1):
```
reward = daily_return - 0.5 * daily_return^2 / target_vol^2 - turnover_penalty
```
Where:
- `daily_return` = portfolio-weighted return for the step
- `target_vol` = annualized target volatility (configurable, default 0.15)
- `turnover_penalty` = 0.20 * sum(|w_new - w_old|) (20bps/day per plan section 6)

**Constructor parameters**:
- `data: pd.DataFrame` - FinRL-format OHLCV with `date`, `tic`, `close`, `volume` columns
- `n_assets: int` - number of assets (derived from unique `tic` values)
- `initial_capital: float = 100_000`
- `target_vol: float = 0.15` - annualized target volatility for reward scaling
- `turnover_penalty: float = 0.002` - 20bps penalty per unit turnover
- `max_position_change: float = 0.10` - maximum per-step allocation change (plan section 6)
- `seed: int | None = None`

**Episode mechanics**:
- `reset()`: Initialize equal-weight portfolio, set data pointer to random offset (>= lookback)
- `step(action)`: Apply softmax to raw action, clip position changes to `max_position_change`, compute portfolio return, calculate reward, advance data pointer
- Episode terminates when data is exhausted or max steps reached (truncated)

**Safety constraints** (plan section 6):
- Maximum position change per step: 10% of portfolio (clamp delta between old and new weights)
- Turnover penalty baked into reward function at 20bps/day

### Integration with existing code

The existing `environments.py` already has `ExecutionEnv`, `SizingEnv`, and `AlphaSelectionEnv`. `PortfolioOptEnv` follows the same patterns:
- Inherits from `gym.Env`
- Uses `np.random.default_rng(seed)` for reproducibility
- Accepts optional `data: pd.DataFrame` with synthetic fallback
- Returns `(obs, reward, done, truncated, info)` from `step()`

## Test Requirements

1. **Shape validation**: After `reset()`, observation shape matches `observation_space.shape`
2. **Action normalization**: Raw action `[0.3, 0.7, 0.5]` produces weights summing to 1.0
3. **Position change clamp**: Action requesting 50% reallocation is clamped to 10% max change
4. **Reward sign**: Positive portfolio return with low turnover produces positive reward
5. **Turnover penalty**: Identical consecutive actions produce zero turnover penalty; maximally different actions produce maximum penalty
6. **Episode termination**: Environment terminates when data is exhausted
7. **Synthetic mode**: Environment works without data (generates synthetic returns)

## Acceptance Criteria

- [ ] `PortfolioOptEnv` is registered in `environments.py` and importable from `quantstack.finrl.environments`
- [ ] Observation and action spaces are correctly defined with proper dtypes and bounds
- [ ] Softmax normalization ensures weights always sum to 1.0
- [ ] Position change per step is clamped to 10% maximum
- [ ] Reward function matches the formula: `daily_return - 0.5 * daily_return^2 / target_vol^2 - turnover_penalty`
- [ ] Environment passes Gymnasium `check_env()` validation
- [ ] Works with both real data (FinRL DataFrame format) and synthetic data fallback
