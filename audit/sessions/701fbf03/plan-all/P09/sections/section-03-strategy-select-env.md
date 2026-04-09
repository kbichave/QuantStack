# Section 03: Strategy Selection Environment

## Objective

Create a `StrategySelectEnv` Gymnasium environment that learns optimal allocation across active strategies based on regime, per-strategy performance metrics, and market conditions. This differs from the existing `AlphaSelectionEnv` (which selects a single alpha signal) by producing continuous allocation weights across strategies and optimizing for portfolio Sharpe ratio.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/finrl/environments.py` | **Modify** | Add `StrategySelectEnv` class |

## Implementation Details

### StrategySelectEnv Class

**State space** (from plan section 3.3):
- Regime: one-hot encoding (4 values: trending_up, trending_down, ranging, unknown)
- Per-strategy metrics (for each of N active strategies):
  - IC over last 21 days (float)
  - Sharpe over last 21 days (float)
  - Win rate over last 21 days (float)
  - Current allocation weight (float)
- Market-level features:
  - Volatility level (float, normalized)
  - Correlation structure summary (float, average pairwise correlation)
  - VIX percentile (float)
  - Market breadth (float)

Total observation dim: `4 + 4 * n_strategies + 4`

**Action space**: `Box(low=0, high=1, shape=(n_strategies,))` with softmax normalization to produce allocation weights that sum to 1.0. This matches plan section 3.3: "Continuous allocation per active strategy (softmax)".

**Reward**: Rolling 21-day portfolio Sharpe ratio (plan section 3.3). Computed as:
```
reward = mean(portfolio_returns[-21:]) / (std(portfolio_returns[-21:]) + 1e-8) * sqrt(252)
```

**Constructor parameters**:
- `strategy_names: list[str] | None = None` - list of strategy identifiers (defaults to a representative set)
- `strategy_returns: dict[str, list[float]] | None = None` - historical returns per strategy (for replay)
- `lookback: int = 21` - rolling window for IC/Sharpe computation
- `max_steps: int = 252` - one trading year per episode
- `seed: int | None = None`

**Episode mechanics**:
- `reset()`: Initialize equal-weight allocation, warm up strategy return history for `lookback` steps
- `step(action)`: Apply softmax to action, compute weighted portfolio return, track 21-day rolling window, compute Sharpe reward
- Regime transitions: use Markov chain with 5% transition probability (same pattern as `AlphaSelectionEnv`)

### Distinction from AlphaSelectionEnv

| Aspect | AlphaSelectionEnv | StrategySelectEnv |
|--------|-------------------|-------------------|
| Action | Discrete (pick one alpha) | Continuous (allocate across all) |
| Reward | Single-step return + regret | Rolling 21-day Sharpe |
| Scope | Alpha signals | Trading strategies |
| Use case | Signal selection | Capital allocation |

Both can coexist. `StrategySelectEnv` is the higher-level allocator that sits on top of the strategy layer.

## Test Requirements

1. **Shape validation**: Observation dim matches `4 + 4 * n_strategies + 4`
2. **Softmax normalization**: Action weights always sum to 1.0
3. **Sharpe reward**: After 21 steps of positive portfolio returns, reward is positive
4. **Sharpe reward sign**: Volatile negative returns produce negative Sharpe reward
5. **Regime transitions**: Over 1000 steps, all 4 regimes are visited
6. **Strategy replay**: When `strategy_returns` are provided, environment replays those returns
7. **PPO compatibility**: Continuous action space works with PPO training

## Acceptance Criteria

- [ ] `StrategySelectEnv` is importable from `quantstack.finrl.environments`
- [ ] Continuous action space with softmax normalization
- [ ] Reward is rolling 21-day portfolio Sharpe ratio
- [ ] State includes regime one-hot, per-strategy IC/Sharpe/win-rate/allocation, and market features
- [ ] Passes Gymnasium `check_env()` validation
- [ ] Works with synthetic data and with provided strategy return histories
- [ ] Model type is PPO (verified in training integration later in section-04)
