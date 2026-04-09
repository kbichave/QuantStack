# P09: Reinforcement Learning Pipeline

**Objective:** Transform FinRL from stubs to a production RL pipeline for portfolio optimization, order execution, and strategy selection.

**Scope:** finrl/, ml/, tools/langchain/finrl_tools.py

**Depends on:** P03 (ML pipeline)

**Enables:** P14 (Advanced ML)

**Effort estimate:** 2-3 weeks

---

## What Changes

### 9.1 RL Environment Design
Three RL environments targeting different problems:

**Portfolio Optimization Environment:**
- State: returns, volume, technical indicators, regime, current allocation
- Action: target portfolio weights (continuous, sum-to-1)
- Reward: risk-adjusted return (Sharpe, with transaction cost penalty)
- Model: PPO or SAC (continuous action space)

**Order Execution Environment:**
- State: remaining quantity, time remaining, market depth, volume profile, spread
- Action: fraction of remaining to execute now
- Reward: negative implementation shortfall (vs TWAP benchmark)
- Model: DQN or PPO

**Strategy Selection Environment:**
- State: regime, recent performance per strategy, vol level, correlation structure
- Action: allocation to each active strategy
- Reward: portfolio Sharpe ratio
- Model: PPO

### 9.2 FinRL Integration
- FinRL already in pyproject.toml (torch, stable-baselines3, gymnasium)
- `Dockerfile.finrl` exists for GPU training
- 11 FinRL tools are stubbed — implement:
  - `finrl_train_model` — environment setup + training
  - `finrl_evaluate` — backtest evaluation
  - `finrl_ensemble` — ensemble of RL + traditional models
  - `finrl_predict` — real-time inference
  - `finrl_list_models` — model registry integration
  - `finrl_compare` — compare RL vs non-RL strategies
  - `finrl_status` — training job monitoring
  - `finrl_promote` — RL model → production

### 9.3 Training Infrastructure
- Overnight training on finrl-worker container (2GB+ RAM, GPU if available)
- Walk-forward validation: train on T-252:T-21, validate on T-21:T
- Model checkpointing every 10K steps
- Early stopping on validation Sharpe degradation

### 9.4 Safety Constraints
- RL agent outputs are ADVISORY — pass through risk gate like all other signals
- Maximum position change per step: 10% of portfolio
- Penalty for turnover > 20%/day in reward function
- Paper trading for 30 days before any RL signal contributes to live decisions

## Key Packages
- `stable-baselines3` (existing) — PPO, SAC, DQN
- `gymnasium` (existing) — environment framework
- `finrl` (existing) — financial RL environments
- Consider: `elegantrl` (performance), `RLlib` (distributed training)

## Build-vs-Buy Assessment
- **FinRL** (existing): Good starting point, active development, 8K+ stars
- **ElegantRL**: Better performance, GPU-optimized, but smaller community
- **RLlib**: Production-grade distributed, overkill for single-machine
- **Recommendation:** Start with FinRL (already integrated), migrate to ElegantRL if performance is insufficient

## Acceptance Criteria

1. Portfolio optimization environment trains and produces valid allocations
2. Walk-forward validation shows RL Sharpe > random allocation
3. All 11 FinRL tools implemented and functional
4. RL signals pass through risk gate (not bypassing)
5. 30-day paper trading before live contribution
