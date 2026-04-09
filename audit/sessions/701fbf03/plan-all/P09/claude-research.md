# P09 Research: Reinforcement Learning Pipeline

## Codebase Research

### What Exists
- **FinRL dependencies**: torch, stable-baselines3, gymnasium in pyproject.toml
- **FinRL Dockerfile**: `Dockerfile.finrl` for GPU training
- **FinRL config**: `src/quantstack/finrl/config.py` — environment configs, model hyperparams
- **FinRL tools**: 11 stubbed tools in `src/quantstack/tools/langchain/` (all return NotImplementedError)
- **ML pipeline (P03)**: model registry, walk-forward validation, hyperparameter optimization via Optuna
- **Training service**: `src/quantstack/ml/training_service.py` — existing ML training infrastructure

### What's Needed (Gaps)
1. **Gymnasium environments**: No custom environments exist — need PortfolioOptEnv, ExecutionOptEnv, StrategySelectEnv
2. **RL training pipeline**: No training loop — need walk-forward RL training with early stopping
3. **FinRL tool implementations**: All 11 tools are stubs — need at least 8 implemented
4. **RL model registry integration**: Model registry exists but no RL-specific metadata
5. **Safety constraints**: No RL-specific position/turnover limits
6. **RL signal collector**: No collector to integrate RL predictions into signal synthesis

## Domain Research

### RL for Trading — Practical Considerations
- PPO is the standard for continuous action spaces (portfolio allocation)
- DQN works well for discrete actions (execution fractions)
- Training on historical data has overfitting risk — walk-forward validation is essential
- Reward shaping matters: risk-adjusted returns (Sharpe-like) train better than raw returns
- Turnover penalty prevents the agent from learning high-frequency churn

### Stable-Baselines3 Integration
- SB3 provides PPO, SAC, DQN out of the box
- Custom gymnasium environment is the main integration point
- Callbacks for early stopping, checkpointing, logging
- CPU training is feasible for moderate state/action spaces (< 100 features, < 50 assets)
