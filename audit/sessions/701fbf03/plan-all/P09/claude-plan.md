# P09 Implementation Plan: Reinforcement Learning Pipeline

## 1. Background

QuantStack has FinRL dependencies (torch, stable-baselines3, gymnasium) in pyproject.toml, a Dockerfile.finrl for GPU training, and 11 stubbed FinRL tools. P03 (ML pipeline) provides the training infrastructure (model registry, walk-forward validation, hyperparameter optimization). P09 transforms FinRL from stubs to a production RL pipeline.

## 2. Anti-Goals

- **Do NOT build custom RL algorithms** — use stable-baselines3 PPO/SAC/DQN implementations
- **Do NOT require GPU for training** — CPU training must work (GPU is optional optimization)
- **Do NOT allow RL to bypass risk gate** — RL outputs are ADVISORY, same as all other signals
- **Do NOT deploy RL to live without 30-day paper validation** — hard gate

## 3. RL Environments

### 3.1 Portfolio Optimization Environment

Custom gymnasium environment `PortfolioOptEnv`:
- **State space**: (n_assets, n_features) — returns, volume, technical indicators, regime, current allocation
- **Action space**: Continuous, sum-to-1 target weights (Box space, softmax normalization)
- **Reward**: risk-adjusted return = daily_return - 0.5 * daily_return² / target_vol² - turnover_penalty
- **Model**: PPO (continuous action, stable training)

### 3.2 Order Execution Environment

Custom gymnasium environment `ExecutionOptEnv`:
- **State space**: remaining_qty, time_remaining, spread, volume_profile, recent_fills
- **Action space**: Discrete — fraction buckets (0%, 10%, 25%, 50%, 100% of remaining)
- **Reward**: negative implementation shortfall vs TWAP benchmark
- **Model**: DQN (discrete actions, simpler training)

### 3.3 Strategy Selection Environment

Custom gymnasium environment `StrategySelectEnv`:
- **State space**: regime, per-strategy IC/Sharpe over last 21d, vol level, correlation structure
- **Action space**: Continuous allocation per active strategy (softmax)
- **Reward**: portfolio Sharpe ratio (rolling 21-day)
- **Model**: PPO

## 4. Training Infrastructure

### 4.1 Walk-Forward Validation

- Train window: T-252 to T-21 (1 year minus 1 month buffer)
- Validation window: T-21 to T (last month)
- Retrain: weekly (overnight, on finrl-worker container)
- Checkpoint: every 10K steps to `models/rl/` directory

### 4.2 Early Stopping

Stop training when:
- Validation Sharpe degrades for 3 consecutive checkpoints
- Max training steps reached (default 500K)
- NaN in loss function

### 4.3 Model Registry Integration

Store RL models in existing ML model registry:
- Model type: `rl_ppo`, `rl_sac`, `rl_dqn`
- Metadata: environment name, training episodes, final Sharpe, action space dim
- Versioning: same A/B promotion path as ML models (P03)

## 5. FinRL Tool Implementation

Implement 8 core tools (3 monitoring/utility tools can remain stubs):

```python
finrl_train_model(env_name, model_type, hyperparams) -> training_result
finrl_evaluate(model_id, eval_window) -> performance_metrics
finrl_ensemble(model_ids, weights) -> ensemble_prediction
finrl_predict(model_id, current_state) -> action_recommendation
finrl_list_models() -> model_registry_entries
finrl_compare(model_id_1, model_id_2) -> comparison_metrics
finrl_status(job_id) -> training_job_status
finrl_promote(model_id) -> promotion_result
```

## 6. Safety Constraints

- RL signal as additional collector in signal synthesis (weight = 0.15, same as ML)
- Maximum position change per step: 10% of portfolio
- Turnover penalty: 20%/day in reward function
- 30-day paper trading before any RL contributes to live decisions
- Feature flag: `rl_signal_enabled()`, default False

## 7. Schema

- `rl_training_runs`: (id, env_name, model_type, start_time, end_time, episodes, final_sharpe, status)
- `rl_model_checkpoints`: (id, training_run_id, step, sharpe, file_path)

## 8. Testing

- Environment: step through episode, verify state/action/reward shapes
- Training: train for 100 episodes, verify convergence (reward increasing)
- Inference: predict produces valid action in action space
- Safety: verify risk gate filters RL recommendations
