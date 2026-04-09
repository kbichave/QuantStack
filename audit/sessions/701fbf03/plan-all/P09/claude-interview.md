# P09 Self-Interview: Reinforcement Learning Pipeline

## Q1: How do you prevent RL overfitting on historical data?
**A:** Three mechanisms: (a) walk-forward validation — train on T-252 to T-21, validate on T-21 to T, retrain weekly; (b) early stopping — halt when validation Sharpe degrades for 3 consecutive checkpoints; (c) regularization in reward — turnover penalty (20%/day) prevents the agent from memorizing specific trades.

## Q2: What happens when RL recommends an action that conflicts with risk gate limits?
**A:** RL output is ADVISORY, not authoritative. It goes through the signal synthesis pipeline as one more collector (weight = 0.15). The final trade decision still passes through risk gate. If RL says "go 100% AAPL" but risk gate caps single-stock at 5%, the gate wins. This is a hard rule.

## Q3: How do the three RL environments interact?
**A:** They don't directly interact. Each solves a different optimization problem: PortfolioOptEnv optimizes asset allocation weights, ExecutionOptEnv optimizes order execution timing, StrategySelectEnv optimizes strategy allocation. They can run independently. StrategySelectEnv output feeds into the meta-allocator (P10), PortfolioOptEnv informs fund_manager sizing, ExecutionOptEnv informs the execution pipeline (P02).

## Q4: CPU training feasibility — what are the realistic training times?
**A:** For PortfolioOptEnv with ~50 assets and ~20 features: state space is 50×20 = 1,000 dims. PPO with 500K steps, batch size 64, takes ~2-4 hours on CPU. This fits in the overnight compute window. ExecutionOptEnv is smaller (state ~10 dims, discrete actions), trains in <30 min. GPU is nice-to-have but not required.

## Q5: How does the RL signal collector integrate with existing synthesis?
**A:** New `rl_signal` collector follows the standard collector pattern: `collect(symbol, ...)` → dict with signal fields. It loads the latest model checkpoint, runs inference on current features, and returns predicted action (allocation weight for PortfolioOptEnv, or execution fraction for ExecutionOptEnv). IC is tracked like all other collectors. Feature flag `rl_signal_enabled()` gates it.

## Q6: What's the 30-day paper validation gate?
**A:** RL models cannot contribute to live trading decisions for the first 30 days after deployment. During this period, RL predictions are recorded but have zero weight in synthesis. After 30 days, if IC > 0.02 and no anomalies detected, weight is gradually increased from 0 to 0.15 over 7 days.

## Q7: How do you handle the sim-to-real gap (training in simulation vs trading live)?
**A:** The walk-forward validation partially addresses this — we validate on recent unseen data. Additionally: (a) the reward function includes transaction costs and slippage estimates from P02's TCA data, (b) the action space is constrained (max 10% position change per step) to avoid unrealistic jumps, (c) the turnover penalty discourages strategies that only work in frictionless simulation.

## Q8: What model registry metadata is needed for RL models?
**A:** Extend existing model_registry with: `model_category='rl'`, `env_name` (which environment), `training_episodes`, `final_sharpe`, `action_space_dim`, `state_space_dim`, `sb3_algorithm` (PPO/DQN/SAC). This uses the same A/B promotion path as ML models (P03) — shadow test before promotion.
