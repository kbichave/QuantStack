# P09 Spec: Reinforcement Learning Pipeline

## Deliverables

### D1: Gymnasium Environments
- PortfolioOptEnv: continuous action space (asset weights), PPO
- ExecutionOptEnv: discrete action space (execution fractions), DQN
- StrategySelectEnv: continuous action space (strategy weights), PPO

### D2: Training Infrastructure
- Walk-forward validation (train T-252 to T-21, validate T-21 to T)
- Early stopping on validation Sharpe degradation
- Weekly retrain overnight
- Checkpoint every 10K steps

### D3: FinRL Tool Implementation
- 8 core tools: train, evaluate, ensemble, predict, list, compare, status, promote
- Uses SB3 PPO/DQN under the hood

### D4: Signal Collector Integration
- New rl_signal collector following standard pattern
- Feature flag gated, default disabled
- 30-day paper validation before weight > 0
- IC tracking from day 1

### D5: Safety Constraints
- Max position change per step: 10% of portfolio
- Turnover penalty: 20%/day in reward
- RL output is advisory — risk gate has final authority

## Dependencies
- P03 (ML Pipeline): model registry, walk-forward validation framework
- P02 (Execution): TCA data for realistic reward function
