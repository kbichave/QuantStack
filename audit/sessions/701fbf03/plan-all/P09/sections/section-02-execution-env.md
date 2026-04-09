# Section 02: Order Execution Environment

## Objective

Enhance the existing `ExecutionEnv` in `environments.py` to align with the P09 plan specification for the order execution optimization environment. The existing implementation is functional but needs refinements to match the plan's reward formulation (negative implementation shortfall vs TWAP benchmark) and to support DQN training for the RL pipeline.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/finrl/environments.py` | **Modify** | Update `ExecutionEnv` reward to use negative implementation shortfall vs TWAP benchmark |

## Implementation Details

### Current State

`ExecutionEnv` already exists with:
- State space: 8 features (qty_frac, time_frac, price_dev, spread, volatility, volume_ratio, vwap_dev, shortfall)
- Action space: Discrete(5) with fractions {0: 0%, 1: 10%, 2: 25%, 3: 50%, 4: 100%}
- Reward: `-impact * 100 + completion_bonus - time_penalty + progress`

### Required Changes

The plan specifies the reward should be **negative implementation shortfall vs TWAP benchmark**. The current reward is ad-hoc (impact cost + bonuses). Update to:

**Reward function**:
```
reward = -implementation_shortfall_vs_twap
```
Where:
- `implementation_shortfall = (avg_fill_price - arrival_price) / arrival_price`
- `twap_benchmark = average close price over the execution horizon`
- `relative_shortfall = implementation_shortfall - twap_shortfall`

This measures how well the agent executes relative to a naive TWAP strategy.

**Additional state features** (from plan section 3.2):
- `remaining_qty` - already present as `qty_frac`
- `time_remaining` - already present as `time_frac`
- `spread` - already present
- `volume_profile` - already present as `volume_ratio`
- `recent_fills` - add a rolling window of last 3 fill sizes as fraction of total

Update observation space from shape `(8,)` to `(11,)` to include the 3 recent fill features.

**TWAP tracking**:
- Compute TWAP price as running mean of close prices over the episode
- Track cumulative TWAP cost for benchmark comparison
- Report both absolute shortfall and relative-to-TWAP in `info` dict

### Backward compatibility

The constructor signature should remain compatible. New parameters have defaults:
- `twap_reward: bool = True` - use TWAP-relative reward (new default)

## Test Requirements

1. **TWAP benchmark**: Over a flat-price episode, TWAP shortfall is near zero
2. **Reward sign**: Agent that front-loads execution in a rising market gets negative reward (bought expensive)
3. **Observation shape**: Updated shape `(11,)` matches observation_space
4. **Recent fills**: After 3 fills, recent_fills features are non-zero
5. **Completion**: Agent that fills 100% gets no time penalty
6. **DQN compatibility**: Action space is `Discrete(5)`, compatible with DQN training

## Acceptance Criteria

- [ ] Reward function uses negative implementation shortfall relative to TWAP benchmark
- [ ] Observation space includes recent fill history (3 additional features)
- [ ] TWAP price is tracked across the episode and reported in `info`
- [ ] `info` dict includes `twap_shortfall`, `agent_shortfall`, and `relative_shortfall`
- [ ] Backward compatible: existing tests still pass with updated defaults
- [ ] Passes Gymnasium `check_env()` validation
