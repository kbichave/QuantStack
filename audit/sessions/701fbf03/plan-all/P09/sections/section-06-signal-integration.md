# Section 06: Signal Engine Integration

## Objective

Add an RL signal collector to the signal engine so that RL model predictions contribute to the synthesis pipeline as an additional voter. The RL signal gets a weight of 0.15 (same as ML), gated behind a feature flag.

## Dependencies

- **section-05-finrl-tools**: RL prediction infrastructure must be functional

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/signal_engine/collectors/rl_signal.py` | **Create** | New collector that runs RL inference and returns signal dict |
| `src/quantstack/signal_engine/synthesis.py` | **Modify** | Add `rl` weight to regime weight profiles, integrate RL collector output |
| `src/quantstack/config/feedback_flags.py` | **Modify** | Add `rl_signal_enabled()` feature flag |

## Implementation Details

### RL Signal Collector

Create `src/quantstack/signal_engine/collectors/rl_signal.py` following the pattern of `ml_signal.py`:

```python
async def collect_rl_signal(symbol: str, store: DataStore) -> dict[str, Any]:
```

**Returns** (when a live/shadow RL model exists for the symbol):
```python
{
    "rl_prediction": float,       # 0-1 predicted direction probability
    "rl_direction": str,          # "bullish" | "bearish" | "neutral"
    "rl_confidence": float,       # 0-1 confidence score
    "rl_model_type": str,         # e.g., "rl_ppo"
    "rl_model_id": str,           # model registry ID
    "rl_shadow": bool,            # True if model is still in shadow
    "rl_action_raw": Any,         # raw action from the model
}
```

**Returns** `{}` when:
- No RL model is registered for the symbol's environment type
- Feature flag `rl_signal_enabled()` is False
- Model inference fails (log warning, never raise)

**Logic**:
1. Check `rl_signal_enabled()` flag -- return `{}` if disabled
2. Query `ModelRegistry` for live or shadow models of type `rl_ppo` or `rl_sac`
3. Pick the most recent live model (or most recent shadow if no live exists)
4. Construct observation vector from DataStore data for the symbol
5. Call `FinRLTrainer.predict()` to get action and confidence
6. Map action to direction: for portfolio envs, action > 0.55 = bullish, < 0.45 = bearish, else neutral
7. Return signal dict

### Synthesis Weight Integration

The existing `_WEIGHT_PROFILES` in `synthesis.py` have 6 voters: `trend`, `rsi`, `macd`, `bb`, `sentiment`, `ml`. Add `rl` as a 7th voter.

Update each regime profile to include an `rl` weight of 0.15, redistributing from the weakest indicators in each regime (same approach used for ML integration):

```python
_WEIGHT_PROFILES = {
    "trending_up": {
        "trend": 0.30, "rsi": 0.08, "macd": 0.17, "bb": 0.05,
        "sentiment": 0.10, "ml": 0.15, "rl": 0.15,
    },
    # ... similar for other regimes
}
```

The `rl` weight behaves like `ml`: when no RL signal is available, the weight is redistributed proportionally to the other active voters.

### Feature Flag

Add to `feedback_flags.py`:

```python
def rl_signal_enabled() -> bool:
    """P09: Reinforcement learning signal collector."""
    return _flag("FEEDBACK_RL_SIGNAL")
```

Default is `False` (safe-off). The RL collector returns `{}` immediately when disabled, so there is zero overhead on the synthesis pipeline when RL is not active.

### Integration with synthesis flow

In the synthesis function (wherever collectors are gathered), add the RL collector call:
```python
if rl_signal_enabled():
    rl_data = await collect_rl_signal(symbol, store)
    if rl_data:
        signals["rl"] = rl_data
```

The synthesizer already handles missing voters by redistributing their weight. No special handling is needed when RL returns `{}`.

## Test Requirements

1. **Flag disabled**: `collect_rl_signal()` returns `{}` when `FEEDBACK_RL_SIGNAL` is not set
2. **No model**: Returns `{}` when no RL model is registered
3. **Shadow model**: Returns signal with `rl_shadow=True` when only shadow model exists
4. **Live model**: Returns signal with `rl_shadow=False` when live model exists
5. **Weight redistribution**: When RL signal is absent, weights sum to 1.0 without RL
6. **Weight with RL**: When RL signal is present, all weights including RL sum to 1.0
7. **Direction mapping**: Confidence > 0.55 maps to bullish, < 0.45 to bearish

## Acceptance Criteria

- [ ] `collect_rl_signal()` exists and follows the same pattern as `collect_ml_signal()`
- [ ] Feature flag `rl_signal_enabled()` defaults to False
- [ ] RL weight is 0.15 in all regime weight profiles
- [ ] RL signal is collected only when feature flag is enabled
- [ ] Missing RL signal causes weight redistribution (no NaN or missing weights)
- [ ] Shadow predictions are tagged and passed through (not filtered)
- [ ] Never raises exceptions to the synthesis pipeline
