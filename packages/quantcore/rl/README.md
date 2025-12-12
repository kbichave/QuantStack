# QuantCore Reinforcement Learning Module

This module provides reinforcement learning components for trading research.

## Maturity Status

| Component | Status | Data Required | Known Limitations |
|-----------|--------|---------------|-------------------|
| **Base Classes** | Stable | N/A | Core abstractions |
| `RLAgent` | Stable | N/A | Abstract base class |
| `RLEnvironment` | Stable | N/A | Abstract base class |
| `ReplayBuffer` | Stable | N/A | Experience storage |
| **Environments** | | | |
| `ExecutionEnvironment` | Stable | OHLCV with volume | Works well with data |
| `OptionsEnvironment` | Stable | OHLCV + features | Works well with data |
| `SpreadEnvironment` | **Experimental** | spread_data | USD/curve features stubbed |
| `SizingEnvironment` | **Experimental** | data + signals | Falls back to synthetic |
| `AlphaSelectionEnvironment` | **Experimental** | market_data (optional) | Fully synthetic alpha returns |
| **Agents** | | | |
| `SpreadArbitrageAgent` | Experimental | SpreadEnvironment | DQN with heuristic fallback |
| `ExecutionAgent` | Experimental | ExecutionEnvironment | DQN-based |
| `SizingAgent` | Experimental | SizingEnvironment | PPO-based |

## Stability Definitions

- **Stable**: Works correctly with appropriate input data. Suitable for research use.
- **Experimental**: May require synthetic data fallbacks. Use for exploration only.
  Results may not reflect real market dynamics without proper data.

## Usage Guidelines

### Providing Real Data

For best results, always provide real market data to environments:

```python
import pandas as pd
from quantcore.rl.spread import SpreadEnvironment

# Load spread data with required columns
spread_data = pd.DataFrame({
    "spread": wti_prices - brent_prices,
    "wti": wti_prices,      # Optional: for correlation
    "brent": brent_prices,  # Optional: for correlation
    # "usd": usd_index,     # Optional: for USD regime
    # "curve": curve_shape, # Optional: for contango/backwardation
}, index=dates)

env = SpreadEnvironment(spread_data=spread_data)
```

### Checking Data Requirements

Each environment has a `DATA_REQUIREMENTS` attribute documenting what it needs:

```python
from quantcore.rl.spread import SpreadEnvironment

print(SpreadEnvironment.DATA_REQUIREMENTS.required_columns)
# ['spread']

print(SpreadEnvironment.DATA_REQUIREMENTS.optional_columns)
# ['wti', 'brent', 'usd', 'curve']
```

### Validating Input Data

Use the validation module to check your data:

```python
from quantcore.validation import DataFrameValidator

result = DataFrameValidator.validate_spread_data(spread_data)
if not result.is_valid:
    raise ValueError(f"Invalid data: {result.errors}")
result.log_warnings()
```

## What Changed (Phase 0 Fixes)

### Fixed: Random Feature Generation

Previously, several state features in `SpreadEnvironment` returned random values:

```python
# OLD (BROKEN) - returned random noise instead of computed features
def _get_volatility_regime(self) -> float:
    return np.random.beta(2, 5)  # Random!
```

Now these features are computed from actual data:

```python
# NEW (FIXED) - computes from spread returns
def _get_volatility_regime(self) -> float:
    # Uses precomputed rolling volatility percentile
    if self._volatility_cache is not None:
        return float(self._volatility_cache.iloc[self.data_idx])
    return 0.5  # Neutral for insufficient data
```

### Added: Explicit Warnings

Environments now log warnings when:
- Required data is missing
- Falling back to synthetic generation
- Optional features are unavailable

### Added: Data Validation

New validation utilities catch common data issues early:
- Missing required columns
- NaN/inf values
- Negative prices
- Unsorted indices

## Limitations

1. **Not Production-Ready**: This module is for research and backtesting only.
   Do not use for live trading without extensive additional validation.

2. **Sample Efficiency**: RL algorithms require significant data. Results with
   limited data may be misleading.

3. **Reward Engineering**: Performance is highly sensitive to reward function
   design. The provided reward functions are starting points, not optimal.

4. **Market Impact**: Impact models are simplified. Real execution costs
   may differ significantly.

5. **Synthetic Fallbacks**: Experimental environments fall back to synthetic
   data when real data is unavailable. This is for API compatibility only;
   results are not meaningful for trading decisions.

## Future Work

- [ ] Add proper USD index data integration
- [ ] Implement futures curve shape calculation
- [ ] Add more sophisticated impact models
- [ ] Improve sample efficiency with offline RL
- [ ] Add comprehensive evaluation metrics


