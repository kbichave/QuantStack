# Section 02: Position Sizing via Confidence Interval Width

## Objective

Use conformal prediction interval width to modulate position size. Narrow intervals indicate high model conviction (larger positions); wide intervals indicate uncertainty (smaller positions). This connects ML uncertainty directly to risk management.

## Dependencies

- **section-01-conformal-prediction** — requires `ConformalPredictor` and `ConformalResult`

## Files to Create/Modify

### New Files

- **`src/quantstack/ml/conviction_sizing.py`** — Confidence-interval-based position size scalar.

### Modified Files

- **`src/quantstack/signal_engine/collectors/ml_signal.py`** — Extend ML signal output to include CI-based conviction fields.
- **`src/quantstack/signal_engine/synthesis.py`** — Incorporate CI width into conviction scoring (if P05 conviction system is present).

## Implementation Details

### `src/quantstack/ml/conviction_sizing.py`

```
def ci_size_scalar(
    ci_lower: float,
    ci_upper: float,
    max_ci_width: float,
    floor: float = 0.2,
) -> float:
    """Compute position size scalar from prediction interval width.

    Formula: size_scalar = 1.0 - (ci_width / max_ci_width), floored at `floor`.

    Args:
        ci_lower: Lower bound of the prediction interval.
        ci_upper: Upper bound of the prediction interval.
        max_ci_width: Maximum CI width observed in the calibration set
                      (used for normalization).
        floor: Minimum scalar — never go below this. Default 0.2 (20% of full size).

    Returns:
        Float in [floor, 1.0]. Higher = more conviction = larger position.
    """
```

```
def compute_max_ci_width(conformal_predictor: ConformalPredictor, X_cal: np.ndarray) -> float:
    """Compute the maximum 90% CI width across the calibration set.
    
    This is used as the normalization denominator for ci_size_scalar.
    Should be computed once per model retrain and cached with model metadata.
    """
```

### ML Signal Collector Extension

Add the following keys to the dict returned by `collect_ml_signal`:
- `ml_ci_lower_90`: float — lower bound of 90% prediction interval
- `ml_ci_upper_90`: float — upper bound of 90% prediction interval
- `ml_ci_width_90`: float — width of 90% prediction interval
- `ml_ci_conviction`: float — ci_size_scalar output (0.2 to 1.0)

Logic: After loading the model, also load the associated `ConformalPredictor` (saved alongside the model). If no conformal calibration exists, omit these keys (backwards compatible).

### Synthesis Integration

In `synthesis.py`, when the ML signal includes `ml_ci_conviction`:
- Multiply the ML vote's effective weight by `ml_ci_conviction`
- This means a wide-CI prediction has its synthesis influence reduced proportionally
- Redistribute the reduced weight to other voters (proportionally)

## Test Requirements

### `tests/unit/ml/test_conviction_sizing.py`

1. **Scalar range**: `ci_size_scalar` always returns value in `[floor, 1.0]`.
2. **Narrow CI = high scalar**: CI width 0 yields scalar 1.0.
3. **Wide CI = low scalar**: CI width >= max_ci_width yields scalar = floor.
4. **Floor enforcement**: Even with extremely wide CI, scalar never goes below floor.
5. **Zero max_ci_width**: Handles edge case (should return floor or raise ValueError).
6. **Integration test**: ML signal collector returns CI fields when conformal model exists.

## Acceptance Criteria

- [ ] `ci_size_scalar` correctly maps CI width to [0.2, 1.0] range
- [ ] ML signal collector includes CI conviction fields when conformal model is available
- [ ] Backwards compatible — missing conformal model does not break existing signal flow
- [ ] Synthesis weights adjust based on CI conviction
- [ ] All unit tests pass
