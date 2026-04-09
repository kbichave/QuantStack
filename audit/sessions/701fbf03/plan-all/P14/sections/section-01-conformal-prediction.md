# Section 01: Conformal Prediction Wrapper

## Objective

Add calibrated prediction intervals to existing LightGBM/XGBoost models using conformal prediction (MAPIE). This provides guaranteed coverage — a 90% confidence interval covers at least 90% of outcomes — giving the system a principled uncertainty estimate for every prediction.

## Files to Create/Modify

### New Files

- **`src/quantstack/ml/conformal.py`** — Conformal prediction wrapper around existing tree-based models.

### Modified Files

- **`src/quantstack/ml/model_registry.py`** — Extend `ModelVersion` dataclass and DB queries to store coverage metrics (`coverage_80`, `coverage_90`, `coverage_95`, `avg_interval_width`).

## Implementation Details

### `src/quantstack/ml/conformal.py`

```
class ConformalPredictor:
    """Wraps a trained sklearn-compatible model with MAPIE conformal intervals."""

    def __init__(self, base_model, method: str = "plus"):
        """
        Args:
            base_model: Fitted LightGBM/XGBoost/CatBoost estimator.
            method: MAPIE method — "plus" (jackknife+) recommended for financial data.
        """

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """Calibrate on held-out calibration set (NOT the training set).
        
        Uses MapieRegressor with method="plus" (jackknife+).
        The calibration set should be the most recent temporal slice
        (respecting walk-forward discipline — no future leakage).
        """

    def predict(self, X: np.ndarray, alpha: list[float] | None = None) -> ConformalResult:
        """Return point prediction + prediction intervals.
        
        Args:
            X: Feature matrix.
            alpha: Miscoverage rates. Default [0.20, 0.10, 0.05] for 80/90/95% CIs.
            
        Returns:
            ConformalResult with .point, .intervals dict keyed by coverage level.
        """

    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute empirical coverage and average width on test set.
        
        Returns dict with keys:
            coverage_80, coverage_90, coverage_95: empirical coverage rates
            avg_width_80, avg_width_90, avg_width_95: mean interval widths
            calibration_ok: bool — True if all coverages within 3% of target
        """
```

```
@dataclass
class ConformalResult:
    point: np.ndarray
    intervals: dict[str, tuple[np.ndarray, np.ndarray]]  # coverage_level -> (lower, upper)
```

### Key Design Decisions

1. **Calibration set is temporal, not random.** Use the last N rows of the training window as calibration data. Random splits would leak future information.
2. **MAPIE `method="plus"` (jackknife+)** — provides finite-sample validity and is more efficient than the naive split conformal method.
3. **`ConformalPredictor` wraps, not replaces.** The base model is unchanged; conformal calibration is a post-hoc layer.

### Model Registry Extension

Add columns to `model_registry` table:
- `coverage_80 FLOAT` — empirical coverage at 80% level
- `coverage_90 FLOAT` — empirical coverage at 90% level  
- `coverage_95 FLOAT` — empirical coverage at 95% level
- `avg_interval_width FLOAT` — average 90% CI width (narrower = more informative)

Add to `ModelVersion` dataclass:
- `coverage_80: float | None`
- `coverage_90: float | None`
- `coverage_95: float | None`
- `avg_interval_width: float | None`

Add calibration flag logic: log a warning if empirical coverage deviates >3% from target (e.g., 90% target but only 85% empirical coverage).

## Dependencies

- **PyPI**: `mapie` (MAPIE library for conformal prediction)
- **Internal**: `quantstack.ml.trainer` (provides fitted models), `quantstack.ml.model_registry`

## Test Requirements

### `tests/unit/ml/test_conformal.py`

1. **Coverage guarantee test**: Train LightGBM on synthetic data, calibrate ConformalPredictor, verify 90% CI covers >= 87% of test outcomes (3% tolerance for finite samples).
2. **Interval monotonicity**: 95% CI width >= 90% CI width >= 80% CI width.
3. **Edge case — constant predictions**: If model predicts the same value for all inputs, intervals should still be valid.
4. **Edge case — small calibration set**: Verify graceful behavior with <30 calibration samples (should warn but not crash).
5. **evaluate_coverage returns correct structure**: Verify all expected keys present with correct types.

## Acceptance Criteria

- [ ] `ConformalPredictor` wraps any sklearn-compatible model and produces calibrated intervals
- [ ] Coverage guarantee holds on held-out test data (90% CI covers >= 87%)
- [ ] `evaluate_coverage()` returns empirical coverage rates and average widths
- [ ] `ModelVersion` dataclass and DB schema extended with coverage fields
- [ ] Calibration deviation >3% triggers a logged warning
- [ ] All unit tests pass
- [ ] No GPU required — runs on CPU
