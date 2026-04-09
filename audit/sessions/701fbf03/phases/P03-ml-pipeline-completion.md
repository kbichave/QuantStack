# P03: ML Pipeline Completion

**Objective:** Transform the ML pipeline from hardcoded defaults to a production-grade system with hyperparameter optimization, model versioning, A/B testing, robust concept drift detection, and feature importance validation.

**Scope:** ml/, learning/drift_detector.py, tools/langchain/ml_tools.py

**Depends on:** None

**Enables:** P09 (RL Trading), P13 (Causal Alpha), P14 (Advanced ML)

**Effort estimate:** 1-2 weeks

---

## What Changes

### 3.1 Hyperparameter Optimization (QS-M1)

**Problem:** `ml/trainer.py:46-63` uses hardcoded `learning_rate=0.05, max_depth=6, n_estimators=500`.

**Implementation:**
```python
# Add to ml/trainer.py
import optuna

def optimize_hyperparameters(
    X_train, y_train, X_val, y_val,
    model_type: str = "lightgbm",
    n_trials: int = 100,
    cv_folds: int = 5,
) -> dict:
    """Bayesian optimization with purged cross-validation."""
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        # Use purged CV from core/validation/purged_cv.py
        return purged_cv_score(X_train, y_train, params, cv_folds)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=600)
    return study.best_params
```

**Files:**
- `src/quantstack/ml/trainer.py` — add Optuna integration
- `pyproject.toml` — add `optuna` dependency

### 3.2 Model Versioning & Registry (QS-M3)

**Problem:** Models saved as `{symbol}_latest.joblib` — latest overwrites previous.

**Implementation:**
```sql
-- New table: model_registry
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    model_type TEXT NOT NULL,      -- lightgbm, xgboost, catboost
    version INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL,
    train_ic FLOAT,
    oos_ic FLOAT,
    train_sharpe FLOAT,
    oos_sharpe FLOAT,
    config_hash TEXT,              -- SHA256 of hyperparameters
    model_path TEXT NOT NULL,      -- filesystem path
    is_active BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    UNIQUE(symbol, model_type, version)
);
```

**Save as:** `{symbol}_{model_type}_v{version}_{ic:.4f}.joblib`
**Keep:** Last 5 versions per symbol per model type
**A/B:** Run shadow predictions from previous version, compare IC weekly

**Files:**
- `src/quantstack/db.py` — add `model_registry` table
- `src/quantstack/ml/ml_signal.py` — version-aware save/load
- New: `src/quantstack/ml/model_registry.py` — registry CRUD

### 3.3 A/B Testing Framework

**Implementation:**
- Active model serves predictions for trading
- Shadow model (previous version) runs predictions in parallel but doesn't trade
- Weekly comparison: if shadow model IC > active model IC for 2 consecutive weeks → swap
- Automatic rollback: if active model IC drops >50% from training IC → revert to previous

**Files:**
- New: `src/quantstack/ml/ab_testing.py`
- `src/quantstack/ml/ml_signal.py` — add shadow prediction path

### 3.4 Enhanced Concept Drift Detection (QS-M2)

**Problem:** Only PSI on 6 hardcoded features.

**Add:**
1. **Rolling IC per feature** — alert when IC drops >50% from training period
2. **Label distribution monitoring** — alert when win rate drops below 40%
3. **Feature interaction drift** — correlation between top features shifted
4. **Per-feature PSI thresholds** — calibrated from historical distributions (not generic 0.10/0.25)

**Files:**
- `src/quantstack/learning/drift_detector.py` — expand beyond PSI
- `scripts/scheduler.py` — add daily drift check job

### 3.5 Feature Importance Validation (QS-M4)

**Problem:** Only MDI (built-in) implemented. MDI biased toward high-cardinality.

**Add:** MDA (Mean Decrease Accuracy) + SHAP consensus
- Feature is "important" only if ranked top-20 by ≥2/3 methods
- Discard features important by only one method

**Files:**
- `src/quantstack/ml/feature_importance.py` — implement MDA, SHAP consensus
- `src/quantstack/ml/trainer.py` — run importance validation post-training

### 3.6 Regime-Stratified Cross-Validation (QS-M5)

**Problem:** Fixed `test_size=0.2` regardless of regime.

**Fix:** Each fold must contain proportional representation of all regime types.

**Files:**
- `src/quantstack/core/validation/purged_cv.py` — add regime stratification parameter

### 3.7 Implement Stubbed ML Tools

**Implement the 5 stubbed ML tools:**
- `train_model` — trigger Optuna-optimized training
- `predict_ml_signal` — version-aware prediction
- `check_concept_drift` — enhanced drift detection
- `compute_deflated_sharpe_ratio` — from `core/validation/`
- `compute_probability_of_overfitting` — from `core/validation/`

**Files:**
- `src/quantstack/tools/langchain/ml_tools.py` — implement all 5

## Tests

| Test | What It Verifies |
|------|-----------------|
| `test_optuna_improves_over_defaults` | Optimized params produce higher CV score |
| `test_model_registry_versioning` | Multiple versions stored, latest flagged active |
| `test_ab_swap_on_better_shadow` | Shadow model promoted when IC exceeds active for 2 weeks |
| `test_drift_detection_rolling_ic` | Alert fires when feature IC drops 50% |
| `test_feature_importance_consensus` | Feature needs 2/3 methods to be "important" |
| `test_regime_stratified_cv` | Each fold has proportional regime representation |

## Acceptance Criteria

1. `optuna` hyperparameter search runs with purged CV
2. `model_registry` table tracks all trained models with metrics
3. Shadow model predictions logged alongside active predictions
4. Drift detector monitors rolling IC, label distribution, feature interactions
5. All 5 ML tools functional (not stubbed)

## Risk

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Optuna search takes too long | Medium | 100 trials max, 600s timeout |
| A/B testing doubles compute | Low | Shadow predictions are fast (inference only) |
| Model registry grows unbounded | Low | Keep last 5 versions, archive older |

## References

- CTO Audit: QS-M1 through QS-M5
- Build-vs-buy: See `../build-vs-buy/hyperparameter-optimization.md`
