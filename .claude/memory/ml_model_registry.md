# ML Model Registry

Tracks all trained ML models in the system. Updated during /workshop and /reflect sessions.

## Format

| Name | Type | Symbol | Timeframe | Features | OOS Accuracy | Trained | Last Validated | Status | Notes |
|------|------|--------|-----------|----------|-------------|---------|---------------|--------|-------|
| — | — | — | — | — | — | — | — | — | No models trained yet |

## Model Types Available

| Type | Module | Use case |
|------|--------|---------|
| LightGBM classifier | `quantcore.models.trainer.ModelTrainer(model_type="lightgbm")` | Direction prediction |
| XGBoost classifier | `quantcore.models.trainer.ModelTrainer(model_type="xgboost")` | Direction prediction |
| CatBoost classifier | `quantcore.models.trainer.ModelTrainer(model_type="catboost")` | Direction prediction |
| HierarchicalEnsemble | `quantcore.models.ensemble.HierarchicalEnsemble` | Multi-timeframe aggregation |
| TFT Regime | `quantcore.hierarchy.regime.tft_regime.TFTRegimeModel` | Soft regime probabilities |
| HMM Regime | `quantcore.hierarchy.regime.hmm_model.HMMRegimeModel` | State transitions + stability |
| Bayesian Changepoint | `quantcore.hierarchy.regime.changepoint.BayesianChangepointDetector` | Regime shift detection |

## When to Train

- Rule-based workshop strategy failed 2+ iterations with Sharpe < 0.5
- `get_regime()` confidence consistently < 0.6 → train HMM for better regime probs
- /reflect shows IC accuracy degrading → SHAP analysis to understand feature drift

## Status Values

- `active` — model is in use, performance within OOS baseline
- `degraded` — live accuracy < OOS accuracy - 20%, flag for retraining
- `retired` — no longer used
- `experimental` — trained but not yet integrated into any strategy
