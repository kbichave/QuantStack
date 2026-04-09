"""Tests for Optuna hyperparameter optimization in ModelTrainer."""

import numpy as np
import pandas as pd
import pytest

from quantstack.ml.trainer import ModelTrainer, TrainingConfig


def _make_synthetic_data(n_samples: int = 500, n_features: int = 10, seed: int = 42):
    """Generate synthetic binary classification data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=dates,
    )
    # Target correlated with first 3 features
    signal = X.iloc[:, :3].sum(axis=1)
    y = pd.Series((signal > signal.median()).astype(int), index=dates, name="label")
    return X, y


class TestOptunaOptimization:
    def test_optimize_returns_valid_config(self):
        """Optuna returns a TrainingConfig with different values from defaults."""
        X, y = _make_synthetic_data()
        config = TrainingConfig(
            model_type="lightgbm",
            use_optuna=True,
            optuna_n_trials=5,
            optuna_timeout=60,
        )
        trainer = ModelTrainer(config)
        optimized = trainer.optimize_hyperparameters(X, y)

        assert isinstance(optimized, TrainingConfig)
        assert optimized.model_type == "lightgbm"
        # At least one param should differ from defaults
        defaults = TrainingConfig()
        changed = (
            optimized.learning_rate != defaults.learning_rate
            or optimized.max_depth != defaults.max_depth
            or optimized.n_estimators != defaults.n_estimators
        )
        assert changed, "Optuna should produce non-default params"

    def test_optuna_cv_score_reasonable(self):
        """Optimized model achieves AUC > 0.5 (better than random)."""
        X, y = _make_synthetic_data()
        config = TrainingConfig(
            model_type="lightgbm",
            use_optuna=True,
            optuna_n_trials=5,
            optuna_timeout=60,
        )
        trainer = ModelTrainer(config)
        result = trainer.train(X, y)

        mean_cv = np.nanmean(result.cv_scores)
        assert mean_cv > 0.5, f"CV AUC {mean_cv} should be > 0.5"

    def test_optuna_with_xgboost(self):
        """Optuna works with XGBoost model type."""
        X, y = _make_synthetic_data(n_samples=300)
        config = TrainingConfig(
            model_type="xgboost",
            use_optuna=True,
            optuna_n_trials=3,
            optuna_timeout=30,
        )
        trainer = ModelTrainer(config)
        optimized = trainer.optimize_hyperparameters(X, y)
        assert optimized.model_type == "xgboost"

    def test_optuna_with_catboost(self):
        """Optuna works with CatBoost model type."""
        X, y = _make_synthetic_data(n_samples=300)
        config = TrainingConfig(
            model_type="catboost",
            use_optuna=True,
            optuna_n_trials=3,
            optuna_timeout=30,
        )
        trainer = ModelTrainer(config)
        optimized = trainer.optimize_hyperparameters(X, y)
        assert optimized.model_type == "catboost"
        assert optimized.max_depth <= 10  # CatBoost cap

    def test_optuna_disabled_uses_defaults(self):
        """With use_optuna=False, training uses default hyperparameters."""
        X, y = _make_synthetic_data()
        config = TrainingConfig(model_type="lightgbm", use_optuna=False)
        trainer = ModelTrainer(config)
        result = trainer.train(X, y)

        # Config should still be the defaults
        assert result.config.learning_rate == 0.05
        assert result.config.max_depth == 6

    def test_optuna_timeout_respected(self):
        """Optuna respects the timeout parameter."""
        import time

        X, y = _make_synthetic_data()
        config = TrainingConfig(
            model_type="lightgbm",
            use_optuna=True,
            optuna_n_trials=1000,  # Would take forever
            optuna_timeout=3,  # But timeout is 3 seconds
        )
        trainer = ModelTrainer(config)

        start = time.time()
        trainer.optimize_hyperparameters(X, y)
        elapsed = time.time() - start

        # Should finish in well under 30s (timeout + overhead)
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"
