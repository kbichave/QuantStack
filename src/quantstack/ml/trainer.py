"""
Model training pipeline for trade classification.

Supports LightGBM, XGBoost, and CatBoost with hyperparameter tuning.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: Literal["lightgbm", "xgboost", "catboost"] = "lightgbm"

    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1

    # Cross-validation
    n_splits: int = 5

    # Model hyperparameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # Class imbalance
    class_weight: str | None = "balanced"

    # Early stopping
    early_stopping_rounds: int = 50

    # Random state
    random_state: int = 42

    # Feature selection
    min_feature_importance: float = 0.001

    def to_lgb_params(self) -> dict:
        """Convert to LightGBM parameters."""
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

    def to_xgb_params(self) -> dict:
        """Convert to XGBoost parameters.

        Note: Do NOT pass objective="binary:logistic" explicitly --
        XGBClassifier sets it internally.  Passing it manually causes
        sklearn 1.8+ to misidentify the estimator as a regressor,
        which breaks cross_val_score with scoring='roc_auc'.
        """
        return {
            "eval_metric": "auc",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": 1,  # Will be set dynamically
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }


@dataclass
class TrainingResult:
    """Result of model training."""

    model: Any
    feature_names: list[str]
    feature_importance: dict[str, float]
    metrics: dict[str, float]
    cv_scores: list[float]
    scaler: StandardScaler | None = None
    config: TrainingConfig | None = None
    feature_group_importance: dict[str, float] | None = None
    feature_to_group: dict[str, str] | None = None


class ModelTrainer:
    """
    Trainer for ML classification models.

    Features:
    - Time-series aware cross-validation
    - Class imbalance handling
    - Feature importance analysis
    - Early stopping
    """

    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.scaler = StandardScaler()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
        scale_features: bool = True,
        feature_to_group: dict[str, str] | None = None,
    ) -> TrainingResult:
        """
        Train a model.

        Args:
            X: Feature matrix
            y: Target labels (0/1)
            feature_names: Optional feature names
            scale_features: Whether to scale features
            feature_to_group: Optional mapping of feature names to group tags

        Returns:
            TrainingResult with trained model and metrics
        """
        logger.info(f"Training {self.config.model_type} model on {len(X)} samples")

        # Handle feature names
        if feature_names is None:
            feature_names = list(X.columns)

        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)

        if len(X_clean) == 0:
            raise ValueError("No valid samples after data preparation")

        # Scale if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            X_clean = pd.DataFrame(X_scaled, columns=feature_names, index=X_clean.index)

        # Split data
        X_train, X_test, y_train, y_test = self._time_series_split(X_clean, y_clean)

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Train class balance: {y_train.mean():.3f}")

        # Train model
        if self.config.model_type == "lightgbm":
            model = self._train_lightgbm(X_train, y_train, X_test, y_test)
        elif self.config.model_type == "xgboost":
            model = self._train_xgboost(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Cross-validation scores
        cv_scores = self._cross_validate(model, X_clean, y_clean)

        # Evaluate on test set
        metrics = self._evaluate(model, X_test, y_test)

        # Feature importance
        importance = self._get_feature_importance(model, feature_names)

        # Compute group importance if mapping provided
        group_importance = None
        if feature_to_group is not None:
            group_importance = self.compute_group_importances(
                importance, feature_to_group
            )
            logger.info("Feature group importances:")
            for group, imp in sorted(
                group_importance.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                logger.info(f"  {group}: {imp * 100:.2f}%")

        logger.info(f"Test AUC: {metrics['auc']:.4f}")
        logger.info(f"CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        return TrainingResult(
            model=model,
            feature_names=feature_names,
            feature_importance=importance,
            metrics=metrics,
            cv_scores=cv_scores,
            scaler=scaler,
            config=self.config,
            feature_group_importance=group_importance,
            feature_to_group=feature_to_group,
        )

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data by removing NaN/inf values."""
        # Combine for aligned filtering
        combined = pd.concat([X, y], axis=1)

        # Remove rows with any NaN or inf
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna()

        if len(combined) == 0:
            logger.warning("No valid samples after cleaning")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Split back
        X_clean = combined.iloc[:, :-1]
        y_clean = combined.iloc[:, -1]

        logger.info(f"Data prepared: {len(X)} -> {len(X_clean)} samples")

        return X_clean, y_clean

    def _time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data respecting time series order."""
        n = len(X)
        test_size = int(n * self.config.test_size)
        train_size = n - test_size

        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        return X_train, X_test, y_train, y_test

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        params = self.config.to_lgb_params()

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        return model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        params = self.config.to_xgb_params()

        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        return model

    def _cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[float]:
        """Perform time-series cross-validation with purged CV.

        Uses PurgedKFoldCV (Lopez de Prado methodology) with 1% embargo
        to prevent data leakage at fold boundaries.

        Manual CV loop instead of sklearn's cross_val_score to avoid
        XGBoost 2.x + sklearn 1.8 compatibility issue where XGBClassifier
        is misidentified as a regressor during scoring.
        """
        # Build fold splits
        try:
            from quantstack.core.validation.purged_cv import PurgedKFoldCV

            purged_cv = PurgedKFoldCV(
                n_splits=self.config.n_splits,
                embargo_pct=0.01,
            )
            splits = list(purged_cv.split(X))
            cv_iter = [(s.train_indices, s.test_indices) for s in splits]
        except (ImportError, Exception) as exc:
            logger.warning(
                f"PurgedKFoldCV failed ({exc}), falling back to TimeSeriesSplit"
            )
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            cv_iter = list(tscv.split(X))

        # Manual fold evaluation -- avoids sklearn's response_method detection
        # which misidentifies XGBClassifier 2.x as a regressor.
        import copy

        scores: list[float] = []
        for train_idx, test_idx in cv_iter:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = copy.deepcopy(model)
            try:
                fold_model.fit(X_tr, y_tr)
                y_prob = fold_model.predict_proba(X_te)[:, 1]
                fold_auc = roc_auc_score(y_te, y_prob)
                scores.append(fold_auc)
            except Exception as fold_exc:
                logger.warning(f"CV fold failed: {fold_exc}")
                scores.append(np.nan)

        return scores

    def _evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model on test set."""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        return metrics

    def _get_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Get feature importance from model."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(feature_names))

        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return dict(
            sorted(
                zip(feature_names, importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    @staticmethod
    def compute_group_importances(
        feature_importance: dict[str, float],
        feature_to_group: dict[str, str],
    ) -> dict[str, float]:
        """
        Aggregate feature importance by group.

        Args:
            feature_importance: Dict mapping feature_name -> importance
            feature_to_group: Dict mapping feature_name -> group_tag

        Returns:
            Dict mapping group_tag -> aggregated_importance
        """
        group_importance = {}

        for feature_name, importance in feature_importance.items():
            group = feature_to_group.get(feature_name, "other")

            if group not in group_importance:
                group_importance[group] = 0.0

            group_importance[group] += importance

        # Normalize to sum to 1.0
        total = sum(group_importance.values())
        if total > 0:
            group_importance = {k: v / total for k, v in group_importance.items()}

        return dict(
            sorted(
                group_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save_model(
        self,
        result: TrainingResult,
        path: str,
    ) -> None:
        """Save trained model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(result, f)

        # Save metadata as JSON
        metadata = {
            "feature_names": result.feature_names,
            "metrics": result.metrics,
            "cv_scores": result.cv_scores,
            "feature_importance": result.feature_importance,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> TrainingResult:
        """Load trained model from file."""
        with open(path, "rb") as f:
            result = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return result
