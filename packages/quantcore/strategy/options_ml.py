"""
ML-based options strategy.

Provides baseline ML strategy for comparison with rule-based and RL.
Uses gradient boosting for direction prediction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.strategy.base import (
    Strategy,
    MarketState,
    TargetPosition,
    DataRequirements,
    PositionDirection,
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML strategy will use fallback.")


@dataclass
class MLPrediction:
    """ML model prediction."""

    direction: PositionDirection
    confidence: float
    probability: float
    features_used: int


class MLDirectionStrategy(Strategy):
    """
    ML-based direction prediction strategy.

    Uses gradient boosting to predict next-day return direction.
    Shares the same sizing/contract selection as rule-based and RL.
    """

    def __init__(
        self,
        name: str = "MLDirection",
        probability_threshold: float = 0.55,
        min_confidence: float = 0.2,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize ML strategy.

        Args:
            name: Strategy name
            probability_threshold: Min probability to take position
            min_confidence: Min confidence (scaled from probability)
            feature_columns: Specific features to use (None = all)
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        self.min_confidence = min_confidence
        self.feature_columns = feature_columns

        self.model = None
        self.scaler = None
        self._is_trained = False
        self._feature_names: List[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Train the ML model.

        Args:
            X: Feature DataFrame
            y: Target labels (1 = up, 0 = down)
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for training")
            return {"error": "sklearn not available"}

        # Select features
        if self.feature_columns:
            X = X[self.feature_columns]

        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            logger.error(f"Insufficient training data: {len(X)} samples")
            return {"error": "insufficient data"}

        self._feature_names = list(X.columns)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        self._is_trained = True

        # Compute metrics
        train_accuracy = self.model.score(X_scaled, y)
        train_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Feature importance
        importance = dict(zip(self._feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(
            f"ML model trained: accuracy={train_accuracy:.3f}, samples={len(X)}"
        )
        logger.info(f"Top features: {[f[0] for f in top_features]}")

        return {
            "train_accuracy": train_accuracy,
            "train_samples": len(X),
            "features_used": len(self._feature_names),
            "top_features": top_features,
        }

    def predict(self, features: Dict[str, float]) -> MLPrediction:
        """
        Make prediction from features.

        Args:
            features: Feature dictionary

        Returns:
            MLPrediction with direction and confidence
        """
        if not self._is_trained or self.model is None:
            return MLPrediction(
                direction=PositionDirection.FLAT,
                confidence=0.0,
                probability=0.5,
                features_used=0,
            )

        # Extract features in correct order
        X = []
        missing = 0
        for col in self._feature_names:
            val = features.get(col, np.nan)
            if pd.isna(val):
                val = 0.0
                missing += 1
            X.append(val)

        if missing > len(self._feature_names) * 0.3:
            # Too many missing features
            return MLPrediction(
                direction=PositionDirection.FLAT,
                confidence=0.0,
                probability=0.5,
                features_used=len(self._feature_names) - missing,
            )

        # Scale and predict
        X_scaled = self.scaler.transform([X])
        probability = self.model.predict_proba(X_scaled)[0, 1]

        # Convert probability to direction and confidence
        if probability > self.probability_threshold:
            direction = PositionDirection.LONG
            confidence = 2 * (probability - 0.5)  # Scale to [0, 1]
        elif probability < (1 - self.probability_threshold):
            direction = PositionDirection.SHORT
            confidence = 2 * (0.5 - probability)  # Scale to [0, 1]
        else:
            direction = PositionDirection.FLAT
            confidence = 0.0

        return MLPrediction(
            direction=direction,
            confidence=confidence,
            probability=probability,
            features_used=len(self._feature_names) - missing,
        )

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """Generate signals using ML prediction."""
        prediction = self.predict(state.features)

        if prediction.direction == PositionDirection.FLAT:
            return []

        if prediction.confidence < self.min_confidence:
            return []

        # Apply regime adjustment
        confidence = prediction.confidence
        if state.regime:
            # Reduce confidence when fighting regime
            if (
                state.regime.trend_regime == "BEAR"
                and prediction.direction == PositionDirection.LONG
            ):
                confidence *= 0.6
            elif (
                state.regime.trend_regime == "BULL"
                and prediction.direction == PositionDirection.SHORT
            ):
                confidence *= 0.6

        # Check earnings gate
        if state.days_to_earnings is not None and state.days_to_earnings <= 5:
            confidence *= 0.7

        if confidence < self.min_confidence:
            return []

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=prediction.direction,
                confidence=confidence,
                reason=f"ML prediction: p={prediction.probability:.3f}, features={prediction.features_used}",
                signal_strength=prediction.probability,
            )
        ]

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1H", "4H", "1D", "1W"],
            need_options_chain=True,
            need_earnings_calendar=True,
            lookback_bars=252,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        state = super().get_state()
        state["is_trained"] = self._is_trained
        state["feature_names"] = self._feature_names
        # Note: actual model serialization would use joblib/pickle
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore model state."""
        super().set_state(state)
        self._is_trained = state.get("is_trained", False)
        self._feature_names = state.get("feature_names", [])


def prepare_ml_dataset(
    df: pd.DataFrame,
    target_column: str = "label_long",
    feature_prefix: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare dataset for ML training.

    Args:
        df: DataFrame with features and labels
        target_column: Name of target column
        feature_prefix: Only include features with this prefix
        exclude_columns: Columns to exclude

    Returns:
        Tuple of (X features, y target)
    """
    exclude = exclude_columns or []
    exclude.extend(["open", "high", "low", "close", "volume", target_column])

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude]

    if feature_prefix:
        feature_cols = [c for c in feature_cols if c.startswith(feature_prefix)]

    # Remove string/object columns
    feature_cols = [
        c
        for c in feature_cols
        if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    X = df[feature_cols]
    y = df[target_column] if target_column in df.columns else pd.Series(index=df.index)

    return X, y


def walk_forward_train(
    df: pd.DataFrame,
    strategy: MLDirectionStrategy,
    train_size: int = 252,
    test_size: int = 21,
    retrain_freq: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward training and evaluation.

    Args:
        df: Full dataset with features and labels
        strategy: ML strategy to train
        train_size: Training window size
        test_size: Test window size
        retrain_freq: How often to retrain

    Returns:
        DataFrame with predictions and actuals
    """
    X, y = prepare_ml_dataset(df)

    results = []

    for i in range(train_size, len(df) - test_size, retrain_freq):
        # Train window
        train_X = X.iloc[i - train_size : i]
        train_y = y.iloc[i - train_size : i]

        # Train
        strategy.train(train_X, train_y)

        # Test window
        test_end = min(i + test_size, len(df))
        for j in range(i, test_end):
            features = X.iloc[j].to_dict()
            prediction = strategy.predict(features)

            results.append(
                {
                    "date": df.index[j],
                    "prediction": prediction.direction.value,
                    "probability": prediction.probability,
                    "confidence": prediction.confidence,
                    "actual": y.iloc[j] if j < len(y) else np.nan,
                }
            )

    return pd.DataFrame(results)


def train_with_proper_split(
    df: pd.DataFrame,
    strategy: MLDirectionStrategy,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    target_column: str = "label_long",
) -> Dict[str, Any]:
    """
    Train with proper train/val/test split to prevent data leakage.

    CRITICAL: Test data is NEVER used for training or hyperparameter tuning.

    Data split (temporal order preserved):
    - Train: 60% oldest data (for model fitting)
    - Validation: 20% (for hyperparameter tuning)
    - Test: 20% newest (HOLDOUT - final evaluation only)

    Args:
        df: DataFrame with features and labels
        strategy: ML strategy to train
        train_pct: Percentage of data for training (default 60%)
        val_pct: Percentage of data for validation (default 20%)
        target_column: Name of target column

    Returns:
        Dictionary with train/val/test metrics
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not available for training")
        return {"error": "sklearn not available"}

    X, y = prepare_ml_dataset(df, target_column=target_column)

    if X.empty or y.empty:
        return {"error": "Empty dataset"}

    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    # Strict temporal split - verify no overlap
    assert train_end > 0, "Train set is empty"
    assert val_end > train_end, "Validation set is empty"
    assert val_end < n, "Test set is empty"

    # Split data
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]  # HOLDOUT - never touch during tuning
    y_test = y.iloc[val_end:]

    logger.info(
        f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    logger.info(f"Train dates: {df.index[0]} to {df.index[train_end-1]}")
    logger.info(f"Val dates: {df.index[train_end]} to {df.index[val_end-1]}")
    logger.info(f"Test dates: {df.index[val_end]} to {df.index[-1]} (HOLDOUT)")

    # Hyperparameter tuning on VALIDATION set only
    best_params = tune_ml_hyperparameters(X_train, y_train, X_val, y_val)

    # Retrain on train+val with best params
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    train_metrics = strategy.train(
        X_trainval,
        y_trainval,
        n_estimators=best_params.get("n_estimators", 100),
        max_depth=best_params.get("max_depth", 5),
        learning_rate=best_params.get("learning_rate", 0.1),
    )

    # Final evaluation on TEST set (never seen during training/tuning)
    test_predictions = []
    test_actuals = []

    for i in range(len(X_test)):
        features = X_test.iloc[i].to_dict()
        pred = strategy.predict(features)
        test_predictions.append(1 if pred.direction == PositionDirection.LONG else 0)
        test_actuals.append(y_test.iloc[i])

    test_accuracy = np.mean(np.array(test_predictions) == np.array(test_actuals))

    return {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_params": best_params,
        "train_accuracy": train_metrics.get("train_accuracy", 0),
        "test_accuracy": test_accuracy,
        "test_predictions": test_predictions,
        "test_actuals": test_actuals,
    }


def tune_ml_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Optional[Dict] = None,
) -> Dict:
    """
    Tune hyperparameters using grid search on VALIDATION set only.

    CRITICAL: Test data is never used here - only train and validation.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        param_grid: Optional parameter grid

    Returns:
        Best parameters dictionary
    """
    if not SKLEARN_AVAILABLE:
        return {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

    param_grid = param_grid or {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle NaN in labels
    train_mask = ~y_train.isna()
    val_mask = ~y_val.isna()

    X_train_clean = X_train_scaled[train_mask]
    y_train_clean = y_train[train_mask]
    X_val_clean = X_val_scaled[val_mask]
    y_val_clean = y_val[val_mask]

    if len(y_train_clean) < 50 or len(y_val_clean) < 20:
        logger.warning("Insufficient data for tuning, using defaults")
        return {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

    best_score = -np.inf
    best_params = {}

    logger.info("Starting hyperparameter tuning on validation set...")

    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                try:
                    model = GradientBoostingClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=42,
                    )
                    model.fit(X_train_clean, y_train_clean)
                    score = model.score(X_val_clean, y_val_clean)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                        }
                except Exception as e:
                    logger.debug(f"Failed with params {n_est}/{depth}/{lr}: {e}")
                    continue

    logger.info(f"Best params: {best_params}, validation accuracy: {best_score:.3f}")

    return (
        best_params
        if best_params
        else {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    )
