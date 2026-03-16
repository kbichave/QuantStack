"""
SHAP-based model explainer for interpretability.

Provides feature importance analysis and prediction explanations.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, explanations will be limited")

from quantcore.models.trainer import TrainingResult


class SHAPExplainer:
    """
    SHAP-based model explainer.

    Features:
    - Global feature importance
    - Local prediction explanations
    - Feature interaction analysis
    """

    def __init__(self, training_result: TrainingResult):
        """
        Initialize explainer.

        Args:
            training_result: Trained model result
        """
        self.model = training_result.model
        self.feature_names = training_result.feature_names
        self.scaler = training_result.scaler
        self._explainer: Any | None = None
        self._shap_values: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        sample_size: int = 1000,
    ) -> None:
        """
        Fit the SHAP explainer.

        Args:
            X: Feature data for background distribution
            sample_size: Number of samples for background
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return

        # Sample if too large
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_sample)
        else:
            X_scaled = X_sample.values

        # Create explainer based on model type
        try:
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("Created TreeExplainer")
        except Exception as e:
            logger.warning(f"TreeExplainer failed, using KernelExplainer: {e}")
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_scaled[:100],
            )

    def explain_global(
        self,
        X: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Get global feature importance using SHAP.

        Args:
            X: Feature data

        Returns:
            Dictionary of feature importance
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            # Fall back to model's feature importance
            if hasattr(self.model, "feature_importances_"):
                return dict(
                    zip(
                        self.feature_names,
                        self.model.feature_importances_,
                        strict=False,
                    )
                )
            return {}

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Calculate SHAP values
        shap_values = self._explainer.shap_values(X_scaled)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)

        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return dict(
            sorted(
                zip(self.feature_names, importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def explain_prediction(
        self,
        X: pd.DataFrame,
        idx: int = 0,
    ) -> dict[str, float]:
        """
        Explain a single prediction.

        Args:
            X: Feature data
            idx: Index of prediction to explain

        Returns:
            Dictionary of feature contributions
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return {}

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Get SHAP values for this prediction
        shap_values = self._explainer.shap_values(X_scaled[idx : idx + 1])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return dict(zip(self.feature_names, shap_values[0], strict=False))

    def get_top_contributors(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        n_features: int = 10,
    ) -> list[dict]:
        """
        Get top contributing features for a prediction.

        Args:
            X: Feature data
            idx: Prediction index
            n_features: Number of top features

        Returns:
            List of feature contribution dictionaries
        """
        contributions = self.explain_prediction(X, idx)

        # Sort by absolute contribution
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:n_features]

        result = []
        for feat, contrib in sorted_contrib:
            result.append(
                {
                    "feature": feat,
                    "contribution": contrib,
                    "direction": "positive" if contrib > 0 else "negative",
                    "value": float(X.iloc[idx].get(feat, 0)),
                }
            )

        return result

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
    ) -> None:
        """
        Create SHAP summary plot.

        Args:
            X: Feature data
            max_display: Maximum features to display
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            logger.warning("SHAP plotting not available")
            return

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        shap_values = self._explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

    def get_feature_interactions(
        self,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
    ) -> np.ndarray | None:
        """
        Get interaction values between two features.

        Args:
            X: Feature data
            feature1: First feature name
            feature2: Second feature name

        Returns:
            Interaction values array
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return None

        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            logger.warning(f"Feature not found: {feature1} or {feature2}")
            return None

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        try:
            interaction_values = self._explainer.shap_interaction_values(X_scaled)

            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1]

            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)

            return interaction_values[:, idx1, idx2]
        except Exception as e:
            logger.warning(f"Could not compute interactions: {e}")
            return None
