"""
Hierarchical ensemble for multi-timeframe prediction.

Combines predictions from different timeframes with alignment weighting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.config.timeframes import TIMEFRAME_HIERARCHY, Timeframe
from quantcore.models.predictor import Predictor
from quantcore.models.trainer import TrainingResult


@dataclass
class EnsembleConfig:
    """Configuration for hierarchical ensemble."""
    # Timeframe weights for ensemble
    timeframe_weights: dict[Timeframe, float] = None

    # Alignment score weighting
    use_alignment_weighting: bool = True

    # Minimum individual TF probability
    min_tf_probability: float = 0.4

    # Aggregation method
    aggregation: str = "weighted_average"  # or "minimum", "geometric_mean"

    def __post_init__(self):
        if self.timeframe_weights is None:
            # Default: execution TF (1H) gets highest weight
            self.timeframe_weights = {
                Timeframe.W1: 0.15,
                Timeframe.D1: 0.25,
                Timeframe.H4: 0.25,
                Timeframe.H1: 0.35,
            }


class HierarchicalEnsemble:
    """
    Ensemble model combining predictions from multiple timeframes.

    The ensemble:
    1. Gets predictions from each timeframe's model
    2. Weights by timeframe importance
    3. Adjusts by cross-TF alignment score
    4. Produces final probability
    """

    def __init__(self, config: EnsembleConfig | None = None):
        """
        Initialize ensemble.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.models: dict[Timeframe, dict[str, TrainingResult]] = {}
        self.predictors: dict[Timeframe, dict[str, Predictor]] = {}

    def add_model(
        self,
        timeframe: Timeframe,
        direction: str,
        training_result: TrainingResult,
    ) -> None:
        """
        Add a model for a timeframe/direction.

        Args:
            timeframe: Timeframe
            direction: "long" or "short"
            training_result: Trained model result
        """
        if timeframe not in self.models:
            self.models[timeframe] = {}
            self.predictors[timeframe] = {}

        self.models[timeframe][direction] = training_result
        self.predictors[timeframe][direction] = Predictor(training_result)

        logger.info(f"Added {direction} model for {timeframe.value}")

    def predict_with_alignment(
        self,
        features: dict[Timeframe, pd.DataFrame],
        direction: str,
        alignment_score: float = 1.0,
    ) -> pd.Series:
        """
        Predict with alignment-weighted ensemble.

        Args:
            features: Feature DataFrames per timeframe
            direction: Trade direction ("long" or "short")
            alignment_score: Cross-TF alignment score (0-1)

        Returns:
            Series of ensemble probabilities
        """
        # Get predictions from each timeframe
        tf_predictions = {}

        for tf in TIMEFRAME_HIERARCHY:
            if tf in self.predictors and direction in self.predictors[tf]:
                predictor = self.predictors[tf][direction]
                if tf in features and not features[tf].empty:
                    # Align indices
                    aligned_features = self._align_to_base_tf(
                        features[tf],
                        features.get(Timeframe.H1, features[tf]),
                    )
                    tf_predictions[tf] = predictor.predict_proba(aligned_features)

        if not tf_predictions:
            logger.warning("No predictions available from any timeframe")
            return pd.Series([0.5] * len(features.get(Timeframe.H1, pd.DataFrame())))

        # Aggregate predictions
        if self.config.aggregation == "weighted_average":
            ensemble_prob = self._weighted_average(tf_predictions, alignment_score)
        elif self.config.aggregation == "minimum":
            ensemble_prob = self._minimum_prob(tf_predictions)
        elif self.config.aggregation == "geometric_mean":
            ensemble_prob = self._geometric_mean(tf_predictions, alignment_score)
        else:
            ensemble_prob = self._weighted_average(tf_predictions, alignment_score)

        return pd.Series(ensemble_prob)

    def predict_single(
        self,
        features: dict[Timeframe, pd.Series],
        direction: str,
        alignment_score: float = 1.0,
    ) -> tuple[float, dict[Timeframe, float]]:
        """
        Predict for a single bar.

        Args:
            features: Feature Series per timeframe
            direction: Trade direction
            alignment_score: Alignment score

        Returns:
            Tuple of (ensemble_probability, per_tf_probabilities)
        """
        tf_probs = {}

        for tf, feat_series in features.items():
            if tf in self.predictors and direction in self.predictors[tf]:
                predictor = self.predictors[tf][direction]
                X = pd.DataFrame([feat_series.to_dict()])
                tf_probs[tf] = float(predictor.predict_proba(X)[0])

        if not tf_probs:
            return 0.5, {}

        # Calculate ensemble probability
        ensemble_prob = self._single_bar_ensemble(tf_probs, alignment_score)

        return ensemble_prob, tf_probs

    def _weighted_average(
        self,
        tf_predictions: dict[Timeframe, np.ndarray],
        alignment_score: float,
    ) -> np.ndarray:
        """Weighted average of timeframe predictions."""
        # Get base length
        base_len = max(len(p) for p in tf_predictions.values())

        weighted_sum = np.zeros(base_len)
        total_weight = 0.0

        for tf, preds in tf_predictions.items():
            weight = self.config.timeframe_weights.get(tf, 0.25)

            # Pad predictions if needed
            if len(preds) < base_len:
                padded = np.full(base_len, 0.5)
                padded[:len(preds)] = preds
                preds = padded

            weighted_sum += weight * preds
            total_weight += weight

        if total_weight > 0:
            avg_prob = weighted_sum / total_weight
        else:
            avg_prob = np.full(base_len, 0.5)

        # Apply alignment weighting
        if self.config.use_alignment_weighting:
            # High alignment -> trust ensemble more
            # Low alignment -> move toward neutral
            avg_prob = alignment_score * avg_prob + (1 - alignment_score) * 0.5

        return avg_prob

    def _minimum_prob(
        self,
        tf_predictions: dict[Timeframe, np.ndarray],
    ) -> np.ndarray:
        """Take minimum probability across timeframes (conservative)."""
        base_len = max(len(p) for p in tf_predictions.values())

        # Stack all predictions
        all_preds = []
        for preds in tf_predictions.values():
            if len(preds) < base_len:
                padded = np.full(base_len, 0.5)
                padded[:len(preds)] = preds
                preds = padded
            all_preds.append(preds)

        return np.min(np.stack(all_preds), axis=0)

    def _geometric_mean(
        self,
        tf_predictions: dict[Timeframe, np.ndarray],
        alignment_score: float,
    ) -> np.ndarray:
        """Geometric mean of predictions."""
        base_len = max(len(p) for p in tf_predictions.values())

        log_sum = np.zeros(base_len)
        count = 0

        for _tf, preds in tf_predictions.items():
            if len(preds) < base_len:
                padded = np.full(base_len, 0.5)
                padded[:len(preds)] = preds
                preds = padded

            # Clip to avoid log(0)
            preds = np.clip(preds, 0.001, 0.999)
            log_sum += np.log(preds)
            count += 1

        if count > 0:
            geom_mean = np.exp(log_sum / count)
        else:
            geom_mean = np.full(base_len, 0.5)

        # Apply alignment weighting
        if self.config.use_alignment_weighting:
            geom_mean = alignment_score * geom_mean + (1 - alignment_score) * 0.5

        return geom_mean

    def _single_bar_ensemble(
        self,
        tf_probs: dict[Timeframe, float],
        alignment_score: float,
    ) -> float:
        """Ensemble for single bar."""
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, prob in tf_probs.items():
            weight = self.config.timeframe_weights.get(tf, 0.25)
            weighted_sum += weight * prob
            total_weight += weight

        if total_weight > 0:
            avg_prob = weighted_sum / total_weight
        else:
            avg_prob = 0.5

        # Apply alignment weighting
        if self.config.use_alignment_weighting:
            avg_prob = alignment_score * avg_prob + (1 - alignment_score) * 0.5

        return avg_prob

    def _align_to_base_tf(
        self,
        df_higher: pd.DataFrame,
        df_base: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align higher TF predictions to base TF index."""
        return df_higher.reindex(df_base.index, method="ffill").fillna(0.5)

    def get_model_summary(self) -> dict[str, dict]:
        """Get summary of all models in ensemble."""
        summary = {}

        for tf in TIMEFRAME_HIERARCHY:
            if tf in self.models:
                summary[tf.value] = {}
                for direction, result in self.models[tf].items():
                    summary[tf.value][direction] = {
                        "auc": result.metrics.get("auc", 0),
                        "cv_auc_mean": np.mean(result.cv_scores),
                        "n_features": len(result.feature_names),
                        "top_features": list(result.feature_importance.keys())[:5],
                    }

        return summary

