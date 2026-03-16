"""
Inference engine for trained models.

Provides prediction interface with confidence calibration.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.models.trainer import TrainingResult
from quantcore.config.timeframes import Timeframe


class Predictor:
    """
    Prediction interface for trained models.
    
    Features:
    - Probability prediction
    - Confidence calibration
    - Batch prediction support
    - Feature validation
    """
    
    def __init__(self, training_result: TrainingResult):
        """
        Initialize predictor.
        
        Args:
            training_result: Result from model training
        """
        self.model = training_result.model
        self.feature_names = training_result.feature_names
        self.scaler = training_result.scaler
        self.feature_importance = training_result.feature_importance
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        validate_features: bool = True,
    ) -> np.ndarray:
        """
        Predict probability of positive class.
        
        Args:
            X: Feature matrix
            validate_features: Whether to validate feature alignment
            
        Returns:
            Array of probabilities
        """
        if validate_features:
            X = self._align_features(X)
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        return probas
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for positive class
            
        Returns:
            Array of predictions (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def predict_single(
        self,
        features: Dict[str, float],
    ) -> Tuple[int, float]:
        """
        Predict for a single sample.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, probability)
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        proba = self.predict_proba(X)[0]
        pred = int(proba >= 0.5)
        
        return pred, proba
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure features are in correct order."""
        # Check for missing features
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            logger.warning(f"Missing {len(missing)} features, filling with 0")
            for col in missing:
                X[col] = 0
        
        # Reorder to expected order
        return X[self.feature_names]
    
    def get_top_features(
        self,
        X: pd.DataFrame,
        n_features: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get top contributing features for predictions.
        
        Args:
            X: Feature matrix
            n_features: Number of top features to return
            
        Returns:
            Dictionary with feature contributions per sample
        """
        # Get top features by importance
        top_features = list(self.feature_importance.keys())[:n_features]
        
        result = {}
        for idx, row in X.iterrows():
            result[idx] = {
                feat: float(row.get(feat, 0))
                for feat in top_features
            }
        
        return result


class MultiTimeframePredictor:
    """
    Predictor for multi-timeframe models.
    
    Manages predictions across different timeframes.
    """
    
    def __init__(self):
        """Initialize multi-TF predictor."""
        self.predictors: Dict[Timeframe, Dict[str, Predictor]] = {}
    
    def add_predictor(
        self,
        timeframe: Timeframe,
        direction: str,  # "long" or "short"
        training_result: TrainingResult,
    ) -> None:
        """
        Add a predictor for a timeframe/direction.
        
        Args:
            timeframe: Timeframe for this predictor
            direction: Trade direction ("long" or "short")
            training_result: Training result with model
        """
        if timeframe not in self.predictors:
            self.predictors[timeframe] = {}
        
        self.predictors[timeframe][direction] = Predictor(training_result)
        logger.info(f"Added predictor for {timeframe.value} {direction}")
    
    def predict(
        self,
        data: Dict[Timeframe, pd.DataFrame],
        direction: str,
    ) -> Dict[Timeframe, np.ndarray]:
        """
        Predict across all timeframes.
        
        Args:
            data: Feature data per timeframe
            direction: Trade direction
            
        Returns:
            Probabilities per timeframe
        """
        results = {}
        
        for tf, features in data.items():
            if tf in self.predictors and direction in self.predictors[tf]:
                predictor = self.predictors[tf][direction]
                results[tf] = predictor.predict_proba(features)
            else:
                results[tf] = np.full(len(features), 0.5)  # Neutral
        
        return results
    
    def predict_single_bar(
        self,
        features: Dict[Timeframe, pd.Series],
        direction: str,
    ) -> Dict[Timeframe, float]:
        """
        Predict for a single bar across timeframes.
        
        Args:
            features: Feature series per timeframe
            direction: Trade direction
            
        Returns:
            Probability per timeframe
        """
        results = {}
        
        for tf, feat_series in features.items():
            if tf in self.predictors and direction in self.predictors[tf]:
                predictor = self.predictors[tf][direction]
                X = pd.DataFrame([feat_series.to_dict()])
                results[tf] = float(predictor.predict_proba(X)[0])
            else:
                results[tf] = 0.5
        
        return results

