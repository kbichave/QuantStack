"""
Temporal Fusion Transformer for regime prediction.

ML-based regime classification using attention mechanisms.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class TFTRegimeState(Enum):
    """TFT regime states."""

    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3


@dataclass
class TFTRegimeResult:
    """Result from TFT regime prediction."""

    predicted_regime: TFTRegimeState
    regime_probabilities: Dict[TFTRegimeState, float]
    confidence: float
    attention_weights: Optional[np.ndarray]
    feature_importance: Dict[str, float]


# Simple TFT-inspired model (full TFT is complex, this is a simplified version)
if TORCH_AVAILABLE:

    class SimpleTFTModel(nn.Module):
        """
        Simplified Temporal Fusion Transformer for regime prediction.

        Uses:
        - LSTM for temporal encoding
        - Multi-head attention for capturing dependencies
        - Gated residual networks for feature processing
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_regimes: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.n_features = n_features
            self.hidden_size = hidden_size
            self.n_regimes = n_regimes

            # Feature embedding
            self.feature_embed = nn.Linear(n_features, hidden_size)

            # LSTM encoder
            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )

            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                hidden_size,
                n_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Gated residual network
            self.grn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

            # Output layer
            self.output = nn.Linear(hidden_size, n_regimes)

            # Feature importance (learnable)
            self.feature_weights = nn.Parameter(torch.ones(n_features))

        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Forward pass.

            Args:
                x: Input tensor [batch, seq_len, n_features]
                return_attention: Whether to return attention weights

            Returns:
                Regime logits and optionally attention weights
            """
            # Feature embedding with importance weighting
            weighted_x = x * self.feature_weights.unsqueeze(0).unsqueeze(0)
            embedded = self.feature_embed(weighted_x)

            # LSTM encoding
            lstm_out, _ = self.lstm(embedded)

            # Self-attention
            attn_out, attn_weights = self.attention(
                lstm_out,
                lstm_out,
                lstm_out,
                need_weights=return_attention,
            )

            # Gated residual connection
            grn_out = self.grn(attn_out)
            gate_out = self.gate(attn_out)
            gated = self.layer_norm(attn_out + gate_out * grn_out)

            # Take last timestep
            final = gated[:, -1, :]

            # Output
            logits = self.output(final)

            if return_attention:
                return logits, attn_weights
            return logits, None

        def get_feature_importance(self) -> Dict[str, float]:
            """Get learned feature importance."""
            weights = self.feature_weights.detach().cpu().numpy()
            weights = np.abs(weights) / np.sum(np.abs(weights))
            return {f"feature_{i}": float(w) for i, w in enumerate(weights)}


class TFTRegimeModel:
    """
    TFT-based regime prediction model.

    Uses a simplified Temporal Fusion Transformer for regime classification.

    Features:
    - Multi-horizon learning
    - Attention-based feature importance
    - Interpretable regime predictions
    """

    FEATURE_NAMES = [
        "returns",
        "volatility",
        "volume_change",
        "momentum_5",
        "momentum_20",
        "rsi_proxy",
        "bb_position",
        "trend_strength",
    ]

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
    ):
        """
        Initialize TFT regime model.

        Args:
            lookback: Sequence length for model
            hidden_size: Hidden layer size
            n_heads: Number of attention heads
            n_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            epochs: Training epochs
        """
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model: Optional["SimpleTFTModel"] = None
        self.is_fitted = False
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(
        self, df: pd.DataFrame, labels: Optional[pd.Series] = None
    ) -> "TFTRegimeModel":
        """
        Fit TFT model.

        Args:
            df: DataFrame with OHLCV data
            labels: Optional regime labels (if None, uses self-supervised)

        Returns:
            Self for chaining
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback")
            return self

        # Prepare features
        features = self._prepare_features(df)

        if features is None or len(features) < self.lookback + 10:
            logger.warning("Insufficient data for TFT training")
            return self

        # Generate labels if not provided (self-supervised)
        if labels is None:
            labels = self._generate_labels(df, features)

        # Create sequences
        X, y = self._create_sequences(features, labels)

        if len(X) < 10:
            logger.warning("Insufficient sequences for training")
            return self

        try:
            # Initialize model
            self.model = SimpleTFTModel(
                n_features=len(self.FEATURE_NAMES),
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
            )

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)

            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                logits, _ = self.model(X_tensor)
                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    logger.debug(
                        f"TFT Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}"
                    )

            self.model.eval()
            self.is_fitted = True
            logger.info("TFT model trained successfully")

        except Exception as e:
            logger.error(f"TFT training failed: {e}")
            self.is_fitted = False

        return self

    def predict(self, df: pd.DataFrame) -> TFTRegimeResult:
        """
        Predict regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            TFTRegimeResult with prediction
        """
        if not self.is_fitted or self.model is None:
            return self._fallback_predict(df)

        features = self._prepare_features(df)

        if features is None or len(features) < self.lookback:
            return self._fallback_predict(df)

        try:
            # Get last sequence
            seq = features[-self.lookback :]
            X = torch.FloatTensor(seq).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                logits, attn_weights = self.model(X, return_attention=True)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            # Get predicted regime
            pred_idx = int(np.argmax(probs))
            predicted_regime = TFTRegimeState(pred_idx)

            # Regime probabilities
            regime_probs = {TFTRegimeState(i): float(probs[i]) for i in range(4)}

            # Feature importance
            feature_importance = self.model.get_feature_importance()
            named_importance = {
                name: feature_importance.get(f"feature_{i}", 0)
                for i, name in enumerate(self.FEATURE_NAMES)
            }

            return TFTRegimeResult(
                predicted_regime=predicted_regime,
                regime_probabilities=regime_probs,
                confidence=float(np.max(probs)),
                attention_weights=(
                    attn_weights.numpy() if attn_weights is not None else None
                ),
                feature_importance=named_importance,
            )

        except Exception as e:
            logger.error(f"TFT prediction failed: {e}")
            return self._fallback_predict(df)

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for TFT."""
        if len(df) < 30:
            return None

        features = pd.DataFrame(index=df.index)

        # Returns
        features["returns"] = df["close"].pct_change()

        # Volatility
        features["volatility"] = df["close"].pct_change().rolling(20).std()

        # Volume change
        if "volume" in df.columns:
            features["volume_change"] = df["volume"].pct_change()
        else:
            features["volume_change"] = 0

        # Momentum
        features["momentum_5"] = df["close"].pct_change(5)
        features["momentum_20"] = df["close"].pct_change(20)

        # RSI proxy (simplified)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features["rsi_proxy"] = gain / (gain + loss + 1e-10) - 0.5

        # Bollinger Band position
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        features["bb_position"] = (df["close"] - sma) / (2 * std + 1e-10)

        # Trend strength (simplified ADX proxy)
        features["trend_strength"] = features["momentum_20"].abs()

        # Drop NaN
        features = features.dropna()

        if len(features) < 10:
            return None

        # Standardize
        values = features.values
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(values, axis=0)
            self.scaler_std = np.std(values, axis=0)
            self.scaler_std[self.scaler_std == 0] = 1

        values = (values - self.scaler_mean) / self.scaler_std
        values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)

        return values

    def _generate_labels(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
    ) -> np.ndarray:
        """Generate regime labels (self-supervised)."""
        # Use simple rules to label regimes
        returns = df["close"].pct_change().dropna().values
        volatility = df["close"].pct_change().rolling(20).std().dropna().values

        # Align lengths
        min_len = min(len(returns), len(volatility), len(features))
        returns = returns[-min_len:]
        volatility = volatility[-min_len:]

        labels = np.zeros(min_len, dtype=np.int64)

        # Get thresholds
        ret_med = np.median(returns)
        vol_med = np.median(volatility)

        for i in range(min_len):
            # Look at recent window
            start = max(0, i - 10)
            avg_ret = np.mean(returns[start : i + 1])
            avg_vol = np.mean(volatility[start : i + 1])

            if avg_ret > ret_med and avg_vol < vol_med:
                labels[i] = TFTRegimeState.TRENDING_UP.value
            elif avg_ret < -ret_med and avg_vol < vol_med:
                labels[i] = TFTRegimeState.TRENDING_DOWN.value
            elif avg_vol > vol_med * 1.5:
                labels[i] = TFTRegimeState.VOLATILE.value
            else:
                labels[i] = TFTRegimeState.RANGING.value

        return labels

    def _create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        min_len = min(len(features), len(labels))
        features = features[-min_len:]
        labels = labels[-min_len:]

        X, y = [], []

        for i in range(self.lookback, min_len):
            X.append(features[i - self.lookback : i])
            y.append(labels[i])

        return np.array(X), np.array(y)

    def _fallback_predict(self, df: pd.DataFrame) -> TFTRegimeResult:
        """Fallback when TFT is not available."""
        # Simple rule-based prediction
        returns = df["close"].pct_change()
        vol = returns.rolling(20).std()

        avg_ret = returns.tail(20).mean()
        avg_vol = vol.iloc[-1]
        hist_vol = vol.mean()

        if avg_ret > 0.001 and avg_vol < hist_vol:
            regime = TFTRegimeState.TRENDING_UP
        elif avg_ret < -0.001 and avg_vol < hist_vol:
            regime = TFTRegimeState.TRENDING_DOWN
        elif avg_vol > hist_vol * 1.5:
            regime = TFTRegimeState.VOLATILE
        else:
            regime = TFTRegimeState.RANGING

        probs = {s: 0.2 for s in TFTRegimeState}
        probs[regime] = 0.6

        return TFTRegimeResult(
            predicted_regime=regime,
            regime_probabilities=probs,
            confidence=0.6,
            attention_weights=None,
            feature_importance={
                name: 1.0 / len(self.FEATURE_NAMES) for name in self.FEATURE_NAMES
            },
        )
