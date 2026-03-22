"""
TFT-based return predictor — adapts the SimpleTFTModel for regression.

Reuses the LSTM + attention + GRN architecture from
``quantcore.hierarchy.regime.tft_regime`` but replaces the classification head
with a multi-horizon regression head that predicts 1d, 5d, and 20d forward
returns simultaneously.

Usage:
    predictor = TFTReturnPredictor(n_features=12, sequence_length=60)
    predictor.train(X_sequences, y_returns)   # X: (N, 60, 12), y: (N, 3)
    preds = predictor.predict(X_new)          # -> (N, 3) for 1d/5d/20d
    predictor.save("models/tft_returns.pt")
    predictor.load("models/tft_returns.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


HORIZONS = [1, 5, 20]  # forward return horizons in trading days


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class TFTReturnModel(nn.Module):
        """
        Simplified TFT for multi-horizon return regression.

        Architecture mirrors SimpleTFTModel (tft_regime.py):
          feature_embed -> LSTM -> MultiheadAttention -> GRN -> regression head
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_horizons: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size

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
                hidden_size, n_heads, dropout=dropout, batch_first=True
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

            # Regression head: one output per horizon
            self.regression_head = nn.Linear(hidden_size, n_horizons)

            # Learnable feature importance
            self.feature_weights = nn.Parameter(torch.ones(n_features))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor [batch, seq_len, n_features]

            Returns:
                Predicted returns [batch, n_horizons]
            """
            weighted_x = x * self.feature_weights.unsqueeze(0).unsqueeze(0)
            embedded = self.feature_embed(weighted_x)

            lstm_out, _ = self.lstm(embedded)

            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

            grn_out = self.grn(attn_out)
            gate_out = self.gate(attn_out)
            gated = self.layer_norm(attn_out + gate_out * grn_out)

            final = gated[:, -1, :]
            return self.regression_head(final)


# ---------------------------------------------------------------------------
# Dataclass for prediction results
# ---------------------------------------------------------------------------


@dataclass
class TFTReturnPrediction:
    """Return prediction from TFT model."""

    returns_1d: float
    returns_5d: float
    returns_20d: float
    feature_importance: dict[str, float]


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------


class TFTReturnPredictor:
    """
    Train and predict multi-horizon returns using TFT architecture.

    Args:
        n_features: Number of input features per bar.
        sequence_length: Number of historical bars per sample.
        hidden_size: LSTM / attention hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of LSTM layers.
        learning_rate: Adam learning rate.
        epochs: Training epochs.
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int = 60,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
    ):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model: TFTReturnModel | None = None  # type: ignore[assignment]
        self.is_fitted = False
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Train the TFT return predictor.

        Args:
            X: Input sequences, shape (N, sequence_length, n_features).
            y: Target returns, shape (N, 3) for 1d/5d/20d horizons.
            feature_names: Optional feature names for interpretability.

        Returns:
            Training metrics: {train_mse, train_mae}.
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for TFT training")
            return {"train_mse": float("nan"), "train_mae": float("nan")}

        if len(X) < 20:
            logger.warning(f"Insufficient samples for TFT training: {len(X)}")
            return {"train_mse": float("nan"), "train_mae": float("nan")}

        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[2])]

        # Standardize features
        flat = X.reshape(-1, X.shape[2])
        self.scaler_mean = np.mean(flat, axis=0)
        self.scaler_std = np.std(flat, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)

        self.model = TFTReturnModel(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            n_horizons=y.shape[1] if y.ndim > 1 else 1,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"TFT Return Epoch {epoch + 1}/{self.epochs}, "
                    f"MSE: {loss.item():.6f}"
                )

        self.model.eval()
        self.is_fitted = True

        with torch.no_grad():
            final_preds = self.model(X_tensor).numpy()
        mse = float(np.mean((final_preds - y) ** 2))
        mae = float(np.mean(np.abs(final_preds - y)))

        logger.info(f"TFT return predictor trained: MSE={mse:.6f}, MAE={mae:.6f}")
        return {"train_mse": round(mse, 6), "train_mae": round(mae, 6)}

    def predict(self, X: np.ndarray) -> TFTReturnPrediction | np.ndarray:
        """
        Predict returns for input sequences.

        Args:
            X: Input sequences, shape (N, sequence_length, n_features) or
               (sequence_length, n_features) for a single sample.

        Returns:
            TFTReturnPrediction for single sample, or ndarray (N, 3) for batch.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call train() first.")

        single = X.ndim == 2
        if single:
            X = X[np.newaxis, ...]

        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()

        if single:
            importance = self._get_feature_importance()
            return TFTReturnPrediction(
                returns_1d=float(preds[0, 0]),
                returns_5d=float(preds[0, 1]) if preds.shape[1] > 1 else 0.0,
                returns_20d=float(preds[0, 2]) if preds.shape[1] > 2 else 0.0,
                feature_importance=importance,
            )
        return preds

    def _get_feature_importance(self) -> dict[str, float]:
        """Extract learned feature importance from model weights."""
        if self.model is None:
            return {}
        weights = self.model.feature_weights.detach().cpu().numpy()
        weights = np.abs(weights)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return {
            name: round(float(w), 4) for name, w in zip(self.feature_names, weights)
        }

    def save(self, path: str) -> None:
        """Save model state to disk."""
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("No model to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "feature_names": self.feature_names,
        }
        torch.save(state, str(save_path))
        logger.info(f"TFT return predictor saved to {save_path}")

    def load(self, path: str) -> None:
        """Load model state from disk."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to load model")

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        self.n_features = state["n_features"]
        self.sequence_length = state["sequence_length"]
        self.hidden_size = state["hidden_size"]
        self.n_heads = state["n_heads"]
        self.n_layers = state["n_layers"]
        self.scaler_mean = state["scaler_mean"]
        self.scaler_std = state["scaler_std"]
        self.feature_names = state["feature_names"]

        self.model = TFTReturnModel(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        )
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.is_fitted = True
        logger.info(f"TFT return predictor loaded from {path}")
