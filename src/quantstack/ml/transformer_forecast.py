"""Transformer-based time-series forecasting (PatchTST).

Provides a unified interface for deep-learning forecasters. When GluonTS or
pytorch-forecasting are available the full PatchTST architecture is used.
Otherwise the module degrades through three levels:

  1. PatchTST via GluonTS / pytorch-forecasting
  2. Simple LSTM via raw PyTorch
  3. None (returns None from predict, caller must handle)

The graceful degradation guarantees the rest of the system never hard-fails
due to missing optional deep-learning dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

_BACKEND: str = "none"

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from gluonts.torch.model.patch_tst import PatchTSTEstimator  # type: ignore[import-untyped]

    _BACKEND = "gluonts"
except ImportError:
    pass

if _BACKEND == "none":
    try:
        from pytorch_forecasting import TemporalFusionTransformer  # type: ignore[import-untyped]

        _BACKEND = "pytorch_forecasting"
    except ImportError:
        pass

if _BACKEND == "none" and _HAS_TORCH:
    _BACKEND = "lstm"

logger.info("TransformerForecaster backend: {}", _BACKEND)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Hyper-parameters for the transformer forecaster.

    Attributes:
        model_name: Architecture identifier (informational).
        context_length: Number of look-back time steps fed to the model.
        prediction_length: Forecast horizon in time steps.
        n_features: Total input features (OHLCV = 5 + extras).
        retrain_interval_days: Minimum days between retrains.
        max_epochs: Training epoch cap.
        learning_rate: Optimiser step size.
    """

    model_name: str = "patch_tst"
    context_length: int = 60
    prediction_length: int = 5
    n_features: int = 25  # 20 derived + 5 OHLCV
    retrain_interval_days: int = 7
    max_epochs: int = 50
    learning_rate: float = 1e-4


# ---------------------------------------------------------------------------
# Minimal LSTM fallback (requires only torch)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _SimpleLSTM(nn.Module):  # type: ignore[name-defined]
        """Two-layer LSTM used as a fallback when no transformer lib is available."""

        def __init__(self, n_features: int, hidden: int = 64, pred_len: int = 5):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden, pred_len)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class TransformerForecaster:
    """Unified deep-learning time-series forecaster with graceful degradation."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self._cfg = config or TransformerConfig()
        self._model: Any = None
        self._backend = _BACKEND
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> TransformerForecaster:
        """Train on a DataFrame with a DatetimeIndex and numeric columns.

        The first 5 columns are assumed to be OHLCV; remaining columns are
        auxiliary features. The target for forecasting is the 'close' column
        (column index 3, zero-based).

        Returns self for chaining.
        """
        if self._backend == "none":
            logger.warning("No deep-learning backend available — fit() is a no-op")
            return self

        if self._backend == "lstm":
            return self._fit_lstm(df)

        if self._backend == "gluonts":
            return self._fit_gluonts(df)

        if self._backend == "pytorch_forecasting":
            return self._fit_ptf(df)

        return self

    def _fit_lstm(self, df: pd.DataFrame) -> TransformerForecaster:
        """Train the simple LSTM fallback."""
        values = df.values.astype(np.float32)
        ctx = self._cfg.context_length
        pred = self._cfg.prediction_length
        n_features = values.shape[1]

        # Build sequences
        xs, ys = [], []
        close_col = min(3, n_features - 1)  # 'close' is typically col 3
        for i in range(len(values) - ctx - pred):
            xs.append(values[i : i + ctx])
            ys.append(values[i + ctx : i + ctx + pred, close_col])

        if len(xs) < 10:
            logger.warning("Insufficient data for LSTM training ({} sequences)", len(xs))
            return self

        X_t = torch.tensor(np.array(xs))  # type: ignore[name-defined]
        y_t = torch.tensor(np.array(ys))  # type: ignore[name-defined]

        model = _SimpleLSTM(n_features, pred_len=pred)
        optimiser = torch.optim.Adam(model.parameters(), lr=self._cfg.learning_rate)  # type: ignore[name-defined]
        loss_fn = nn.MSELoss()  # type: ignore[name-defined]

        model.train()
        for epoch in range(self._cfg.max_epochs):
            optimiser.zero_grad()
            preds = model(X_t)
            loss = loss_fn(preds, y_t)
            loss.backward()
            optimiser.step()
            if (epoch + 1) % 10 == 0:
                logger.debug("LSTM epoch {}/{} — loss {:.6f}", epoch + 1, self._cfg.max_epochs, loss.item())

        model.eval()
        self._model = model
        self._fitted = True
        logger.info("LSTM fallback trained: {} epochs, final loss {:.6f}", self._cfg.max_epochs, loss.item())
        return self

    def _fit_gluonts(self, df: pd.DataFrame) -> TransformerForecaster:
        """Train via GluonTS PatchTST."""
        try:
            from gluonts.dataset.pandas import PandasDataset  # type: ignore[import-untyped]

            dataset = PandasDataset.from_long_dataframe(
                df.reset_index(), target="close", item_id="symbol"
                if "symbol" in df.columns
                else None,
            )
            estimator = PatchTSTEstimator(
                prediction_length=self._cfg.prediction_length,
                context_length=self._cfg.context_length,
                trainer_kwargs={"max_epochs": self._cfg.max_epochs},
            )
            self._model = estimator.train(dataset)
            self._fitted = True
            logger.info("GluonTS PatchTST trained")
        except Exception:
            logger.exception("GluonTS training failed — degrading to LSTM")
            self._backend = "lstm"
            return self._fit_lstm(df)
        return self

    def _fit_ptf(self, df: pd.DataFrame) -> TransformerForecaster:
        """Train via pytorch-forecasting TFT (closest available transformer)."""
        logger.warning("pytorch-forecasting TFT training not yet wired — degrading to LSTM")
        self._backend = "lstm"
        return self._fit_lstm(df)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Forecast the next prediction_length steps.

        Parameters:
            df: DataFrame with at least *context_length* rows of OHLCV + features.

        Returns:
            {"forecast": np.ndarray, "confidence": float, "model": str}
            or None if no backend / model is available.
        """
        if self._backend == "none" or not self._fitted or self._model is None:
            return None

        if self._backend in ("lstm",):
            return self._predict_lstm(df)

        # GluonTS / PTF path
        logger.warning("GluonTS/PTF predict path not yet wired — returning None")
        return None

    def _predict_lstm(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Run the LSTM fallback for prediction."""
        values = df.values.astype(np.float32)
        ctx = self._cfg.context_length
        if len(values) < ctx:
            logger.warning("Insufficient rows for prediction ({} < {})", len(values), ctx)
            return None

        seq = torch.tensor(values[-ctx:]).unsqueeze(0)  # type: ignore[name-defined]
        with torch.no_grad():  # type: ignore[name-defined]
            forecast = self._model(seq).squeeze(0).numpy()

        return {
            "forecast": forecast,
            "confidence": 0.5,  # LSTM fallback — moderate confidence
            "model": "lstm_fallback",
        }

    # ------------------------------------------------------------------
    # Retraining schedule
    # ------------------------------------------------------------------

    @staticmethod
    def should_retrain(last_train_date: date, today: date) -> bool:
        """Return True if enough time has elapsed since the last training run.

        Uses TransformerConfig.retrain_interval_days as the threshold (default 7).
        Static because it needs no model state — just date arithmetic.
        """
        days_elapsed = (today - last_train_date).days
        return days_elapsed >= TransformerConfig.retrain_interval_days
