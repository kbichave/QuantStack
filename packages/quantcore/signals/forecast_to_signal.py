"""
Forecast to Signal Conversion.

Transform ML forecasts into tradeable position signals.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    # Normalization
    normalize: bool = True
    lookback: int = 252  # For z-score normalization

    # Position sizing
    position_sizing: str = "proportional"  # "proportional", "binary", "quantile"
    max_position: float = 1.0
    min_position: float = -1.0

    # Thresholds
    entry_threshold: float = 0.0  # Minimum forecast for entry
    exit_threshold: float = 0.0  # Forecast level to close position

    # Smoothing
    smooth: bool = False
    smooth_halflife: int = 3


def normalize_forecast(
    forecast: pd.Series,
    lookback: int = 252,
    method: str = "zscore",
) -> pd.Series:
    """
    Normalize forecast to have stable scale.

    Args:
        forecast: Raw forecast series
        lookback: Rolling window for normalization
        method: "zscore", "rank", or "minmax"

    Returns:
        Normalized forecast in [-1, 1] range
    """
    if method == "zscore":
        mean = forecast.rolling(lookback, min_periods=20).mean()
        std = forecast.rolling(lookback, min_periods=20).std()
        normalized = (forecast - mean) / (std + 1e-10)
        # Clip to [-3, 3] then scale to [-1, 1]
        normalized = normalized.clip(-3, 3) / 3

    elif method == "rank":
        # Percentile rank within lookback window
        normalized = (
            forecast.rolling(lookback, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            * 2
            - 1
        )  # Scale to [-1, 1]

    elif method == "minmax":
        roll_min = forecast.rolling(lookback, min_periods=20).min()
        roll_max = forecast.rolling(lookback, min_periods=20).max()
        normalized = 2 * (forecast - roll_min) / (roll_max - roll_min + 1e-10) - 1

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.fillna(0)


def forecast_to_position(
    forecast: pd.Series,
    config: Optional[SignalConfig] = None,
) -> pd.Series:
    """
    Convert forecast to position signal.

    Args:
        forecast: Forecast series (higher = bullish)
        config: Signal configuration

    Returns:
        Position signal in [min_position, max_position]
    """
    config = config or SignalConfig()

    # Normalize if requested
    if config.normalize:
        signal = normalize_forecast(forecast, config.lookback)
    else:
        signal = forecast

    # Apply position sizing method
    if config.position_sizing == "proportional":
        # Linear scaling
        position = signal.clip(config.min_position, config.max_position)

    elif config.position_sizing == "binary":
        # Binary positions at thresholds
        position = pd.Series(0.0, index=signal.index)
        position[signal > config.entry_threshold] = config.max_position
        position[signal < -config.entry_threshold] = config.min_position

    elif config.position_sizing == "quantile":
        # Position based on forecast quantile
        quantile = signal.rolling(252, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x).mean()
        )
        position = (quantile - 0.5) * 2 * config.max_position

    else:
        raise ValueError(f"Unknown position sizing: {config.position_sizing}")

    # Apply smoothing
    if config.smooth:
        position = position.ewm(halflife=config.smooth_halflife).mean()

    return position


class ForecastToSignal:
    """
    Full pipeline from forecasts to signals.

    Handles:
    - Multiple forecast sources
    - Forecast combination
    - Signal normalization
    - Position sizing

    Example:
        converter = ForecastToSignal(
            combination_method="equal_weight",
            position_sizing="proportional",
        )

        signals = converter.convert({
            "ml_model": ml_forecasts,
            "momentum": momentum_signal,
        })
    """

    def __init__(
        self,
        combination_method: str = "equal_weight",
        position_sizing: str = "proportional",
        max_position: float = 1.0,
        normalize: bool = True,
    ):
        """
        Initialize converter.

        Args:
            combination_method: How to combine forecasts ("equal_weight", "ic_weight")
            position_sizing: Position sizing method
            max_position: Maximum position size
            normalize: Whether to normalize forecasts
        """
        self.combination_method = combination_method
        self.config = SignalConfig(
            position_sizing=position_sizing,
            max_position=max_position,
            min_position=-max_position,
            normalize=normalize,
        )

        self.forecast_weights: Dict[str, float] = {}
        self.ic_history: Dict[str, pd.Series] = {}

    def combine_forecasts(
        self,
        forecasts: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Combine multiple forecasts into single signal.

        Args:
            forecasts: Dictionary of forecast name -> series
            weights: Optional explicit weights

        Returns:
            Combined forecast
        """
        if not forecasts:
            raise ValueError("No forecasts provided")

        # Align all forecasts
        aligned = pd.DataFrame(forecasts)

        if weights:
            w = pd.Series(weights)
        elif self.combination_method == "equal_weight":
            w = pd.Series({k: 1.0 / len(forecasts) for k in forecasts})
        elif self.combination_method == "ic_weight":
            # Weight by recent IC
            w = self._compute_ic_weights(aligned)
        else:
            w = pd.Series({k: 1.0 / len(forecasts) for k in forecasts})

        # Normalize weights
        w = w / w.sum()
        self.forecast_weights = w.to_dict()

        # Combine
        combined = (aligned * w).sum(axis=1)

        return combined

    def _compute_ic_weights(
        self,
        forecasts: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        lookback: int = 63,
    ) -> pd.Series:
        """Compute IC-based weights."""
        # Default to equal weights if no returns
        if returns is None:
            return pd.Series({c: 1.0 for c in forecasts.columns})

        weights = {}
        for col in forecasts.columns:
            # Compute rolling IC
            ic = forecasts[col].rolling(lookback).corr(returns.shift(-1))
            ic_mean = ic.iloc[-lookback:].mean()
            weights[col] = max(0, ic_mean)  # Only positive IC

        return pd.Series(weights)

    def convert(
        self,
        forecasts: Dict[str, pd.Series],
        returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Convert forecasts to position signal.

        Args:
            forecasts: Dictionary of forecast series
            returns: Optional returns for IC weighting

        Returns:
            Position signal
        """
        combined = self.combine_forecasts(forecasts)
        position = forecast_to_position(combined, self.config)

        return position

    def get_weights(self) -> Dict[str, float]:
        """Get current forecast weights."""
        return self.forecast_weights


def generate_signals_walkforward(
    forecasts: pd.DataFrame,
    returns: pd.Series,
    train_window: int = 252,
    test_window: int = 21,
) -> pd.Series:
    """
    Generate signals with walkforward optimization.

    Re-estimates weights at each step to avoid lookahead.

    Args:
        forecasts: DataFrame of forecast columns
        returns: Return series for IC calculation
        train_window: Training window size
        test_window: Test window size

    Returns:
        Out-of-sample signals
    """
    signals = pd.Series(index=forecasts.index, dtype=float)

    for end in range(train_window, len(forecasts), test_window):
        # Training period
        train_start = end - train_window
        train_forecasts = forecasts.iloc[train_start:end]
        train_returns = returns.iloc[train_start:end]

        # Compute IC-weighted combination
        weights = {}
        for col in forecasts.columns:
            ic = train_forecasts[col].corr(train_returns.shift(-1))
            weights[col] = max(0, ic)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(forecasts.columns) for k in forecasts.columns}

        # Apply to test period
        test_end = min(end + test_window, len(forecasts))
        test_forecasts = forecasts.iloc[end:test_end]

        combined = sum(test_forecasts[col] * w for col, w in weights.items())
        signals.iloc[end:test_end] = combined

    # Convert to positions
    return forecast_to_position(signals)
