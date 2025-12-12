# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Type definitions and aliases for QuantCore.

This module provides standardized type aliases used throughout the library
to ensure consistent typing and improve code readability.

Example
-------
>>> from quantcore._typing import OHLCV, Signal, Returns
>>> def compute_signal(data: OHLCV) -> Signal:
...     return data["close"].pct_change()
"""

from __future__ import annotations

from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


# =============================================================================
# Generic Type Variables
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# Numeric type variable
Numeric = TypeVar("Numeric", int, float, np.floating, np.integer)

# DataFrame/Series type variables
DF = TypeVar("DF", bound=pd.DataFrame)
S = TypeVar("S", bound=pd.Series)


# =============================================================================
# Array Types
# =============================================================================

# NumPy array types
Float64Array: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

# Generic numeric array
NumericArray: TypeAlias = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# =============================================================================
# DataFrame Column Types (TypedDict for structured DataFrames)
# =============================================================================


class OHLCVColumns(TypedDict):
    """Standard OHLCV DataFrame column specification."""

    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series


class OHLCVOptionalColumns(TypedDict, total=False):
    """Optional columns that may be present in OHLCV data."""

    adjusted_close: pd.Series
    vwap: pd.Series
    trades: pd.Series
    turnover: pd.Series


# =============================================================================
# Primary Data Types
# =============================================================================

# OHLCV DataFrame - the core data structure for price data
# Should have DatetimeIndex and columns: open, high, low, close, volume
OHLCV: TypeAlias = pd.DataFrame

# Time series signal (e.g., alpha signals, indicators)
Signal: TypeAlias = pd.Series

# Return series (typically log returns or simple returns)
Returns: TypeAlias = pd.Series

# Price series (close prices, typically)
Prices: TypeAlias = pd.Series

# Volume series
Volume: TypeAlias = pd.Series

# Timestamp types
Timestamp: TypeAlias = pd.Timestamp | datetime | str
DateRange: TypeAlias = Tuple[Timestamp, Timestamp]


# =============================================================================
# Trading Types
# =============================================================================

# Position sizing (shares/contracts or fractional)
Position: TypeAlias = float | int

# Position direction
Direction: TypeAlias = Literal["long", "short", "flat"]

# Order side
OrderSide: TypeAlias = Literal["buy", "sell"]

# Order type
OrderType: TypeAlias = Literal["market", "limit", "stop", "stop_limit"]

# Asset class
AssetClass: TypeAlias = Literal["equity", "futures", "options", "fx", "crypto"]

# Timeframe literals
TimeframeLiteral: TypeAlias = Literal[
    "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
]


# =============================================================================
# Feature Types
# =============================================================================

# Feature matrix (rows = timestamps, columns = features)
FeatureMatrix: TypeAlias = pd.DataFrame

# Feature names
FeatureNames: TypeAlias = List[str] | Tuple[str, ...]

# Feature importance dictionary
FeatureImportance: TypeAlias = Dict[str, float]


# =============================================================================
# Model Types
# =============================================================================

# Prediction output
Prediction: TypeAlias = pd.Series | NDArray[np.floating[Any]]

# Probability output (for classification)
Probability: TypeAlias = pd.Series | NDArray[np.floating[Any]]

# Model parameters dictionary
ModelParams: TypeAlias = Dict[str, Any]

# Hyperparameters
Hyperparameters: TypeAlias = Dict[str, int | float | str | bool | None]


# =============================================================================
# Backtest Types
# =============================================================================


class TradeRecord(TypedDict):
    """Record of a single trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: Direction
    size: float
    pnl: float
    return_pct: float
    commission: float


class BacktestMetrics(TypedDict):
    """Standard backtest performance metrics."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float


# Trade list
Trades: TypeAlias = List[TradeRecord]

# Equity curve (cumulative returns or portfolio value)
EquityCurve: TypeAlias = pd.Series


# =============================================================================
# Risk Types
# =============================================================================

# Risk metrics
VaR: TypeAlias = float  # Value at Risk
CVaR: TypeAlias = float  # Conditional VaR (Expected Shortfall)
Volatility: TypeAlias = float  # Annualized volatility

# Drawdown series
Drawdown: TypeAlias = pd.Series

# Correlation matrix
CorrelationMatrix: TypeAlias = pd.DataFrame


# =============================================================================
# Configuration Types
# =============================================================================

# Generic configuration dictionary
Config: TypeAlias = Dict[str, Any]

# Strategy parameters
StrategyParams: TypeAlias = Dict[
    str, int | float | str | bool | List[Any] | Dict[str, Any]
]


# =============================================================================
# Callback Types
# =============================================================================

# Progress callback (current, total, message)
ProgressCallback: TypeAlias = Callable[[int, int, str], None]

# Event callback
EventCallback: TypeAlias = Callable[[Dict[str, Any]], None]

# Trade callback
TradeCallback: TypeAlias = Callable[[TradeRecord], None]


# =============================================================================
# Protocol Definitions (Structural Typing)
# =============================================================================


class SupportsPredict(Protocol):
    """Protocol for objects that support prediction."""

    def predict(self, X: FeatureMatrix) -> Prediction:
        """Generate predictions from features."""
        ...


class SupportsFit(Protocol):
    """Protocol for objects that support fitting/training."""

    def fit(self, X: FeatureMatrix, y: Signal) -> "SupportsFit":
        """Fit the model to data."""
        ...


class SupportsTransform(Protocol):
    """Protocol for objects that support transformation."""

    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Transform the input data."""
        ...


class SupportsScore(Protocol):
    """Protocol for objects that support scoring."""

    def score(self, X: FeatureMatrix, y: Signal) -> float:
        """Score the model on data."""
        ...


class Strategy(Protocol):
    """Protocol for trading strategy implementations."""

    def generate_signals(self, data: OHLCV) -> Signal:
        """Generate trading signals from OHLCV data."""
        ...


class FeatureComputer(Protocol):
    """Protocol for feature computation classes."""

    def compute(self, data: OHLCV) -> FeatureMatrix:
        """Compute features from OHLCV data."""
        ...

    def get_feature_names(self) -> FeatureNames:
        """Return list of feature names."""
        ...


class RiskManager(Protocol):
    """Protocol for risk management implementations."""

    def compute_position_size(
        self,
        signal: float,
        price: float,
        volatility: float,
        portfolio_value: float,
    ) -> Position:
        """Compute position size based on risk parameters."""
        ...


# =============================================================================
# Result Types (for function returns)
# =============================================================================


class FitResult(NamedTuple):
    """Result from model fitting."""

    model: Any
    train_score: float
    val_score: float | None
    feature_importance: FeatureImportance | None


class BacktestResult(NamedTuple):
    """Result from backtesting."""

    metrics: BacktestMetrics
    trades: Trades
    equity_curve: EquityCurve
    signals: Signal


class OptimizationResult(NamedTuple):
    """Result from parameter optimization."""

    best_params: Hyperparameters
    best_score: float
    all_results: List[Dict[str, Any]]


# =============================================================================
# Convenience Type Guards
# =============================================================================


def is_ohlcv(df: pd.DataFrame) -> bool:
    """Check if DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close", "volume"}
    return required.issubset(set(df.columns.str.lower()))


def is_datetime_index(df: pd.DataFrame) -> bool:
    """Check if DataFrame has DatetimeIndex."""
    return isinstance(df.index, pd.DatetimeIndex)


def is_valid_ohlcv(df: pd.DataFrame) -> bool:
    """Check if DataFrame is valid OHLCV with DatetimeIndex."""
    return is_ohlcv(df) and is_datetime_index(df)


# =============================================================================
# Exported Names
# =============================================================================

__all__ = [
    # Generic type vars
    "T",
    "T_co",
    "T_contra",
    "Numeric",
    "DF",
    "S",
    # Array types
    "Float64Array",
    "Float32Array",
    "Int64Array",
    "BoolArray",
    "NumericArray",
    # Primary data types
    "OHLCV",
    "Signal",
    "Returns",
    "Prices",
    "Volume",
    "Timestamp",
    "DateRange",
    # Trading types
    "Position",
    "Direction",
    "OrderSide",
    "OrderType",
    "AssetClass",
    "TimeframeLiteral",
    # Feature types
    "FeatureMatrix",
    "FeatureNames",
    "FeatureImportance",
    # Model types
    "Prediction",
    "Probability",
    "ModelParams",
    "Hyperparameters",
    # Backtest types
    "TradeRecord",
    "BacktestMetrics",
    "Trades",
    "EquityCurve",
    # Risk types
    "VaR",
    "CVaR",
    "Volatility",
    "Drawdown",
    "CorrelationMatrix",
    # Config types
    "Config",
    "StrategyParams",
    # Callback types
    "ProgressCallback",
    "EventCallback",
    "TradeCallback",
    # Protocols
    "SupportsPredict",
    "SupportsFit",
    "SupportsTransform",
    "SupportsScore",
    "Strategy",
    "FeatureComputer",
    "RiskManager",
    # Result types
    "FitResult",
    "BacktestResult",
    "OptimizationResult",
    # Type guards
    "is_ohlcv",
    "is_datetime_index",
    "is_valid_ohlcv",
]
