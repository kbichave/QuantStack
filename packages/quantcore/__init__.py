# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore: Institutional-Grade Quantitative Trading Research Platform.

A comprehensive library for systematic trading research, backtesting,
and signal generation with support for multiple asset classes.

Example
-------
>>> import quantcore as qc
>>>
>>> # Access timeframes
>>> print(qc.Timeframe.DAILY)
>>>
>>> # Import submodules
>>> from quantcore import data, features, backtesting
>>>
>>> # Use specific classes
>>> from quantcore.features.technical_indicators import TechnicalIndicators

Modules
-------
backtesting
    Event-driven backtesting engine with realistic cost modeling.
data
    Data fetching, storage (DuckDB), and preprocessing utilities.
features
    200+ technical indicators and feature engineering tools.
models
    ML model training, ensemble methods, and prediction.
research
    Statistical tests, alpha decay analysis, and walkforward validation.
risk
    Position sizing, drawdown controls, and risk management.
rl
    Reinforcement learning agents for trading.
microstructure
    Order book simulation, impact models, and execution algorithms.
math
    Stochastic processes, Kalman filters, and optimization utilities.
"""

from __future__ import annotations


__version__ = "0.1.0"
__author__ = "Kshitij Bichave"
__license__ = "Apache-2.0"


# =============================================================================
# Direct Submodule Imports
# =============================================================================

# Core submodules - always imported
from quantcore import (
    analysis,
    backtesting,
    config,
    data,
    equity,
    execution,
    features,
    hierarchy,
    labeling,
    math,
    microstructure,
    models,
    options,
    research,
    risk,
    rl,
    signals,
    strategy,
    utils,
    validation,
    visualization,
)

# =============================================================================
# Convenience Imports
# =============================================================================

# Configuration
from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS

# Types
from quantcore._typing import (
    OHLCV,
    Signal,
    Returns,
    FeatureMatrix,
    BacktestMetrics,
    TradeRecord,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Submodules
    "analysis",
    "backtesting",
    "config",
    "data",
    "equity",
    "execution",
    "features",
    "hierarchy",
    "labeling",
    "math",
    "microstructure",
    "models",
    "options",
    "research",
    "risk",
    "rl",
    "signals",
    "strategy",
    "utils",
    "validation",
    "visualization",
    # Configuration
    "Timeframe",
    "TIMEFRAME_PARAMS",
    # Types
    "OHLCV",
    "Signal",
    "Returns",
    "FeatureMatrix",
    "BacktestMetrics",
    "TradeRecord",
]
