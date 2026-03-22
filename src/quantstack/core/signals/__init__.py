# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Signals Module - Converting Forecasts to Tradeable Signals.

Provides the pipeline from ML forecasts to positions:
- Forecast to signal conversion
- Cost-adjusted signal filtering
- Signal evaluation and performance metrics
"""

from quantstack.core.signals.cost_adjuster import (
    CostAdjuster,
    minimum_edge_filter,
    net_of_cost_signal,
)
from quantstack.core.signals.forecast_to_signal import (
    ForecastToSignal,
    forecast_to_position,
    normalize_forecast,
)
from quantstack.core.signals.signal_evaluator import (
    SignalEvaluator,
    compute_ic,
    compute_sharpe,
    compute_turnover,
)

__all__ = [
    "ForecastToSignal",
    "forecast_to_position",
    "normalize_forecast",
    "CostAdjuster",
    "net_of_cost_signal",
    "minimum_edge_filter",
    "SignalEvaluator",
    "compute_ic",
    "compute_sharpe",
    "compute_turnover",
]
