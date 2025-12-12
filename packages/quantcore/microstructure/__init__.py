# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Microstructure Simulation Module.

Provides limit order book simulation and execution algorithms:
- Order book with bid/ask queues
- Price-time priority matching engine
- Market impact models
- Execution algorithms (TWAP, VWAP, IS minimization)
"""

from quantcore.microstructure.order_book import OrderBook, Order, OrderType, Side
from quantcore.microstructure.matching_engine import MatchingEngine, Fill
from quantcore.microstructure.impact_models import (
    square_root_impact,
    ImpactModel,
    estimate_kyle_lambda,
)
from quantcore.microstructure.execution_algos import (
    TWAPExecutor,
    VWAPExecutor,
    ISExecutor,
)
from quantcore.microstructure.simulator import MarketSimulator

__all__ = [
    "OrderBook",
    "Order",
    "OrderType",
    "Side",
    "MatchingEngine",
    "Fill",
    "square_root_impact",
    "ImpactModel",
    "estimate_kyle_lambda",
    "TWAPExecutor",
    "VWAPExecutor",
    "ISExecutor",
    "MarketSimulator",
]
