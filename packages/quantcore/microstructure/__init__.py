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

from quantcore.microstructure.execution_algos import (
    ISExecutor,
    TWAPExecutor,
    VWAPExecutor,
)
from quantcore.microstructure.impact_models import (
    ImpactModel,
    estimate_kyle_lambda,
    square_root_impact,
)
from quantcore.microstructure.matching_engine import Fill, MatchingEngine
from quantcore.microstructure.microstructure_features import (
    MicrostructureFeatureEngine,
    MicrostructureFeatures,
)
from quantcore.microstructure.order_book import Order, OrderBook, OrderType, Side
from quantcore.microstructure.order_book_reconstructor import (
    BookSnapshot,
    OrderBookReconstructor,
)
from quantcore.microstructure.simulator import MarketSimulator

__all__ = [
    # Simulation
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
    # Live microstructure
    "OrderBookReconstructor",
    "BookSnapshot",
    "MicrostructureFeatureEngine",
    "MicrostructureFeatures",
]
