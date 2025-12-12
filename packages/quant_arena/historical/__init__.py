# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Historical QuantArena - Simulation harness for multi-agent trading.

This module provides the historical simulation engine that:
- Replays daily OHLCV data from earliest available to present
- Runs quantpod agents at each daily bar
- Tracks portfolio evolution with full P&L attribution
- Logs agent messages for chat timeline visualization

Usage:
    python -m quant_arena.historical.run
"""

from quant_arena.historical.config import HistoricalConfig
from quant_arena.historical.universe import SymbolUniverse
from quant_arena.historical.clock import HistoricalClock
from quant_arena.historical.sim_broker import SimBroker
from quant_arena.historical.data_loader import DataLoader
from quant_arena.historical.engine import HistoricalEngine

__all__ = [
    "HistoricalConfig",
    "SymbolUniverse",
    "HistoricalClock",
    "SimBroker",
    "DataLoader",
    "HistoricalEngine",
]
