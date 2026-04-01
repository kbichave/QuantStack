# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared data models — lightweight dataclasses used across layers.

Models here must have zero intra-project dependencies so any layer can
import them without creating upward dependency violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Tick:
    """
    A single market tick from the data bus.

    Produced by MarketDataBus and consumed by TickExecutor.
    """

    symbol: str
    price: float
    volume: int
    bid: float | None = None
    ask: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price


@dataclass
class TradeReflection:
    """A single trade outcome with market context."""

    symbol: str
    strategy_id: str
    action: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    realized_pnl_pct: float
    holding_days: int
    regime_at_entry: str
    regime_at_exit: str = "unknown"
    conviction_at_entry: float = 0.0
    signals_at_entry: str = ""  # serialized key signals
    signals_at_exit: str = ""
    lesson: str = ""  # auto-generated or empty
    timestamp: str = ""
