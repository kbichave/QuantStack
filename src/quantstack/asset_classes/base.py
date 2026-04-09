# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class that every asset-class implementation must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantstack.asset_classes.types import AssetClassType, PositionLimits, TradingSchedule


class AssetClass(ABC):
    """Contract for an asset-class plugin.

    Concrete subclasses (equity, crypto, futures, …) implement the four
    abstract methods so the rest of the platform can treat all asset classes
    uniformly — risk gate, signal engine, and execution adapters all dispatch
    through this interface.
    """

    @property
    @abstractmethod
    def asset_type(self) -> AssetClassType: ...

    @abstractmethod
    def get_trading_schedule(self) -> TradingSchedule: ...

    @abstractmethod
    def get_position_limits(self) -> PositionLimits: ...

    @abstractmethod
    def get_signal_collector_names(self) -> list[str]:
        """Return the names of signal collectors relevant to this asset class."""
        ...

    @abstractmethod
    def validate_order(self, symbol: str, qty: float, price: float) -> tuple[bool, str]:
        """Return ``(True, "")`` if the order is valid, else ``(False, reason)``."""
        ...
