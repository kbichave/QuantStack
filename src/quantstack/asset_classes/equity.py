# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Equity asset-class implementation (US equities via Alpaca / eTrade)."""

from __future__ import annotations

from datetime import time

from quantstack.asset_classes.base import AssetClass
from quantstack.asset_classes.types import AssetClassType, PositionLimits, TradingSchedule

_EQUITY_SCHEDULE = TradingSchedule(
    open_time=time(9, 30),
    close_time=time(16, 0),
    timezone="America/New_York",
    days_active=(0, 1, 2, 3, 4),  # Mon–Fri
)

_EQUITY_LIMITS = PositionLimits(
    max_pct_equity=0.05,
    max_notional=50_000,
    max_positions=20,
    max_leverage=1.0,
)


class EquityAssetClass(AssetClass):
    """US equity markets — regular trading hours, no leverage."""

    @property
    def asset_type(self) -> AssetClassType:
        return AssetClassType.EQUITY

    def get_trading_schedule(self) -> TradingSchedule:
        return _EQUITY_SCHEDULE

    def get_position_limits(self) -> PositionLimits:
        return _EQUITY_LIMITS

    def get_signal_collector_names(self) -> list[str]:
        return [
            "momentum",
            "mean_reversion",
            "fundamental_value",
            "earnings_quality",
            "technical",
        ]

    def validate_order(self, symbol: str, qty: float, price: float) -> tuple[bool, str]:
        if not symbol or not symbol.strip():
            return False, "Symbol must not be empty"
        if qty <= 0:
            return False, f"Quantity must be positive, got {qty}"
        if price <= 0:
            return False, f"Price must be positive, got {price}"
        return True, ""
