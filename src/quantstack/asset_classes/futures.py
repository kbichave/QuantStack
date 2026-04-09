# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Futures asset-class implementation (CME Globex hours, leveraged)."""

from __future__ import annotations

from datetime import time

from quantstack.asset_classes.base import AssetClass
from quantstack.asset_classes.types import AssetClassType, PositionLimits, TradingSchedule

# CME Globex: Sunday 18:00 ET → Friday 17:00 ET with a daily 17:00–18:00 break.
# Modelled here as an 18:00-open / 17:00-close window per active day (wraps midnight).
_FUTURES_SCHEDULE = TradingSchedule(
    open_time=time(18, 0),
    close_time=time(17, 0),
    timezone="America/New_York",
    days_active=(0, 1, 2, 3, 4, 6),  # Mon–Fri + Sun evening open
)

_FUTURES_LIMITS = PositionLimits(
    max_pct_equity=0.03,
    max_notional=100_000,
    max_positions=5,
    max_leverage=10.0,
)

SUPPORTED_SYMBOLS: frozenset[str] = frozenset({"ES", "NQ", "CL", "GC", "ZN"})


class FuturesAssetClass(AssetClass):
    """CME futures — near-24h on weekdays, leverage up to 10x."""

    @property
    def asset_type(self) -> AssetClassType:
        return AssetClassType.FUTURES

    def get_trading_schedule(self) -> TradingSchedule:
        return _FUTURES_SCHEDULE

    def get_position_limits(self) -> PositionLimits:
        return _FUTURES_LIMITS

    def get_signal_collector_names(self) -> list[str]:
        return ["momentum", "technical", "macro_regime"]

    def validate_order(self, symbol: str, qty: float, price: float) -> tuple[bool, str]:
        if not symbol or not symbol.strip():
            return False, "Symbol must not be empty"
        root = symbol.split("/")[0].upper().rstrip("0123456789")
        if root not in SUPPORTED_SYMBOLS:
            return (
                False,
                f"Unsupported futures symbol root {root!r}. "
                f"Supported: {sorted(SUPPORTED_SYMBOLS)}",
            )
        if qty <= 0:
            return False, f"Quantity must be positive, got {qty}"
        if price <= 0:
            return False, f"Price must be positive, got {price}"
        return True, ""
