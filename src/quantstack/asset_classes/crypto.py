# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Crypto asset-class implementation (24/7 trading, conservative limits)."""

from __future__ import annotations

from datetime import time

from quantstack.asset_classes.base import AssetClass
from quantstack.asset_classes.types import AssetClassType, PositionLimits, TradingSchedule

_CRYPTO_SCHEDULE = TradingSchedule(
    open_time=time(0, 0),
    close_time=time(0, 0),
    timezone="UTC",
    days_active=(0, 1, 2, 3, 4, 5, 6),  # Every day
    is_24h=True,
)

_CRYPTO_LIMITS = PositionLimits(
    max_pct_equity=0.02,
    max_notional=10_000,
    max_positions=5,
    max_leverage=1.0,
)

_VALID_SUFFIXES = ("USD", "USDT")


class CryptoAssetClass(AssetClass):
    """Cryptocurrency markets — 24/7, tighter position limits."""

    @property
    def asset_type(self) -> AssetClassType:
        return AssetClassType.CRYPTO

    def get_trading_schedule(self) -> TradingSchedule:
        return _CRYPTO_SCHEDULE

    def get_position_limits(self) -> PositionLimits:
        return _CRYPTO_LIMITS

    def get_signal_collector_names(self) -> list[str]:
        return ["momentum", "technical", "social_sentiment"]

    def validate_order(self, symbol: str, qty: float, price: float) -> tuple[bool, str]:
        if not symbol or not symbol.strip():
            return False, "Symbol must not be empty"
        if not symbol.endswith(_VALID_SUFFIXES):
            return False, f"Crypto symbol must end with one of {_VALID_SUFFIXES}, got {symbol!r}"
        if qty <= 0:
            return False, f"Quantity must be positive, got {qty}"
        if price <= 0:
            return False, f"Price must be positive, got {price}"
        return True, ""
