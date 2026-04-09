# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Asset-class enums, trading schedules, and position-limit definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo


class AssetClassType(str, Enum):
    """Supported asset classes across the QuantStack platform."""

    EQUITY = "equity"
    OPTIONS = "options"
    FUTURES = "futures"
    CRYPTO = "crypto"
    FOREX = "forex"
    FIXED_INCOME = "fixed_income"


@dataclass(frozen=True)
class TradingSchedule:
    """When a given asset class can be traded.

    For 24-hour markets (crypto, some forex pairs) set ``is_24h=True``
    and the open/close times are ignored.
    """

    open_time: time
    close_time: time
    timezone: str
    days_active: tuple[int, ...]  # 0=Mon … 6=Sun (ISO weekday - 1)
    is_24h: bool = False

    def is_open(self, dt: datetime) -> bool:
        """Return *True* if *dt* falls within the trading window.

        ``dt`` is converted to the schedule's timezone before comparison.
        """
        tz = ZoneInfo(self.timezone)
        local = dt.astimezone(tz)

        if local.weekday() not in self.days_active:
            return False

        if self.is_24h:
            return True

        local_time = local.time()

        # Handle schedules that cross midnight (e.g. futures Sun 18:00 → Fri 17:00
        # are modelled per-day, so open > close means the window wraps midnight).
        if self.open_time <= self.close_time:
            return self.open_time <= local_time < self.close_time
        # Wraps midnight: open in the evening, close next morning.
        return local_time >= self.open_time or local_time < self.close_time


@dataclass(frozen=True)
class PositionLimits:
    """Hard caps enforced by the risk gate for a given asset class."""

    max_pct_equity: float   # e.g. 0.05 → 5 % of total equity per position
    max_notional: float     # absolute dollar cap per position
    max_positions: int      # max concurrent open positions
    max_leverage: float = 1.0
