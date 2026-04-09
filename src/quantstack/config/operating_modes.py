# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Operating mode system (P15) — clock-driven mode detection for the autonomous fund.

The system operates in different modes depending on the time of day:
  - MARKET_HOURS: 09:30–16:00 ET Mon–Fri. Trading active, research backgrounded.
  - EXTENDED_HOURS: 16:00–20:00 ET Mon–Fri. No new trades, monitor existing.
  - OVERNIGHT_WEEKEND: 20:00 ET Fri – 09:30 ET Mon, or 20:00–09:30 weeknights.
    Full research compute, no trading.
  - CRYPTO_FUTURES: 24/7 if ENABLE_CRYPTO_FUTURES=true. Crypto-specific trading
    with minimal research overhead.

Mode detection is a pure function of clock time in US/Eastern. No LLM calls,
no network. The scheduler and graph orchestrator read the current mode to decide
which graphs to activate and how to allocate compute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any

from loguru import logger

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

_ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class OperatingMode(str, Enum):
    """Which mode the autonomous fund is currently in."""

    MARKET_HOURS = "market_hours"
    EXTENDED_HOURS = "extended_hours"
    OVERNIGHT_WEEKEND = "overnight_weekend"
    CRYPTO_FUTURES = "crypto_futures"


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a single operating mode."""

    mode: OperatingMode
    graphs_active: tuple[str, ...]
    trading_enabled: bool
    research_compute_priority: float
    position_monitoring: bool


@dataclass
class ModeTransition:
    """Record of a mode transition for audit logging."""

    from_mode: OperatingMode
    to_mode: OperatingMode
    timestamp: datetime


# ---------------------------------------------------------------------------
# Default configurations per mode
# ---------------------------------------------------------------------------

DEFAULT_MODE_CONFIGS: dict[OperatingMode, ModeConfig] = {
    OperatingMode.MARKET_HOURS: ModeConfig(
        mode=OperatingMode.MARKET_HOURS,
        graphs_active=("trading", "supervisor", "research"),
        trading_enabled=True,
        research_compute_priority=0.2,
        position_monitoring=True,
    ),
    OperatingMode.EXTENDED_HOURS: ModeConfig(
        mode=OperatingMode.EXTENDED_HOURS,
        graphs_active=("supervisor",),
        trading_enabled=False,
        research_compute_priority=0.3,
        position_monitoring=True,
    ),
    OperatingMode.OVERNIGHT_WEEKEND: ModeConfig(
        mode=OperatingMode.OVERNIGHT_WEEKEND,
        graphs_active=("research", "supervisor"),
        trading_enabled=False,
        research_compute_priority=1.0,
        position_monitoring=False,
    ),
    OperatingMode.CRYPTO_FUTURES: ModeConfig(
        mode=OperatingMode.CRYPTO_FUTURES,
        graphs_active=("trading", "supervisor"),
        trading_enabled=True,
        research_compute_priority=0.1,
        position_monitoring=True,
    ),
}

# Market boundaries in US/Eastern
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_EXTENDED_CLOSE = time(20, 0)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def detect_current_mode(now: datetime | None = None) -> OperatingMode:
    """Determine the operating mode from the current clock time in US/Eastern.

    Pure function of time — no network calls, no DB reads. Suitable for
    calling every loop iteration without side effects.

    If ENABLE_CRYPTO_FUTURES=true and we're outside equity market + extended
    hours, crypto mode takes precedence over overnight/weekend.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)

    weekday = now.weekday()  # 0=Monday, 6=Sunday
    t = now.time()

    crypto_enabled = os.environ.get("ENABLE_CRYPTO_FUTURES", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Weekend: Saturday (5) or Sunday (6)
    is_weekend = weekday >= 5

    if is_weekend:
        if crypto_enabled:
            return OperatingMode.CRYPTO_FUTURES
        return OperatingMode.OVERNIGHT_WEEKEND

    # Weekday time windows
    if _MARKET_OPEN <= t < _MARKET_CLOSE:
        return OperatingMode.MARKET_HOURS

    if _MARKET_CLOSE <= t < _EXTENDED_CLOSE:
        return OperatingMode.EXTENDED_HOURS

    # Before market open or after extended close
    if crypto_enabled:
        return OperatingMode.CRYPTO_FUTURES
    return OperatingMode.OVERNIGHT_WEEKEND


def get_mode_config(mode: OperatingMode | None = None) -> ModeConfig:
    """Return the ModeConfig for the given (or current) operating mode."""
    if mode is None:
        mode = detect_current_mode()
    return DEFAULT_MODE_CONFIGS[mode]
