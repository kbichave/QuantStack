# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-asset expansion framework (P12).

Re-exports the core types and registry so callers can do::

    from quantstack.asset_classes import AssetClassType, register, get
"""

from __future__ import annotations

from quantstack.asset_classes.base import AssetClass
from quantstack.asset_classes.registry import enabled, get, is_registered, register
from quantstack.asset_classes.types import AssetClassType, PositionLimits, TradingSchedule

__all__ = [
    "AssetClass",
    "AssetClassType",
    "PositionLimits",
    "TradingSchedule",
    "enabled",
    "get",
    "is_registered",
    "register",
]
