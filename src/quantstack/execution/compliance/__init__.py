# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compliance utilities — calendar, wash-sale rules, market-hours checks,
pre-trade SEC checks (PDT, margin), and post-trade hooks (wash sales, tax lots)."""

from quantstack.execution.compliance.calendar import (
    calendar_day_offset,
    get_default_calendar,
    is_during_market_hours,
    rolling_business_day_window,
    trading_day_for,
    wash_sale_window_end,
)
from quantstack.execution.compliance.posttrade import TaxLotManager, WashSaleTracker
from quantstack.execution.compliance.pretrade import (
    ComplianceResult,
    MarginCalculator,
    PDTChecker,
)

__all__ = [
    "calendar_day_offset",
    "get_default_calendar",
    "is_during_market_hours",
    "rolling_business_day_window",
    "trading_day_for",
    "wash_sale_window_end",
    "ComplianceResult",
    "MarginCalculator",
    "PDTChecker",
    "TaxLotManager",
    "WashSaleTracker",
]
