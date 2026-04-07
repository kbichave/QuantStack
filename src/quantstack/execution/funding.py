# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Borrowing / funding cost model.

Computes daily margin interest for positions that use borrowed capital.
The calculator is pure math — it does NOT touch the database.  The caller
(typically ``PortfolioState.accrue_daily_funding``) persists the results.
"""

from __future__ import annotations

import os

MARGIN_ANNUAL_RATE_DEFAULT = 0.08  # 8% APR
TRADING_DAYS_PER_YEAR = 252


class FundingCostCalculator:
    """Compute margin interest charges on borrowed capital."""

    def __init__(self, annual_rate: float | None = None):
        if annual_rate is not None:
            self.annual_rate = annual_rate
        else:
            self.annual_rate = float(
                os.getenv("MARGIN_ANNUAL_RATE", str(MARGIN_ANNUAL_RATE_DEFAULT))
            )

    def daily_interest(self, margin_used: float) -> float:
        """Compute one day's margin interest.  Returns 0.0 if *margin_used* <= 0."""
        if margin_used <= 0:
            return 0.0
        return margin_used * self.annual_rate / TRADING_DAYS_PER_YEAR

    def accrue_funding_costs(
        self, positions: list,
    ) -> list[tuple[str, float]]:
        """Compute daily funding cost for every position with margin_used > 0.

        Returns a list of ``(symbol, daily_cost)`` tuples.
        Does **not** write to the database — the caller persists.
        """
        result: list[tuple[str, float]] = []
        for pos in positions:
            margin = getattr(pos, "margin_used", 0.0)
            if margin > 0:
                cost = self.daily_interest(margin)
                result.append((getattr(pos, "symbol", ""), cost))
        return result
