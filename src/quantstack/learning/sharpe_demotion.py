# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Live vs Backtest Sharpe Demotion — Section 12 of deep-implement blueprint.

Computes rolling 21-day live Sharpe and triggers demotion when live performance
falls below 50% of backtest Sharpe for 21 consecutive days.

Pure functions — accept data as parameters, no DB calls inside compute logic.
"""

from __future__ import annotations

import math
import os


def compute_live_sharpe(daily_returns: list[float]) -> float | None:
    """
    Compute annualized Sharpe ratio from daily returns.

    Formula: Sharpe = (mean / std) * sqrt(252)

    Args:
        daily_returns: List of daily returns (e.g., [0.01, -0.005, 0.02])

    Returns:
        Annualized Sharpe ratio, or None if fewer than 21 values.
        Returns float('inf') if mean > 0 and std = 0.
        Returns float('-inf') if mean < 0 and std = 0.
        Returns 0.0 if mean = 0 and std = 0.
    """
    if len(daily_returns) < 21:
        return None

    mean_return = sum(daily_returns) / len(daily_returns)

    # Compute standard deviation manually
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
    std_return = math.sqrt(variance)

    # Handle edge case: std ≈ 0 (use tolerance for floating-point precision)
    if std_return < 1e-10:
        if mean_return > 1e-10:
            return float('inf')
        elif mean_return < -1e-10:
            return float('-inf')
        else:
            return 0.0

    # Annualize: sqrt(252) trading days per year
    sharpe = (mean_return / std_return) * math.sqrt(252)
    return sharpe


def check_sharpe_demotion(
    live_sharpe: float | None,
    backtest_sharpe: float,
    consecutive_days: int = 21,
) -> dict | None:
    """
    Check if live Sharpe has fallen below 50% of backtest Sharpe for 21+ days.

    Args:
        live_sharpe: Rolling 21-day live Sharpe (None if insufficient data)
        backtest_sharpe: Backtest Sharpe from training
        consecutive_days: Number of days with degraded performance (default 21)

    Returns:
        Demotion info dict if triggered, None otherwise.
        Dict contains: {"triggered": True, "live_sharpe", "backtest_sharpe", "threshold", "consecutive_days"}
    """
    # Check config flag
    if os.getenv("FEEDBACK_SHARPE_DEMOTION", "false").lower() not in ("true", "1", "yes"):
        return None

    # Cold start: skip if insufficient data
    if live_sharpe is None:
        return None

    # Gate: must have 21+ consecutive days
    if consecutive_days < 21:
        return None

    # Trigger: live < 50% of backtest
    threshold = 0.5 * backtest_sharpe
    if live_sharpe < threshold:
        return {
            "triggered": True,
            "live_sharpe": live_sharpe,
            "backtest_sharpe": backtest_sharpe,
            "threshold": threshold,
            "consecutive_days": consecutive_days,
        }

    return None
