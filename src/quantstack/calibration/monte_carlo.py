"""Monte Carlo path simulator for threshold calibration.

Reusable by both daily halt calibration (bootstrap P&L paths) and
Kelly fraction calibration (equity curve paths).
"""

from __future__ import annotations

import numpy as np


def simulate_paths(
    daily_returns: np.ndarray,
    n_paths: int = 10_000,
    n_days: int = 252,
    kelly_fraction: float = 1.0,
    halt_threshold: float | None = None,
) -> np.ndarray:
    """Simulate equity curve paths by resampling from observed daily returns.

    Args:
        daily_returns: Historical daily return observations to resample from.
        n_paths: Number of Monte Carlo paths.
        n_days: Trading days per path.
        kelly_fraction: Position sizing fraction applied to each return.
        halt_threshold: If set, trading halts for the remainder of the month
            when intra-month drawdown exceeds this value.

    Returns:
        Array of shape (n_paths, n_days) with cumulative equity values
        (starting at 1.0).
    """
    if len(daily_returns) == 0:
        return np.ones((n_paths, n_days))

    rng = np.random.default_rng(42)

    # Resample returns with replacement
    sampled_idx = rng.integers(0, len(daily_returns), size=(n_paths, n_days))
    sampled_returns = daily_returns[sampled_idx] * kelly_fraction

    # Build equity curves
    equity = np.ones((n_paths, n_days))

    for day in range(n_days):
        if day == 0:
            equity[:, day] = 1.0 + sampled_returns[:, day]
        else:
            equity[:, day] = equity[:, day - 1] * (1.0 + sampled_returns[:, day])

        if halt_threshold is not None:
            # Check intra-month drawdown (month = 21 trading days)
            month_start_day = (day // 21) * 21
            month_start_equity = equity[:, month_start_day] if month_start_day < day else equity[:, 0]
            month_dd = (equity[:, day] - month_start_equity) / month_start_equity

            # Halt trading for paths where monthly drawdown exceeds threshold
            halted = month_dd < -halt_threshold
            if day + 1 < n_days:
                # Zero out returns for halted paths for rest of month
                days_left_in_month = 21 - (day % 21) - 1
                for future_day in range(day + 1, min(day + 1 + days_left_in_month, n_days)):
                    sampled_returns[halted, future_day] = 0.0

    return equity


def compute_max_drawdowns(equity_paths: np.ndarray) -> np.ndarray:
    """Compute maximum drawdown for each equity path.

    Args:
        equity_paths: Shape (n_paths, n_days) equity curves.

    Returns:
        Shape (n_paths,) array of max drawdowns (positive values, e.g. 0.15 = 15%).
    """
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (running_max - equity_paths) / running_max
    return np.max(drawdowns, axis=1)


def compute_monthly_max_drawdowns(equity_paths: np.ndarray, days_per_month: int = 21) -> np.ndarray:
    """Compute monthly maximum drawdowns across all paths.

    Returns:
        Flat array of all monthly max drawdowns across all paths and months.
    """
    n_paths, n_days = equity_paths.shape
    n_months = n_days // days_per_month
    monthly_dds = []

    for m in range(n_months):
        start = m * days_per_month
        end = start + days_per_month
        month_equity = equity_paths[:, start:end]
        month_start = month_equity[:, 0:1]
        month_dd = (month_start - month_equity) / month_start
        monthly_dds.append(np.max(month_dd, axis=1))

    return np.concatenate(monthly_dds)
