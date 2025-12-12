"""
Monte Carlo simulation for strategy robustness testing.

IMPORTANT: This simulation should be run on a TRUE HOLDOUT dataset that was
NOT used for parameter selection or model training. Using parameters that
were optimized on the same data invalidates the robustness test.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.backtesting.costs import ProductionCostModel


def run_monte_carlo_simulation(
    spread_df: pd.DataFrame,
    params: Dict[str, float],
    initial_capital: float = 100000,
    n_simulations: int = 1000,
    timing_noise_days: int = 1,
    holdout_start: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to test strategy robustness.

    CRITICAL: This should run on TRUE HOLDOUT data that was never used for
    parameter selection. If params were tuned on data overlapping with the
    simulation period, results will be meaninglessly optimistic.

    Randomly perturbs:
    - Entry/exit timing by ±timing_noise_days
    - Slippage (random within range from 0.8x to 1.5x base)

    Args:
        spread_df: DataFrame with spread data and spread_zscore column
        params: Strategy parameters (from hyperparameter tuning)
        initial_capital: Starting capital
        n_simulations: Number of Monte Carlo simulations to run
        timing_noise_days: Maximum timing perturbation in days
        holdout_start: Start date for holdout data (e.g., "2023-01-01").
                       If None, defaults to "2021-01-01" but will show warning.

    Returns:
        Dictionary with simulation statistics
    """
    valid_data = spread_df.dropna(subset=["spread_zscore"]).copy()

    if len(valid_data) < 100:
        return {"error": "Insufficient data"}

    # Determine holdout period
    if holdout_start is None:
        holdout_start = "2021-01-01"
        logger.warning(
            "Monte Carlo: No holdout_start specified, defaulting to 2021-01-01"
        )
        logger.warning("  If params were tuned on 2021+ data, results may be overfit!")

    # Filter to holdout period only
    valid_data = valid_data[valid_data.index >= holdout_start]

    if len(valid_data) < 100:
        return {
            "error": f"Insufficient holdout data after {holdout_start} (need 100+, got {len(valid_data)})"
        }

    logger.info(
        f"Monte Carlo running on holdout: {valid_data.index[0].date()} to {valid_data.index[-1].date()} ({len(valid_data)} bars)"
    )

    cost_model = ProductionCostModel(slippage_model="volatility")
    results = []

    for sim in range(n_simulations):
        # Add timing noise
        timing_shift = np.random.randint(-timing_noise_days, timing_noise_days + 1)

        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []

        # Random slippage multiplier for this simulation
        slippage_mult = np.random.uniform(0.8, 1.5)

        for i in range(60, len(valid_data) - abs(timing_shift)):
            idx = i + timing_shift if i + timing_shift < len(valid_data) else i
            row = valid_data.iloc[idx]
            zscore = row["spread_zscore"]
            spread = row["spread"]

            # Get volatility for cost calculation
            if idx >= 20:
                window = valid_data["spread"].iloc[idx - 20 : idx]
                vol = window.std() / (abs(window.mean()) + 1e-8)
            else:
                vol = 0.02

            n_contracts = params.get("position_size", 2000) // 1000

            entry_z = params.get("entry_zscore", 2.0)
            exit_z = params.get("exit_zscore", 0.0)
            stop_z = params.get("stop_loss_zscore", 5.0)

            # Entry
            if position == 0:
                if zscore < -entry_z:
                    position = 1
                    entry_price = spread
                    cost = (
                        cost_model.calculate_total_cost(
                            n_contracts, vol, is_round_trip=False
                        )
                        * slippage_mult
                    )
                    capital -= cost
                elif zscore > entry_z:
                    position = -1
                    entry_price = spread
                    cost = (
                        cost_model.calculate_total_cost(
                            n_contracts, vol, is_round_trip=False
                        )
                        * slippage_mult
                    )
                    capital -= cost

            # Exit
            elif position == 1:
                if (stop_z and zscore < -stop_z) or zscore > exit_z:
                    pnl = (spread - entry_price) * n_contracts * 1000
                    cost = (
                        cost_model.calculate_total_cost(
                            n_contracts, vol, is_round_trip=False
                        )
                        * slippage_mult
                    )
                    capital += pnl - cost
                    trades.append(pnl - cost)
                    position = 0

            elif position == -1:
                if (stop_z and zscore > stop_z) or zscore < -exit_z:
                    pnl = (entry_price - spread) * n_contracts * 1000
                    cost = (
                        cost_model.calculate_total_cost(
                            n_contracts, vol, is_round_trip=False
                        )
                        * slippage_mult
                    )
                    capital += pnl - cost
                    trades.append(pnl - cost)
                    position = 0

        # Close open position at end
        if position != 0:
            spread = valid_data["spread"].iloc[-1]
            if position == 1:
                pnl = (spread - entry_price) * n_contracts * 1000
            else:
                pnl = (entry_price - spread) * n_contracts * 1000
            vol = 0.02
            cost = cost_model.calculate_total_cost(
                n_contracts, vol, is_round_trip=False
            )
            capital += pnl - cost
            trades.append(pnl - cost)

        total_return = (capital - initial_capital) / initial_capital * 100
        win_rate = sum(1 for t in trades if t > 0) / max(1, len(trades)) * 100

        results.append(
            {
                "total_return": total_return,
                "final_capital": capital,
                "n_trades": len(trades),
                "win_rate": win_rate,
            }
        )

    # Calculate statistics
    returns = [r["total_return"] for r in results]
    win_rates = [r["win_rate"] for r in results]

    return {
        "n_simulations": n_simulations,
        "holdout_start": holdout_start,
        "holdout_bars": len(valid_data),
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "median_return": np.median(returns),
        "percentile_5": np.percentile(returns, 5),
        "percentile_25": np.percentile(returns, 25),
        "percentile_75": np.percentile(returns, 75),
        "percentile_95": np.percentile(returns, 95),
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "prob_positive": sum(1 for r in returns if r > 0) / len(returns) * 100,
        "mean_win_rate": np.mean(win_rates),
        "min_win_rate": np.min(win_rates),
        "max_win_rate": np.max(win_rates),
        "all_returns": returns,
    }
