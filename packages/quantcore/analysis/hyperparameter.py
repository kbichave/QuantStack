"""
Hyperparameter tuning for trading strategies.

IMPORTANT: Uses proper 3-way split to avoid data leakage:
- Train: Model fitting (not used here, but reserved for ML models)
- Validation: Parameter selection (choose best hyperparameters here)
- Test: Final evaluation ONLY (never used for selection)
"""

from itertools import product
from typing import Any, Dict, Tuple

import pandas as pd
from loguru import logger

from quantcore.backtesting.costs import ProductionCostModel
from quantcore.backtesting.engine import run_backtest_with_params


def tune_hyperparameters(
    spread_df: pd.DataFrame,
    initial_capital: float = 100000,
    train_end: str = "2018-01-01",
    val_end: str = "2021-01-01",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Tune strategy hyperparameters using grid search with proper 3-way split.

    CRITICAL: Parameters are selected based on VALIDATION set performance only.
    Test set is used ONLY for final unbiased evaluation.

    Split Structure:
        Train: data < train_end (for ML model fitting, reserved)
        Validation: train_end <= data < val_end (for parameter selection)
        Test: data >= val_end (for final evaluation ONLY)

    Uses PRODUCTION-GRADE cost model with realistic slippage.

    Args:
        spread_df: DataFrame with spread data and spread_zscore column
        initial_capital: Starting capital for backtests
        train_end: End date for training period (exclusive)
        val_end: End date for validation period (exclusive), start of test

    Returns:
        Tuple of (best_params, results_dict) where results_dict contains:
            - validation_results: Results on validation set (used for selection)
            - test_results: Results on test set (unbiased final evaluation)
            - all_results: Full grid search results
    """
    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning (3-Way Split, No Lookahead)")
    logger.info("=" * 60)

    if spread_df.empty or len(spread_df) < 500:
        logger.error("Insufficient data for hyperparameter tuning")
        return {}, {}

    # CRITICAL: Proper 3-way train/validation/test split
    valid_data = spread_df.dropna(subset=["spread_zscore"]).copy()

    train_data = valid_data[valid_data.index < train_end]
    val_data = valid_data[
        (valid_data.index >= train_end) & (valid_data.index < val_end)
    ]
    test_data = valid_data[valid_data.index >= val_end]

    # Validate split sizes
    if len(train_data) < 252:
        logger.warning(
            f"Train set small ({len(train_data)} bars). Consider adjusting train_end."
        )

    if len(val_data) < 252:
        logger.error(
            f"Validation set too small ({len(val_data)} bars). Need at least 252 bars."
        )
        return {}, {}

    if len(test_data) < 200:
        logger.error(
            f"Test set too small ({len(test_data)} bars). Need at least 200 bars."
        )
        return {}, {}

    logger.info("PROPER 3-WAY TEMPORAL SPLIT (No Lookahead Bias)")
    logger.info(
        f"  Train:      {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data):,} bars) - Reserved for ML"
    )
    logger.info(
        f"  Validation: {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data):,} bars) - Parameter Selection"
    )
    logger.info(
        f"  Test:       {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data):,} bars) - Final Eval ONLY"
    )

    cost_model = ProductionCostModel(slippage_model="volatility")
    logger.info(
        f"Production cost model: ~${cost_model.calculate_total_cost(2):.2f} per 2-contract RT"
    )

    # Parameter grid
    param_grid = {
        "entry_zscore": [1.5, 2.0, 2.5],
        "exit_zscore": [0.0, 0.5],
        "position_size": [2000],
        "stop_loss_zscore": [4.0, 5.0],
    }

    combinations = list(
        product(
            param_grid["entry_zscore"],
            param_grid["exit_zscore"],
            param_grid["position_size"],
            param_grid["stop_loss_zscore"],
        )
    )

    logger.info(
        f"Testing {len(combinations)} parameter combinations on VALIDATION set..."
    )

    best_val_sharpe = -999
    best_params = {}
    best_val_results = {}
    all_results = []

    for entry_z, exit_z, pos_size, stop_z in combinations:
        # Skip invalid parameter combinations
        if exit_z >= entry_z:
            continue
        if stop_z is not None and stop_z <= entry_z:
            continue

        n_contracts = pos_size // 1000
        spread_cost = cost_model.cost_per_barrel(n_contracts)

        # Run backtest on VALIDATION data only for parameter selection
        val_results = run_backtest_with_params(
            val_data, initial_capital, entry_z, exit_z, pos_size, spread_cost, stop_z
        )

        val_sharpe = val_results["sharpe_ratio"]

        all_results.append(
            {
                "entry_zscore": entry_z,
                "exit_zscore": exit_z,
                "position_size": pos_size,
                "spread_cost": spread_cost,
                "stop_loss_zscore": stop_z,
                "val_sharpe": val_sharpe,
                "val_return": val_results["total_return_pct"],
                "val_trades": val_results["total_trades"],
                "val_win_rate": val_results["win_rate"],
            }
        )

        # Select best params based on VALIDATION sharpe (NOT test!)
        if val_sharpe > best_val_sharpe and val_results["total_trades"] >= 3:
            best_val_sharpe = val_sharpe
            best_params = {
                "entry_zscore": entry_z,
                "exit_zscore": exit_z,
                "position_size": pos_size,
                "spread_cost": spread_cost,
                "stop_loss_zscore": stop_z,
            }
            best_val_results = val_results

    # Print top validation results
    all_results_sorted = sorted(
        all_results, key=lambda x: x["val_sharpe"], reverse=True
    )

    logger.success("Top Parameter Combinations (by VALIDATION Sharpe):")
    header = f"{'Entry Z':>8} {'Exit Z':>8} {'Size':>8} {'Cost/bbl':>10} {'Stop Z':>8} | {'Val SR':>10} {'Val Ret':>10} {'Trades':>8}"
    logger.info(header)
    logger.info("-" * 95)

    for r in all_results_sorted[:5]:
        stop_str = f"{r['stop_loss_zscore']:.1f}" if r["stop_loss_zscore"] else "None"
        logger.info(
            f"{r['entry_zscore']:>8.1f} {r['exit_zscore']:>8.1f} {r['position_size']:>8} ${r['spread_cost']:>9.4f} {stop_str:>8} | {r['val_sharpe']:>10.2f} {r['val_return']:>9.2f}% {r['val_trades']:>8}"
        )

    if not best_params:
        logger.error("No valid parameter combination found")
        return {}, {}

    logger.success("Best parameters (selected on VALIDATION set):")
    for k, v in best_params.items():
        if k == "spread_cost":
            logger.info(f"    {k}: ${v:.4f}/barrel")
        else:
            logger.info(f"    {k}: {v}")

    # NOW evaluate on TEST set - this is the unbiased final result
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON UNSEEN TEST SET")
    logger.info("=" * 60)

    test_results = run_backtest_with_params(
        test_data,
        initial_capital,
        best_params["entry_zscore"],
        best_params["exit_zscore"],
        best_params["position_size"],
        best_params["spread_cost"],
        best_params["stop_loss_zscore"],
    )

    logger.info(
        f"  Test Period: {test_data.index[0].date()} to {test_data.index[-1].date()}"
    )
    logger.info(f"  Test Sharpe: {test_results['sharpe_ratio']:.2f}")
    logger.info(f"  Test Return: {test_results['total_return_pct']:.1f}%")
    logger.info(f"  Test Trades: {test_results['total_trades']}")
    logger.info(f"  Test Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Test Max DD: {test_results['max_drawdown']:.1f}%")

    # Warn if there's significant degradation from validation to test
    val_sharpe = best_val_results["sharpe_ratio"]
    test_sharpe = test_results["sharpe_ratio"]

    if test_sharpe < val_sharpe * 0.5:
        logger.warning(
            f"WARNING: Test Sharpe ({test_sharpe:.2f}) << Validation Sharpe ({val_sharpe:.2f})"
        )
        logger.warning("This suggests possible overfitting to validation period.")

    logger.info("=" * 60)

    # Return comprehensive results
    results = {
        "validation_results": best_val_results,
        "test_results": test_results,
        "all_grid_results": all_results,
        "split_info": {
            "train_start": str(train_data.index[0].date()),
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end,
            "test_start": val_end,
            "test_end": str(test_data.index[-1].date()),
            "train_bars": len(train_data),
            "val_bars": len(val_data),
            "test_bars": len(test_data),
        },
    }

    return best_params, results
