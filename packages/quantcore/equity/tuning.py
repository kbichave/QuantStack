"""
Per-ticker hyperparameter tuning.

Tunes strategy parameters on validation data for each ticker individually.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.equity.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from quantcore.equity.backtester import backtest_signals


@dataclass
class TunedParams:
    """Tuned hyperparameters for a ticker."""

    ticker: str
    mean_reversion: Dict[str, float]
    momentum: Dict[str, float]
    trend_following: Dict[str, float]
    best_strategy: str
    validation_pnl: float


def tune_mean_reversion(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    initial_equity: float = 100000,
) -> Tuple[Dict[str, float], float]:
    """
    Tune MeanReversion hyperparameters on validation data.

    Grid search over:
    - zscore_threshold: [1.5, 2.0, 2.5, 3.0]
    - reversion_delta: [0.1, 0.2, 0.3]
    """
    best_pnl = float("-inf")
    best_params = {"zscore_threshold": 2.0, "reversion_delta": 0.2}

    for zscore in [1.5, 2.0, 2.5, 3.0]:
        for delta in [0.1, 0.2, 0.3]:
            strategy = MeanReversionStrategy(
                zscore_threshold=zscore,
                reversion_delta=delta,
            )
            signals = strategy.generate_signals(features)
            result = backtest_signals(signals, prices, initial_equity=initial_equity)

            if result.total_pnl > best_pnl:
                best_pnl = result.total_pnl
                best_params = {"zscore_threshold": zscore, "reversion_delta": delta}

    return best_params, best_pnl


def tune_momentum(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    initial_equity: float = 100000,
) -> Tuple[Dict[str, float], float]:
    """
    Tune Momentum hyperparameters on validation data.

    Grid search over:
    - rsi_oversold: [20, 25, 30, 35]
    - rsi_overbought: [65, 70, 75, 80]
    """
    best_pnl = float("-inf")
    best_params = {"rsi_oversold": 30, "rsi_overbought": 70}

    for oversold in [20, 25, 30, 35]:
        for overbought in [65, 70, 75, 80]:
            strategy = MomentumStrategy(
                rsi_oversold=oversold,
                rsi_overbought=overbought,
            )
            signals = strategy.generate_signals(features)
            result = backtest_signals(signals, prices, initial_equity=initial_equity)

            if result.total_pnl > best_pnl:
                best_pnl = result.total_pnl
                best_params = {"rsi_oversold": oversold, "rsi_overbought": overbought}

    return best_params, best_pnl


def tune_ticker_params(
    ticker: str,
    val_features: pd.DataFrame,
    val_prices: pd.DataFrame,
    initial_equity: float = 100000,
) -> TunedParams:
    """
    Tune all strategy parameters for a single ticker.

    Args:
        ticker: Symbol
        val_features: Validation features
        val_prices: Validation prices
        initial_equity: Initial equity for backtesting

    Returns:
        TunedParams with best parameters for each strategy
    """
    logger.info(f"  Tuning hyperparameters for {ticker}...")

    # Tune each strategy
    mr_params, mr_pnl = tune_mean_reversion(val_features, val_prices, initial_equity)
    mom_params, mom_pnl = tune_momentum(val_features, val_prices, initial_equity)

    # TrendFollowing doesn't have tunable params currently
    tf_params = {}
    strategy = TrendFollowingStrategy()
    signals = strategy.generate_signals(val_features)
    tf_result = backtest_signals(signals, val_prices, initial_equity=initial_equity)
    tf_pnl = tf_result.total_pnl

    # Find best strategy
    pnls = {"MeanReversion": mr_pnl, "Momentum": mom_pnl, "TrendFollowing": tf_pnl}
    best_strategy = max(pnls.items(), key=lambda x: x[1])[0]
    best_pnl = pnls[best_strategy]

    logger.info(
        f"    MeanReversion: z={mr_params['zscore_threshold']}, delta={mr_params['reversion_delta']} -> ${mr_pnl:,.0f}"
    )
    logger.info(
        f"    Momentum: oversold={mom_params['rsi_oversold']}, overbought={mom_params['rsi_overbought']} -> ${mom_pnl:,.0f}"
    )
    logger.info(f"    TrendFollowing: (default) -> ${tf_pnl:,.0f}")
    logger.info(f"    Best: {best_strategy}")

    return TunedParams(
        ticker=ticker,
        mean_reversion=mr_params,
        momentum=mom_params,
        trend_following=tf_params,
        best_strategy=best_strategy,
        validation_pnl=best_pnl,
    )


def tune_all_tickers(
    symbol_data: Dict[str, Any],
    calculate_data_split: callable,
    initial_equity: float = 100000,
) -> Dict[str, TunedParams]:
    """
    Tune hyperparameters for all tickers.

    Args:
        symbol_data: Dict mapping symbol -> SymbolData
        calculate_data_split: Function to calculate data split
        initial_equity: Initial equity

    Returns:
        Dict mapping ticker -> TunedParams
    """
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETER TUNING (Per-Ticker on Validation Data)")
    logger.info("=" * 60)

    tuned_params = {}

    for symbol, data in symbol_data.items():
        if data.features is None or data.features.empty:
            continue

        split = calculate_data_split(len(data.features))

        # Use validation data for tuning
        val_features = data.features.iloc[split.val_start : split.val_end]
        val_prices = data.ohlcv.iloc[split.val_start : split.val_end]

        if len(val_prices) < 20:
            logger.warning(
                f"  {symbol}: Insufficient validation data ({len(val_prices)} bars)"
            )
            continue

        tuned = tune_ticker_params(symbol, val_features, val_prices, initial_equity)
        tuned_params[symbol] = tuned

    return tuned_params
