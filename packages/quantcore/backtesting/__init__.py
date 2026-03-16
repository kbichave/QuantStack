"""Backtesting module for WTI trading strategies."""

from quantcore.backtesting.costs import ProductionCostModel
from quantcore.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    calculate_metrics,
    run_backtest,
    run_backtest_with_params,
)
from quantcore.backtesting.realistic_engine import (
    FillRecord,
    OrderBookSnapshot,
    RealisticBacktestConfig,
    RealisticBacktestEngine,
    RealisticBacktestResult,
)
from quantcore.backtesting.reports import PerformanceReport
from quantcore.backtesting.strategies import (
    backtest_bollinger_bands,
    backtest_buy_hold,
    backtest_changepoint_strategy,
    backtest_ensemble_strategy,
    backtest_hmm_strategy,
    backtest_macd_strategy,
    backtest_momentum_strategy,
    backtest_rl_enhanced_strategy,
    backtest_rl_spread_strategy,
    backtest_rsi_strategy,
    backtest_sma_crossover,
    backtest_spread_strategy,
    backtest_tft_strategy,
    run_strategy_comparison,
)

__all__ = [
    # Standard backtesting
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "PerformanceReport",
    "ProductionCostModel",
    # Realistic backtesting with microstructure
    "RealisticBacktestConfig",
    "RealisticBacktestEngine",
    "RealisticBacktestResult",
    "OrderBookSnapshot",
    "FillRecord",
    "run_backtest_with_params",
    "run_backtest",
    "calculate_metrics",
    "backtest_spread_strategy",
    "backtest_sma_crossover",
    "backtest_bollinger_bands",
    "backtest_rsi_strategy",
    "backtest_momentum_strategy",
    "backtest_macd_strategy",
    "backtest_hmm_strategy",
    "backtest_changepoint_strategy",
    "backtest_tft_strategy",
    "backtest_ensemble_strategy",
    "backtest_buy_hold",
    "backtest_rl_spread_strategy",
    "backtest_rl_enhanced_strategy",
    "run_strategy_comparison",
]
