"""
Equity trading module.

Provides:
- Equity signal strategies (MeanReversion, Momentum, TrendFollowing, RRG, Composite)
- Simple equity backtester (100 shares, $0 commission)
- ML strategy for direction prediction
- RL strategy (PPO-based)
- Per-ticker hyperparameter tuning
- Pipeline orchestration
- Report generation
"""

from quantcore.equity.strategies import (
    EquityStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
    RRGStrategy,
    CompositeStrategy,
)

from quantcore.equity.backtester import (
    TradeRecord,
    BacktestResult,
    backtest_signals,
)

from quantcore.equity.reports import (
    TickerStrategyResult,
    StrategyResult,
    PipelineReport,
    generate_text_report,
)

from quantcore.equity.ml_strategy import run_ml_strategy

from quantcore.equity.tuning import (
    TunedParams,
    tune_ticker_params,
    tune_all_tickers,
)

# RL is optional (requires gymnasium + stable-baselines3)
try:
    from quantcore.equity.rl_strategy import (
        EquityTradingEnv,
        run_rl_strategy,
        RL_AVAILABLE,
    )
except ImportError:
    RL_AVAILABLE = False
    EquityTradingEnv = None
    run_rl_strategy = None

from quantcore.equity.pipeline import (
    run_pipeline,
    run_rule_based_strategies,
)

__all__ = [
    # Strategies
    "EquityStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "TrendFollowingStrategy",
    "RRGStrategy",
    "CompositeStrategy",
    # Backtester
    "TradeRecord",
    "BacktestResult",
    "backtest_signals",
    # Reports
    "TickerStrategyResult",
    "StrategyResult",
    "PipelineReport",
    "generate_text_report",
    # ML
    "run_ml_strategy",
    # RL
    "RL_AVAILABLE",
    "EquityTradingEnv",
    "run_rl_strategy",
    # Tuning
    "TunedParams",
    "tune_ticker_params",
    "tune_all_tickers",
    # Pipeline
    "run_pipeline",
    "run_rule_based_strategies",
]
