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

from quantstack.core.equity.backtester import (
    BacktestResult,
    TradeRecord,
    backtest_signals,
)
from quantstack.core.equity.ml_strategy import run_ml_strategy
from quantstack.core.equity.reports import (
    PipelineReport,
    StrategyResult,
    TickerStrategyResult,
    generate_text_report,
)
from quantstack.core.equity.strategies import (
    CompositeStrategy,
    EquityStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RRGStrategy,
    TrendFollowingStrategy,
)
from quantstack.core.equity.tuning import (
    TunedParams,
    tune_all_tickers,
    tune_ticker_params,
)

# RL is optional (requires gymnasium + stable-baselines3)
try:
    from quantstack.core.equity.rl_strategy import (
        RL_AVAILABLE,
        EquityTradingEnv,
        run_rl_strategy,
    )
except ImportError:
    RL_AVAILABLE = False
    EquityTradingEnv = None
    run_rl_strategy = None

from quantstack.core.equity.pipeline import (
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
