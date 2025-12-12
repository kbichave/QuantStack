"""
Quant Research Metrics Integration.

Bridges quant_research/ modules into the trading pipelines.
Provides a unified API for signal evaluation and cost analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from loguru import logger

# Import from quant_research modules
from quantcore.research.stat_tests import (
    adf_test,
    lagged_cross_correlation,
    harvey_liu_correction,
)
from quantcore.research.alpha_decay import AlphaDecayAnalyzer
from quantcore.research.cost_model import TransactionCostModel
from quantcore.research.walkforward import WalkForwardValidator

# Import from signals module
from quantcore.signals.signal_evaluator import (
    SignalEvaluator,
    compute_ic,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_turnover,
)
from quantcore.signals.cost_adjuster import CostAdjuster, CostParams


@dataclass
class QuantResearchReport:
    """Comprehensive quant research metrics report."""

    # Signal quality
    ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0  # IC Information Ratio

    # Alpha decay
    alpha_decay_curve: pd.Series = field(default_factory=pd.Series)
    half_life_bars: int = 0

    # Performance (gross)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0

    # Performance (net of costs)
    sharpe_net: float = 0.0
    sortino_net: float = 0.0
    total_costs_bps: float = 0.0

    # Turnover
    annual_turnover: float = 0.0
    avg_holding_days: float = 0.0

    # Statistical tests
    is_stationary: bool = False
    adf_pvalue: float = 1.0

    # Cost breakdown
    commission_cost: float = 0.0
    spread_cost: float = 0.0
    impact_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "signal_quality": {
                "ic": self.ic,
                "ic_std": self.ic_std,
                "ic_ir": self.ic_ir,
                "half_life_bars": self.half_life_bars,
            },
            "performance_gross": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "annual_return": self.annual_return,
                "annual_volatility": self.annual_volatility,
            },
            "performance_net": {
                "sharpe_net": self.sharpe_net,
                "sortino_net": self.sortino_net,
                "total_costs_bps": self.total_costs_bps,
            },
            "turnover": {
                "annual_turnover": self.annual_turnover,
                "avg_holding_days": self.avg_holding_days,
            },
            "costs": {
                "commission_cost": self.commission_cost,
                "spread_cost": self.spread_cost,
                "impact_cost": self.impact_cost,
            },
            "statistical_tests": {
                "is_stationary": self.is_stationary,
                "adf_pvalue": self.adf_pvalue,
            },
        }


def run_signal_diagnostics(
    signal: pd.Series,
    returns: pd.Series,
    cost_bps: float = 5.0,
) -> QuantResearchReport:
    """
    Run comprehensive signal diagnostics.

    Args:
        signal: Position signal series
        returns: Asset return series
        cost_bps: Transaction cost in basis points

    Returns:
        QuantResearchReport with all metrics
    """
    report = QuantResearchReport()

    # Align data
    common_idx = signal.index.intersection(returns.index)
    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]

    if len(signal) < 50:
        logger.warning("Insufficient data for signal diagnostics")
        return report

    # 1. Signal quality metrics
    evaluator = SignalEvaluator()
    metrics = evaluator.evaluate(signal, returns, cost_bps=0)

    report.ic = evaluator.ic
    ic_series = evaluator.ic_series.dropna()
    report.ic_std = ic_series.std() if len(ic_series) > 0 else 0
    report.ic_ir = report.ic / report.ic_std if report.ic_std > 0 else 0

    # 2. Gross performance
    report.sharpe_ratio = metrics.sharpe_ratio
    report.sortino_ratio = metrics.sortino_ratio
    report.max_drawdown = metrics.max_drawdown
    report.annual_return = metrics.annualized_return
    report.annual_volatility = metrics.volatility

    # 3. Net performance
    net_metrics = evaluator.evaluate(signal, returns, cost_bps=cost_bps)
    report.sharpe_net = net_metrics.sharpe_ratio
    report.sortino_net = net_metrics.sortino_ratio

    # 4. Turnover analysis
    report.annual_turnover = compute_turnover(signal)
    position_changes = (signal.diff().abs() > 0.01).sum()
    report.avg_holding_days = len(signal) / max(position_changes, 1)

    # 5. Cost breakdown
    total_turnover = signal.diff().abs().sum()
    report.total_costs_bps = total_turnover * cost_bps
    report.commission_cost = total_turnover * 1.0  # 1 bps commission
    report.spread_cost = total_turnover * 2.0  # 2 bps spread
    report.impact_cost = total_turnover * 2.0  # 2 bps impact

    # 6. Alpha decay
    decay = run_alpha_decay_analysis(signal, returns)
    report.alpha_decay_curve = decay["decay_curve"]
    report.half_life_bars = decay["half_life"]

    # 7. Statistical tests
    try:
        adf_result = adf_test(returns)
        report.is_stationary = adf_result.is_stationary
        report.adf_pvalue = adf_result.p_value
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}")

    return report


def run_alpha_decay_analysis(
    signal: pd.Series,
    returns: pd.Series,
    max_lag: int = 20,
) -> Dict[str, Any]:
    """
    Analyze how alpha decays over holding periods.

    Args:
        signal: Position signal
        returns: Asset returns
        max_lag: Maximum holding period to test

    Returns:
        Dictionary with decay curve and half-life
    """
    # Compute IC at each lag
    ics = []
    for lag in range(1, max_lag + 1):
        forward_ret = returns.rolling(lag).sum().shift(-lag)
        ic = signal.corr(forward_ret)
        ics.append(ic if not np.isnan(ic) else 0)

    decay_curve = pd.Series(ics, index=range(1, max_lag + 1))

    # Find half-life (where IC drops to half of initial)
    initial_ic = abs(ics[0]) if len(ics) > 0 else 0
    half_life = max_lag

    for i, ic in enumerate(ics):
        if abs(ic) < initial_ic / 2:
            half_life = i + 1
            break

    return {
        "decay_curve": decay_curve,
        "half_life": half_life,
        "initial_ic": initial_ic,
    }


def compute_cost_adjusted_returns(
    signal: pd.Series,
    returns: pd.Series,
    commission_bps: float = 1.0,
    spread_bps: float = 2.0,
    impact_bps: float = 2.0,
) -> Dict[str, pd.Series]:
    """
    Compute returns with realistic transaction costs.

    Args:
        signal: Position signal
        returns: Asset returns
        commission_bps: Commission cost
        spread_bps: Bid-ask spread cost
        impact_bps: Market impact cost

    Returns:
        Dictionary with gross and net return series
    """
    # Gross returns
    gross_returns = signal.shift(1) * returns

    # Cost calculation
    turnover = signal.diff().abs()
    total_cost_bps = commission_bps + spread_bps + impact_bps
    costs = turnover * total_cost_bps / 10000

    # Net returns
    net_returns = gross_returns - costs

    # Cumulative
    cum_gross = (1 + gross_returns.fillna(0)).cumprod()
    cum_net = (1 + net_returns.fillna(0)).cumprod()

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "costs": costs,
        "cumulative_gross": cum_gross,
        "cumulative_net": cum_net,
        "total_cost": costs.sum() * 10000,  # In bps
    }


def run_walkforward_backtest(
    features: pd.DataFrame,
    labels: pd.Series,
    model_fn,
    train_window: int = 252,
    test_window: int = 21,
    cost_bps: float = 5.0,
) -> Dict[str, Any]:
    """
    Run walkforward backtest with rolling optimization.

    Args:
        features: Feature DataFrame
        labels: Label series (returns)
        model_fn: Function that trains model and returns predictions
        train_window: Training window size
        test_window: Out-of-sample test window
        cost_bps: Transaction costs

    Returns:
        Dictionary with walkforward results
    """
    predictions = pd.Series(index=features.index, dtype=float)

    for end in range(train_window, len(features), test_window):
        # Training data
        train_start = max(0, end - train_window)
        X_train = features.iloc[train_start:end]
        y_train = labels.iloc[train_start:end]

        # Test data
        test_end = min(end + test_window, len(features))
        X_test = features.iloc[end:test_end]

        # Train and predict
        try:
            preds = model_fn(X_train, y_train, X_test)
            predictions.iloc[end:test_end] = preds
        except Exception as e:
            logger.warning(f"Model training failed at {end}: {e}")
            predictions.iloc[end:test_end] = 0

    # Evaluate
    report = run_signal_diagnostics(predictions, labels, cost_bps)

    return {
        "predictions": predictions,
        "report": report,
        "n_windows": (len(features) - train_window) // test_window,
    }


def compare_strategies(
    strategies: Dict[str, pd.Series],
    returns: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compare multiple strategies.

    Args:
        strategies: Dict of strategy name -> signal series
        returns: Asset returns
        cost_bps: Transaction costs

    Returns:
        DataFrame comparing strategies
    """
    results = []

    for name, signal in strategies.items():
        report = run_signal_diagnostics(signal, returns, cost_bps)

        results.append(
            {
                "strategy": name,
                "ic": report.ic,
                "sharpe_gross": report.sharpe_ratio,
                "sharpe_net": report.sharpe_net,
                "sortino_net": report.sortino_net,
                "max_dd": report.max_drawdown,
                "turnover": report.annual_turnover,
                "alpha_halflife": report.half_life_bars,
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("strategy")


def generate_quant_report_section(
    signal: pd.Series,
    returns: pd.Series,
    strategy_name: str = "Strategy",
    cost_bps: float = 5.0,
) -> str:
    """
    Generate text report section for quant metrics.

    Args:
        signal: Position signal
        returns: Asset returns
        strategy_name: Name for the strategy
        cost_bps: Transaction costs

    Returns:
        Formatted report string
    """
    report = run_signal_diagnostics(signal, returns, cost_bps)

    lines = [
        f"\n{'='*60}",
        f"QUANT RESEARCH METRICS: {strategy_name}",
        f"{'='*60}",
        "",
        "SIGNAL QUALITY:",
        f"  Information Coefficient (IC): {report.ic:.4f}",
        f"  IC Std Dev: {report.ic_std:.4f}",
        f"  IC Information Ratio: {report.ic_ir:.2f}",
        f"  Alpha Half-Life: {report.half_life_bars} bars",
        "",
        "PERFORMANCE (GROSS):",
        f"  Sharpe Ratio: {report.sharpe_ratio:.2f}",
        f"  Sortino Ratio: {report.sortino_ratio:.2f}",
        f"  Max Drawdown: {report.max_drawdown:.2%}",
        f"  Annual Return: {report.annual_return:.2%}",
        f"  Annual Volatility: {report.annual_volatility:.2%}",
        "",
        "PERFORMANCE (NET OF COSTS):",
        f"  Sharpe Ratio (Net): {report.sharpe_net:.2f}",
        f"  Sortino Ratio (Net): {report.sortino_net:.2f}",
        f"  Total Costs: {report.total_costs_bps:.1f} bps",
        "",
        "TURNOVER:",
        f"  Annual Turnover: {report.annual_turnover:.2f}x",
        f"  Avg Holding Period: {report.avg_holding_days:.1f} days",
        "",
        "STATISTICAL TESTS:",
        f"  Returns Stationary: {report.is_stationary}",
        f"  ADF p-value: {report.adf_pvalue:.4f}",
        f"{'='*60}",
    ]

    return "\n".join(lines)
