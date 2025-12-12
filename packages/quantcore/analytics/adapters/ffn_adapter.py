# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FFN adapter for portfolio analytics and performance metrics.

Provides:
- Comprehensive portfolio statistics (Sharpe, Sortino, Calmar, etc.)
- Factor analysis and attribution
- Drawdown analysis
- Performance tear sheet generation

FFN (Financial Functions) is a battle-tested library for portfolio analysis.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger


def compute_portfolio_stats_ffn(
    equity_curve: Union[pd.Series, List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    benchmark: Optional[Union[pd.Series, List[float]]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive portfolio statistics using ffn.

    Args:
        equity_curve: Portfolio equity values over time (indexed by date ideally)
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        periods_per_year: Trading periods per year (252 for daily, 52 for weekly)
        benchmark: Optional benchmark returns for relative metrics

    Returns:
        Dictionary with comprehensive performance metrics:
            - returns: Total return, CAGR, annualized return
            - risk: Volatility, max drawdown, VaR, CVaR
            - ratios: Sharpe, Sortino, Calmar, Information ratio
            - drawdown: Max DD, avg DD, recovery time
            - rolling: Rolling Sharpe, rolling vol
    """
    # Convert to pandas Series if needed
    if isinstance(equity_curve, (list, np.ndarray)):
        equity = pd.Series(equity_curve)
    else:
        equity = equity_curve.copy()

    # Ensure numeric
    equity = pd.to_numeric(equity, errors="coerce").dropna()

    if len(equity) < 2:
        return {"error": "Insufficient data points for statistics"}

    # Calculate returns
    returns = equity.pct_change().dropna()

    if len(returns) < 1:
        return {"error": "Could not calculate returns"}

    try:
        import ffn

        # Create PerformanceStats object
        perf = ffn.PerformanceStats(equity)

        # Extract all statistics
        stats = {
            # Return metrics
            "total_return": (
                float(perf.total_return)
                if hasattr(perf, "total_return")
                else _total_return(equity)
            ),
            "cagr": (
                float(perf.cagr)
                if hasattr(perf, "cagr")
                else _cagr(equity, periods_per_year)
            ),
            "daily_mean": float(returns.mean()),
            "daily_std": float(returns.std()),
            "annualized_return": float(returns.mean() * periods_per_year),
            "annualized_volatility": float(returns.std() * np.sqrt(periods_per_year)),
            # Risk metrics
            "max_drawdown": (
                float(perf.max_drawdown)
                if hasattr(perf, "max_drawdown")
                else _max_drawdown(equity)
            ),
            "avg_drawdown": (
                float(perf.avg_drawdown)
                if hasattr(perf, "avg_drawdown")
                else _avg_drawdown(equity)
            ),
            "avg_drawdown_days": (
                float(perf.avg_drawdown_days)
                if hasattr(perf, "avg_drawdown_days")
                else None
            ),
            # Risk-adjusted ratios
            "sharpe_ratio": (
                float(perf.daily_sharpe)
                if hasattr(perf, "daily_sharpe")
                else _sharpe_ratio(returns, risk_free_rate, periods_per_year)
            ),
            "sortino_ratio": (
                float(perf.daily_sortino)
                if hasattr(perf, "daily_sortino")
                else _sortino_ratio(returns, risk_free_rate, periods_per_year)
            ),
            "calmar_ratio": (
                float(perf.calmar)
                if hasattr(perf, "calmar")
                else _calmar_ratio(equity, periods_per_year)
            ),
            # Distribution metrics
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "positive_days_pct": float((returns > 0).sum() / len(returns) * 100),
            # VaR metrics
            "var_95": float(np.percentile(returns, 5)),
            "var_99": float(np.percentile(returns, 1)),
            "cvar_95": float(returns[returns <= np.percentile(returns, 5)].mean()),
            # Time metrics
            "num_periods": len(equity),
            "periods_per_year": periods_per_year,
        }

        # Add monthly stats if available
        try:
            monthly_returns = _resample_returns(returns, "M")
            if len(monthly_returns) > 1:
                stats["monthly_mean"] = float(monthly_returns.mean())
                stats["monthly_std"] = float(monthly_returns.std())
                stats["best_month"] = float(monthly_returns.max())
                stats["worst_month"] = float(monthly_returns.min())
                stats["positive_months_pct"] = float(
                    (monthly_returns > 0).sum() / len(monthly_returns) * 100
                )
        except Exception:
            pass

        # Add drawdown details
        dd_info = _drawdown_details(equity)
        stats.update(dd_info)

        return stats

    except ImportError:
        logger.warning("ffn not available, using internal calculations")
        return _compute_stats_internal(
            equity, returns, risk_free_rate, periods_per_year
        )


def _compute_stats_internal(
    equity: pd.Series,
    returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: int,
) -> Dict[str, Any]:
    """Internal implementation without ffn."""
    return {
        "total_return": _total_return(equity),
        "cagr": _cagr(equity, periods_per_year),
        "daily_mean": float(returns.mean()),
        "daily_std": float(returns.std()),
        "annualized_return": float(returns.mean() * periods_per_year),
        "annualized_volatility": float(returns.std() * np.sqrt(periods_per_year)),
        "max_drawdown": _max_drawdown(equity),
        "avg_drawdown": _avg_drawdown(equity),
        "sharpe_ratio": _sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": _sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": _calmar_ratio(equity, periods_per_year),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "positive_days_pct": float((returns > 0).sum() / len(returns) * 100),
        "var_95": float(np.percentile(returns, 5)),
        "var_99": float(np.percentile(returns, 1)),
        "cvar_95": float(returns[returns <= np.percentile(returns, 5)].mean()),
        "num_periods": len(equity),
        "periods_per_year": periods_per_year,
        **_drawdown_details(equity),
    }


def compute_factor_stats_ffn(
    returns: Union[pd.Series, List[float], np.ndarray],
    benchmark_returns: Union[pd.Series, List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Compute factor/benchmark-relative statistics.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns (e.g., SPY)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Dictionary with relative metrics:
            - alpha, beta
            - information_ratio
            - tracking_error
            - up/down capture
            - correlation
    """
    # Convert to Series
    if isinstance(returns, (list, np.ndarray)):
        ret = pd.Series(returns)
    else:
        ret = returns.copy()

    if isinstance(benchmark_returns, (list, np.ndarray)):
        bench = pd.Series(benchmark_returns)
    else:
        bench = benchmark_returns.copy()

    # Align series
    common_idx = ret.index.intersection(bench.index)
    if len(common_idx) == 0:
        # Try positional alignment
        min_len = min(len(ret), len(bench))
        ret = ret.iloc[:min_len]
        bench = bench.iloc[:min_len]
    else:
        ret = ret.loc[common_idx]
        bench = bench.loc[common_idx]

    if len(ret) < 10:
        return {"error": "Insufficient overlapping data points"}

    # Convert to excess returns
    rf_period = risk_free_rate / periods_per_year
    excess_ret = ret - rf_period
    excess_bench = bench - rf_period

    # Beta and Alpha (CAPM regression)
    cov_matrix = np.cov(ret, bench)
    if cov_matrix[1, 1] > 0:
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        beta = 0.0

    alpha = ret.mean() - (rf_period + beta * (bench.mean() - rf_period))
    alpha_annualized = alpha * periods_per_year

    # Tracking error
    active_returns = ret - bench
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)

    # Information ratio
    if tracking_error > 0:
        information_ratio = active_returns.mean() * periods_per_year / tracking_error
    else:
        information_ratio = 0.0

    # Correlation
    correlation = ret.corr(bench)

    # Up/Down capture
    up_periods = bench > 0
    down_periods = bench < 0

    if up_periods.sum() > 0:
        up_capture = ret[up_periods].sum() / bench[up_periods].sum() * 100
    else:
        up_capture = 0.0

    if down_periods.sum() > 0:
        down_capture = ret[down_periods].sum() / bench[down_periods].sum() * 100
    else:
        down_capture = 0.0

    # R-squared
    r_squared = correlation**2

    # Treynor ratio
    if abs(beta) > 0.001:
        treynor_ratio = (ret.mean() * periods_per_year - risk_free_rate) / beta
    else:
        treynor_ratio = 0.0

    return {
        "alpha": float(alpha_annualized),
        "beta": float(beta),
        "correlation": float(correlation),
        "r_squared": float(r_squared),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "treynor_ratio": float(treynor_ratio),
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": (
            float(up_capture / down_capture) if down_capture != 0 else None
        ),
        "active_return": float(active_returns.mean() * periods_per_year),
    }


def generate_tearsheet_data(
    equity_curve: Union[pd.Series, List[float], np.ndarray],
    benchmark: Optional[Union[pd.Series, List[float]]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Generate comprehensive tearsheet data for reporting.

    Args:
        equity_curve: Portfolio equity values
        benchmark: Optional benchmark for comparison
        risk_free_rate: Risk-free rate
        periods_per_year: Periods per year

    Returns:
        Dictionary with all data needed for a performance tearsheet:
            - summary_stats: Key metrics
            - monthly_returns: Monthly return table
            - drawdown_periods: List of drawdown periods
            - rolling_metrics: Rolling Sharpe, vol, etc.
            - return_distribution: Histogram data
            - factor_exposure: If benchmark provided
    """
    # Convert inputs
    if isinstance(equity_curve, (list, np.ndarray)):
        equity = pd.Series(equity_curve)
    else:
        equity = equity_curve.copy()

    equity = pd.to_numeric(equity, errors="coerce").dropna()
    returns = equity.pct_change().dropna()

    # Summary statistics
    summary = compute_portfolio_stats_ffn(
        equity, risk_free_rate, periods_per_year, benchmark
    )

    # Monthly returns table
    try:
        monthly_returns = _resample_returns(returns, "M")
        monthly_table = _build_monthly_table(monthly_returns)
    except Exception:
        monthly_table = {}

    # Drawdown periods
    dd_periods = _get_drawdown_periods(equity)

    # Rolling metrics (21-day rolling window)
    rolling_window = min(21, len(returns) // 2)
    if rolling_window > 5:
        rolling_sharpe = _rolling_sharpe(
            returns, rolling_window, risk_free_rate, periods_per_year
        )
        rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(periods_per_year)
    else:
        rolling_sharpe = pd.Series()
        rolling_vol = pd.Series()

    # Return distribution
    hist_counts, hist_edges = np.histogram(returns, bins=50)
    return_dist = {
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
    }

    result = {
        "summary_stats": summary,
        "monthly_table": monthly_table,
        "drawdown_periods": dd_periods,
        "rolling_sharpe": rolling_sharpe.tolist() if len(rolling_sharpe) > 0 else [],
        "rolling_volatility": rolling_vol.tolist() if len(rolling_vol) > 0 else [],
        "return_distribution": return_dist,
        "equity_curve": equity.tolist(),
        "cumulative_returns": ((1 + returns).cumprod() - 1).tolist(),
    }

    # Add factor exposure if benchmark provided
    if benchmark is not None:
        if isinstance(benchmark, (list, np.ndarray)):
            bench_ret = pd.Series(benchmark).pct_change().dropna()
        else:
            bench_ret = benchmark.pct_change().dropna()

        # Align
        min_len = min(len(returns), len(bench_ret))
        factor_stats = compute_factor_stats_ffn(
            returns.iloc[:min_len],
            bench_ret.iloc[:min_len],
            risk_free_rate,
            periods_per_year,
        )
        result["factor_exposure"] = factor_stats

    return result


# ============================================================================
# Internal helper functions
# ============================================================================


def _total_return(equity: pd.Series) -> float:
    """Calculate total return."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) - 1)


def _cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Calculate CAGR."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0

    total_ret = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / periods_per_year

    if n_years <= 0 or total_ret <= 0:
        return 0.0

    return float(total_ret ** (1 / n_years) - 1)


def _max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown."""
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _avg_drawdown(equity: pd.Series) -> float:
    """Calculate average drawdown."""
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    dd_values = drawdown[drawdown < 0]
    if len(dd_values) == 0:
        return 0.0
    return float(dd_values.mean())


def _sharpe_ratio(
    returns: pd.Series, risk_free_rate: float, periods_per_year: int
) -> float:
    """Calculate Sharpe ratio."""
    if returns.std() == 0:
        return 0.0

    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    return float(excess_returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _sortino_ratio(
    returns: pd.Series, risk_free_rate: float, periods_per_year: int
) -> float:
    """Calculate Sortino ratio."""
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period

    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = np.sqrt((downside_returns**2).mean())
    return float(excess_returns.mean() / downside_std * np.sqrt(periods_per_year))


def _calmar_ratio(equity: pd.Series, periods_per_year: int) -> float:
    """Calculate Calmar ratio."""
    cagr = _cagr(equity, periods_per_year)
    max_dd = abs(_max_drawdown(equity))

    if max_dd == 0:
        return 0.0
    return float(cagr / max_dd)


def _drawdown_details(equity: pd.Series) -> Dict[str, Any]:
    """Get detailed drawdown information."""
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    # Find max drawdown point
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()

    # Find peak before max DD
    peak_idx = running_max.loc[:max_dd_idx].idxmax()

    # Find recovery (if any)
    try:
        recovery_idx = equity.loc[max_dd_idx:][
            equity.loc[max_dd_idx:] >= running_max.loc[max_dd_idx]
        ].index[0]
        recovery_duration = len(equity.loc[max_dd_idx:recovery_idx])
    except (IndexError, KeyError):
        recovery_idx = None
        recovery_duration = None

    return {
        "max_dd_start": str(peak_idx) if peak_idx is not None else None,
        "max_dd_trough": str(max_dd_idx) if max_dd_idx is not None else None,
        "max_dd_recovery": str(recovery_idx) if recovery_idx is not None else None,
        "max_dd_duration": (
            len(equity.loc[peak_idx:max_dd_idx]) if peak_idx is not None else None
        ),
        "max_dd_recovery_duration": recovery_duration,
    }


def _resample_returns(returns: pd.Series, freq: str) -> pd.Series:
    """Resample returns to specified frequency."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        # Create a date range
        returns.index = pd.date_range(
            start="2020-01-01", periods=len(returns), freq="D"
        )

    return (1 + returns).resample(freq).prod() - 1


def _build_monthly_table(monthly_returns: pd.Series) -> Dict[str, Any]:
    """Build monthly return table by year."""
    if len(monthly_returns) == 0:
        return {}

    df = monthly_returns.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(
        values="return", index="year", columns="month", aggfunc="sum"
    )
    pivot["YTD"] = pivot.sum(axis=1)

    return pivot.to_dict()


def _get_drawdown_periods(equity: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N drawdown periods."""
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    # Find drawdown periods
    periods = []
    in_drawdown = False
    start_idx = None

    for i, (idx, dd) in enumerate(drawdown.items()):
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_idx = idx
        elif dd >= 0 and in_drawdown:
            in_drawdown = False
            period_dd = drawdown.loc[start_idx:idx]
            periods.append(
                {
                    "start": str(start_idx),
                    "trough": str(period_dd.idxmin()),
                    "end": str(idx),
                    "max_dd": float(period_dd.min()),
                    "duration": len(period_dd),
                }
            )

    # Sort by max drawdown and return top N
    periods.sort(key=lambda x: x["max_dd"])
    return periods[:top_n]


def _rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rf_period = risk_free_rate / periods_per_year
    excess = returns - rf_period

    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    return sharpe.dropna()
