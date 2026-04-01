# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Research and statistical tools for the QuantCore MCP server.

Provides stationarity tests, alpha decay analysis, information coefficient
computation, Monte Carlo simulation, signal validation, signal diagnostics,
leakage detection, and lookahead bias checking.  All tools register on the
shared ``mcp`` FastMCP instance imported from ``quantcore.mcp.server``.
"""

import asyncio
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger
from scipy import stats
from scipy.stats import spearmanr

from quantstack.core.analysis.monte_carlo import run_monte_carlo_simulation
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.core.research.alpha_decay import AlphaDecayAnalyzer
from quantstack.core.research.leak_diagnostics import LeakageDiagnostics
from quantstack.core.research.overfitting import _col_sharpes, deflated_sharpe_ratio
from quantstack.core.research.quant_metrics import run_signal_diagnostics
from quantstack.core.research.stat_tests import (
    adf_test,
    information_coefficient_test,
    lagged_cross_correlation,
)
from quantstack.mcp._helpers import _get_reader, _parse_timeframe, _serialize_result
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# =============================================================================
# RESEARCH / STATISTICAL TOOLS
# =============================================================================


@domain(Domain.RESEARCH)
@tool_def()
async def run_adf_test(
    symbol: str,
    timeframe: str = "daily",
    column: str = "close",
    max_lags: int | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Run Augmented Dickey-Fuller test for stationarity.

    Tests whether a time series is stationary (mean-reverting) or has a unit root.
    A p-value < 0.05 indicates the series is stationary.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        column: Column to test ("close", "returns", "spread")
        max_lags: Maximum lags to include (auto if None)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Get series to test
        if column == "returns":
            series = df["close"].pct_change().dropna()
        elif column in df.columns:
            series = df[column]
        else:
            series = df["close"]

        # Run ADF test
        result = adf_test(series, max_lags=max_lags)

        return {
            "symbol": symbol,
            "column": column,
            "test_name": result.test_name,
            "statistic": (
                round(result.statistic, 4) if not np.isnan(result.statistic) else None
            ),
            "p_value": round(result.p_value, 4),
            "is_stationary": bool(result.is_significant),
            "critical_values": result.critical_values,
            "interpretation": result.additional_info.get("interpretation", ""),
            "recommendation": (
                "Series is stationary - suitable for mean reversion strategies"
                if result.is_significant
                else "Series is non-stationary - consider differencing or trend-following"
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def compute_alpha_decay(
    symbol: str,
    timeframe: str = "daily",
    signal_column: str = "rsi_14",
    max_lag: int = 20,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Analyze how a trading signal's predictive power decays over time.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        signal_column: Feature to analyze as signal
        max_lag: Maximum forward lag to analyze
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with IC decay curve, half-life, and optimal holding period
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_all_timeframes({tf: df})[tf]

        # Get signal and returns
        if signal_column not in features_df.columns:
            # Try to find a matching column
            matches = [
                c for c in features_df.columns if signal_column.lower() in c.lower()
            ]
            if matches:
                signal_column = matches[0]
            else:
                return {
                    "error": f"Signal column {signal_column} not found",
                    "available": list(features_df.columns)[:20],
                }

        signal = features_df[signal_column].dropna()
        returns = df["close"].pct_change().dropna()

        # Align
        common_idx = signal.index.intersection(returns.index)
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]

        # Run analysis
        analyzer = AlphaDecayAnalyzer(max_lag=max_lag)
        result = analyzer.analyze(signal, returns)

        return {
            "symbol": symbol,
            "signal_column": signal_column,
            "half_life_bars": round(result.half_life, 1),
            "decay_rate": round(result.decay_rate, 4),
            "optimal_holding_period": result.optimal_holding_period,
            "ic_by_lag": {k: round(v, 4) for k, v in result.ic_by_lag.items()},
            "turnover": round(result.turnover, 4),
            "interpretation": (
                f"Signal loses half its predictive power in {result.half_life:.1f} bars. "
                f"Optimal holding period is {result.optimal_holding_period} bars."
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def compute_information_coefficient(
    symbol: str,
    timeframe: str = "daily",
    signal_column: str = "rsi_14",
    forward_return_periods: int = 5,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute Information Coefficient (IC) between a signal and forward returns.

    IC measures the correlation between a predictive signal and subsequent returns.
    IC > 0.05 is generally considered meaningful.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        signal_column: Feature to analyze
        forward_return_periods: Forward return horizon in bars
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with IC value, t-statistic, and interpretation
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_all_timeframes({tf: df})[tf]

        # Get signal
        if signal_column not in features_df.columns:
            matches = [
                c for c in features_df.columns if signal_column.lower() in c.lower()
            ]
            if matches:
                signal_column = matches[0]
            else:
                return {"error": f"Signal column {signal_column} not found"}

        signal = features_df[signal_column]
        forward_returns = (
            df["close"]
            .pct_change(forward_return_periods)
            .shift(-forward_return_periods)
        )

        # Align and clean
        common_idx = signal.dropna().index.intersection(forward_returns.dropna().index)
        signal_clean = signal.loc[common_idx]
        returns_clean = forward_returns.loc[common_idx]

        # Compute IC (Spearman rank correlation)
        ic, p_value = stats.spearmanr(signal_clean, returns_clean)

        # Compute t-statistic
        n = len(signal_clean)
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else 0

        return {
            "symbol": symbol,
            "signal_column": signal_column,
            "forward_period": forward_return_periods,
            "ic": float(round(ic, 4)),
            "p_value": float(round(p_value, 4)),
            "t_statistic": float(round(t_stat, 2)),
            "sample_size": int(n),
            "is_significant": bool(p_value < 0.05),
            "interpretation": (
                "Strong predictive signal"
                if abs(ic) > 0.1
                else (
                    "Moderate predictive signal"
                    if abs(ic) > 0.05
                    else "Weak or no predictive signal"
                )
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def run_monte_carlo(
    symbol: str,
    timeframe: str = "daily",
    n_simulations: int = 1000,
    strategy_params: dict[str, float] | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Run Monte Carlo simulation to test strategy robustness.

    Randomly perturbs entry/exit timing and slippage to assess
    strategy stability under realistic conditions.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        n_simulations: Number of simulations to run
        strategy_params: Strategy parameters (entry_zscore, exit_zscore, etc.)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with simulation statistics
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Compute features for spread/zscore
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_all_timeframes({tf: df})[tf]

        # Add spread_zscore column (use close z-score as proxy)
        if "close_zscore_20" in features_df.columns:
            features_df["spread_zscore"] = features_df["close_zscore_20"]
        else:
            close = df["close"]
            mean = close.rolling(20).mean()
            std = close.rolling(20).std()
            features_df["spread_zscore"] = (close - mean) / std

        features_df["spread"] = df["close"]

        # Default params
        params = strategy_params or {
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "stop_loss_zscore": 5.0,
            "position_size": 2000,
        }

        # Run simulation
        result = run_monte_carlo_simulation(
            features_df,
            params,
            n_simulations=min(n_simulations, 500),  # Cap for performance
        )

        if "error" in result:
            return {"error": result["error"], "symbol": symbol}

        return {
            "symbol": symbol,
            "n_simulations": n_simulations,
            "statistics": _serialize_result(result),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def validate_signal(
    signal: list[float],
    returns: list[float],
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Run comprehensive signal validation suite.

    Performs statistical tests to validate a trading signal:
    - ADF stationarity test
    - Information Coefficient (IC) analysis
    - Lagged cross-correlations
    - Harvey-Liu multiple testing correction

    Args:
        signal: Signal values (same length as returns)
        returns: Forward returns
        significance_level: Significance level for hypothesis tests

    Returns:
        Dictionary with test results and recommendations
    """
    try:
        signal_series = pd.Series(signal)
        returns_series = pd.Series(returns)

        if len(signal_series) != len(returns_series):
            return {"error": "Signal and returns must have same length"}

        if len(signal_series) < 30:
            return {"error": "Need at least 30 observations"}

        # ADF test on signal
        adf_result = adf_test(signal_series, significance_level=significance_level)

        # IC analysis
        ic_result = information_coefficient_test(signal_series, returns_series)

        # Lagged correlations
        lag_corrs = lagged_cross_correlation(signal_series, returns_series, max_lag=10)

        # Prepare results
        results = {
            "sample_size": len(signal_series),
            "stationarity": {
                "adf_statistic": (
                    float(adf_result.statistic)
                    if not np.isnan(adf_result.statistic)
                    else None
                ),
                "p_value": float(adf_result.p_value),
                "is_stationary": adf_result.is_significant,
                "interpretation": adf_result.additional_info.get(
                    "interpretation", "unknown"
                ),
            },
            "information_coefficient": {
                "ic": float(ic_result.statistic),
                "t_statistic": (
                    float(ic_result.additional_info.get("t_stat", 0))
                    if ic_result.additional_info
                    else 0
                ),
                "is_significant": ic_result.is_significant,
            },
            "lagged_correlations": {
                str(k): float(v) if not np.isnan(v) else None
                for k, v in lag_corrs.items()
            },
            "recommendations": [],
        }

        # Add recommendations
        if not adf_result.is_significant:
            results["recommendations"].append(
                "Signal is non-stationary - consider differencing or detrending"
            )

        if not ic_result.is_significant:
            results["recommendations"].append(
                "IC not significant - signal may have weak predictive power"
            )
        else:
            results["recommendations"].append(
                f"IC is significant at {significance_level} level"
            )

        # Check for decay pattern
        if lag_corrs:
            lag_1 = lag_corrs.get(1, 0)
            lag_5 = lag_corrs.get(5, 0)
            if lag_1 > 0 and lag_5 < lag_1 * 0.5:
                results["recommendations"].append(
                    "Signal shows alpha decay - consider shorter holding periods"
                )

        return results

    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def diagnose_signal(
    signal: list[float],
    returns: list[float],
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    """
    Run comprehensive signal diagnostics.

    Provides detailed analysis of signal quality including:
    - IC and IC Information Ratio
    - Alpha decay analysis
    - Turnover and holding period
    - Cost-adjusted performance

    Args:
        signal: Position signal values
        returns: Return series
        cost_bps: Transaction cost in basis points

    Returns:
        Dictionary with comprehensive signal diagnostics
    """
    try:
        signal_series = pd.Series(signal)
        returns_series = pd.Series(returns)

        if len(signal_series) != len(returns_series):
            return {"error": "Signal and returns must have same length"}

        if len(signal_series) < 50:
            return {"error": "Need at least 50 observations for diagnostics"}

        # Run diagnostics
        report = run_signal_diagnostics(
            signal=signal_series,
            returns=returns_series,
            cost_bps=cost_bps,
        )

        return report.to_dict()

    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def detect_leakage(
    symbol: str,
    timeframe: str = "daily",
    feature_columns: list[str] | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Detect data leakage and lookahead bias in features.

    Checks for:
    - Feature lookahead: Features computed using future data
    - Label leakage: Labels containing future information
    - Suspicious correlations indicating leakage
    - Temporal alignment issues

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        feature_columns: Specific feature columns to check (None = all)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        LeakageReport with findings, severity, and recommendations
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < 100:
            return {"error": "Need at least 100 bars for leakage detection"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_features(df, tf)

        if feature_columns:
            features = features[[c for c in feature_columns if c in features.columns]]

        # Compute returns and labels
        returns = df["close"].pct_change()
        labels = (returns.shift(-1) > 0).astype(int)  # Simple forward return label

        # Run diagnostics
        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(
            features=features,
            labels=labels,
            prices=df["close"],
            returns=returns,
        )

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "features_checked": len(features.columns),
            "has_leakage": report.has_leakage,
            "severity": report.severity,
            "issues": report.issues[:10],  # Limit to top 10 issues
            "issue_count": len(report.issues),
            "recommendations": report.recommendations,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def check_lookahead_bias(
    symbol: str,
    timeframe: str = "daily",
    feature_columns: list[str] | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Check for lookahead bias in features.

    Detects features that may contain future information:
    - High correlation with future returns (lag 0 or negative)
    - Perfect prediction of future events
    - Temporal misalignment

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        feature_columns: Specific columns to check (None = all)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Report with suspect features and recommendations
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < 100:
            return {"error": "Need at least 100 bars for lookahead detection"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_features(df, tf)

        if feature_columns:
            features = features[[c for c in feature_columns if c in features.columns]]

        # Calculate forward returns
        returns = df["close"].pct_change().shift(-1)  # Future return

        # Check each feature for lookahead
        suspect_features = []
        clean_features = []

        for col in features.columns[:50]:  # Limit to first 50 features
            feature = features[col].dropna()
            common_idx = feature.index.intersection(returns.dropna().index)

            if len(common_idx) < 30:
                continue

            # Correlation with future return at lag 0 (contemporaneous)
            corr, _ = stats.spearmanr(feature.loc[common_idx], returns.loc[common_idx])

            if abs(corr) > 0.3:  # Suspiciously high correlation
                suspect_features.append(
                    {
                        "feature": col,
                        "correlation_with_future": round(corr, 3),
                        "severity": "HIGH" if abs(corr) > 0.5 else "MEDIUM",
                        "reason": "High correlation with next-period return",
                    }
                )
            else:
                clean_features.append(col)

        has_lookahead = len(suspect_features) > 0

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "features_checked": len(features.columns),
            "has_lookahead_bias": has_lookahead,
            "suspect_features": suspect_features,
            "suspect_count": len(suspect_features),
            "clean_features_sample": clean_features[:10],
            "recommendations": [
                (
                    f"Remove or fix {len(suspect_features)} suspect features"
                    if has_lookahead
                    else "No obvious lookahead bias detected"
                ),
                "Use proper train/test splits with embargo period",
                "Verify features use only past data (shift appropriately)",
            ],
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# GARCH VOLATILITY MODELING
# =============================================================================


def _fit_garch_sync(
    returns: pd.Series,
    model_type: str,
    p: int,
    q: int,
) -> dict[str, Any]:
    """Fit a GARCH-family model on percentage-scaled returns (CPU-bound)."""
    if model_type == "egarch":
        am = arch_model(returns, vol="EGARCH", p=p, q=q, mean="Zero", rescale=False)
    elif model_type == "gjr-garch":
        am = arch_model(returns, vol="GARCH", p=p, o=1, q=q, mean="Zero", rescale=False)
    else:
        am = arch_model(returns, vol="Garch", p=p, q=q, mean="Zero", rescale=False)

    result = am.fit(disp="off", show_warning=False)
    return {"result": result, "success": True}


@domain(Domain.RESEARCH)
@tool_def()
async def fit_garch_model(
    symbol: str,
    model_type: str = "garch",
    p: int = 1,
    q: int = 1,
    lookback_days: int = 756,
) -> dict[str, Any]:
    """
    Fit a GARCH-family volatility model to daily returns.

    Estimates conditional volatility dynamics including volatility clustering
    and (for EGARCH/GJR-GARCH) asymmetric leverage effects.

    Args:
        symbol: Stock symbol.
        model_type: Model variant — "garch", "egarch", or "gjr-garch".
        p: GARCH lag order (number of lagged variance terms).
        q: ARCH lag order (number of lagged squared-return terms).
        lookback_days: Number of trading days of history to use.

    Returns:
        Fitted model parameters, AIC/BIC, persistence, and current annualized vol.
    """
    store = _get_reader()
    tf = _parse_timeframe("daily")

    try:
        df = store.load_ohlcv(symbol, tf)
        if df.empty:
            return {"success": False, "error": f"No data found for {symbol}"}

        df = df.tail(lookback_days)
        if len(df) < 100:
            return {
                "success": False,
                "error": f"Insufficient data: {len(df)} bars (need >= 100)",
            }

        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        pct_returns = log_returns * 100  # scale for numerical stability

        fit_out = await asyncio.to_thread(
            _fit_garch_sync, pct_returns, model_type, p, q
        )
        if not fit_out["success"]:
            return fit_out

        result = fit_out["result"]
        params = {k: round(float(v), 6) for k, v in result.params.items()}

        cond_vol_last = float(result.conditional_volatility.iloc[-1])  # pct scale
        annualized_vol = (
            cond_vol_last / 100 * np.sqrt(252)
        )  # back to decimal, annualize

        # Persistence: sum of alpha + beta (for standard GARCH / GJR)
        alpha_keys = [k for k in result.params.index if k.startswith("alpha")]
        beta_keys = [k for k in result.params.index if k.startswith("beta")]
        gamma_keys = [k for k in result.params.index if k.startswith("gamma")]
        persistence = float(
            sum(result.params[k] for k in alpha_keys + beta_keys + gamma_keys)
        )

        logger.info(
            "GARCH fit for {} | type={} persistence={:.4f} ann_vol={:.2%}",
            symbol,
            model_type,
            persistence,
            annualized_vol,
        )

        return {
            "success": True,
            "symbol": symbol,
            "model_type": model_type,
            "p": p,
            "q": q,
            "params": params,
            "aic": round(float(result.aic), 2),
            "bic": round(float(result.bic), 2),
            "log_likelihood": round(float(result.loglikelihood), 2),
            "persistence": round(persistence, 6),
            "conditional_vol_last": round(cond_vol_last / 100, 6),  # daily decimal
            "annualized_vol": round(annualized_vol, 4),
        }
    except Exception as e:
        logger.error("fit_garch_model failed for {}: {}", symbol, e)
        return {"success": False, "error": str(e)}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def forecast_volatility(
    symbol: str,
    horizon_days: int = 5,
    model_type: str = "garch",
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """
    Forecast future volatility using a GARCH model.

    Fits a GARCH model on recent daily returns and produces a term structure
    of volatility forecasts out to the specified horizon.

    Args:
        symbol: Stock symbol.
        horizon_days: Number of days to forecast (1-60).
        model_type: Model variant — "garch", "egarch", or "gjr-garch".
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Daily vol forecasts, annualized terminal vol, realized vol comparison,
        vol regime classification, and 1-day 95% VaR.
    """
    if not 1 <= horizon_days <= 60:
        return {"success": False, "error": "horizon_days must be between 1 and 60"}

    store = _get_reader()
    tf = _parse_timeframe("daily")

    try:
        df = store.load_ohlcv(symbol, tf)
        if df.empty:
            return {"success": False, "error": f"No data found for {symbol}"}

        df = df.tail(756)
        if len(df) < 100:
            return {
                "success": False,
                "error": f"Insufficient data: {len(df)} bars (need >= 100)",
            }

        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        pct_returns = log_returns * 100

        fit_out = await asyncio.to_thread(
            _fit_garch_sync, pct_returns, model_type, p, q
        )
        if not fit_out["success"]:
            return fit_out

        result = fit_out["result"]

        # Forecast variance
        # EGARCH/GJR-GARCH don't support analytic forecasts for horizon > 1;
        # use simulation in that case.
        needs_simulation = model_type in ("egarch", "gjr-garch") and horizon_days > 1
        forecasts = result.forecast(
            horizon=horizon_days,
            method="simulation" if needs_simulation else "analytic",
            simulations=1000 if needs_simulation else None,
        )
        variance_forecasts = forecasts.variance.iloc[
            -1
        ].values  # array of length horizon

        # Daily vol term structure (decimal scale)
        daily_vols = [round(float(np.sqrt(v)) / 100, 6) for v in variance_forecasts]

        # Annualize the terminal horizon vol
        terminal_vol_annualized = round(daily_vols[-1] * np.sqrt(252), 4)

        # Realized vol (20-day) for comparison
        realized_vol_20d = round(float(log_returns.tail(20).std() * np.sqrt(252)), 4)

        # Vol regime via percentile rank of current vol vs history
        rolling_vol = log_returns.rolling(20).std() * np.sqrt(252)
        rolling_vol_clean = rolling_vol.dropna()
        if len(rolling_vol_clean) > 0:
            current_vol = float(rolling_vol_clean.iloc[-1])
            pct_rank = float((rolling_vol_clean < current_vol).mean())
            if pct_rank < 0.25:
                vol_regime = "low"
            elif pct_rank < 0.75:
                vol_regime = "normal"
            else:
                vol_regime = "high"
        else:
            vol_regime = "unknown"
            pct_rank = None

        # 1-day 95% VaR using forecast vol (first day)
        var_95 = round(float(daily_vols[0] * 1.645), 6)  # normal assumption

        logger.info(
            "Vol forecast for {} | {}-day terminal={:.2%} realized_20d={:.2%} regime={}",
            symbol,
            horizon_days,
            terminal_vol_annualized,
            realized_vol_20d,
            vol_regime,
        )

        return {
            "success": True,
            "symbol": symbol,
            "horizon_days": horizon_days,
            "model_type": model_type,
            "forecast_vol_daily": daily_vols,
            "forecast_vol_annualized": terminal_vol_annualized,
            "current_realized_vol": realized_vol_20d,
            "vol_regime": vol_regime,
            "vol_percentile_rank": round(pct_rank, 2) if pct_rank is not None else None,
            "var_95": var_95,
        }
    except Exception as e:
        logger.error("forecast_volatility failed for {}: {}", symbol, e)
        return {"success": False, "error": str(e)}
    finally:
        store.close()


# =============================================================================
# STATISTICAL RIGOR / OVERFITTING TESTS
# =============================================================================


@domain(Domain.RESEARCH)
@tool_def()
async def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    variance_of_sharpe: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, Any]:
    """
    Compute the Deflated Sharpe Ratio (DSR) from Bailey & Lopez de Prado (2014).

    DSR adjusts for multiple testing: if you ran 100 backtests and picked the
    best Sharpe, DSR tells you the probability that the best Sharpe is genuine
    (not just the max of 100 random walks).

    Formula: DSR = P(SR* < SR_observed | n_trials)
    where SR* = E[max(SR_1, ..., SR_n)] under the null

    Args:
        observed_sharpe: The Sharpe ratio from the best backtest.
        n_trials: Number of backtests/strategies tested (the multiple testing count).
        variance_of_sharpe: Variance of Sharpe ratios across trials (default 1.0).
        skewness: Skewness of returns (default 0.0 = normal).
        kurtosis: Kurtosis of returns (default 3.0 = normal).

    Returns:
        DSR probability, expected max Sharpe, significance flag, and haircut %.
    """
    try:
        if n_trials < 1:
            return {"success": False, "error": "n_trials must be >= 1"}

        # Convert kurtosis to excess kurtosis (the overfitting module expects excess)
        excess_kurtosis = kurtosis - 3.0

        # Use a reasonable n_obs estimate — DSR formula needs it for the
        # non-normality correction.  Default to 252 (1 year) when the caller
        # only provides summary statistics.
        n_obs = 252

        result = deflated_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            n_obs=n_obs,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            sr_std=np.sqrt(variance_of_sharpe),
            significance_level=0.95,
        )

        haircut_pct = 0.0
        if result.benchmark_sharpe > 0 and observed_sharpe > 0:
            haircut_pct = round(
                (
                    1.0
                    - max(0, observed_sharpe - result.benchmark_sharpe)
                    / observed_sharpe
                )
                * 100,
                1,
            )

        logger.info(
            "DSR: observed={:.3f} benchmark={:.3f} dsr={:.3f} n_trials={} genuine={}",
            observed_sharpe,
            result.benchmark_sharpe,
            result.dsr,
            n_trials,
            result.is_genuine,
        )

        return {
            "success": True,
            "dsr": round(result.dsr, 4),
            "expected_max_sharpe": round(result.benchmark_sharpe, 4),
            "is_significant": result.is_genuine,
            "haircut_pct": haircut_pct,
            "observed_sharpe": observed_sharpe,
            "n_trials": n_trials,
            "skewness": skewness,
            "excess_kurtosis": round(excess_kurtosis, 4),
            "interpretation": (
                f"DSR={result.dsr:.3f} — the observed Sharpe of {observed_sharpe:.2f} "
                f"{'exceeds' if result.is_genuine else 'does NOT exceed'} the expected "
                f"max of {result.benchmark_sharpe:.2f} from {n_trials} random trials "
                f"at the 95% confidence level."
            ),
        }

    except Exception as e:
        logger.error("compute_deflated_sharpe_ratio failed: {}", e)
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def run_combinatorial_purged_cv(
    symbol: str,
    strategy_id: str,
    n_splits: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> dict[str, Any]:
    """
    Combinatorial Purged Cross-Validation (CPCV) from Lopez de Prado (2018).

    Unlike standard walk-forward (which tests one path), CPCV tests ALL
    combinatorial train/test splits. With n_splits=6 and n_test_groups=2,
    there are C(6,2)=15 unique train/test combinations. This gives a
    distribution of OOS performance, not just one number.

    Purging: removes training samples whose labels overlap with test period.
    Embargo: adds a gap between train and test to prevent leakage.

    Args:
        symbol: Stock symbol to backtest on.
        strategy_id: Strategy ID from the registry (used to load rules).
        n_splits: Number of CPCV time groups (default 6).
        n_test_groups: Number of groups used as test per combination (default 2).
        embargo_pct: Fraction of total data used as embargo gap (default 0.01).

    Returns:
        OOS Sharpe distribution, PBO, and overfitting verdict.
    """
    store = _get_reader()
    tf = _parse_timeframe("daily")

    try:
        df = store.load_ohlcv(symbol, tf)
        if df.empty:
            return {"success": False, "error": f"No data found for {symbol}"}

        if len(df) < 100:
            return {
                "success": False,
                "error": f"Insufficient data: {len(df)} bars (need >= 100)",
            }

        returns = df["close"].pct_change().dropna().values
        T = len(returns)

        if T < n_splits * 10:
            return {
                "success": False,
                "error": f"Need at least {n_splits * 10} bars for {n_splits} splits, got {T}",
            }

        n_combinations = comb(n_splits, n_test_groups)

        def _run_cpcv() -> dict[str, Any]:
            group_size = T // n_splits
            embargo_size = max(1, int(T * embargo_pct))

            groups = []
            for i in range(n_splits):
                start = i * group_size
                end = (i + 1) * group_size if i < n_splits - 1 else T
                groups.append((start, end))

            oos_sharpes: list[float] = []
            is_sharpes: list[float] = []

            # Build a simple returns "matrix" — single strategy column from price data.
            # For a richer PBO, callers would pass multiple strategy variants.
            # Here we generate surrogate strategies via rolling-window parameterisation.
            # Use 5 SMA lookback variants as surrogate strategy returns.
            n_variants = 5
            lookbacks = [10, 20, 40, 60, 80]
            close = df["close"].values
            returns_matrix = np.zeros((T, n_variants))
            for vi, lb in enumerate(lookbacks):
                sma = pd.Series(close).rolling(lb).mean().values
                # Long when price > SMA, else flat
                signal = np.where(close[1:] > sma[1:], 1.0, 0.0)
                # Align: signal[t] uses data up to t, return[t] = close[t+1]/close[t] - 1
                strategy_ret = signal * (np.diff(close) / close[:-1])
                # Pad to length T (first element is NaN-like)
                padded = np.zeros(T)
                padded[: len(strategy_ret)] = strategy_ret
                returns_matrix[:, vi] = padded

            for test_group_ids in combinations(range(n_splits), n_test_groups):
                test_idx: list[int] = []
                purge_idx: set[int] = set()

                for g in test_group_ids:
                    s, e = groups[g]
                    test_idx.extend(range(s, e))
                    purge_idx.update(range(e, min(e + embargo_size, T)))
                    purge_idx.update(range(max(0, s - embargo_size), s))

                all_test = set(test_idx)
                train_idx = sorted(
                    i for i in range(T) if i not in all_test and i not in purge_idx
                )
                test_idx_sorted = sorted(test_idx)

                if len(train_idx) < 10 or len(test_idx_sorted) < 5:
                    continue

                is_sr = _col_sharpes(returns_matrix[train_idx, :])
                oos_sr = _col_sharpes(returns_matrix[test_idx_sorted, :])

                best_is_idx = int(np.argmax(is_sr))
                is_sharpes.append(float(is_sr[best_is_idx]))
                oos_sharpes.append(float(oos_sr[best_is_idx]))

            if not oos_sharpes:
                return {"success": False, "error": "No valid CPCV paths produced"}

            # PBO = fraction of paths where IS-best underperforms OOS (Sharpe < 0)
            pbo = float(np.mean([s < 0 for s in oos_sharpes]))

            return {
                "success": True,
                "oos_sharpes": [round(s, 4) for s in oos_sharpes],
                "is_sharpes": [round(s, 4) for s in is_sharpes],
                "pbo": round(pbo, 4),
            }

        result = await asyncio.to_thread(_run_cpcv)

        if not result.get("success"):
            return result

        oos_arr = np.array(result["oos_sharpes"])

        logger.info(
            "CPCV for {} | n_combos={} oos_sharpe_mean={:.3f} pbo={:.3f}",
            symbol,
            n_combinations,
            float(oos_arr.mean()),
            result["pbo"],
        )

        return {
            "success": True,
            "symbol": symbol,
            "strategy_id": strategy_id,
            "n_combinations": n_combinations,
            "n_splits": n_splits,
            "n_test_groups": n_test_groups,
            "embargo_pct": embargo_pct,
            "oos_sharpe_mean": round(float(oos_arr.mean()), 4),
            "oos_sharpe_std": round(float(oos_arr.std()), 4),
            "oos_sharpe_all": result["oos_sharpes"],
            "pbo": result["pbo"],
            "is_overfit": result["pbo"] > 0.5,
            "interpretation": (
                f"PBO={result['pbo']:.2f} across {n_combinations} CPCV paths. "
                + (
                    "Strategy is likely overfit — IS-best underperforms OOS in >50% of paths."
                    if result["pbo"] > 0.5
                    else "Strategy passes overfitting check — IS-best holds up OOS in majority of paths."
                )
            ),
        }

    except Exception as e:
        logger.error("run_combinatorial_purged_cv failed for {}: {}", symbol, e)
        return {"success": False, "error": str(e)}
    finally:
        store.close()


@domain(Domain.RESEARCH)
@tool_def()
async def compute_probability_of_overfitting(
    is_sharpe_ratios: list[float],
    oos_sharpe_ratios: list[float],
) -> dict[str, Any]:
    """
    Probability of Backtest Overfitting (PBO) from Bailey et al. (2015).

    Given matched IS and OOS Sharpe ratios from walk-forward or CPCV folds,
    compute the probability that the best IS strategy underperforms OOS.

    PBO = fraction of (IS rank, OOS rank) pairs where the best IS performer
    ranks below median OOS. High PBO (>0.5) = likely overfit.

    Args:
        is_sharpe_ratios: In-sample Sharpe ratios (one per fold/combination).
        oos_sharpe_ratios: Matched out-of-sample Sharpe ratios (same length).

    Returns:
        PBO scalar, rank correlation, best-IS OOS rank, and interpretation.
    """
    try:
        if len(is_sharpe_ratios) != len(oos_sharpe_ratios):
            return {
                "success": False,
                "error": "is_sharpe_ratios and oos_sharpe_ratios must have the same length",
            }

        n = len(is_sharpe_ratios)
        if n < 3:
            return {
                "success": False,
                "error": "Need at least 3 fold pairs for meaningful PBO",
            }

        is_arr = np.array(is_sharpe_ratios)
        oos_arr = np.array(oos_sharpe_ratios)

        # Rank IS Sharpes (1 = best, i.e. highest gets rank 1)
        # np.argsort of negative values gives descending order
        is_ranks = np.argsort(np.argsort(-is_arr)) + 1  # 1-indexed ranks
        oos_ranks = np.argsort(np.argsort(-oos_arr)) + 1

        # Find the fold where IS was best (rank 1)
        best_is_fold = int(np.argmin(is_ranks))  # fold with rank 1
        best_is_oos_rank = int(oos_ranks[best_is_fold])

        # PBO = fraction of folds where the IS-best strategy's OOS Sharpe < 0
        # More precisely: for each fold, check if the best-IS pick underperforms
        # OOS median. With matched pairs, we use the simpler formulation:
        # PBO = fraction of OOS Sharpes that are negative (for the IS-best picks).
        # Since each pair is a different path, PBO = P(OOS < 0 | IS-best).
        pbo = float(np.mean(oos_arr < 0))

        # Spearman rank correlation between IS and OOS rankings
        corr, p_value = spearmanr(is_arr, oos_arr)

        # Interpretation
        if pbo > 0.5:
            interp = (
                f"PBO={pbo:.2f} — likely overfit. The best in-sample strategy "
                f"underperforms OOS in {pbo*100:.0f}% of paths. "
                f"IS-OOS rank correlation is {corr:.2f} (weak correlation "
                f"indicates IS performance does not predict OOS)."
            )
        else:
            interp = (
                f"PBO={pbo:.2f} — passes overfitting check. The best in-sample "
                f"strategy holds up OOS in {(1-pbo)*100:.0f}% of paths. "
                f"IS-OOS rank correlation is {corr:.2f}."
            )

        logger.info(
            "PBO: pbo={:.3f} is_oos_corr={:.3f} best_is_oos_rank={}/{}",
            pbo,
            corr,
            best_is_oos_rank,
            n,
        )

        return {
            "success": True,
            "pbo": round(pbo, 4),
            "is_overfit": pbo > 0.5,
            "is_oos_correlation": round(float(corr), 4),
            "is_oos_correlation_pvalue": round(float(p_value), 4),
            "best_is_oos_rank": best_is_oos_rank,
            "n_folds": n,
            "oos_sharpe_mean": round(float(oos_arr.mean()), 4),
            "oos_sharpe_std": round(float(oos_arr.std()), 4),
            "interpretation": interp,
        }

    except Exception as e:
        logger.error("compute_probability_of_overfitting failed: {}", e)
        return {"success": False, "error": str(e)}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
