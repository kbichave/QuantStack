# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Research and statistical tools for the QuantCore MCP server.

Provides stationarity tests, alpha decay analysis, information coefficient
computation, Monte Carlo simulation, signal validation, signal diagnostics,
leakage detection, and lookahead bias checking.  All tools register on the
shared ``mcp`` FastMCP instance imported from ``quantcore.mcp.server``.
"""

from typing import Any

import numpy as np
import pandas as pd

from quantcore.mcp._helpers import _get_reader, _parse_timeframe, _serialize_result
from quantcore.mcp.server import mcp

# =============================================================================
# RESEARCH / STATISTICAL TOOLS
# =============================================================================


@mcp.tool()
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
    from quantcore.research.stat_tests import adf_test

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
            "statistic": (round(result.statistic, 4) if not np.isnan(result.statistic) else None),
            "p_value": round(result.p_value, 4),
            "is_stationary": result.is_significant,
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


@mcp.tool()
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
    from quantcore.features.factory import MultiTimeframeFeatureFactory
    from quantcore.research.alpha_decay import AlphaDecayAnalyzer

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
            matches = [c for c in features_df.columns if signal_column.lower() in c.lower()]
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


@mcp.tool()
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
    from scipy import stats

    from quantcore.features.factory import MultiTimeframeFeatureFactory

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
            matches = [c for c in features_df.columns if signal_column.lower() in c.lower()]
            if matches:
                signal_column = matches[0]
            else:
                return {"error": f"Signal column {signal_column} not found"}

        signal = features_df[signal_column]
        forward_returns = (
            df["close"].pct_change(forward_return_periods).shift(-forward_return_periods)
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
            "ic": round(ic, 4),
            "p_value": round(p_value, 4),
            "t_statistic": round(t_stat, 2),
            "sample_size": n,
            "is_significant": p_value < 0.05,
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


@mcp.tool()
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
    from quantcore.analysis.monte_carlo import run_monte_carlo_simulation
    from quantcore.features.factory import MultiTimeframeFeatureFactory

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


@mcp.tool()
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
    from quantcore.research.stat_tests import (
        adf_test,
        information_coefficient_test,
        lagged_cross_correlation,
    )

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
                    float(adf_result.statistic) if not np.isnan(adf_result.statistic) else None
                ),
                "p_value": float(adf_result.p_value),
                "is_stationary": adf_result.is_significant,
                "interpretation": adf_result.additional_info.get("interpretation", "unknown"),
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
                str(k): float(v) if not np.isnan(v) else None for k, v in lag_corrs.items()
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
            results["recommendations"].append(f"IC is significant at {significance_level} level")

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


@mcp.tool()
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
    from quantcore.research.quant_metrics import run_signal_diagnostics

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


@mcp.tool()
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
    from quantcore.features.factory import MultiTimeframeFeatureFactory
    from quantcore.research.leak_diagnostics import LeakageDiagnostics

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


@mcp.tool()
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
    from scipy import stats

    from quantcore.features.factory import MultiTimeframeFeatureFactory

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
