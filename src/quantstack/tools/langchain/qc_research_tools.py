"""Research and statistical tools for LangGraph agents."""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field


@tool
async def run_adf_test(
    symbol: Annotated[str, Field(description="Ticker symbol to test for stationarity, e.g. 'AAPL', 'SPY', or a spread symbol")],
    timeframe: Annotated[str, Field(description="Candle interval for the time series: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    column: Annotated[str, Field(description="Column to test for unit root: 'close', 'returns', or 'spread' (for cointegration pairs)")] = "close",
    max_lags: Annotated[Optional[int], Field(description="Maximum number of lags to include in the ADF regression; auto-selected by AIC if None")] = None,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Calculates the Augmented Dickey-Fuller (ADF) test statistic and p-value to determine whether a time series is stationary or contains a unit root. Use when testing for mean reversion, validating cointegration residuals in stat-arb pairs, or checking stationarity assumptions before model fitting. A p-value below 0.05 indicates the series is stationary. Returns JSON with ADF test statistic, p-value, critical values, and interpretation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_alpha_decay(
    symbol: Annotated[str, Field(description="Ticker symbol to analyze alpha decay for, e.g. 'AAPL', 'SPY'")],
    timeframe: Annotated[str, Field(description="Candle interval for the analysis: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    signal_column: Annotated[str, Field(description="Feature column name to analyze as the predictive signal, e.g. 'rsi_14', 'macd_hist', 'ic_score'")] = "rsi_14",
    max_lag: Annotated[int, Field(description="Maximum forward lag (in bars) to compute the information coefficient decay curve")] = 20,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Calculates how a trading signal's predictive power (information coefficient) decays over successive forward lags to determine optimal holding period and signal half-life. Use when evaluating alpha persistence, sizing holding horizons for momentum or mean-reversion strategies, or comparing IC decay across candidate signals. Returns JSON with IC decay curve, alpha half-life, and optimal holding period recommendation.
    """
    try:
        import math
        import numpy as np
        from scipy.stats import spearmanr
        from quantstack.data.storage import DataStore

        store = DataStore()
        df = store.load_ohlcv(symbol.upper(), timeframe=timeframe)
        if df is None or len(df) < max_lag + 30:
            return json.dumps({"error": f"Insufficient data for {symbol} ({timeframe})"})

        if end_date:
            df = df[df.index <= end_date]

        if signal_column not in df.columns:
            return json.dumps({"error": f"Column '{signal_column}' not found. Available: {list(df.columns)}"})

        signal = df[signal_column].dropna().values
        close = df["close"].values

        decay_curve = []
        peak_ic = 0.0
        peak_lag = 1

        for lag in range(1, max_lag + 1):
            if len(signal) <= lag + 5:
                break
            fwd_return = (close[lag:] - close[:-lag]) / close[:-lag]
            sig = signal[:len(fwd_return)]
            mask = np.isfinite(sig) & np.isfinite(fwd_return)
            if mask.sum() < 10:
                continue
            corr, pval = spearmanr(sig[mask], fwd_return[mask])
            if not math.isfinite(corr):
                continue
            decay_curve.append({"lag": lag, "ic": round(float(corr), 6), "p_value": round(float(pval), 6)})
            if abs(corr) > abs(peak_ic):
                peak_ic = float(corr)
                peak_lag = lag

        # Estimate half-life: first lag where IC drops below peak/2
        half_life = None
        if decay_curve and abs(peak_ic) > 0.01:
            for pt in decay_curve:
                if pt["lag"] > peak_lag and abs(pt["ic"]) < abs(peak_ic) / 2:
                    half_life = pt["lag"]
                    break

        return json.dumps({
            "symbol": symbol,
            "signal_column": signal_column,
            "decay_curve": decay_curve,
            "peak_ic": round(peak_ic, 6),
            "peak_lag": peak_lag,
            "half_life_bars": half_life,
            "optimal_holding_period": peak_lag,
            "interpretation": (
                f"Peak IC={peak_ic:.4f} at lag {peak_lag}. "
                + (f"Half-life: {half_life} bars." if half_life else "IC does not decay below half within max_lag.")
            ),
            "n_observations": len(signal),
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, default=str)


@tool
async def compute_information_coefficient(
    symbol: Annotated[str, Field(description="Ticker symbol to compute IC for, e.g. 'AAPL', 'SPY', 'TSLA'")],
    timeframe: Annotated[str, Field(description="Candle interval for computing returns: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    signal_column: Annotated[str, Field(description="Feature column to correlate with forward returns, e.g. 'rsi_14', 'macd_hist', 'sentiment_score'")] = "rsi_14",
    forward_return_periods: Annotated[int, Field(description="Number of forward bars for return calculation, e.g. 5 for 5-day forward returns")] = 5,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Computes the Information Coefficient (IC) — rank correlation between a predictive signal and forward returns — to measure signal quality. Use when evaluating whether a feature (e.g. RSI, sentiment, factor score) has genuine predictive power for alpha generation. IC above 0.05 is generally meaningful; also provides t-statistic for statistical significance. Returns JSON with IC value, t-statistic, p-value, and interpretation.
    """
    try:
        import math
        import numpy as np
        from scipy.stats import spearmanr
        from quantstack.data.storage import DataStore

        store = DataStore()
        df = store.load_ohlcv(symbol.upper(), timeframe=timeframe)
        if df is None or len(df) < forward_return_periods + 30:
            return json.dumps({"error": f"Insufficient data for {symbol} ({timeframe})"})

        if end_date:
            df = df[df.index <= end_date]

        if signal_column not in df.columns:
            return json.dumps({"error": f"Column '{signal_column}' not found. Available: {list(df.columns)}"})

        signal = df[signal_column].dropna().values
        close = df["close"].values
        n = len(signal)
        fwd = forward_return_periods

        if n <= fwd + 5:
            return json.dumps({"error": f"Not enough data points ({n}) for {fwd}-period forward returns"})

        fwd_return = (close[fwd:] - close[:-fwd]) / close[:-fwd]
        sig = signal[:len(fwd_return)]
        mask = np.isfinite(sig) & np.isfinite(fwd_return)

        if mask.sum() < 20:
            return json.dumps({"error": f"Only {mask.sum()} valid observations — need 20+"})

        corr, pvalue = spearmanr(sig[mask], fwd_return[mask])
        if not math.isfinite(corr):
            return json.dumps({"error": "Non-finite correlation — likely constant signal"})

        n_obs = int(mask.sum())
        denom = math.sqrt(1 - corr**2) if abs(corr) < 1.0 else 1e-10
        t_stat = corr * math.sqrt(max(0, n_obs - 2)) / denom

        if abs(corr) > 0.05 and pvalue < 0.05:
            interpretation = f"Significant IC ({corr:.4f}, p={pvalue:.4f}). Signal has predictive power."
        elif abs(corr) > 0.02:
            interpretation = f"Weak IC ({corr:.4f}, p={pvalue:.4f}). Marginal signal — may not survive costs."
        else:
            interpretation = f"No meaningful IC ({corr:.4f}, p={pvalue:.4f}). Signal is noise."

        return json.dumps({
            "symbol": symbol,
            "signal_column": signal_column,
            "forward_return_periods": fwd,
            "ic": round(float(corr), 6),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(pvalue), 6),
            "n_observations": n_obs,
            "interpretation": interpretation,
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, default=str)


@tool
async def run_monte_carlo(
    symbol: Annotated[str, Field(description="Ticker symbol to run Monte Carlo simulation on, e.g. 'AAPL', 'SPY'")],
    timeframe: Annotated[str, Field(description="Candle interval for the backtest simulation: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    n_simulations: Annotated[int, Field(description="Number of Monte Carlo simulation paths to generate (higher = more precise confidence intervals)")] = 1000,
    strategy_params: Annotated[Optional[dict[str, float]], Field(description="Strategy parameter dict (e.g. entry_zscore, exit_zscore, stop_loss_pct) to perturb during simulation")] = None,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Runs Monte Carlo simulation to assess strategy robustness by randomly perturbing entry/exit timing, slippage, and parameter values across thousands of paths. Use when validating whether a backtest result is statistically robust or fragile, stress-testing a strategy under realistic execution conditions, or computing confidence intervals for Sharpe ratio and drawdown. Returns JSON with simulation statistics including percentile P&L, drawdown distribution, and robustness score.
    """
    try:
        from dataclasses import asdict

        from quantstack.core.analysis.bootstrap_mc import bootstrap_sharpe_ci
        from quantstack.tools._helpers import _get_reader

        reader = _get_reader()
        price_data = reader.get_prices(symbol.upper(), timeframe=timeframe, end_date=end_date)

        if price_data is None or price_data.empty or len(price_data) < 60:
            return json.dumps({"error": f"Insufficient price data for {symbol}"})

        returns = price_data["close"].pct_change().dropna()
        if len(returns) < 30:
            return json.dumps({"error": "Too few return observations for MC"})

        mc_result = bootstrap_sharpe_ci(
            returns,
            n_simulations=n_simulations,
        )

        output = asdict(mc_result)
        output["symbol"] = symbol.upper()
        output["gate_5th_pctile_passes"] = mc_result.ci_5 >= 0.3
        return json.dumps(output, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def validate_signal(
    signal: Annotated[list[float], Field(description="Array of signal values (predictions or scores) to validate; must be same length as returns")],
    returns: Annotated[list[float], Field(description="Array of forward returns corresponding to each signal observation")],
    significance_level: Annotated[float, Field(description="Significance level (alpha) for hypothesis tests, e.g. 0.05 for 95% confidence")] = 0.05,
) -> str:
    """Runs a comprehensive signal validation suite including ADF stationarity test, Information Coefficient (IC) analysis, lagged cross-correlations, and Harvey-Liu multiple testing correction to determine whether a trading signal has genuine predictive power. Use when you need to validate a new alpha signal before deploying it in a strategy, or when checking for spurious correlations. Returns JSON with test results, significance flags, and deployment recommendations.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def diagnose_signal(
    signal: Annotated[list[float], Field(description="Array of position signal values (e.g. -1 to +1 weights) to diagnose")],
    returns: Annotated[list[float], Field(description="Array of return series corresponding to each signal observation")],
    cost_bps: Annotated[float, Field(description="Transaction cost in basis points (e.g. 5.0 = 0.05%) for cost-adjusted performance calculation")] = 5.0,
) -> str:
    """Provides comprehensive signal diagnostics including IC, IC Information Ratio (ICIR), alpha decay analysis, turnover rate, implied holding period, and cost-adjusted net performance. Use when you need a full health check on a trading signal before strategy deployment, or when comparing signal quality across multiple candidate alphas. Calculates both gross and net-of-cost Sharpe ratio. Returns JSON with IC, ICIR, decay profile, turnover, holding period, and cost-adjusted metrics.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def detect_leakage(
    symbol: Annotated[str, Field(description="Ticker symbol to check for data leakage, e.g. 'AAPL', 'SPY'")],
    timeframe: Annotated[str, Field(description="Candle interval for the feature data: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    feature_columns: Annotated[Optional[list[str]], Field(description="Specific feature column names to audit for leakage; checks all features if None")] = None,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Detects data leakage and lookahead bias in feature pipelines by checking for feature lookahead (features computed with future data), label leakage, suspicious zero-lag correlations, and temporal alignment issues. Use when auditing ML training pipelines, validating backtest integrity, or diagnosing suspiciously high in-sample performance. Critical for preventing overfitting and ensuring out-of-sample validity. Returns JSON with leakage findings, severity levels, and remediation recommendations.
    """
    try:
        import numpy as np

        from quantstack.core.validation.leakage import LeakageDetector
        from quantstack.tools._helpers import _get_reader

        reader = _get_reader()
        price_data = reader.get_prices(symbol.upper(), timeframe=timeframe, end_date=end_date)

        if price_data is None or price_data.empty or len(price_data) < 50:
            return json.dumps({"error": f"Insufficient price data for {symbol}"})

        numeric_cols = price_data.select_dtypes(include=[np.number]).columns.tolist()
        if feature_columns:
            numeric_cols = [c for c in feature_columns if c in numeric_cols]
        if not numeric_cols:
            return json.dumps({"error": "No numeric feature columns found"})

        X = price_data[numeric_cols].dropna()
        y = price_data["close"].pct_change().shift(-1).reindex(X.index).dropna()
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]
        if len(X) < 50:
            return json.dumps({"error": "Too few rows after alignment"})

        detector = LeakageDetector()
        report = detector.run_all_tests(X, y)
        return json.dumps(report, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def check_lookahead_bias(
    symbol: Annotated[str, Field(description="Ticker symbol to audit for lookahead bias, e.g. 'AAPL', 'SPY'")],
    timeframe: Annotated[str, Field(description="Candle interval for the feature data: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    feature_columns: Annotated[Optional[list[str]], Field(description="Specific feature column names to check for lookahead; audits all features if None")] = None,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Checks for lookahead bias in feature columns by detecting high correlation with future returns at lag 0 or negative lags, perfect prediction of future events, and temporal misalignment between features and labels. Use when validating backtest pipelines, auditing ML features for leakage, or investigating suspiciously good model performance that may not generalize out-of-sample. Returns JSON with suspect features, bias severity, and remediation recommendations.
    """
    try:
        import numpy as np
        from scipy.stats import spearmanr

        from quantstack.core.validation.leakage import FeatureShiftTest
        from quantstack.tools._helpers import _get_reader

        reader = _get_reader()
        price_data = reader.get_prices(symbol.upper(), timeframe=timeframe, end_date=end_date)

        if price_data is None or price_data.empty or len(price_data) < 50:
            return json.dumps({"error": f"Insufficient price data for {symbol}"})

        numeric_cols = price_data.select_dtypes(include=[np.number]).columns.tolist()
        if feature_columns:
            numeric_cols = [c for c in feature_columns if c in numeric_cols]
        if not numeric_cols:
            return json.dumps({"error": "No numeric feature columns found"})

        fwd_ret = price_data["close"].pct_change().shift(-1)
        suspects: list[dict] = []

        for col in numeric_cols:
            series = price_data[col].dropna()
            common = series.index.intersection(fwd_ret.dropna().index)
            if len(common) < 30:
                continue
            corr, pval = spearmanr(series.loc[common], fwd_ret.loc[common])
            if abs(corr) > 0.3 and pval < 0.01:
                suspects.append({
                    "feature": col,
                    "lag0_correlation": round(float(corr), 4),
                    "p_value": round(float(pval), 6),
                    "severity": "CRITICAL" if abs(corr) > 0.6 else "WARNING",
                    "recommendation": (
                        "Likely lookahead — feature correlates strongly with "
                        "next-bar returns. Verify temporal alignment."
                    ),
                })

        # Also run statistical shift test
        X = price_data[numeric_cols].dropna()
        y = fwd_ret.reindex(X.index).dropna()
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]
        shift_findings = []
        if len(X) >= 50:
            shift_test = FeatureShiftTest()
            shift_result = shift_test.run(X, y)
            shift_findings = shift_result if isinstance(shift_result, list) else [shift_result]

        return json.dumps({
            "symbol": symbol.upper(),
            "features_checked": len(numeric_cols),
            "suspects": suspects,
            "shift_test_findings": shift_findings,
            "verdict": "CLEAN" if not suspects else "LOOKAHEAD_DETECTED",
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def fit_garch_model(
    symbol: Annotated[str, Field(description="Ticker symbol to fit the GARCH volatility model on, e.g. 'AAPL', 'SPY'")],
    model_type: Annotated[str, Field(description="GARCH model variant: 'garch' (symmetric), 'egarch' (exponential), or 'gjr-garch' (asymmetric leverage)")] = "garch",
    p: Annotated[int, Field(description="GARCH lag order — number of lagged conditional variance terms in the model")] = 1,
    q: Annotated[int, Field(description="ARCH lag order — number of lagged squared-return (innovation) terms in the model")] = 1,
    lookback_days: Annotated[int, Field(description="Number of trading days of historical returns to use for model fitting, e.g. 756 for ~3 years")] = 756,
) -> str:
    """Fits a GARCH-family conditional volatility model (GARCH, EGARCH, or GJR-GARCH) to daily returns, estimating volatility clustering dynamics and asymmetric leverage effects. Use when modeling time-varying volatility for options pricing, risk management, position sizing, or regime detection. Provides model selection via AIC/BIC and persistence diagnostics. Returns JSON with fitted parameters, AIC/BIC, volatility persistence, and current annualized vol estimate.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def forecast_volatility(
    symbol: Annotated[str, Field(description="Ticker symbol to forecast volatility for, e.g. 'AAPL', 'SPY', 'QQQ'")],
    horizon_days: Annotated[int, Field(description="Number of trading days to forecast forward (1-60), e.g. 5 for one-week vol forecast")] = 5,
    model_type: Annotated[str, Field(description="GARCH model variant: 'garch' (symmetric), 'egarch' (exponential), or 'gjr-garch' (asymmetric leverage)")] = "garch",
    p: Annotated[int, Field(description="GARCH lag order — number of lagged conditional variance terms")] = 1,
    q: Annotated[int, Field(description="ARCH lag order — number of lagged squared-return (innovation) terms")] = 1,
) -> str:
    """Forecasts future volatility term structure using a fitted GARCH model on recent daily returns, producing daily conditional vol estimates out to the specified horizon. Use when pricing options, computing Value-at-Risk (VaR), sizing positions based on expected volatility, or classifying the current vol regime (low/normal/high/extreme). Compares forecast against realized vol. Returns JSON with daily vol forecasts, annualized terminal vol, realized vol comparison, vol regime classification, and 1-day 95% VaR.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_deflated_sharpe_ratio(
    observed_sharpe: Annotated[float, Field(description="Observed Sharpe ratio from the best backtest or strategy selection")],
    n_trials: Annotated[int, Field(description="Number of backtests or strategies tested — the multiple testing count for DSR correction")],
    variance_of_sharpe: Annotated[float, Field(description="Variance of Sharpe ratios across all trials; defaults to 1.0 under normal assumptions")] = 1.0,
    skewness: Annotated[float, Field(description="Skewness of the return distribution; 0.0 assumes normality")] = 0.0,
    kurtosis: Annotated[float, Field(description="Kurtosis of the return distribution; 3.0 assumes normality (mesokurtic)")] = 3.0,
) -> str:
    """Computes the Deflated Sharpe Ratio (DSR) from Bailey and Lopez de Prado (2014) to correct for multiple testing bias in backtest optimization. Use when you have selected the best strategy from N backtests and need to assess the probability that the observed Sharpe ratio is genuine rather than the expected maximum of N random walks. Critical for detecting overfitting and p-hacking in strategy research. Returns JSON with DSR probability, expected max Sharpe under null, significance flag, and Sharpe haircut percentage.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_combinatorial_purged_cv(
    symbol: Annotated[str, Field(description="Ticker symbol to backtest the strategy on, e.g. 'AAPL', 'SPY'")],
    strategy_id: Annotated[str, Field(description="Strategy ID from the registry used to load trading rules and parameters for backtesting")],
    n_splits: Annotated[int, Field(description="Number of CPCV time groups to partition the data into (e.g. 6 yields C(6,2)=15 combinations)")] = 6,
    n_test_groups: Annotated[int, Field(description="Number of groups used as out-of-sample test set per combination (e.g. 2 for pairwise splits)")] = 2,
    embargo_pct: Annotated[float, Field(description="Fraction of total data used as embargo gap between train and test to prevent leakage (e.g. 0.01 = 1%)")] = 0.01,
) -> str:
    """Runs Combinatorial Purged Cross-Validation (CPCV) from Lopez de Prado (2018) to test strategy robustness across all combinatorial train/test splits with temporal purging and embargo. Use when validating a strategy beyond standard walk-forward analysis, computing the probability of backtest overfitting (PBO), or assessing out-of-sample Sharpe ratio distribution. Unlike single-path walk-forward, CPCV generates C(n,k) unique split combinations for comprehensive OOS evaluation. Returns JSON with OOS Sharpe distribution, PBO estimate, and overfitting verdict.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_probability_of_overfitting(
    is_sharpe_ratios: Annotated[list[float], Field(description="Array of in-sample Sharpe ratios, one per walk-forward fold or CPCV combination")],
    oos_sharpe_ratios: Annotated[list[float], Field(description="Array of matched out-of-sample Sharpe ratios, same length and order as is_sharpe_ratios")],
) -> str:
    """Calculates the Probability of Backtest Overfitting (PBO) from Bailey et al. (2015) using matched in-sample and out-of-sample Sharpe ratios from walk-forward or CPCV folds. Use when assessing whether a strategy's in-sample performance generalizes or is merely curve-fitted noise. PBO above 0.5 indicates likely overfitting. Computes rank correlation between IS and OOS performance to detect selection bias. Returns JSON with PBO scalar, rank correlation, best-IS OOS rank, and interpretation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
