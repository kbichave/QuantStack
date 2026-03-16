# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Equity-portfolio risk metrics for simulation results.

Provides Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected
Shortfall) at the 95% and 99% confidence levels, using both the historical
(empirical) method and the parametric (normal) method.

Historical simulation context: regulatory capital models (Basel III) require
Expected Shortfall at 97.5%; internal risk limits typically use 95% and 99%
1-day VaR as daily stop-loss guardrails.

Usage:
    from quant_arena.historical.risk_metrics import compute_risk_metrics

    snapshots = broker.get_daily_snapshots()
    report = compute_risk_metrics(snapshots)
    print(f"1-day 99% VaR: ${report.var_99_1day:,.0f}")
    print(f"Expected Shortfall (99%): ${report.cvar_99_1day:,.0f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VaRReport:
    """
    Value-at-Risk and Expected Shortfall for the simulated equity curve.

    All dollar figures represent potential loss (positive = loss amount).
    Percentages are expressed as decimals (0.05 = 5%).
    """

    # --- Historical (empirical) method ---
    # Directly reads the empirical return distribution; no distributional assumption.
    var_95_hist: float          # 1-day 5th-percentile loss (dollar)
    var_99_hist: float          # 1-day 1st-percentile loss (dollar)
    cvar_95_hist: float         # Expected loss in the 5% tail (dollar)
    cvar_99_hist: float         # Expected loss in the 1% tail (dollar)

    # --- Parametric (normal) method ---
    # Assumes returns are normally distributed; underestimates tail risk for
    # fat-tailed returns but is fast and standard for daily reporting.
    var_95_param: float
    var_99_param: float

    # --- Return percentiles (for diagnostics) ---
    pct_1: float                # 1st percentile of daily returns (e.g. -0.032)
    pct_5: float                # 5th percentile of daily returns
    skewness: float
    excess_kurtosis: float

    # --- Period-scaled (10-day / monthly, using sqrt-of-time scaling) ---
    var_99_10day: float         # 10-day 99% VaR (dollar)
    var_99_monthly: float       # 21-day 99% VaR (dollar)

    # --- Context ---
    n_observations: int
    equity_at_calculation: float

    # --- Stress scenarios ---
    stress_scenarios: Dict[str, float] = field(default_factory=dict)
    # e.g. {"2008_financial_crisis": -18400.0, "2020_covid_crash": -9200.0}


# Historical 1-month peak-to-trough losses for common stress scenarios.
# Source: public market data (approximate monthly peak-to-trough returns).
_STRESS_SCENARIOS: Dict[str, float] = {
    "2008_lehman_month":      -0.175,   # S&P 500, October 2008
    "2020_covid_crash_month": -0.125,   # S&P 500, March 2020
    "2011_euro_crisis_month": -0.070,   # S&P 500, August 2011
    "2022_rate_shock_month":  -0.082,   # S&P 500, September 2022
    "2000_dot_com_month":     -0.095,   # S&P 500, March 2001
}


def compute_risk_metrics(
    daily_snapshots: list,  # List[PortfolioState]
    seed: int = 42,
) -> VaRReport:
    """
    Compute VaR and CVaR from the simulated daily equity snapshots.

    Args:
        daily_snapshots: Output of SimBroker.get_daily_snapshots().
        seed: Random seed (unused currently; reserved for MC extension).

    Returns:
        VaRReport with historical and parametric VaR/CVaR.
    """
    if len(daily_snapshots) < 10:
        equity = daily_snapshots[-1].equity if daily_snapshots else 0.0
        return _empty_report(equity)

    equities = np.array([s.equity for s in daily_snapshots], dtype=float)
    current_equity = float(equities[-1])

    # Daily returns
    returns = np.diff(equities) / equities[:-1]
    n = len(returns)

    if n < 5:
        return _empty_report(current_equity)

    # --- Historical VaR / CVaR ---
    # VaR at confidence c = -percentile(returns, 1-c)
    # CVaR = mean of returns below the VaR threshold
    p1 = float(np.percentile(returns, 1))
    p5 = float(np.percentile(returns, 5))

    # Convert return percentiles to dollar losses at current equity
    var_99_hist = abs(p1) * current_equity
    var_95_hist = abs(p5) * current_equity

    tail_99 = returns[returns <= p1]
    tail_95 = returns[returns <= p5]

    cvar_99_hist = abs(float(np.mean(tail_99))) * current_equity if len(tail_99) > 0 else var_99_hist
    cvar_95_hist = abs(float(np.mean(tail_95))) * current_equity if len(tail_95) > 0 else var_95_hist

    # --- Parametric (normal) VaR ---
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    # z-scores for normal distribution
    z_95 = 1.6449   # norm.ppf(0.95)
    z_99 = 2.3263   # norm.ppf(0.99)

    var_95_param = (mean_r - z_95 * std_r) * current_equity * -1
    var_99_param = (mean_r - z_99 * std_r) * current_equity * -1

    # Floor at 0 (parametric can go negative if mean is very positive)
    var_95_param = max(0.0, var_95_param)
    var_99_param = max(0.0, var_99_param)

    # --- Period scaling (sqrt-of-time approximation) ---
    var_99_10day = var_99_hist * math.sqrt(10)
    var_99_monthly = var_99_hist * math.sqrt(21)

    # --- Distributional stats ---
    from scipy import stats as scipy_stats
    skewness = float(scipy_stats.skew(returns))
    excess_kurtosis = float(scipy_stats.kurtosis(returns))

    # --- Stress scenarios (apply historical shocks to current equity) ---
    stress = {}
    for scenario_name, monthly_return in _STRESS_SCENARIOS.items():
        stress[scenario_name] = round(monthly_return * current_equity, 2)

    return VaRReport(
        var_95_hist=round(var_95_hist, 2),
        var_99_hist=round(var_99_hist, 2),
        cvar_95_hist=round(cvar_95_hist, 2),
        cvar_99_hist=round(cvar_99_hist, 2),
        var_95_param=round(var_95_param, 2),
        var_99_param=round(var_99_param, 2),
        pct_1=round(p1, 6),
        pct_5=round(p5, 6),
        skewness=round(skewness, 4),
        excess_kurtosis=round(excess_kurtosis, 4),
        var_99_10day=round(var_99_10day, 2),
        var_99_monthly=round(var_99_monthly, 2),
        n_observations=n,
        equity_at_calculation=round(current_equity, 2),
        stress_scenarios=stress,
    )


def _empty_report(equity: float) -> VaRReport:
    """Return a zeroed VaRReport when insufficient data is available."""
    return VaRReport(
        var_95_hist=0.0,
        var_99_hist=0.0,
        cvar_95_hist=0.0,
        cvar_99_hist=0.0,
        var_95_param=0.0,
        var_99_param=0.0,
        pct_1=0.0,
        pct_5=0.0,
        skewness=0.0,
        excess_kurtosis=0.0,
        var_99_10day=0.0,
        var_99_monthly=0.0,
        n_observations=0,
        equity_at_calculation=equity,
        stress_scenarios={},
    )


def format_var_report(report: VaRReport) -> str:
    """Return a human-readable risk report string."""
    lines = [
        "=== Risk Metrics (VaR / CVaR) ===",
        f"Equity at calculation : ${report.equity_at_calculation:>12,.0f}",
        f"Observations          : {report.n_observations} days",
        "",
        "1-Day Historical VaR:",
        f"  95% VaR             : ${report.var_95_hist:>10,.0f}",
        f"  99% VaR             : ${report.var_99_hist:>10,.0f}",
        "",
        "1-Day Expected Shortfall (CVaR):",
        f"  95% CVaR            : ${report.cvar_95_hist:>10,.0f}",
        f"  99% CVaR            : ${report.cvar_99_hist:>10,.0f}",
        "",
        "1-Day Parametric (Normal) VaR:",
        f"  95% VaR             : ${report.var_95_param:>10,.0f}",
        f"  99% VaR             : ${report.var_99_param:>10,.0f}",
        "",
        "Scaled VaR (sqrt-of-time):",
        f"  10-day 99% VaR      : ${report.var_99_10day:>10,.0f}",
        f"  Monthly 99% VaR     : ${report.var_99_monthly:>10,.0f}",
        "",
        "Return Distribution:",
        f"  Skewness            : {report.skewness:>8.3f}",
        f"  Excess kurtosis     : {report.excess_kurtosis:>8.3f}",
        "",
        "Stress Scenarios (monthly impact at current equity):",
    ]
    for name, pnl in sorted(report.stress_scenarios.items(), key=lambda x: x[1]):
        lines.append(f"  {name:<35} ${pnl:>10,.0f}")
    return "\n".join(lines)
