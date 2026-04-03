"""Portfolio risk monitoring: 5 deterministic checks run every cycle.

Each function is a pure computation that reads positions/returns and produces
a risk assessment. No LLM calls. These are wired as supervisor graph nodes
but are testable standalone.

Escalation levels:
  0-5% DD   -> normal
  5-10% DD  -> RISK_SIZING_OVERRIDE (50% sizing reduction)
  10-15% DD -> RISK_ENTRY_HALT (no new entries)
  15-20% DD -> RISK_LIQUIDATION (exit-only mode)
  >20% DD   -> RISK_EMERGENCY (kill switch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Fallback thresholds (used when Section 02 calibration not available)
_DEFAULT_VAR_95_THRESHOLD = 0.02
_DEFAULT_VAR_99_THRESHOLD = 0.05
_DEFAULT_CVAR_99_THRESHOLD = 0.07
_DEFAULT_MAX_POSITION_PCT = 0.15
_DEFAULT_FACTOR_BETA_SOFT = 0.5
_DEFAULT_MARKET_BETA_HARD = 2.0
_DEFAULT_CORR_CHANGE_THRESHOLD = 0.3
_DEFAULT_AVG_CORR_LIMIT = 0.5


@dataclass
class RiskSnapshot:
    """Point-in-time portfolio risk snapshot."""

    total_equity: float
    gross_exposure: float
    net_exposure: float
    largest_position_pct: float
    position_count: int
    daily_pnl: float
    var_95: float | None = None
    var_99: float | None = None
    cvar_99: float | None = None
    portfolio_dd_pct: float | None = None
    market_beta: float | None = None
    momentum_beta: float | None = None
    value_beta: float | None = None
    avg_pairwise_corr: float | None = None
    escalation_level: str | None = None
    warnings: list[str] | None = None


def compute_risk_snapshot(
    positions: list[dict],
    total_equity: float,
    daily_pnl: float,
) -> RiskSnapshot:
    """Compute a point-in-time portfolio risk snapshot from positions.

    Each position dict should have: symbol, quantity, current_price, side, avg_cost.
    """
    if not positions:
        return RiskSnapshot(
            total_equity=total_equity,
            gross_exposure=0.0,
            net_exposure=0.0,
            largest_position_pct=0.0,
            position_count=0,
            daily_pnl=daily_pnl,
        )

    exposures = []
    for p in positions:
        value = abs(p.get("quantity", 0) * p.get("current_price", 0))
        sign = 1 if p.get("side", "long") == "long" else -1
        exposures.append({"value": value, "signed": value * sign})

    gross = sum(e["value"] for e in exposures)
    net = sum(e["signed"] for e in exposures)
    largest_pct = max(e["value"] for e in exposures) / total_equity if total_equity > 0 else 0

    return RiskSnapshot(
        total_equity=total_equity,
        gross_exposure=gross,
        net_exposure=net,
        largest_position_pct=largest_pct,
        position_count=len(positions),
        daily_pnl=daily_pnl,
    )


def compute_factor_exposures(
    position_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    weights: dict[str, float],
    window: int = 60,
) -> dict[str, float]:
    """Compute portfolio-level factor betas from rolling regression.

    Args:
        position_returns: Daily returns per position (columns = symbols).
        factor_returns: Daily returns for factors (columns = market, momentum, value).
        weights: Position weights (symbol -> weight).
        window: Rolling window in days.

    Returns:
        Dict with market_beta, momentum_beta, value_beta.
    """
    if len(position_returns) < window or factor_returns.empty:
        return {"market_beta": None, "momentum_beta": None, "value_beta": None}

    # Portfolio return = weighted sum
    portfolio_ret = sum(
        position_returns.get(sym, pd.Series(0, index=position_returns.index)) * w
        for sym, w in weights.items()
    )

    # Use last `window` days
    port_tail = portfolio_ret.iloc[-window:]
    betas = {}

    for factor in ["market", "momentum", "value"]:
        if factor not in factor_returns.columns:
            betas[f"{factor}_beta"] = None
            continue

        factor_tail = factor_returns[factor].iloc[-window:]
        if len(factor_tail) < window:
            betas[f"{factor}_beta"] = None
            continue

        # OLS regression: portfolio = alpha + beta * factor
        aligned = pd.DataFrame({"port": port_tail, "factor": factor_tail}).dropna()
        if len(aligned) < 20:
            betas[f"{factor}_beta"] = None
            continue

        cov = np.cov(aligned["port"], aligned["factor"])
        var_factor = cov[1, 1]
        if var_factor > 0:
            betas[f"{factor}_beta"] = float(cov[0, 1] / var_factor)
        else:
            betas[f"{factor}_beta"] = None

    return betas


def compute_correlation_matrix(
    position_returns: pd.DataFrame,
    window: int = 60,
) -> tuple[pd.DataFrame, float]:
    """Compute pairwise correlation matrix and average correlation.

    Returns:
        (correlation_matrix, avg_pairwise_correlation)
    """
    if position_returns.shape[1] < 2 or len(position_returns) < window:
        return pd.DataFrame(), 0.0

    corr = position_returns.iloc[-window:].corr()
    # Average of off-diagonal elements
    n = len(corr)
    if n < 2:
        return corr, 0.0

    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(corr.values[mask].mean())
    return corr, avg_corr


def compute_var_historical(
    portfolio_returns: pd.Series,
    n_scenarios: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute VaR and CVaR via historical simulation.

    Returns dict with var_95, var_99, cvar_99 (as positive loss fractions).
    """
    returns = portfolio_returns.dropna()
    if len(returns) < 50:
        return {"var_95": None, "var_99": None, "cvar_99": None}

    rng = np.random.default_rng(seed)
    scenarios = rng.choice(returns.values, size=n_scenarios, replace=True)

    var_95 = float(-np.percentile(scenarios, 5))
    var_99 = float(-np.percentile(scenarios, 1))

    losses_beyond_var99 = scenarios[scenarios <= -var_99]
    cvar_99 = float(-losses_beyond_var99.mean()) if len(losses_beyond_var99) > 0 else var_99

    return {"var_95": var_95, "var_99": var_99, "cvar_99": cvar_99}


def check_drawdown_cascade(
    current_equity: float,
    peak_equity: float,
) -> tuple[str | None, str | None]:
    """Determine drawdown escalation level.

    Returns (escalation_level, event_type_name) or (None, None) if normal.
    """
    if peak_equity <= 0:
        return None, None

    dd_pct = (peak_equity - current_equity) / peak_equity

    if dd_pct > 0.20:
        return "emergency", "RISK_EMERGENCY"
    elif dd_pct > 0.15:
        return "liquidation", "RISK_LIQUIDATION"
    elif dd_pct > 0.10:
        return "entry_halt", "RISK_ENTRY_HALT"
    elif dd_pct > 0.05:
        return "sizing_override", "RISK_SIZING_OVERRIDE"

    return None, None


def check_factor_limits(
    betas: dict[str, float | None],
    soft_limit: float = _DEFAULT_FACTOR_BETA_SOFT,
    hard_limit: float = _DEFAULT_MARKET_BETA_HARD,
) -> list[str]:
    """Check factor beta limits and return list of warnings."""
    warnings = []

    market_beta = betas.get("market_beta")
    if market_beta is not None and abs(market_beta) > hard_limit:
        warnings.append(f"HARD: |market_beta|={abs(market_beta):.2f} > {hard_limit}")

    for factor in ["momentum_beta", "value_beta"]:
        beta = betas.get(factor)
        if beta is not None and abs(beta) > soft_limit:
            warnings.append(f"SOFT: |{factor}|={abs(beta):.2f} > {soft_limit}")

    return warnings
