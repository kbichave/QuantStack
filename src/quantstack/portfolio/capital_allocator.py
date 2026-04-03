"""Dynamic capital allocation: strategy-level budgets with scoring.

Institutional firms allocate capital to strategies, not individual trades.
This module computes per-strategy budgets based on a composite score:

    score_i = sharpe_i * capacity_i * (1 - correlation_penalty_i) * regime_fit_i

Budgets are normalized to sum <= total_equity, capped per strategy, and
adjusted for forward-testing status. Weekly rebalancing updates budgets
without forcing liquidation of existing positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


_MIN_TRADES_FOR_LIVE_SHARPE = 10
_ADV_PARTICIPATION_LIMIT = 0.02  # 2% of ADV


def compute_allocation_scores(
    strategies: list[dict],
    closed_trades: pd.DataFrame,
    current_regime: str,
    correlation_matrix: pd.DataFrame,
    adv_data: dict[str, float],
    allocated_capital: float = 100_000,
) -> list[dict]:
    """Compute allocation score for each active strategy.

    Returns list of dicts with keys: strategy_id, score, sharpe_component,
    capacity_component, correlation_penalty, regime_fit.
    """
    results = []

    for strat in strategies:
        sid = strat["strategy_id"]

        # Sharpe component
        strat_trades = closed_trades[closed_trades["strategy_id"] == sid]
        sharpe = _compute_sharpe(strat, strat_trades)

        # Capacity component
        adv = adv_data.get(sid, 0)
        capacity = min(1.0, adv * _ADV_PARTICIPATION_LIMIT / max(allocated_capital, 1))

        # Correlation penalty (avg correlation with other strategies)
        corr_penalty = _compute_correlation_penalty(sid, correlation_matrix)

        # Regime fit
        regime_fit = _compute_regime_fit(strat, current_regime)

        score = max(0.0, sharpe) * capacity * (1 - corr_penalty) * regime_fit

        results.append({
            "strategy_id": sid,
            "score": score,
            "sharpe_component": sharpe,
            "capacity_component": capacity,
            "correlation_penalty": corr_penalty,
            "regime_fit": regime_fit,
        })

    return results


def compute_budgets(
    scores: list[dict],
    total_equity: float,
    max_strategy_allocation: float = 0.25,
    forward_testing_scalar: float | None = None,
    strategy_statuses: dict[str, str] | None = None,
) -> list[dict]:
    """Convert scores to dollar budgets.

    Budgets sum to <= total_equity. No single strategy exceeds
    max_strategy_allocation * total_equity.
    """
    total_score = sum(s["score"] for s in scores)

    if total_score <= 0:
        return [
            {"strategy_id": s["strategy_id"], "budget_pct": 0.0, "budget_dollars": 0.0}
            for s in scores
        ]

    max_dollars = max_strategy_allocation * total_equity
    statuses = strategy_statuses or {}
    ft_scalar = forward_testing_scalar or 1.0

    # First pass: proportional allocation
    budgets = []
    for s in scores:
        raw_pct = s["score"] / total_score
        raw_dollars = total_equity * raw_pct
        capped_dollars = min(raw_dollars, max_dollars)
        budgets.append({
            "strategy_id": s["strategy_id"],
            "raw_dollars": capped_dollars,
            "score": s["score"],
        })

    # Apply forward-testing scalar
    for b in budgets:
        if statuses.get(b["strategy_id"]) == "forward_testing":
            b["raw_dollars"] *= ft_scalar

    # Normalize to not exceed total_equity
    total_allocated = sum(b["raw_dollars"] for b in budgets)
    scale = min(1.0, total_equity / total_allocated) if total_allocated > 0 else 0

    result = []
    for b in budgets:
        dollars = b["raw_dollars"] * scale
        result.append({
            "strategy_id": b["strategy_id"],
            "budget_pct": dollars / total_equity if total_equity > 0 else 0,
            "budget_dollars": dollars,
        })

    return result


def get_strategy_budget_remaining(
    strategy_id: str,
    budget_dollars: float,
    deployed_capital: float,
) -> float:
    """Return capital available for new entries in this strategy."""
    return max(0.0, budget_dollars - deployed_capital)


# -- Internal helpers --


def _compute_sharpe(strat: dict, trades: pd.DataFrame) -> float:
    """Compute Sharpe component for a strategy.

    >= 10 trades: rolling Sharpe from PnL.
    < 10 trades: backtest Sharpe with DSR haircut.
    """
    if len(trades) >= _MIN_TRADES_FOR_LIVE_SHARPE:
        pnl = trades["pnl"].values
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl, ddof=1)
        if std_pnl > 0:
            return float(mean_pnl / std_pnl * np.sqrt(252))
        return 0.0

    # Fallback: backtest Sharpe with DSR haircut
    backtest_sharpe = strat.get("backtest_sharpe", 0.0)
    dsr_penalty = strat.get("dsr_penalty", 0.3)
    return backtest_sharpe / (1 + dsr_penalty)


def _compute_correlation_penalty(strategy_id: str, corr_matrix: pd.DataFrame) -> float:
    """Average correlation of this strategy with all others."""
    if strategy_id not in corr_matrix.columns:
        return 0.0

    row = corr_matrix.loc[strategy_id]
    # Exclude self-correlation
    others = row.drop(strategy_id, errors="ignore")
    if len(others) == 0:
        return 0.0

    return float(np.clip(others.mean(), 0.0, 1.0))


def _compute_regime_fit(strat: dict, current_regime: str) -> float:
    """1.0 if matched, 0.5 if unknown, 0.0 if mismatched."""
    affinity = strat.get("regime_affinity", [])

    if current_regime == "unknown":
        return 0.5

    if current_regime in affinity:
        return 1.0

    return 0.0
