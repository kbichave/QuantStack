"""Risk-parity portfolio optimizer with alpha tilts and constrained optimization.

Components:
1. Ledoit-Wolf covariance estimation (sklearn)
2. Risk-parity base weights (scipy SLSQP)
3. Alpha tilt application
4. Constrained optimization with infeasibility cascade

The optimizer is deterministic — no LLM calls. It runs as a graph node
between risk_sizing and portfolio_review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


@dataclass
class PortfolioConstraints:
    """Constraint parameters for the optimizer."""

    position_min: float = 0.01
    position_max: float = 0.15
    sector_max: float = 0.30
    strategy_max: float = 0.25
    turnover_max: float = 0.20
    gross_exposure_min: float = 0.50
    gross_exposure_max: float = 1.50


# ---------------------------------------------------------------------------
# 7.1 Covariance Estimation
# ---------------------------------------------------------------------------


def estimate_covariance(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance on trailing `window` days of returns.

    Returns a positive-definite covariance matrix as a numpy array.
    """
    tail = returns.iloc[-window:]
    lw = LedoitWolf()
    lw.fit(tail.values)
    return lw.covariance_


# ---------------------------------------------------------------------------
# 7.2 Risk-Parity Base Weights
# ---------------------------------------------------------------------------


def compute_risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Compute risk-parity weights where each asset contributes equal risk.

    Uses scipy SLSQP to minimize variance of risk contributions.
    Returns weight array summing to 1.0.
    """
    n = cov_matrix.shape[0]
    if n == 0:
        return np.array([])

    # Initial guess: inverse-volatility weights
    vols = np.sqrt(np.diag(cov_matrix))
    vols = np.where(vols > 0, vols, 1e-8)
    x0 = (1.0 / vols) / (1.0 / vols).sum()

    def _risk_parity_objective(w):
        """Minimize variance of marginal risk contributions."""
        w = np.maximum(w, 1e-10)
        sigma_p = np.sqrt(w @ cov_matrix @ w)
        if sigma_p < 1e-12:
            return 0.0
        marginal_risk = (cov_matrix @ w) / sigma_p
        risk_contrib = w * marginal_risk
        target = sigma_p / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0) for _ in range(n)]

    result = minimize(
        _risk_parity_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x / result.x.sum()  # Renormalize
        return weights

    # Fallback: inverse-volatility
    logger.warning("Risk-parity optimization did not converge, using inverse-vol weights")
    return x0


# ---------------------------------------------------------------------------
# 7.2 Alpha Tilts
# ---------------------------------------------------------------------------


def apply_alpha_tilts(
    base_weights: np.ndarray,
    alpha_signals: np.ndarray,
    tilt_strength: float = 0.5,
) -> np.ndarray:
    """Tilt risk-parity weights toward assets with stronger alpha signals.

    w_final_i = w_base_i * (1 + tilt_strength * alpha_signal_i)
    Re-normalized to sum to 1.0.
    """
    tilted = base_weights * (1.0 + tilt_strength * alpha_signals)
    total = tilted.sum()
    if total > 0:
        return tilted / total
    return base_weights.copy()


# ---------------------------------------------------------------------------
# 7.2 Constrained Optimization with Infeasibility Cascade
# ---------------------------------------------------------------------------


def optimize_portfolio(
    cov_matrix: np.ndarray,
    alpha_signals: np.ndarray,
    current_weights: np.ndarray,
    sector_map: dict[str, str],
    strategy_map: dict[str, str],
    factor_exposures: np.ndarray | None = None,
    factor_penalty_weight: float = 0.1,
    constraints: PortfolioConstraints | None = None,
) -> tuple[np.ndarray, dict]:
    """Full portfolio optimization pipeline.

    1. Compute risk-parity base weights from cov_matrix
    2. Apply alpha tilts from alpha_signals
    3. Solve constrained optimization with infeasibility cascade
    4. Apply factor exposure penalty (soft, not hard constraint)

    Returns (target_weights, metadata_dict).
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    n = cov_matrix.shape[0]
    if n == 0:
        return np.array([]), {"feasible": True, "relaxation_applied": None}

    # Step 1-2: Risk-parity + alpha tilts as starting point
    rp_weights = compute_risk_parity_weights(cov_matrix)
    tilted = apply_alpha_tilts(rp_weights, alpha_signals, tilt_strength=0.5)

    # Step 3: Constrained optimization with cascade
    symbols = list(sector_map.keys()) if sector_map else [f"SYM{i}" for i in range(n)]

    # Build sector groupings
    sectors: dict[str, list[int]] = {}
    for i, sym in enumerate(symbols[:n]):
        sec = sector_map.get(sym, f"default_{i}")
        sectors.setdefault(sec, []).append(i)

    # Relaxation cascade: turnover -> sector -> position_max
    relaxation_steps = [
        {"turnover_max": constraints.turnover_max * 1.5},  # 20% -> 30%
        {"sector_max": constraints.sector_max + 0.10},      # 30% -> 40%
        {"position_max": constraints.position_max + 0.05},   # 15% -> 20%
    ]

    active_constraints = PortfolioConstraints(
        position_min=constraints.position_min,
        position_max=constraints.position_max,
        sector_max=constraints.sector_max,
        strategy_max=constraints.strategy_max,
        turnover_max=constraints.turnover_max,
        gross_exposure_min=constraints.gross_exposure_min,
        gross_exposure_max=constraints.gross_exposure_max,
    )

    relaxation_applied = None

    for attempt in range(len(relaxation_steps) + 1):
        result = _solve_constrained(
            tilted, cov_matrix, current_weights, sectors, active_constraints,
            factor_exposures, factor_penalty_weight,
        )

        if result is not None:
            meta = {
                "feasible": True,
                "relaxation_applied": relaxation_applied,
                "turnover": float(np.sum(np.abs(result - current_weights))),
            }
            if factor_exposures is not None:
                meta["factor_exposures"] = (result @ factor_exposures).tolist()
            return result, meta

        # Relax constraints
        if attempt < len(relaxation_steps):
            relaxation = relaxation_steps[attempt]
            for key, value in relaxation.items():
                setattr(active_constraints, key, value)
            relaxation_applied = list(relaxation.keys())[0]
            logger.info(f"Portfolio optimizer: relaxing {relaxation_applied} to {list(relaxation.values())[0]}")
        else:
            break

    # All relaxation failed — return current portfolio
    logger.warning("Portfolio optimizer: all relaxation steps failed, returning current weights")
    meta = {
        "feasible": False,
        "relaxation_applied": relaxation_applied,
        "turnover": 0.0,
    }
    return current_weights.copy(), meta


def _solve_constrained(
    target_weights: np.ndarray,
    cov_matrix: np.ndarray,
    current_weights: np.ndarray,
    sectors: dict[str, list[int]],
    constraints: PortfolioConstraints,
    factor_exposures: np.ndarray | None,
    factor_penalty_weight: float,
) -> np.ndarray | None:
    """Attempt constrained optimization. Returns None if infeasible."""
    n = len(target_weights)

    def _objective(w):
        # Minimize distance from tilted risk-parity target
        tracking = np.sum((w - target_weights) ** 2)

        # Risk-parity term: minimize variance of risk contributions
        sigma_p_sq = w @ cov_matrix @ w
        if sigma_p_sq > 1e-12:
            sigma_p = np.sqrt(sigma_p_sq)
            marginal = (cov_matrix @ w) / sigma_p
            rc = w * marginal
            rp_term = np.sum((rc - sigma_p / n) ** 2)
        else:
            rp_term = 0.0

        # Factor penalty (soft constraint)
        factor_term = 0.0
        if factor_exposures is not None and factor_penalty_weight > 0:
            portfolio_exposure = w @ factor_exposures
            factor_term = factor_penalty_weight * np.sum(portfolio_exposure ** 2)

        return tracking + 0.1 * rp_term + factor_term

    # Constraints
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Turnover constraint
    cons.append({
        "type": "ineq",
        "fun": lambda w: constraints.turnover_max - np.sum(np.abs(w - current_weights)),
    })

    # Sector constraints
    for sector_name, indices in sectors.items():
        cons.append({
            "type": "ineq",
            "fun": lambda w, idx=indices: constraints.sector_max - np.sum(w[idx]),
        })

    # Gross exposure
    cons.append({
        "type": "ineq",
        "fun": lambda w: np.sum(np.abs(w)) - constraints.gross_exposure_min,
    })
    cons.append({
        "type": "ineq",
        "fun": lambda w: constraints.gross_exposure_max - np.sum(np.abs(w)),
    })

    bounds = [(constraints.position_min, constraints.position_max) for _ in range(n)]

    result = minimize(
        _objective,
        current_weights.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if result.success:
        w = result.x
        # Verify constraints are actually satisfied (SLSQP can be approximate)
        turnover = np.sum(np.abs(w - current_weights))
        if turnover > constraints.turnover_max + 0.02:
            return None
        for indices in sectors.values():
            if np.sum(w[indices]) > constraints.sector_max + 0.02:
                return None
        return w

    return None
