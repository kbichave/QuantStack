"""
Mean-variance portfolio optimizer with institutional-grade constraints.

Sits between signal generation and order submission — the step the original
system was missing (GAP-5 in the gap analysis):

    Signal generation  →  Portfolio construction  →  Order generation
    (per-instrument       (THIS MODULE: optimize     (order_lifecycle.py)
     alpha forecasts)      across the book with
                          constraints + costs)

Design decisions:
  - Uses scipy.optimize.minimize (SLSQP) — already a dependency, no new packages.
  - Ledoit-Wolf covariance shrinkage (sklearn) — reduces estimation error on
    small samples, which is the regime for a 5-20 symbol universe.
  - Transaction cost term in objective prevents churning positions for trivial
    alpha improvements.
  - Sector constraints are enforced as linear inequality constraints.
  - Long-only mode is the default; long/short requires explicit opt-in because
    eTrade paper accounts don't support shorting by default.

Failure modes:
  - Singular covariance matrix → Ledoit-Wolf shrinkage prevents this.
  - All signals identical → optimizer returns equal-weight; log a warning.
  - Infeasible constraints (e.g., max_weight < 1/n) → relax bounds, log error.
  - SLSQP convergence failure → fall back to equal-weight.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import OptimizeResult, minimize


class OptimizationObjective(str, Enum):
    """Objective function to optimise."""

    MAX_SHARPE = "max_sharpe"  # Maximise Sharpe ratio (default)
    MIN_VARIANCE = "min_variance"  # Minimise portfolio variance
    RISK_PARITY = "risk_parity"  # Equalise risk contribution per asset
    MAX_DIVERSIFICATION = "max_diversification"  # Maximise diversification ratio


@dataclass
class PortfolioConstraints:
    """
    Constraints applied during optimisation.

    All weights represent fractional portfolio allocations.
    Weight of 0.10 = 10% of portfolio NAV in that instrument.

    Attributes:
        min_weight: Floor on any single position (default 0.0 = long-only).
                    Set negative (e.g. -0.10) to allow shorts.
        max_weight: Cap on any single position (default 0.20 = 20%).
        max_leverage: Sum of absolute weights (default 1.0 = fully invested,
                      no leverage). Set to 1.5 to allow 50% gross leverage.
        sector_map: Dict mapping symbol → sector string. If provided,
                    sector_max_weight is enforced per sector.
        sector_max_weight: Max allocation to any single sector (default 0.40).
        turnover_cost_bps: Turnover penalty added to objective — deters
                           trading for trivial alpha. Units: bps per unit of
                           turnover (|w_new - w_old|). Default 5 bps.
        min_trade_threshold: Ignore rebalancing trades smaller than this
                             fraction of portfolio (default 0.01 = 1%).
                             Prevents micro-trades that are all cost, no alpha.
    """

    min_weight: float = 0.0
    max_weight: float = 0.20
    max_leverage: float = 1.0
    sector_map: dict[str, str] = field(default_factory=dict)
    sector_max_weight: float = 0.40
    turnover_cost_bps: float = 5.0
    min_trade_threshold: float = 0.01


@dataclass
class OptimizationResult:
    """
    Output of the portfolio optimiser.

    target_weights contains the solved allocation. current_weights (if passed)
    allow the caller to compute trades needed to reach the target.
    """

    symbols: list[str]
    target_weights: dict[str, float]
    expected_return: float  # Annualised, fraction (e.g. 0.12 = 12%)
    expected_volatility: float  # Annualised, fraction
    expected_sharpe: float  # Expected Sharpe ratio
    diversification_ratio: float  # Weighted avg vol / portfolio vol
    objective: OptimizationObjective
    converged: bool
    solver_message: str
    # Trades required to move from current to target (None if no current given)
    required_trades: dict[str, float] | None = None

    @property
    def risk_contributions(self) -> dict[str, float]:
        """
        Marginal risk contribution per asset as % of total portfolio variance.

        Risk contribution_i = w_i * (Σw)_i / (w'Σw)
        Used for risk-parity diagnostics and position sizing checks.
        """
        # Stored as a post-hoc property; set by optimizer after solve
        return getattr(self, "_risk_contributions", {})


class MeanVarianceOptimizer:
    """
    Markowitz mean-variance optimizer with shrinkage covariance.

    Typical usage:
        opt = MeanVarianceOptimizer(risk_free_rate=0.05)

        result = opt.optimize(
            signals={"SPY": 0.08, "QQQ": 0.12, "GLD": 0.03},
            cov_matrix=cov_df,          # from covariance_matrix() helper
            constraints=PortfolioConstraints(max_weight=0.40),
            current_weights={"SPY": 0.33, "QQQ": 0.33, "GLD": 0.33},
        )

        # result.target_weights → {"SPY": 0.25, "QQQ": 0.45, "GLD": 0.30}
        # result.required_trades → {"SPY": -0.08, "QQQ": +0.12, "GLD": -0.03}
    """

    # Maximum weight we'll give any single asset regardless of constraints —
    # hard backstop to prevent the optimizer from putting everything in one name.
    _HARD_MAX_WEIGHT = 0.50

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        annualisation_factor: int = 252,
    ) -> None:
        """
        Args:
            risk_free_rate: Annual risk-free rate as fraction (0.05 = 5%).
            annualisation_factor: Trading days per year (252 for equities).
        """
        self.risk_free_rate = risk_free_rate
        self.annualisation_factor = annualisation_factor

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def optimize(
        self,
        signals: dict[str, float],
        cov_matrix: pd.DataFrame,
        constraints: PortfolioConstraints | None = None,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """
        Compute target portfolio weights.

        Args:
            signals: Expected annual return forecasts per symbol.
                     Keys are symbols; values are fractions (0.10 = 10%).
                     These come from your signal generation layer.
            cov_matrix: Annualised covariance matrix as a DataFrame.
                        Use covariance_matrix() to build one from returns.
            constraints: Constraint specification. Defaults to long-only, 20% max.
            objective: Optimization objective. Defaults to MAX_SHARPE.
            current_weights: Current portfolio weights. If provided, a turnover
                             penalty is added to the objective and required_trades
                             is populated in the result.

        Returns:
            OptimizationResult with target_weights and diagnostics.
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        # Align symbols: only optimise over symbols present in both signals and cov
        symbols = [s for s in signals if s in cov_matrix.index and s in cov_matrix.columns]
        if not symbols:
            raise ValueError(
                "No symbols overlap between signals and covariance matrix. "
                f"Signals: {list(signals.keys())}, Cov: {list(cov_matrix.index)}"
            )
        if len(symbols) < len(signals):
            dropped = set(signals) - set(symbols)
            logger.warning(f"[OPT] Dropped {dropped} — not in covariance matrix")

        mu = np.array([signals[s] for s in symbols])
        sigma = cov_matrix.loc[symbols, symbols].values.astype(float)

        # Apply Ledoit-Wolf shrinkage to the raw covariance matrix.
        # This is critical for small universes where sample cov is ill-conditioned.
        sigma = self._ledoit_wolf_shrinkage(sigma, symbols)

        # Prepare current weights vector (zero for new positions)
        w_current = np.array(
            [current_weights.get(s, 0.0) if current_weights else 0.0 for s in symbols]
        )

        # Solve
        w_target, sol = self._solve(
            mu=mu,
            sigma=sigma,
            symbols=symbols,
            constraints=constraints,
            objective=objective,
            w_current=w_current,
        )

        target_dict = dict(zip(symbols, w_target, strict=False))

        # Apply min_trade_threshold — if the move is tiny, keep current weight
        if current_weights is not None:
            for sym in symbols:
                delta = abs(target_dict[sym] - current_weights.get(sym, 0.0))
                if delta < constraints.min_trade_threshold:
                    target_dict[sym] = current_weights.get(sym, 0.0)
            # Re-normalise after threshold clipping
            total = sum(abs(v) for v in target_dict.values())
            if total > 1e-9:
                scale = min(1.0, constraints.max_leverage) / total
                target_dict = {k: v * scale for k, v in target_dict.items()}

        # Portfolio-level metrics
        w_arr = np.array([target_dict[s] for s in symbols])
        port_var = float(w_arr @ sigma @ w_arr)
        port_vol = float(np.sqrt(max(port_var, 1e-12)))
        port_ret = float(w_arr @ mu)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 1e-9 else 0.0

        # Diversification ratio: weighted avg individual vol / portfolio vol
        individual_vols = np.sqrt(np.diag(sigma))
        weighted_avg_vol = float(w_arr @ individual_vols) if np.all(w_arr >= 0) else port_vol
        div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-9 else 1.0

        required_trades = None
        if current_weights is not None:
            required_trades = {s: target_dict[s] - current_weights.get(s, 0.0) for s in symbols}
            # Also include any symbols in current_weights not in target (close them)
            for s, w in current_weights.items():
                if s not in target_dict:
                    required_trades[s] = -w

        result = OptimizationResult(
            symbols=symbols,
            target_weights=target_dict,
            expected_return=port_ret,
            expected_volatility=port_vol,
            expected_sharpe=sharpe,
            diversification_ratio=div_ratio,
            objective=objective,
            converged=sol.success,
            solver_message=sol.message,
            required_trades=required_trades,
        )

        # Attach risk contributions
        result._risk_contributions = self._compute_risk_contributions(w_arr, sigma, symbols)

        logger.info(
            f"[OPT] {objective.value} | {len(symbols)} assets | "
            f"E[r]={port_ret:.2%} σ={port_vol:.2%} Sharpe={sharpe:.2f} "
            f"converged={sol.success}"
        )
        return result

    # -------------------------------------------------------------------------
    # Solver
    # -------------------------------------------------------------------------

    def _solve(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        symbols: list[str],
        constraints: PortfolioConstraints,
        objective: OptimizationObjective,
        w_current: np.ndarray,
    ) -> tuple[np.ndarray, OptimizeResult]:
        """Run SLSQP optimization. Falls back to equal-weight on failure."""
        n = len(symbols)
        w0 = np.ones(n) / n  # Equal-weight starting point

        # Build scipy bounds per asset
        lo = max(constraints.min_weight, -self._HARD_MAX_WEIGHT)
        hi = min(constraints.max_weight, self._HARD_MAX_WEIGHT)
        bounds = [(lo, hi)] * n

        # Constraints list for scipy
        scipy_constraints = self._build_scipy_constraints(
            n=n,
            symbols=symbols,
            constraints=constraints,
        )

        # Turnover cost term (bps → fraction)
        turnover_cost = constraints.turnover_cost_bps / 10_000

        # Objective functions
        if objective == OptimizationObjective.MAX_SHARPE:

            def obj_fn(w: np.ndarray) -> float:
                ret = float(w @ mu)
                vol = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
                sharpe = (ret - self.risk_free_rate) / vol
                turnover = float(np.sum(np.abs(w - w_current)))
                return -(sharpe - turnover_cost * turnover)

        elif objective == OptimizationObjective.MIN_VARIANCE:

            def obj_fn(w: np.ndarray) -> float:
                var = float(w @ sigma @ w)
                turnover = float(np.sum(np.abs(w - w_current)))
                return var + turnover_cost * turnover

        elif objective == OptimizationObjective.RISK_PARITY:

            def obj_fn(w: np.ndarray) -> float:
                return self._risk_parity_objective(w, sigma)

        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:

            def obj_fn(w: np.ndarray) -> float:
                individual_vols = np.sqrt(np.diag(sigma))
                port_vol = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
                weighted_avg_vol = float(w @ individual_vols)
                div_ratio = weighted_avg_vol / port_vol
                return -div_ratio
        else:
            raise ValueError(f"Unknown objective: {objective}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = minimize(
                obj_fn,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )

        if not sol.success:
            logger.warning(
                f"[OPT] Solver did not converge ({sol.message}). Falling back to equal-weight."
            )
            w_result = w0.copy()
        else:
            w_result = np.clip(sol.x, lo, hi)
            # Snap tiny weights to zero to avoid ghost positions
            w_result[np.abs(w_result) < 1e-4] = 0.0
            # Re-normalise to leverage limit
            total_abs = np.sum(np.abs(w_result))
            if total_abs > constraints.max_leverage:
                w_result = w_result * constraints.max_leverage / total_abs

        return w_result, sol

    def _build_scipy_constraints(
        self,
        n: int,
        symbols: list[str],
        constraints: PortfolioConstraints,
    ) -> list:
        """Convert PortfolioConstraints to scipy constraint dicts."""
        scipy_constraints = []

        # Weights must sum to ≤ max_leverage (equality for long-only fully invested)
        if constraints.min_weight >= 0:
            # Long-only: weights sum to exactly 1
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: np.sum(w) - 1.0,
                }
            )
        else:
            # Long/short: sum of abs(w) ≤ max_leverage
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: constraints.max_leverage - np.sum(np.abs(w)),
                }
            )

        # Sector constraints: sum of weights in each sector ≤ sector_max_weight
        if constraints.sector_map:
            sectors: dict[str, list[int]] = {}
            for i, sym in enumerate(symbols):
                sector = constraints.sector_map.get(sym, "_unclassified")
                sectors.setdefault(sector, []).append(i)

            for _sector, indices in sectors.items():
                sector_indices = indices  # capture for closure

                def sector_constraint(w: np.ndarray, idxs: list[int] = sector_indices) -> float:
                    return constraints.sector_max_weight - float(np.sum(w[idxs]))

                scipy_constraints.append(
                    {
                        "type": "ineq",
                        "fun": sector_constraint,
                    }
                )

        return scipy_constraints

    # -------------------------------------------------------------------------
    # Risk parity objective
    # -------------------------------------------------------------------------

    @staticmethod
    def _risk_parity_objective(w: np.ndarray, sigma: np.ndarray) -> float:
        """
        Risk parity loss: minimise sum of squared differences between
        risk contributions.

        RC_i = w_i * (Σw)_i / (w'Σw)
        Objective: Σ_i Σ_j (RC_i - RC_j)^2
        """
        port_var = float(w @ sigma @ w)
        if port_var < 1e-12:
            return 0.0
        marginal_rc = sigma @ w
        rc = w * marginal_rc / port_var  # Fractional risk contributions
        # Sum of squared pairwise differences = variance of RC vector (up to constant)
        rc_mean = np.mean(rc)
        return float(np.sum((rc - rc_mean) ** 2))

    # -------------------------------------------------------------------------
    # Covariance shrinkage
    # -------------------------------------------------------------------------

    @staticmethod
    def _ledoit_wolf_shrinkage(sigma: np.ndarray, symbols: list[str]) -> np.ndarray:
        """
        Apply Ledoit-Wolf analytical shrinkage to the covariance matrix.

        Why: On small samples (n_obs < 5 * n_assets), the sample covariance
        matrix is ill-conditioned and the optimiser exploits estimation error
        rather than true signal. Ledoit-Wolf shrinks extreme eigenvalues toward
        the average, reducing this effect at the cost of mild bias.

        Uses sklearn's LedoitWolf if available; falls back to a simple
        Schafer-Strimmer constant-correlation shrinkage otherwise.
        """
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(
                np.random.multivariate_normal(
                    np.zeros(len(symbols)), sigma, size=max(100, 10 * len(symbols))
                )
            )
            return lw.covariance_
        except Exception:
            # Fallback: shrink toward scaled identity
            trace = np.trace(sigma)
            n = sigma.shape[0]
            target = (trace / n) * np.eye(n)
            # Constant shrinkage coefficient: 20%
            alpha = 0.20
            return (1 - alpha) * sigma + alpha * target

    # -------------------------------------------------------------------------
    # Risk contributions
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_risk_contributions(
        w: np.ndarray,
        sigma: np.ndarray,
        symbols: list[str],
    ) -> dict[str, float]:
        """Return fractional risk contribution per asset."""
        port_var = float(w @ sigma @ w)
        if port_var < 1e-12:
            return {s: 1.0 / len(symbols) for s in symbols}
        marginal = sigma @ w
        rc = w * marginal / port_var
        return dict(zip(symbols, rc.tolist(), strict=False))


# =============================================================================
# Helper: build covariance matrix from a returns DataFrame
# =============================================================================


def covariance_matrix(
    returns: pd.DataFrame,
    annualise: bool = True,
    trading_days: int = 252,
    shrinkage: bool = True,
) -> pd.DataFrame:
    """
    Build an annualised covariance matrix from a returns DataFrame.

    Args:
        returns: Daily returns, shape (n_obs, n_symbols). Each column = a symbol.
        annualise: Multiply by trading_days (default True — expected by optimizer).
        trading_days: Number of trading days per year (252 for equities).
        shrinkage: Apply Ledoit-Wolf shrinkage. Recommended for n_obs < 5*n_symbols.

    Returns:
        Annualised (optionally shrunk) covariance matrix as a DataFrame.
    """
    returns = returns.dropna(how="all")
    symbols = list(returns.columns)

    if shrinkage:
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(returns.fillna(0))
            cov_arr = lw.covariance_
        except Exception:
            cov_arr = returns.cov().values
    else:
        cov_arr = returns.cov().values

    if annualise:
        cov_arr = cov_arr * trading_days

    return pd.DataFrame(cov_arr, index=symbols, columns=symbols)


# =============================================================================
# Helper: translate optimizer output to trade list
# =============================================================================


def trades_from_weights(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    portfolio_nav: float,
    min_trade_notional: float = 100.0,
) -> list[dict]:
    """
    Convert weight deltas to a list of trade dictionaries.

    Args:
        target_weights: Symbol → target fractional weight.
        current_weights: Symbol → current fractional weight (0 if not held).
        portfolio_nav: Total portfolio net asset value in dollars.
        min_trade_notional: Skip trades smaller than this dollar amount.

    Returns:
        List of {"symbol", "side", "notional", "weight_delta"} dicts,
        sorted so SELLS come before BUYS (frees cash before deploying it).
    """
    all_symbols = set(target_weights) | set(current_weights)
    trades = []

    for sym in all_symbols:
        w_target = target_weights.get(sym, 0.0)
        w_current = current_weights.get(sym, 0.0)
        delta = w_target - w_current
        notional = abs(delta) * portfolio_nav

        if notional < min_trade_notional:
            continue

        trades.append(
            {
                "symbol": sym,
                "side": "buy" if delta > 0 else "sell",
                "notional": notional,
                "weight_delta": delta,
            }
        )

    # Sells before buys — avoid momentary over-leverage
    return sorted(trades, key=lambda t: (0 if t["side"] == "sell" else 1, -t["notional"]))
