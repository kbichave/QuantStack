"""
Almgren-Chriss (2000) optimal execution cost model.

Estimates expected transaction cost for a given order size, volatility, and
liquidity. Used as the adaptive baseline for the execution RL agent's reward
function — replaces the flat 5 bps heuristic.

The model decomposes execution cost into:
  - Permanent impact: price moves against you as the market absorbs information
  - Temporary impact: price moves against you during execution, then reverts
  - Timing risk: volatility-driven uncertainty cost of spreading execution over time

References:
  Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions."
  Journal of Risk, 3(2), 5-39.

Default coefficients are calibrated to US large-cap equities (XOM, MSFT, IBM)
at participation rates < 1% ADV. For higher participation or small-caps,
recalibrate gamma/eta from fill data via calibrate_from_fills().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ACCostBreakdown:
    """Breakdown of Almgren-Chriss expected execution cost."""

    permanent_impact_bps: float
    temporary_impact_bps: float
    timing_risk_bps: float
    total_bps: float
    participation_pct: float  # order_shares / daily_volume * 100


def almgren_chriss_expected_cost_bps(
    order_shares: float,
    daily_volume: float,
    daily_volatility: float,
    execution_horizon_days: float = 1 / 6.5,  # ~1 trading hour
    gamma: float = 0.1,
    eta: float = 0.01,
) -> float:
    """
    Expected one-way execution cost in basis points.

    Args:
        order_shares: Number of shares to execute.
        daily_volume: Average daily volume (shares).
        daily_volatility: Daily return volatility (e.g., 0.02 = 2%).
        execution_horizon_days: Fraction of a trading day for execution.
            1/6.5 ≈ 1 hour (6.5 hours in a US trading day).
        gamma: Permanent impact coefficient. Higher = more lasting price impact.
        eta: Temporary impact coefficient. Higher = more transient slippage.

    Returns:
        Expected cost in basis points (>= 0).
    """
    if order_shares <= 0 or daily_volume <= 0 or daily_volatility <= 0:
        return 0.0

    breakdown = almgren_chriss_cost_breakdown(
        order_shares, daily_volume, daily_volatility,
        execution_horizon_days, gamma, eta,
    )
    return breakdown.total_bps


def almgren_chriss_cost_breakdown(
    order_shares: float,
    daily_volume: float,
    daily_volatility: float,
    execution_horizon_days: float = 1 / 6.5,
    gamma: float = 0.1,
    eta: float = 0.01,
) -> ACCostBreakdown:
    """
    Full cost breakdown with permanent impact, temporary impact, and timing risk.

    See almgren_chriss_expected_cost_bps() for parameter descriptions.
    """
    if order_shares <= 0 or daily_volume <= 0 or daily_volatility <= 0:
        return ACCostBreakdown(
            permanent_impact_bps=0.0,
            temporary_impact_bps=0.0,
            timing_risk_bps=0.0,
            total_bps=0.0,
            participation_pct=0.0,
        )

    T = max(execution_horizon_days, 1e-6)
    sigma = daily_volatility
    X = order_shares
    V = daily_volume

    participation_rate = X / V

    # Permanent impact: market learns from order flow
    # Cost ~ gamma * sigma * (X/V)
    # In bps: multiply by 10_000
    permanent_bps = gamma * sigma * participation_rate * 10_000

    # Temporary impact: transient price displacement during execution
    # Cost ~ eta * sigma * sqrt(X / (V * T))
    # Square-root law: impact scales with sqrt of participation rate per unit time
    temporary_bps = eta * sigma * np.sqrt(participation_rate / T) * 10_000

    # Timing risk: volatility cost of spreading execution over T
    # Risk ~ 0.5 * sigma * sqrt(T) * (X/V)
    # This is the cost of uncertainty — executing faster reduces it but increases impact
    timing_risk_bps = 0.5 * sigma * np.sqrt(T) * participation_rate * 10_000

    total_bps = permanent_bps + temporary_bps + timing_risk_bps

    return ACCostBreakdown(
        permanent_impact_bps=round(permanent_bps, 4),
        temporary_impact_bps=round(temporary_bps, 4),
        timing_risk_bps=round(timing_risk_bps, 4),
        total_bps=round(total_bps, 4),
        participation_pct=round(participation_rate * 100, 6),
    )


# ---------------------------------------------------------------------------
# Optimal trajectory
# ---------------------------------------------------------------------------


def optimal_trajectory(
    total_shares: float,
    n_slices: int,
    daily_volume: float,
    daily_volatility: float,
    risk_aversion: float = 1e-6,
    gamma: float = 0.1,
    eta: float = 0.01,
) -> np.ndarray:
    """
    Almgren-Chriss optimal execution trajectory.

    Returns the number of shares to trade in each time slice to minimize
    expected cost + risk_aversion * variance.

    For risk_aversion → 0: uniform (TWAP-like) trajectory.
    For risk_aversion → ∞: trade everything immediately.

    Args:
        total_shares: Total shares to execute.
        n_slices: Number of time slices.
        daily_volume: Average daily volume.
        daily_volatility: Daily volatility.
        risk_aversion: Lambda — higher = more urgency, front-loaded.
        gamma: Permanent impact coefficient.
        eta: Temporary impact coefficient.

    Returns:
        Array of shape (n_slices,) with shares per slice. Sums to total_shares.
    """
    if n_slices <= 0 or total_shares <= 0:
        return np.array([])

    if n_slices == 1:
        return np.array([total_shares])

    tau = 1.0 / n_slices  # time per slice as fraction of horizon

    # Kappa: urgency parameter balancing impact vs risk
    # kappa = sqrt(risk_aversion * sigma^2 / (eta * sigma / sqrt(V * tau)))
    # Simplified: higher kappa = more aggressive (front-loaded)
    vol_per_slice = daily_volatility * np.sqrt(tau)
    temp_impact_rate = eta * daily_volatility / np.sqrt(max(daily_volume * tau, 1.0))

    if temp_impact_rate <= 0:
        return np.full(n_slices, total_shares / n_slices)

    kappa_sq = risk_aversion * vol_per_slice**2 / temp_impact_rate
    kappa = np.sqrt(max(kappa_sq, 1e-12))

    # Optimal trajectory: n_j = X * sinh(kappa * (N-j) * tau) / sinh(kappa * N * tau)
    # where j is slice index (0-based), N = n_slices
    j = np.arange(n_slices)
    remaining_time = (n_slices - j) * tau
    prev_remaining = (n_slices - j + 1) * tau

    # Shares held at start of each slice
    denom = np.sinh(kappa * n_slices * tau)
    if abs(denom) < 1e-12:
        return np.full(n_slices, total_shares / n_slices)

    holdings_start = total_shares * np.sinh(kappa * remaining_time) / denom
    holdings_end = np.append(holdings_start[1:], 0.0)

    trajectory = holdings_start - holdings_end
    # Ensure non-negative and sums to total
    trajectory = np.maximum(trajectory, 0.0)
    trajectory = trajectory * (total_shares / max(trajectory.sum(), 1e-12))

    return trajectory


# ---------------------------------------------------------------------------
# Calibration from historical fills
# ---------------------------------------------------------------------------


def calibrate_from_fills(
    fill_prices: np.ndarray,
    arrival_prices: np.ndarray,
    order_sizes: np.ndarray,
    daily_volumes: np.ndarray,
    daily_volatilities: np.ndarray,
) -> tuple[float, float]:
    """
    Calibrate gamma and eta from historical fill data using least-squares.

    Fits: slippage_bps = gamma * sigma * (X/V) + eta * sigma * sqrt(X/(V*T))

    Args:
        fill_prices: Actual fill prices.
        arrival_prices: Prices at signal time.
        order_sizes: Shares per order.
        daily_volumes: ADV at time of each order.
        daily_volatilities: Daily vol at time of each order.

    Returns:
        (gamma, eta) — calibrated coefficients.
    """
    n = len(fill_prices)
    if n < 10:
        # Insufficient data — return defaults
        return 0.1, 0.01

    # Observed slippage in bps
    sides = np.sign(order_sizes)  # +1 buy, -1 sell
    slippage_bps = sides * (fill_prices - arrival_prices) / arrival_prices * 10_000

    sigma = daily_volatilities
    participation = np.abs(order_sizes) / np.maximum(daily_volumes, 1.0)

    T = 1 / 6.5  # assume 1-hour execution horizon

    # Feature matrix: [gamma_feature, eta_feature]
    feat_gamma = sigma * participation * 10_000
    feat_eta = sigma * np.sqrt(participation / T) * 10_000

    A = np.column_stack([feat_gamma, feat_eta])
    # Least squares: minimize ||A @ [gamma, eta] - slippage||^2
    # with non-negativity constraint (gamma >= 0, eta >= 0)
    result, _, _, _ = np.linalg.lstsq(A, slippage_bps, rcond=None)

    gamma_cal = max(float(result[0]), 0.001)
    eta_cal = max(float(result[1]), 0.001)

    return gamma_cal, eta_cal


# ---------------------------------------------------------------------------
# High-level convenience: compute_optimal_schedule
# ---------------------------------------------------------------------------


def compute_optimal_schedule(
    total_shares: int,
    time_horizon_minutes: int,
    daily_volume: int,
    daily_volatility_pct: float,
    permanent_impact: float = 0.1,
    temporary_impact: float = 0.01,
    risk_aversion: float = 1e-6,
) -> dict:
    """
    Almgren-Chriss optimal execution schedule with cost estimates.

    Minimizes: E[cost] + risk_aversion * Var[cost]

    The analytical solution uses:
        kappa = sqrt(risk_aversion * sigma^2 / eta)
        schedule_k = N * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    Higher risk_aversion front-loads execution (reduces variance at the cost
    of higher market impact). Lower risk_aversion spreads execution uniformly
    (TWAP-like, lower impact but higher timing risk).

    Args:
        total_shares: Total number of shares to execute.
        time_horizon_minutes: Execution window in minutes (e.g., 60).
        daily_volume: Average daily volume (shares).
        daily_volatility_pct: Daily return volatility in percent (e.g., 1.5 = 1.5%).
        permanent_impact: Gamma coefficient for permanent price impact.
        temporary_impact: Eta coefficient for temporary price impact.
        risk_aversion: Lambda — higher values front-load execution.

    Returns:
        {
            "schedule": list of shares per 5-minute interval,
            "expected_cost_bps": total expected one-way cost,
            "expected_risk_bps": timing risk component,
            "urgency_parameter": computed kappa value,
            "n_intervals": number of execution intervals,
            "interval_minutes": minutes per interval,
        }
    """
    if total_shares <= 0 or daily_volume <= 0:
        return {
            "schedule": [],
            "expected_cost_bps": 0.0,
            "expected_risk_bps": 0.0,
            "urgency_parameter": 0.0,
            "n_intervals": 0,
            "interval_minutes": 0,
        }

    daily_vol = daily_volatility_pct / 100.0
    trading_day_minutes = 390  # 6.5 hours
    horizon_fraction = time_horizon_minutes / trading_day_minutes

    # Use 5-minute intervals
    interval_minutes = 5
    n_intervals = max(1, time_horizon_minutes // interval_minutes)

    # Compute optimal trajectory
    trajectory = optimal_trajectory(
        total_shares=float(total_shares),
        n_slices=n_intervals,
        daily_volume=float(daily_volume),
        daily_volatility=daily_vol,
        risk_aversion=risk_aversion,
        gamma=permanent_impact,
        eta=temporary_impact,
    )

    # Round to integer shares, ensure sum matches
    schedule = np.round(trajectory).astype(int)
    remainder = total_shares - int(schedule.sum())
    if len(schedule) > 0 and remainder != 0:
        schedule[0] += remainder

    # Cost breakdown
    breakdown = almgren_chriss_cost_breakdown(
        order_shares=float(total_shares),
        daily_volume=float(daily_volume),
        daily_volatility=daily_vol,
        execution_horizon_days=horizon_fraction,
        gamma=permanent_impact,
        eta=temporary_impact,
    )

    # Urgency parameter kappa
    tau = 1.0 / n_intervals if n_intervals > 0 else 1.0
    vol_per_slice = daily_vol * np.sqrt(tau)
    temp_rate = temporary_impact * daily_vol / np.sqrt(
        max(float(daily_volume) * tau, 1.0)
    )
    kappa = float(np.sqrt(max(risk_aversion * vol_per_slice ** 2 / max(temp_rate, 1e-12), 1e-12)))

    return {
        "schedule": schedule.tolist(),
        "expected_cost_bps": round(breakdown.total_bps, 4),
        "expected_risk_bps": round(breakdown.timing_risk_bps, 4),
        "urgency_parameter": round(kappa, 6),
        "n_intervals": n_intervals,
        "interval_minutes": interval_minutes,
    }
