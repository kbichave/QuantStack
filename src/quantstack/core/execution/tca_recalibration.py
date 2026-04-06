"""Monthly TCA coefficient recalibration (section-08).

Fits Almgren-Chriss coefficients from historical trade data in tca_results.

Regression form (Almgren et al. 2005):
    normalized_slippage = γ × participation_rate + η × participation_rate^0.6 + ε

Segments: large_cap (ADV > $10M), small_cap (ADV ≤ $10M), market_wide (all).
Minimum 50 trades per segment required.

Called monthly by supervisor. Each run inserts new rows (no upsert — historical
record preserved).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from quantstack.core.execution.tca_engine import (
    ADV_LARGE_CAP_THRESHOLD,
    DEFAULT_BETA,
    DEFAULT_ETA,
    DEFAULT_GAMMA,
)
from quantstack.db import PgConnection

logger = logging.getLogger(__name__)

MIN_TRADES_FOR_FIT: int = 50
FIT_BETA: float = DEFAULT_BETA  # Fixed exponent for the power-law term


@dataclass
class RecalibrationResult:
    """Result of a single segment recalibration."""

    symbol_group: str
    eta: float
    gamma: float
    beta: float
    n_trades: int
    r_squared: float
    skipped: bool = False
    skip_reason: str = ""


def run_tca_recalibration(conn: PgConnection) -> list[RecalibrationResult]:
    """Run monthly OLS recalibration on historical tca_results.

    Returns list of RecalibrationResult (one per segment attempted).
    """
    trades = _fetch_trade_data(conn)
    if not trades:
        logger.info("[TCA-Recal] No trades with forecast data found")
        return []

    results: list[RecalibrationResult] = []

    # Segment trades
    large = [t for t in trades if t["adv_dollars"] > ADV_LARGE_CAP_THRESHOLD]
    small = [t for t in trades if t["adv_dollars"] <= ADV_LARGE_CAP_THRESHOLD]

    for group, segment_trades in [
        ("large_cap", large),
        ("small_cap", small),
        ("market_wide", trades),
    ]:
        result = _fit_segment(group, segment_trades)
        results.append(result)
        if not result.skipped:
            _persist_result(conn, result)

    conn.commit()
    fitted = [r for r in results if not r.skipped]
    logger.info(
        "[TCA-Recal] Completed: %d/%d segments fitted", len(fitted), len(results)
    )
    return results


def _fetch_trade_data(conn: PgConnection) -> list[dict]:
    """Fetch trades with forecast data for recalibration."""
    rows = conn.execute(
        """
        SELECT tr.symbol, tr.shares, tr.shortfall_vs_arrival_bps,
               tr.ac_expected_cost_bps,
               tr.adv_at_trade, tr.daily_vol_at_trade, tr.arrival_price
        FROM tca_results tr
        WHERE tr.ac_expected_cost_bps IS NOT NULL
          AND tr.shares > 0
          AND tr.adv_at_trade > 0
          AND tr.daily_vol_at_trade > 0
          AND tr.arrival_price > 0
        """
    ).fetchall()

    trades = []
    for row in rows:
        symbol, shares, shortfall, _ac_cost, adv, daily_vol, price = row
        if any(v is None for v in (shares, shortfall, adv, daily_vol, price)):
            continue
        trades.append({
            "symbol": symbol,
            "shares": float(shares),
            "shortfall_bps": float(shortfall),
            "adv": float(adv),
            "daily_vol": float(daily_vol),
            "price": float(price),
            "adv_dollars": float(adv) * float(price),
        })
    return trades


def _fit_segment(
    group: str, trades: list[dict]
) -> RecalibrationResult:
    """Fit OLS regression for one segment."""
    if len(trades) < MIN_TRADES_FOR_FIT:
        return RecalibrationResult(
            symbol_group=group,
            eta=DEFAULT_ETA,
            gamma=DEFAULT_GAMMA,
            beta=FIT_BETA,
            n_trades=len(trades),
            r_squared=0.0,
            skipped=True,
            skip_reason=f"Insufficient trades: {len(trades)} < {MIN_TRADES_FOR_FIT}",
        )

    # Build arrays
    participation = np.array([t["shares"] / t["adv"] for t in trades])
    daily_vol = np.array([t["daily_vol"] for t in trades])

    # Normalize slippage to vol units
    # normalized_slippage = shortfall_bps / (daily_vol * 10_000)
    normalized_slippage = np.array([
        t["shortfall_bps"] / (t["daily_vol"] * 10_000)
        if t["daily_vol"] > 0 else 0.0
        for t in trades
    ])

    # Design matrix: X = [participation_rate, participation_rate^0.6] (no intercept)
    X = np.column_stack([
        participation,
        participation ** FIT_BETA,
    ])

    # OLS fit
    result, residuals, _, _ = np.linalg.lstsq(X, normalized_slippage, rcond=None)
    gamma_fit, eta_fit = float(result[0]), float(result[1])

    # Clamp to positive (impact must be non-negative)
    gamma_fit = max(gamma_fit, 0.001)
    eta_fit = max(eta_fit, 0.001)

    # Compute R²
    y_pred = X @ np.array([gamma_fit, eta_fit])
    ss_res = np.sum((normalized_slippage - y_pred) ** 2)
    ss_tot = np.sum((normalized_slippage - np.mean(normalized_slippage)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if r_squared < 0.1:
        logger.warning(
            "[TCA-Recal] %s: R²=%.3f — fit is unreliable (noisy data)", group, r_squared
        )

    return RecalibrationResult(
        symbol_group=group,
        eta=round(eta_fit, 6),
        gamma=round(gamma_fit, 6),
        beta=FIT_BETA,
        n_trades=len(trades),
        r_squared=round(r_squared, 4),
    )


def _persist_result(conn: PgConnection, result: RecalibrationResult) -> None:
    """Write fitted coefficients to tca_coefficients table."""
    conn.execute(
        """
        INSERT INTO tca_coefficients
            (symbol_group, eta, gamma, beta, n_trades_in_fit, r_squared, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """,
        [
            result.symbol_group,
            result.eta,
            result.gamma,
            result.beta,
            result.n_trades,
            result.r_squared,
        ],
    )
    logger.info(
        "[TCA-Recal] %s: η=%.4f γ=%.4f β=%.2f R²=%.3f (n=%d)",
        result.symbol_group, result.eta, result.gamma, result.beta,
        result.r_squared, result.n_trades,
    )
