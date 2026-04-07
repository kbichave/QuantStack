"""Factor exposure monitoring — beta, sector, style, momentum crowding.

Computes portfolio-level factor exposures, persists snapshots, and triggers
configurable drift alerts when thresholds from ``factor_config`` are breached.

Entry point: ``run_factor_exposure_check(positions)`` — called from the
supervisor health_check node every cycle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from quantstack.db import db_conn


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FactorExposureSnapshot:
    """Full factor exposure snapshot for monitoring and persistence."""

    portfolio_beta: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    top_sector: str = ""
    top_sector_pct: float = 0.0
    style_scores: dict[str, float] = field(default_factory=dict)
    momentum_crowding_pct: float = 0.0
    benchmark_symbol: str = "SPY"
    alerts_triggered: int = 0
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Config defaults (also seeded into factor_config table by db.py)
# ---------------------------------------------------------------------------

FACTOR_CONFIG_DEFAULTS: dict[str, str] = {
    "beta_drift_threshold": "0.3",
    "sector_max_pct": "40",
    "momentum_crowding_pct": "70",
    "benchmark_symbol": "SPY",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_factor_config() -> dict[str, str]:
    """Read factor_config table rows into a dict.

    Falls back to FACTOR_CONFIG_DEFAULTS if DB read fails.
    """
    try:
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT config_key, value FROM factor_config"
            ).fetchall()
        config = {r["config_key"]: r["value"] for r in rows}
        if config:
            return config
    except Exception as e:
        logger.warning("[factor] Config read failed, using defaults: %s", e)
    return dict(FACTOR_CONFIG_DEFAULTS)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


async def compute_factor_exposure(
    positions: list[dict],
    benchmark_symbol: str = "SPY",
) -> FactorExposureSnapshot:
    """Compute portfolio factor exposure against benchmark.

    Args:
        positions: List of position dicts with at minimum 'symbol',
            'quantity', 'market_value'. Optional: 'sector', style factor fields.
        benchmark_symbol: Ticker for the benchmark index.

    Returns:
        FactorExposureSnapshot with all metrics.
    """
    now = datetime.now(timezone.utc)

    if not positions:
        return FactorExposureSnapshot(
            benchmark_symbol=benchmark_symbol,
            computed_at=now,
        )

    # --- Sector weights ---
    total_mv = sum(abs(p.get("market_value", 0)) for p in positions)
    sector_mv: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector") or "unknown"
        mv = abs(p.get("market_value", 0))
        sector_mv[sector] = sector_mv.get(sector, 0) + mv

    if total_mv > 0:
        sector_weights = {s: mv / total_mv for s, mv in sector_mv.items()}
    else:
        sector_weights = {"unknown": 1.0}

    top_sector = max(sector_weights, key=sector_weights.get) if sector_weights else ""
    top_sector_pct = sector_weights.get(top_sector, 0) * 100

    # --- Beta computation ---
    portfolio_beta = await _compute_portfolio_beta(positions, benchmark_symbol, total_mv)

    # --- Style scores (simplified) ---
    style_scores = _compute_style_scores(positions, total_mv)

    # --- Momentum crowding ---
    momentum_crowding_pct = _compute_momentum_crowding(positions, total_mv)

    return FactorExposureSnapshot(
        portfolio_beta=portfolio_beta,
        sector_weights=sector_weights,
        top_sector=top_sector,
        top_sector_pct=top_sector_pct,
        style_scores=style_scores,
        momentum_crowding_pct=momentum_crowding_pct,
        benchmark_symbol=benchmark_symbol,
        computed_at=now,
    )


async def _compute_portfolio_beta(
    positions: list[dict],
    benchmark_symbol: str,
    total_mv: float,
) -> float:
    """Compute portfolio beta via weighted-average position betas.

    Falls back to 1.0 if insufficient data.
    """
    if total_mv <= 0:
        return 1.0

    try:
        from quantstack.data.pg_storage import PgDataStore

        store = PgDataStore()
        # Fetch benchmark returns
        bench_df = store.load_ohlcv(benchmark_symbol, "D1", limit=60)
        if bench_df is None or len(bench_df) < 20:
            return 1.0

        bench_returns = bench_df["close"].pct_change().dropna().values

        weighted_beta = 0.0
        for p in positions:
            sym = p.get("symbol", "")
            mv = abs(p.get("market_value", 0))
            weight = mv / total_mv if total_mv > 0 else 0

            try:
                sym_df = store.load_ohlcv(sym, "D1", limit=60)
                if sym_df is None or len(sym_df) < 20:
                    weighted_beta += weight * 1.0
                    continue

                sym_returns = sym_df["close"].pct_change().dropna().values
                min_len = min(len(sym_returns), len(bench_returns))
                if min_len < 10:
                    weighted_beta += weight * 1.0
                    continue

                sr = sym_returns[-min_len:]
                br = bench_returns[-min_len:]
                cov = np.cov(sr, br)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
                weighted_beta += weight * beta
            except Exception:
                weighted_beta += weight * 1.0

        return round(weighted_beta, 4)
    except Exception as e:
        logger.debug("[factor] Beta computation failed, defaulting to 1.0: %s", e)
        return 1.0


def _compute_style_scores(positions: list[dict], total_mv: float) -> dict[str, float]:
    """Compute portfolio-weighted style scores.

    Uses pre-computed per-position scores if available, else returns 0.0.
    """
    scores = {"momentum": 0.0, "value": 0.0, "growth": 0.0, "quality": 0.0}
    if total_mv <= 0:
        return scores

    for p in positions:
        mv = abs(p.get("market_value", 0))
        weight = mv / total_mv
        for factor in scores:
            scores[factor] += weight * p.get(f"{factor}_score", 0.0)

    return {k: round(v, 4) for k, v in scores.items()}


def _compute_momentum_crowding(positions: list[dict], total_mv: float) -> float:
    """Percentage of portfolio in positions with momentum score >= 80th percentile."""
    if total_mv <= 0 or not positions:
        return 0.0

    mom_scores = [p.get("momentum_score", 0.0) for p in positions]
    if not any(s != 0 for s in mom_scores):
        return 0.0

    threshold = np.percentile([s for s in mom_scores if s != 0], 80) if len([s for s in mom_scores if s != 0]) >= 5 else 0.8
    crowded_mv = sum(
        abs(p.get("market_value", 0))
        for p, s in zip(positions, mom_scores)
        if s >= threshold
    )
    return round(crowded_mv / total_mv * 100, 2) if total_mv > 0 else 0.0


# ---------------------------------------------------------------------------
# Drift checking
# ---------------------------------------------------------------------------


async def check_factor_drift(
    exposure: FactorExposureSnapshot,
    config: dict[str, str],
) -> list[dict]:
    """Check factor exposure against configurable thresholds.

    Returns list of alert dicts ready for emit_system_alert().
    """
    alerts: list[dict] = []

    # 1. Beta drift
    beta_threshold = float(config.get("beta_drift_threshold", "0.3"))
    beta_drift = abs(exposure.portfolio_beta - 1.0)
    if beta_drift > beta_threshold:
        severity = "critical" if beta_drift >= 2 * beta_threshold else "warning"
        alerts.append({
            "category": "factor_drift",
            "severity": severity,
            "title": f"Beta drift: {exposure.portfolio_beta:.2f} (drift {beta_drift:.2f} > {beta_threshold})",
            "detail": (
                f"Portfolio beta is {exposure.portfolio_beta:.2f}, drifting "
                f"{beta_drift:.2f} from neutral 1.0. Threshold: {beta_threshold}."
            ),
            "metadata": {
                "portfolio_beta": exposure.portfolio_beta,
                "drift": beta_drift,
                "threshold": beta_threshold,
            },
        })

    # 2. Sector concentration
    sector_max = float(config.get("sector_max_pct", "40"))
    if exposure.top_sector_pct > sector_max:
        alerts.append({
            "category": "factor_drift",
            "severity": "warning",
            "title": f"Sector concentration: {exposure.top_sector} at {exposure.top_sector_pct:.1f}%",
            "detail": (
                f"Top sector {exposure.top_sector} is {exposure.top_sector_pct:.1f}% "
                f"of portfolio, exceeding {sector_max}% threshold."
            ),
            "metadata": {
                "top_sector": exposure.top_sector,
                "top_sector_pct": exposure.top_sector_pct,
                "threshold": sector_max,
            },
        })

    # 3. Momentum crowding
    mom_threshold = float(config.get("momentum_crowding_pct", "70"))
    if exposure.momentum_crowding_pct > mom_threshold:
        alerts.append({
            "category": "factor_drift",
            "severity": "warning",
            "title": f"Momentum crowding: {exposure.momentum_crowding_pct:.1f}%",
            "detail": (
                f"Momentum crowding at {exposure.momentum_crowding_pct:.1f}%, "
                f"exceeding {mom_threshold}% threshold."
            ),
            "metadata": {
                "momentum_crowding_pct": exposure.momentum_crowding_pct,
                "threshold": mom_threshold,
            },
        })

    return alerts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_factor_snapshot(snapshot: FactorExposureSnapshot) -> None:
    """Write a FactorExposureSnapshot row to factor_exposure_history."""
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO factor_exposure_history "
            "(portfolio_beta, sector_weights, style_scores, momentum_crowding_pct, "
            "benchmark_symbol, alerts_triggered, computed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                snapshot.portfolio_beta,
                json.dumps(snapshot.sector_weights),
                json.dumps(snapshot.style_scores),
                snapshot.momentum_crowding_pct,
                snapshot.benchmark_symbol,
                snapshot.alerts_triggered,
                snapshot.computed_at,
            ],
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_factor_exposure_check(positions: list[dict]) -> dict[str, Any]:
    """Full factor exposure cycle: compute, persist, check drift, emit alerts.

    Single entry point for supervisor integration.
    """
    # 1. Load config
    config = load_factor_config()
    benchmark = config.get("benchmark_symbol", "SPY")

    # 2. Compute exposure
    exposure = await compute_factor_exposure(positions, benchmark)

    # 3. Check drift
    drift_alerts = await check_factor_drift(exposure, config)

    # 4. Emit alerts
    for alert_info in drift_alerts:
        try:
            from quantstack.tools.functions.system_alerts import emit_system_alert

            await emit_system_alert(
                category=alert_info["category"],
                severity=alert_info["severity"],
                title=alert_info["title"],
                detail=alert_info["detail"],
                source="factor_exposure",
                metadata=alert_info.get("metadata"),
            )
        except Exception as e:
            logger.warning("[factor] Failed to emit drift alert: %s", e)

    exposure.alerts_triggered = len(drift_alerts)

    # 5. Persist snapshot
    try:
        persist_factor_snapshot(exposure)
    except Exception as e:
        logger.warning("[factor] Failed to persist snapshot: %s", e)

    return {
        "portfolio_beta": exposure.portfolio_beta,
        "top_sector": exposure.top_sector,
        "top_sector_pct": exposure.top_sector_pct,
        "momentum_crowding_pct": exposure.momentum_crowding_pct,
        "alerts_triggered": exposure.alerts_triggered,
    }
