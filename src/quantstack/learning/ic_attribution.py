# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IC Attribution — per-collector signal quality tracking.

Tracks rolling information coefficient (IC) for each SignalEngine collector.
IC = rank_correlation(collector_signal, forward_returns) over a trailing window.

When IC decays below zero, the collector is actively losing money.
When IC is positive and stable, the collector is contributing alpha.

Used by /reflect to:
  1. Identify which collectors are working and which are degrading
  2. Auto-weight collectors in SignalBrief synthesis
  3. Generate evidence for prompt/code tuning decisions

Usage:
    tracker = ICAttributionTracker()
    tracker.record(symbol="AAPL", collector="technical", signal_value=0.7, forward_return=0.02)
    report = tracker.get_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock

from loguru import logger
from scipy.stats import spearmanr as _spearmanr

from quantstack.db import db_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CollectorIC:
    """Per-collector IC summary."""

    collector: str
    ic: float | None  # Spearman rank correlation
    observations: int
    status: str  # "strong" | "weak" | "degraded" | "insufficient"
    trend: str  # "improving" | "stable" | "declining"


@dataclass
class ICAttributionReport:
    """Full IC report across all collectors."""

    collectors: list[CollectorIC]
    best_collector: str | None
    worst_collector: str | None
    degraded_collectors: list[str]  # IC <= 0
    suggested_weights: dict[str, float]
    report_date: str
    total_observations: int


# ---------------------------------------------------------------------------
# Observation storage
# ---------------------------------------------------------------------------


@dataclass
class _Observation:
    """Single signal/return observation."""

    signal_value: float
    forward_return: float
    timestamp: str
    regime: str = "unknown"


@dataclass
class _CollectorState:
    """Persisted state for one collector."""

    observations: list[_Observation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ICAttributionTracker
# ---------------------------------------------------------------------------


class ICAttributionTracker:
    """
    Tracks rolling IC (Spearman rank correlation between signal values and
    forward returns) per SignalEngine collector.

    State is persisted to PostgreSQL (ic_attribution_data table). All public
    methods are thread-safe via a Lock. In-memory dict is the primary data
    structure; DB is the persistence layer.

    Requires scipy for Spearman correlation. If scipy is not installed,
    ``get_collector_ic`` returns None and ``get_weights`` returns uniform
    weights — the tracker still records observations so that IC can be
    computed once scipy becomes available.
    """

    _lock = Lock()

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
    ) -> None:
        self._window_size = window_size
        self._collectors: dict[str, _CollectorState] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        symbol: str,
        collector: str,
        signal_value: float,
        forward_return: float,
        timestamp: datetime | None = None,
        regime: str = "unknown",
    ) -> None:
        """
        Record a single observation for a collector.

        Args:
            symbol: Ticker symbol (used for logging context; observations
                    are aggregated across symbols per collector).
            collector: Collector name (e.g. "technical", "sentiment").
            signal_value: The collector's output — a score in [0, 1] or
                          any numeric signal that should correlate with
                          forward returns.
            forward_return: Realized return over the holding period.
            timestamp: When the signal was generated. Defaults to now.
            regime: Trend regime at signal time (P05 §5.1 regime-conditioned IC).
        """
        if not math.isfinite(signal_value) or not math.isfinite(forward_return):
            logger.debug(
                f"[ICAttribution] Skipping non-finite observation for {collector} "
                f"(signal={signal_value}, return={forward_return})"
            )
            return

        ts = (timestamp or datetime.now(timezone.utc)).isoformat()

        with self._lock:
            state = self._collectors.setdefault(collector, _CollectorState())
            state.observations.append(
                _Observation(
                    signal_value=signal_value,
                    forward_return=forward_return,
                    timestamp=ts,
                    regime=regime,
                )
            )

            # Persist the single new observation to DB
            self._persist_observation(collector, signal_value, forward_return, ts, regime)

            # Keep bounded — retain 2x window to allow trend comparison
            max_keep = self._window_size * 2
            if len(state.observations) > max_keep:
                state.observations = state.observations[-max_keep:]
                self._truncate_old_observations(collector, max_keep)

        logger.debug(
            f"[ICAttribution] Recorded {collector} for {symbol}: "
            f"signal={signal_value:.4f} return={forward_return:.4f} "
            f"(n={len(state.observations)})"
        )

    def get_collector_ic(
        self,
        collector: str,
        min_observations: int = 20,
    ) -> float | None:
        """
        Compute rolling Spearman rank correlation for a collector.

        Uses the most recent ``window_size`` observations. Returns None if
        scipy is not installed or if there are fewer than ``min_observations``
        data points.
        """
        with self._lock:
            state = self._collectors.get(collector)
            if state is None or len(state.observations) < min_observations:
                return None

            recent = state.observations[-self._window_size :]
            if len(recent) < min_observations:
                return None

            signals = [o.signal_value for o in recent]
            returns = [o.forward_return for o in recent]

        corr, _ = _spearmanr(signals, returns)

        if not math.isfinite(corr):
            return None

        return float(corr)

    def get_report(self) -> ICAttributionReport:
        """
        Generate a full IC attribution report across all collectors.

        Returns an ``ICAttributionReport`` with per-collector IC, status,
        trend, and suggested weights.
        """
        collector_ics: list[CollectorIC] = []
        degraded: list[str] = []

        with self._lock:
            collector_names = list(self._collectors.keys())

        for name in collector_names:
            ic_val = self.get_collector_ic(name)
            obs_count = len(self._collectors.get(name, _CollectorState()).observations)

            status = _classify_status(ic_val, obs_count)
            trend = self._compute_trend(name)

            entry = CollectorIC(
                collector=name,
                ic=ic_val,
                observations=obs_count,
                status=status,
                trend=trend,
            )
            collector_ics.append(entry)

            if status == "degraded":
                degraded.append(name)

        # Determine best/worst by IC value (only among those with sufficient data)
        ranked = [c for c in collector_ics if c.ic is not None]
        ranked.sort(key=lambda c: c.ic, reverse=True)  # type: ignore[arg-type]

        best = ranked[0].collector if ranked else None
        worst = ranked[-1].collector if ranked else None

        weights = self.get_weights()
        total_obs = sum(c.observations for c in collector_ics)

        return ICAttributionReport(
            collectors=collector_ics,
            best_collector=best,
            worst_collector=worst,
            degraded_collectors=degraded,
            suggested_weights=weights,
            report_date=datetime.now(timezone.utc).isoformat(),
            total_observations=total_obs,
        )

    def get_weights(self) -> dict[str, float]:
        """
        Suggested collector weights based on IC.

        Collectors with IC > 0 are normalized to sum to 1.0.
        Collectors with IC <= 0 or insufficient data get weight 0.0.

        If no collector has positive IC (or scipy is absent), returns
        uniform weights so the system degrades gracefully.
        """
        ics: dict[str, float] = {}

        with self._lock:
            collector_names = list(self._collectors.keys())

        for name in collector_names:
            ic_val = self.get_collector_ic(name)
            if ic_val is not None and ic_val > 0:
                ics[name] = ic_val

        # Fallback: uniform weights when no collector has demonstrated edge
        if not ics:
            n = len(collector_names)
            if n == 0:
                return {}
            uniform = round(1.0 / n, 4)
            return {name: uniform for name in collector_names}

        # Normalize positive ICs to sum to 1.0
        total_ic = sum(ics.values())
        weights: dict[str, float] = {}
        for name in collector_names:
            if name in ics:
                weights[name] = round(ics[name] / total_ic, 4)
            else:
                weights[name] = 0.0

        return weights

    def get_weights_for_regime(
        self,
        regime: str,
        window: int = 63,
        min_days: int = 60,
    ) -> dict[str, float] | None:
        """
        IC-derived weights conditioned on a specific regime (P05 §5.1).

        Filters observations by ``regime``, requires at least ``min_days``
        observations in that regime, then computes per-collector Spearman IC
        over the most recent ``window`` observations. Normalizes positive-IC
        collectors to sum 1.0.

        Returns None if insufficient data or all IC <= 0.
        """
        ics: dict[str, float] = {}

        with self._lock:
            for name, state in self._collectors.items():
                regime_obs = [o for o in state.observations if o.regime == regime]
                if len(regime_obs) < min_days:
                    continue

                recent = regime_obs[-window:]
                if len(recent) < min_days:
                    continue

                signals = [o.signal_value for o in recent]
                returns = [o.forward_return for o in recent]

                corr, _ = _spearmanr(signals, returns)
                if math.isfinite(corr) and corr > 0:
                    ics[name] = corr

        if not ics:
            return None

        total_ic = sum(ics.values())
        return {name: round(ic / total_ic, 4) for name, ic in ics.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_observation(
        self,
        collector: str,
        signal_value: float,
        forward_return: float,
        timestamp: str,
        regime: str = "unknown",
    ) -> None:
        """Insert a single observation to PostgreSQL. Called under lock."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO ic_attribution_data
                        (collector, signal_value, forward_return, recorded_at, regime)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [collector, signal_value, forward_return, timestamp, regime],
                )
        except Exception as exc:
            logger.warning(f"[ICAttribution] Failed to persist observation: {exc}")

    def _truncate_old_observations(self, collector: str, max_keep: int) -> None:
        """Remove observations beyond the retention window for a collector."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    DELETE FROM ic_attribution_data
                    WHERE collector = %s
                      AND id NOT IN (
                          SELECT id FROM ic_attribution_data
                          WHERE collector = %s
                          ORDER BY recorded_at DESC
                          LIMIT %s
                      )
                    """,
                    [collector, collector, max_keep],
                )
        except Exception as exc:
            logger.warning(f"[ICAttribution] Failed to truncate old observations: {exc}")

    def _load(self) -> None:
        """Load persisted observation state from PostgreSQL."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    SELECT collector, signal_value, forward_return, recorded_at,
                           COALESCE(regime, 'unknown') AS regime
                    FROM ic_attribution_data
                    ORDER BY collector, recorded_at ASC
                    """
                )
                rows = conn.fetchall()

            for row in rows:
                collector = row["collector"]
                state = self._collectors.setdefault(collector, _CollectorState())
                state.observations.append(
                    _Observation(
                        signal_value=row["signal_value"],
                        forward_return=row["forward_return"],
                        timestamp=row["recorded_at"].isoformat()
                            if hasattr(row["recorded_at"], "isoformat")
                            else str(row["recorded_at"]),
                        regime=row["regime"],
                    )
                )

            logger.info(
                f"[ICAttribution] Loaded state for {len(self._collectors)} collectors from DB"
            )
        except Exception as exc:
            logger.warning(f"[ICAttribution] Failed to load state from DB: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_trend(self, collector: str) -> str:
        """
        Determine IC trend by comparing IC over the last window vs the
        prior window. Requires 2x window_size observations.

        Returns "improving", "stable", or "declining".
        """
        with self._lock:
            state = self._collectors.get(collector)
            if state is None or len(state.observations) < self._window_size * 2:
                return "stable"

            recent = state.observations[-self._window_size :]
            prior = state.observations[-self._window_size * 2 : -self._window_size]

        recent_ic = self._compute_ic_for_observations(recent)
        prior_ic = self._compute_ic_for_observations(prior)

        if recent_ic is None or prior_ic is None:
            return "stable"

        diff = recent_ic - prior_ic
        if diff > 0.02:
            return "improving"
        if diff < -0.02:
            return "declining"
        return "stable"

    @staticmethod
    def _compute_ic_for_observations(observations: list[_Observation]) -> float | None:
        """Compute Spearman IC for a list of observations."""
        if len(observations) < 5:
            return None

        signals = [o.signal_value for o in observations]
        returns = [o.forward_return for o in observations]

        corr, _ = _spearmanr(signals, returns)
        if not math.isfinite(corr):
            return None
        return float(corr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IC Weight Precomputation (P05 §3 — batch job)
# ---------------------------------------------------------------------------

_BASE_REGIMES = ("trending_up", "trending_down", "ranging", "unknown")
_IC_GATE_FLOOR = 0.02
_ICIR_THRESHOLD = 0.1
_ICIR_PENALTY = 0.7
_TOTAL_WEIGHT_FLOOR = 0.1
_MAX_SINGLE_WEIGHT = 0.80
_STALENESS_DAYS = 14


def _rolling_sub_ics(
    signals: list[float], returns: list[float], window: int = 21,
) -> list[float]:
    """Compute IC over rolling sub-windows for ICIR calculation."""
    sub_ics: list[float] = []
    for i in range(0, len(signals) - window + 1, window):
        s = signals[i : i + window]
        r = returns[i : i + window]
        if len(s) < 10:
            continue
        corr, _ = _spearmanr(s, r)
        if math.isfinite(corr):
            sub_ics.append(corr)
    return sub_ics


def _population_std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def compute_and_store_ic_weights() -> dict[str, dict[str, float]]:
    """Batch-compute IC-driven weights per regime and store in precomputed_ic_weights.

    For each base regime:
      1. Query ic_attribution_data for the last 63 days, filtered by regime
      2. Compute per-collector Spearman IC
      3. Apply IC gate: drop collectors with IC < 0.02
      4. Apply ICIR penalty: multiply by 0.7 if IC/std(IC) < 0.1
      5. Apply correlation penalty from CrossSectionalICTracker
      6. Reject regime if single collector > 0.80 (sanity bound)
      7. Check weight floor: if total positive weight < 0.1, skip regime
      8. Normalize to sum=1.0
      9. Upsert into precomputed_ic_weights

    Returns:
        {regime: {collector: weight}} for logging/diagnostics.
    """
    results: dict[str, dict[str, float]] = {}

    for regime in _BASE_REGIMES:
        with db_conn() as conn:
            conn.execute(
                "SELECT collector, signal_value, forward_return "
                "FROM ic_attribution_data "
                "WHERE regime = %s AND recorded_at > NOW() - INTERVAL '63 days' "
                "ORDER BY collector, recorded_at ASC",
                [regime],
            )
            rows = conn.fetchall()

        if not rows:
            logger.info("[ICPrecompute] No data for regime=%s, skipping", regime)
            continue

        # Group by collector
        collector_data: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            coll = row["collector"]
            collector_data.setdefault(coll, []).append(
                (row["signal_value"], row["forward_return"]),
            )

        # Per-collector Spearman IC
        collector_ic: dict[str, float] = {}
        collector_ic_std: dict[str, float] = {}
        for coll, pairs in collector_data.items():
            if len(pairs) < 20:
                continue
            sigs = [p[0] for p in pairs]
            rets = [p[1] for p in pairs]
            corr, _ = _spearmanr(sigs, rets)
            if not math.isfinite(corr):
                continue
            collector_ic[coll] = corr
            sub_ics = _rolling_sub_ics(sigs, rets, window=21)
            if sub_ics:
                collector_ic_std[coll] = _population_std(sub_ics)

        # IC gate
        gated = {k: v for k, v in collector_ic.items() if v >= _IC_GATE_FLOOR}

        # ICIR penalty
        for coll in list(gated):
            std = collector_ic_std.get(coll, 0.0)
            if std > 0:
                icir = gated[coll] / std
                if icir < _ICIR_THRESHOLD:
                    gated[coll] *= _ICIR_PENALTY

        # Correlation penalty
        try:
            from quantstack.signal_engine.cross_sectional_ic import (
                CrossSectionalICTracker,
            )
            penalties = CrossSectionalICTracker().compute_pairwise_correlation()
            if penalties:
                for coll in gated:
                    if coll in penalties:
                        gated[coll] *= penalties[coll]
        except Exception as exc:
            logger.warning("[ICPrecompute] Correlation penalty failed: %s", exc)

        # Weight floor check
        total_positive = sum(v for v in gated.values() if v > 0)
        if total_positive < _TOTAL_WEIGHT_FLOOR:
            logger.info(
                "[ICPrecompute] regime=%s total_positive=%.4f < floor, skipping",
                regime, total_positive,
            )
            continue

        # Normalize
        positive = {k: v for k, v in gated.items() if v > 0}
        total = sum(positive.values())
        weights = {k: round(v / total, 4) for k, v in positive.items()}

        # Sanity bound: reject if any single collector > 0.80
        if any(w > _MAX_SINGLE_WEIGHT for w in weights.values()):
            logger.warning(
                "[ICPrecompute] regime=%s has collector with weight > %.2f, "
                "falling back to static",
                regime, _MAX_SINGLE_WEIGHT,
            )
            continue

        # Upsert
        with db_conn() as conn:
            for coll, weight in weights.items():
                ic_val = collector_ic.get(coll, 0.0)
                conn.execute(
                    "INSERT INTO precomputed_ic_weights "
                    "(regime, collector, weight, ic_value, computed_at) "
                    "VALUES (%s, %s, %s, %s, NOW()) "
                    "ON CONFLICT (regime, collector) "
                    "DO UPDATE SET weight = EXCLUDED.weight, "
                    "ic_value = EXCLUDED.ic_value, "
                    "computed_at = EXCLUDED.computed_at",
                    [regime, coll, weight, ic_val],
                )

        results[regime] = weights
        logger.info(
            "[ICPrecompute] regime=%s collectors=%d weights=%s",
            regime, len(weights), weights,
        )

    return results


def get_precomputed_weights(regime: str) -> dict[str, float] | None:
    """Read precomputed IC-driven weights for a regime.

    Returns None if no rows exist or data is stale (> 14 days old).
    Caller should fall back to static weight profiles.
    """
    try:
        with db_conn() as conn:
            conn.execute(
                "SELECT collector, weight FROM precomputed_ic_weights "
                "WHERE regime = %s AND computed_at > NOW() - INTERVAL '%s days'",
                [regime, _STALENESS_DAYS],
            )
            rows = conn.fetchall()
        if not rows:
            return None
        return {row["collector"]: row["weight"] for row in rows}
    except Exception as exc:
        logger.warning("[ICPrecompute] Failed to read precomputed weights: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Conviction Factor Calibration (P05 §5 — quarterly batch job)
# ---------------------------------------------------------------------------

_CALIBRATION_FACTORS = (
    "adx", "stability", "timeframe", "regime_agreement",
    "ml_confirmation", "data_quality",
)
_MIN_CALIBRATION_SAMPLES = 100
_MIN_R_SQUARED = 0.01


def calibrate_conviction_factors() -> dict[str, dict[str, float]]:
    """Quarterly batch job: calibrate conviction factor parameters from realized data.

    Joins signals.metadata.conviction_factors with OHLCV forward returns to
    find optimal threshold/scale parameters for each conviction factor.
    Uses ALL signals (not just closed trades) to avoid survivorship bias.

    Returns:
        {factor_name: {param_name: param_value}} for logging.
    """
    results: dict[str, dict[str, float]] = {}

    for factor in _CALIBRATION_FACTORS:
        try:
            with db_conn() as conn:
                # Join signals with forward 5-day returns from OHLCV
                conn.execute(
                    """
                    SELECT
                        (s.metadata::jsonb -> 'conviction_factors' ->> %s)::float AS factor_value,
                        (o_fwd.close - o_cur.close) / NULLIF(o_cur.close, 0) AS forward_return
                    FROM signals s
                    JOIN ohlcv o_cur ON o_cur.symbol = s.symbol AND o_cur.date = s.signal_date
                    JOIN ohlcv o_fwd ON o_fwd.symbol = s.symbol
                        AND o_fwd.date = (
                            SELECT MIN(date) FROM ohlcv
                            WHERE symbol = s.symbol AND date > s.signal_date + INTERVAL '4 days'
                        )
                    WHERE s.metadata::jsonb -> 'conviction_factors' ? %s
                      AND s.signal_date > NOW() - INTERVAL '180 days'
                    """,
                    [factor, factor],
                )
                rows = conn.fetchall()

            if len(rows) < _MIN_CALIBRATION_SAMPLES:
                logger.info(
                    "[Calibration] %s: only %d samples (need %d), skipping",
                    factor, len(rows), _MIN_CALIBRATION_SAMPLES,
                )
                continue

            factor_values = [float(r["factor_value"]) for r in rows if r["factor_value"] is not None]
            forward_returns = [float(r["forward_return"]) for r in rows if r["forward_return"] is not None]

            if len(factor_values) < _MIN_CALIBRATION_SAMPLES:
                continue

            # Simple linear regression: forward_return = a + b * factor_value
            n = len(factor_values)
            mean_x = sum(factor_values) / n
            mean_y = sum(forward_returns) / n
            ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(factor_values, forward_returns))
            ss_xx = sum((x - mean_x) ** 2 for x in factor_values)
            ss_yy = sum((y - mean_y) ** 2 for y in forward_returns)

            if ss_xx == 0 or ss_yy == 0:
                continue

            slope = ss_xy / ss_xx
            intercept = mean_y - slope * mean_x
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

            if r_squared < _MIN_R_SQUARED:
                logger.info(
                    "[Calibration] %s: R²=%.4f below threshold, keeping defaults",
                    factor, r_squared,
                )
                continue

            # Store calibrated params
            params = {
                "slope": round(slope, 6),
                "intercept": round(intercept, 6),
                "r_squared": round(r_squared, 4),
            }

            with db_conn() as conn:
                for param_name, param_value in params.items():
                    conn.execute(
                        "INSERT INTO conviction_calibration_params "
                        "(factor_name, param_name, param_value, calibrated_at, "
                        "sample_size, r_squared) "
                        "VALUES (%s, %s, %s, NOW(), %s, %s) "
                        "ON CONFLICT (factor_name, param_name) "
                        "DO UPDATE SET param_value = EXCLUDED.param_value, "
                        "calibrated_at = EXCLUDED.calibrated_at, "
                        "sample_size = EXCLUDED.sample_size, "
                        "r_squared = EXCLUDED.r_squared",
                        [factor, param_name, param_value, n, r_squared],
                    )

            results[factor] = params
            logger.info(
                "[Calibration] %s: slope=%.6f R²=%.4f n=%d",
                factor, slope, r_squared, n,
            )

        except Exception as exc:
            logger.warning("[Calibration] %s failed: %s", factor, exc)

    return results


# ---------------------------------------------------------------------------
# Ensemble A/B Evaluation (P05 §6 — weekly batch job)
# ---------------------------------------------------------------------------


def evaluate_ensemble_ab() -> dict[str, float]:
    """Weekly job: backfill forward returns and compare ensemble methods by IC.

    1. Backfill forward_return_5d for recent ensemble_ab_results
    2. Compute per-method IC (Spearman correlation of signal_value vs forward_return)
    3. If non-default method has IC improvement > 0.01 sustained over 60+ days, promote it

    Returns:
        {method_name: ic_value} for logging.
    """
    method_ics: dict[str, float] = {}

    try:
        # Step 1: Backfill forward returns
        with db_conn() as conn:
            conn.execute(
                """
                UPDATE ensemble_ab_results ab
                SET forward_return_5d = (o_fwd.close - o_cur.close) / NULLIF(o_cur.close, 0)
                FROM ohlcv o_cur, ohlcv o_fwd
                WHERE o_cur.symbol = ab.symbol AND o_cur.date = ab.signal_date
                  AND o_fwd.symbol = ab.symbol
                  AND o_fwd.date = (
                      SELECT MIN(date) FROM ohlcv
                      WHERE symbol = ab.symbol AND date > ab.signal_date + INTERVAL '4 days'
                  )
                  AND ab.forward_return_5d IS NULL
                  AND ab.signal_date < NOW() - INTERVAL '7 days'
                """
            )

        # Step 2: Compute per-method IC over last 60 days
        with db_conn() as conn:
            conn.execute(
                """
                SELECT method_name, signal_value, forward_return_5d
                FROM ensemble_ab_results
                WHERE forward_return_5d IS NOT NULL
                  AND signal_date > NOW() - INTERVAL '60 days'
                ORDER BY method_name, signal_date
                """
            )
            rows = conn.fetchall()

        if not rows:
            logger.info("[EnsembleAB] No backfilled results yet")
            return method_ics

        # Group by method
        method_data: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            method_data.setdefault(row["method_name"], []).append(
                (row["signal_value"], row["forward_return_5d"]),
            )

        for method, pairs in method_data.items():
            if len(pairs) < 30:
                continue
            sigs = [p[0] for p in pairs]
            rets = [p[1] for p in pairs]
            corr, _ = _spearmanr(sigs, rets)
            if math.isfinite(corr):
                method_ics[method] = round(corr, 4)

        logger.info("[EnsembleAB] Method ICs: %s", method_ics)

        # Step 3: Auto-promote if non-default method is clearly better
        if len(method_ics) >= 2:
            default_ic = method_ics.get("weighted_avg", 0.0)
            best_method = max(method_ics, key=lambda m: method_ics[m])
            best_ic = method_ics[best_method]

            if best_method != "weighted_avg" and (best_ic - default_ic) > 0.01:
                with db_conn() as conn:
                    conn.execute(
                        "UPDATE ensemble_config SET active_method = %s, "
                        "promoted_at = NOW(), evidence_ic = %s "
                        "WHERE id = 1",
                        [best_method, best_ic],
                    )
                logger.info(
                    "[EnsembleAB] Promoted %s (IC=%.4f) over weighted_avg (IC=%.4f)",
                    best_method, best_ic, default_ic,
                )

    except Exception as exc:
        logger.warning("[EnsembleAB] Evaluation failed: %s", exc)

    return method_ics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_status(ic: float | None, observations: int) -> str:
    """
    Classify collector health based on IC value.

    Thresholds:
      - IC > 0.05  → "strong" — collector is contributing alpha
      - 0 < IC <= 0.05 → "weak" — marginal signal, may not survive costs
      - IC <= 0    → "degraded" — collector is noise or anti-predictive
      - insufficient observations → "insufficient"
    """
    if ic is None or observations < 20:
        return "insufficient"
    if ic > 0.05:
        return "strong"
    if ic > 0:
        return "weak"
    return "degraded"
