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
                )
            )

            # Persist the single new observation to DB
            self._persist_observation(collector, signal_value, forward_return, ts)

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_observation(
        self, collector: str, signal_value: float, forward_return: float, timestamp: str
    ) -> None:
        """Insert a single observation to PostgreSQL. Called under lock."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO ic_attribution_data
                        (collector, signal_value, forward_return, recorded_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [collector, signal_value, forward_return, timestamp],
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
                    SELECT collector, signal_value, forward_return, recorded_at
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
