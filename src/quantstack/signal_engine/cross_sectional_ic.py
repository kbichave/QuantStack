# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Cross-sectional IC tracking and gate (P01: Signal Statistical Rigor).

Computes daily Information Coefficient (Spearman rank correlation) across
all symbols for each collector's vote score vs N-day forward returns.

Distinct from ICAttributionTracker (learning/ic_attribution.py), which
computes per-trade IC. This module computes cross-sectional IC: how well
does each collector's signal *rank* stocks on a given day?

Data pipeline:
  1. synthesis.py persists vote scores → signals.metadata JSONB
  2. Scheduler runs compute_and_store() daily at 17:00 ET
  3. get_ic_gate_status() consumed by synthesis weights (flag-gated)

Tables used:
  - signals (read): vote scores per symbol per day
  - ohlcv (read): close prices for forward return computation
  - signal_ic (write): daily IC metrics per collector per horizon
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import Any

from loguru import logger
from scipy.stats import spearmanr as _spearmanr

from quantstack.db import db_conn
from quantstack.signal_engine.correlation import (
    compute_signal_correlations,
)


class CrossSectionalICTracker:
    """Daily cross-sectional IC computation and gating."""

    def compute_and_store(
        self,
        target_date: date,
        horizons: list[int] | None = None,
    ) -> dict[str, dict[int, float | None]]:
        """Compute cross-sectional IC for all collectors on target_date.

        For each collector extracted from signals.metadata:
          1. Gather (symbol, vote_score) across all symbols on target_date
          2. Compute forward returns from ohlcv close prices
          3. Spearman rank correlation across symbols
          4. Store in signal_ic table

        Args:
            target_date: The date for which signals were recorded.
            horizons: Forward return horizons in trading days (default [1, 5, 21]).

        Returns:
            {collector: {horizon: ic_value}} for all computed ICs.
        """
        if horizons is None:
            horizons = [1, 5, 21]

        # 1. Load vote scores from signals table
        vote_data = self._load_vote_scores(target_date)
        if not vote_data:
            logger.info(
                "[CrossSectionalIC] No vote data for %s — skipping", target_date
            )
            return {}

        # Extract unique symbols and collector names
        symbols = sorted(vote_data.keys())
        if len(symbols) < 5:
            logger.info(
                "[CrossSectionalIC] Only %d symbols on %s — need 5+ for meaningful IC",
                len(symbols), target_date,
            )
            return {}

        collectors = set()
        for votes in vote_data.values():
            collectors.update(votes.keys())
        collectors = sorted(collectors)

        results: dict[str, dict[int, float | None]] = {}

        for horizon in horizons:
            # 2. Load forward returns
            returns = self._load_forward_returns(symbols, target_date, horizon)
            if len(returns) < 5:
                continue

            # Only use symbols that have both votes and returns
            common_symbols = [s for s in symbols if s in returns]
            if len(common_symbols) < 5:
                continue

            for collector in collectors:
                # Build paired arrays: (signal_value, forward_return)
                signals_arr = []
                returns_arr = []
                for sym in common_symbols:
                    vote = vote_data.get(sym, {}).get(collector)
                    if vote is not None and math.isfinite(vote):
                        signals_arr.append(vote)
                        returns_arr.append(returns[sym])

                if len(signals_arr) < 5:
                    continue

                # 3. Spearman rank correlation
                corr, pvalue = _spearmanr(signals_arr, returns_arr)
                if not math.isfinite(corr):
                    continue

                ic_val = round(float(corr), 6)
                n_symbols = len(signals_arr)

                # t-statistic: IC * sqrt(n-2) / sqrt(1-IC^2)
                denom = math.sqrt(1 - corr**2) if abs(corr) < 1.0 else 1e-10
                t_stat = corr * math.sqrt(max(0, n_symbols - 2)) / denom

                results.setdefault(collector, {})[horizon] = ic_val

                # 4. Store in signal_ic table
                self._store_ic(
                    target_date, collector, horizon, ic_val, t_stat, n_symbols,
                )

        logger.info(
            "[CrossSectionalIC] Computed IC for %d collectors × %d horizons on %s",
            len(results), len(horizons), target_date,
        )
        return results

    def get_rolling_ic(
        self,
        collector: str,
        horizon: int = 5,
        window: int = 63,
    ) -> float | None:
        """Rolling mean IC over the last `window` trading days."""
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT rank_ic FROM signal_ic
                    WHERE strategy_id = %s AND horizon_days = %s
                    ORDER BY date DESC LIMIT %s
                    """,
                    [collector, horizon, window],
                ).fetchall()
            if not rows or len(rows) < 5:
                return None
            ics = [float(r[0]) for r in rows if r[0] is not None]
            return round(sum(ics) / len(ics), 6) if ics else None
        except Exception as exc:
            logger.warning("[CrossSectionalIC] get_rolling_ic failed: %s", exc)
            return None

    def get_ic_stability(
        self,
        collector: str,
        horizon: int = 5,
        window: int = 63,
    ) -> float | None:
        """Std dev of daily IC (lower = more stable, higher ICIR)."""
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT rank_ic FROM signal_ic
                    WHERE strategy_id = %s AND horizon_days = %s
                    ORDER BY date DESC LIMIT %s
                    """,
                    [collector, horizon, window],
                ).fetchall()
            if not rows or len(rows) < 10:
                return None
            ics = [float(r[0]) for r in rows if r[0] is not None]
            if len(ics) < 10:
                return None
            mean = sum(ics) / len(ics)
            variance = sum((x - mean) ** 2 for x in ics) / len(ics)
            return round(math.sqrt(variance), 6)
        except Exception as exc:
            logger.warning("[CrossSectionalIC] get_ic_stability failed: %s", exc)
            return None

    def get_ic_gate_status(
        self,
        threshold: float = 0.02,
        min_days: int = 21,
        horizon: int = 5,
        window: int = 63,
    ) -> dict[str, bool]:
        """Return {collector: should_include}.

        A collector is gated (False) if its rolling IC over `window` days
        is below `threshold` AND it has at least `min_days` of data.
        Collectors with insufficient data default to True (include).
        """
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT strategy_id, rank_ic FROM signal_ic
                    WHERE horizon_days = %s
                      AND date >= CURRENT_DATE - %s
                    ORDER BY strategy_id, date DESC
                    """,
                    [horizon, window],
                ).fetchall()

            # Group by collector
            collector_ics: dict[str, list[float]] = {}
            for r in rows:
                name = r[0]
                if r[1] is not None:
                    collector_ics.setdefault(name, []).append(float(r[1]))

            gate: dict[str, bool] = {}
            for name, ics in collector_ics.items():
                if len(ics) < min_days:
                    gate[name] = True  # insufficient data — include
                else:
                    avg_ic = sum(ics) / len(ics)
                    gate[name] = avg_ic >= threshold

            return gate
        except Exception as exc:
            logger.warning("[CrossSectionalIC] get_ic_gate_status failed: %s", exc)
            return {}

    def compute_pairwise_correlation(
        self,
        window_days: int = 63,
    ) -> dict[str, float]:
        """Compute correlation penalties using vote score history.

        Queries signals.metadata JSONB for the last `window_days` days,
        builds per-collector daily vote score time series, then delegates
        to correlation.compute_signal_correlations().

        Returns:
            {collector: penalty_factor} where factor ∈ (0, 1].
        """
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT signal_date, metadata FROM signals
                    WHERE strategy_id = 'synthesis_v1'
                      AND signal_date >= CURRENT_DATE - %s
                    ORDER BY signal_date
                    """,
                    [window_days],
                ).fetchall()

            if not rows:
                return {}

            # Build per-collector daily vote score lists
            # Average across symbols per day per collector
            daily_votes: dict[str, dict[str, list[float]]] = {}  # date -> collector -> [scores]
            for r in rows:
                meta = r[1] if isinstance(r[1], dict) else json.loads(r[1]) if r[1] else {}
                votes = meta.get("votes", {})
                d = str(r[0])
                for collector, score in votes.items():
                    if isinstance(score, (int, float)) and math.isfinite(score):
                        daily_votes.setdefault(d, {}).setdefault(collector, []).append(score)

            # Average per day, build time series
            dates = sorted(daily_votes.keys())
            signal_data: dict[str, list[float]] = {}
            for d in dates:
                for collector, scores in daily_votes.get(d, {}).items():
                    avg = sum(scores) / len(scores)
                    signal_data.setdefault(collector, []).append(avg)

            # Get IC data for weaker-signal identification
            ic_data: dict[str, float] = {}
            for collector in signal_data:
                ic = self.get_rolling_ic(collector, horizon=5, window=window_days)
                if ic is not None:
                    ic_data[collector] = ic

            result = compute_signal_correlations(
                signal_data, ic_data, min_observations=min(21, window_days),
            )

            logger.info(
                "[CrossSectionalIC] Correlation analysis: %d collectors, "
                "%d effective independent signals",
                len(signal_data), result.effective_signal_count,
            )
            return result.penalties

        except Exception as exc:
            logger.warning("[CrossSectionalIC] compute_pairwise_correlation failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_vote_scores(self, target_date: date) -> dict[str, dict[str, float]]:
        """Load {symbol: {collector: vote_score}} from signals table."""
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT symbol, metadata FROM signals
                    WHERE strategy_id = 'synthesis_v1' AND signal_date = %s
                    """,
                    [target_date],
                ).fetchall()

            result: dict[str, dict[str, float]] = {}
            for r in rows:
                symbol = r[0]
                meta = r[1] if isinstance(r[1], dict) else json.loads(r[1]) if r[1] else {}
                votes = meta.get("votes", {})
                if votes:
                    result[symbol] = {
                        k: float(v) for k, v in votes.items()
                        if isinstance(v, (int, float)) and math.isfinite(v)
                    }
            return result
        except Exception as exc:
            logger.warning("[CrossSectionalIC] _load_vote_scores failed: %s", exc)
            return {}

    def _load_forward_returns(
        self,
        symbols: list[str],
        base_date: date,
        horizon_days: int,
    ) -> dict[str, float]:
        """Load forward returns: (close[date+horizon] - close[date]) / close[date]."""
        target_date = base_date + timedelta(days=horizon_days + 2)  # buffer for weekends
        try:
            with db_conn() as conn:
                # Get close prices on base_date and the nearest date after base_date + horizon
                rows = conn.execute(
                    """
                    WITH base AS (
                        SELECT symbol, close
                        FROM ohlcv
                        WHERE timeframe = '1d'
                          AND timestamp::date = %s
                          AND symbol = ANY(%s)
                    ),
                    future AS (
                        SELECT DISTINCT ON (symbol) symbol, close
                        FROM ohlcv
                        WHERE timeframe = '1d'
                          AND timestamp::date > %s
                          AND timestamp::date <= %s
                          AND symbol = ANY(%s)
                        ORDER BY symbol, timestamp DESC
                    )
                    SELECT b.symbol, b.close AS base_close, f.close AS future_close
                    FROM base b
                    JOIN future f ON b.symbol = f.symbol
                    WHERE b.close > 0
                    """,
                    [base_date, symbols, base_date, target_date, symbols],
                ).fetchall()

            returns: dict[str, float] = {}
            for r in rows:
                base_close = float(r[1])
                future_close = float(r[2])
                ret = (future_close - base_close) / base_close
                if math.isfinite(ret):
                    returns[r[0]] = ret
            return returns
        except Exception as exc:
            logger.warning("[CrossSectionalIC] _load_forward_returns failed: %s", exc)
            return {}

    def _store_ic(
        self,
        target_date: date,
        collector: str,
        horizon: int,
        ic_val: float,
        t_stat: float,
        n_symbols: int,
    ) -> None:
        """Upsert IC result into signal_ic table."""
        try:
            # Compute ICIR metrics from history
            icir_21d = self._compute_icir(collector, horizon, 21)
            icir_63d = self._compute_icir(collector, horizon, 63)
            ic_positive_rate = self._compute_positive_rate(collector, horizon, 63)

            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO signal_ic
                        (date, strategy_id, horizon_days, rank_ic,
                         ic_positive_rate, icir_21d, icir_63d, ic_tstat, n_symbols)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, strategy_id, horizon_days) DO UPDATE SET
                        rank_ic = EXCLUDED.rank_ic,
                        ic_positive_rate = EXCLUDED.ic_positive_rate,
                        icir_21d = EXCLUDED.icir_21d,
                        icir_63d = EXCLUDED.icir_63d,
                        ic_tstat = EXCLUDED.ic_tstat,
                        n_symbols = EXCLUDED.n_symbols,
                        updated_at = NOW()
                    """,
                    [
                        target_date, collector, horizon, ic_val,
                        ic_positive_rate, icir_21d, icir_63d, t_stat, n_symbols,
                    ],
                )
        except Exception as exc:
            logger.warning("[CrossSectionalIC] _store_ic failed: %s", exc)

    def _compute_icir(
        self, collector: str, horizon: int, window: int
    ) -> float | None:
        """ICIR = mean(IC) / std(IC) over window."""
        mean_ic = self.get_rolling_ic(collector, horizon, window)
        std_ic = self.get_ic_stability(collector, horizon, window)
        if mean_ic is not None and std_ic is not None and std_ic > 0:
            return round(mean_ic / std_ic, 4)
        return None

    def _compute_positive_rate(
        self, collector: str, horizon: int, window: int
    ) -> float | None:
        """Fraction of days with IC > 0."""
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT rank_ic FROM signal_ic
                    WHERE strategy_id = %s AND horizon_days = %s
                    ORDER BY date DESC LIMIT %s
                    """,
                    [collector, horizon, window],
                ).fetchall()
            if not rows:
                return None
            ics = [float(r[0]) for r in rows if r[0] is not None]
            if not ics:
                return None
            return round(sum(1 for x in ics if x > 0) / len(ics), 4)
        except Exception:
            return None
