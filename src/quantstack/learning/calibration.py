# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Agent confidence calibration tracker.

An agent that says "BUY with 0.85 confidence" needs to be right 85% of
the time for that confidence score to mean anything.

This module:
  1. Records (confidence, outcome) pairs for each agent
  2. Measures calibration error (ECE — Expected Calibration Error)
  3. Adjusts confidence scores at inference time based on historical accuracy
  4. Generates monthly calibration reports

Usage:
    tracker = CalibrationTracker()

    # Record a prediction outcome
    tracker.record(
        agent_name="SuperTrader",
        symbol="SPY",
        stated_confidence=0.80,
        was_correct=True,   # Did the trade make money?
    )

    # Get calibrated confidence
    calibrated = tracker.calibrate(
        agent_name="SuperTrader",
        raw_confidence=0.80,
    )

    # Check if agent is over/underconfident
    report = tracker.calibration_report("SuperTrader")
"""

from __future__ import annotations

from threading import Lock

from loguru import logger

from quantstack.db import PgConnection, pg_conn

# =============================================================================
# CALIBRATION TRACKER
# =============================================================================


class CalibrationTracker:
    """
    Tracks and corrects agent confidence scores over time.

    Uses a simple histogram-based calibration:
      - Bucket stated confidences into 10% bins (0.0-0.1, 0.1-0.2, ...)
      - Track accuracy per bucket per agent
      - At inference time, remap stated confidence to historical accuracy
        for that bucket

    This is Platt scaling in spirit but simpler and interpretable.
    """

    N_BINS = 10  # 10 bins of 0.1 width each

    _lock = Lock()

    def __init__(self) -> None:
        self._init_schema()
        logger.info("CalibrationTracker initialized (PostgreSQL)")

    def _init_schema(self) -> None:
        with self._lock:
            with pg_conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS calibration_records (
                        id              BIGINT PRIMARY KEY,
                        agent_name      VARCHAR NOT NULL,
                        symbol          VARCHAR,
                        action          VARCHAR,
                        stated_confidence DOUBLE PRECISION NOT NULL,
                        confidence_bin  INTEGER NOT NULL,
                        was_correct     BOOLEAN,
                        pnl             DOUBLE PRECISION,
                        recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE SEQUENCE IF NOT EXISTS calibration_seq START 1"
                )

    # -------------------------------------------------------------------------
    # Record
    # -------------------------------------------------------------------------

    def record(
        self,
        agent_name: str,
        stated_confidence: float,
        was_correct: bool | None = None,
        symbol: str | None = None,
        action: str | None = None,
        pnl: float | None = None,
    ) -> None:
        """
        Record the outcome of an agent prediction.

        Args:
            agent_name: Which agent made the prediction
            stated_confidence: The confidence reported (0.0–1.0)
            was_correct: True if prediction was correct (trade made money)
            symbol: Optional ticker symbol
            action: "buy", "sell", "hold"
            pnl: Realized P&L if available
        """
        stated_confidence = max(0.0, min(1.0, stated_confidence))
        bin_idx = min(self.N_BINS - 1, int(stated_confidence * self.N_BINS))

        with self._lock:
            with pg_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO calibration_records
                        (id, agent_name, symbol, action, stated_confidence,
                         confidence_bin, was_correct, pnl)
                    VALUES (nextval('calibration_seq'), ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        agent_name,
                        symbol,
                        action,
                        stated_confidence,
                        bin_idx,
                        was_correct,
                        pnl,
                    ],
                )

    # -------------------------------------------------------------------------
    # Calibrate
    # -------------------------------------------------------------------------

    def calibrate(
        self,
        agent_name: str,
        raw_confidence: float,
        min_bin_samples: int = 10,
    ) -> float:
        """
        Return calibrated confidence based on historical accuracy.

        If there are fewer than `min_bin_samples` records in this bin,
        returns raw_confidence unchanged (not enough data to calibrate).

        Args:
            agent_name: Which agent to calibrate for
            raw_confidence: The stated confidence to calibrate
            min_bin_samples: Minimum records required to apply calibration

        Returns:
            Calibrated confidence (0.0–1.0)
        """
        raw = max(0.0, min(1.0, raw_confidence))
        bin_idx = min(self.N_BINS - 1, int(raw * self.N_BINS))

        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END) as correct
                FROM calibration_records
                WHERE agent_name = ?
                  AND confidence_bin = ?
                  AND was_correct IS NOT NULL
                """,
                [agent_name, bin_idx],
            ).fetchone()

        total = rows[0] if rows else 0
        correct = rows[1] if rows else 0

        if total < min_bin_samples:
            return raw  # Not enough data to calibrate

        historical_accuracy = correct / total
        return round(historical_accuracy, 4)

    # -------------------------------------------------------------------------
    # Calibration report
    # -------------------------------------------------------------------------

    def calibration_report(self, agent_name: str) -> dict:
        """
        Generate a full calibration report for an agent.

        Returns:
            {
                agent_name, total_records,
                bins: [{stated_range, actual_accuracy, n_samples, bias}],
                ece: Expected Calibration Error (0 = perfect),
                verdict: "WELL_CALIBRATED" | "OVERCONFIDENT" | "UNDERCONFIDENT"
            }
        """
        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    confidence_bin,
                    COUNT(*) as n,
                    AVG(CASE WHEN was_correct = TRUE THEN 1.0 ELSE 0.0 END) as accuracy
                FROM calibration_records
                WHERE agent_name = ? AND was_correct IS NOT NULL
                GROUP BY confidence_bin
                ORDER BY confidence_bin
                """,
                [agent_name],
            ).fetchall()

            total_records = conn.execute(
                "SELECT COUNT(*) FROM calibration_records WHERE agent_name = ?",
                [agent_name],
            ).fetchone()[0]

        bins = []
        ece_sum = 0.0
        total_weighted = 0

        for row in rows:
            bin_idx, n, accuracy = row[0], row[1], row[2] or 0.0
            bin_mid = (bin_idx + 0.5) / self.N_BINS  # Midpoint of bin
            bias = (
                accuracy - bin_mid
            )  # positive = underconfident, negative = overconfident

            bins.append(
                {
                    "stated_range": f"{bin_idx / self.N_BINS:.1f}-{(bin_idx + 1) / self.N_BINS:.1f}",
                    "actual_accuracy": round(accuracy, 4),
                    "n_samples": n,
                    "bias": round(bias, 4),
                }
            )

            ece_sum += n * abs(bias)
            total_weighted += n

        ece = ece_sum / total_weighted if total_weighted > 0 else 0.0

        # Determine if overconfident, underconfident, or well-calibrated
        if total_weighted >= 20:
            biases = [b["bias"] for b in bins if bins]
            avg_bias = sum(biases) / len(biases) if biases else 0.0
            if avg_bias < -0.1:
                verdict = "OVERCONFIDENT"
            elif avg_bias > 0.1:
                verdict = "UNDERCONFIDENT"
            elif ece < 0.05:
                verdict = "WELL_CALIBRATED"
            else:
                verdict = "NEEDS_IMPROVEMENT"
        else:
            verdict = "INSUFFICIENT_DATA"

        return {
            "agent_name": agent_name,
            "total_records": total_records,
            "bins": bins,
            "ece": round(ece, 4),
            "verdict": verdict,
        }

    def all_agents_summary(self) -> list[dict]:
        """Summary of calibration for all agents."""
        agents = self.conn.execute(
            "SELECT DISTINCT agent_name FROM calibration_records"
        ).fetchall()
        return [self.calibration_report(row[0]) for row in agents]


# Singleton
_calibration_tracker: CalibrationTracker | None = None


def get_calibration_tracker(db_path: str | None = None) -> CalibrationTracker:
    """Get the singleton CalibrationTracker instance."""
    global _calibration_tracker
    if _calibration_tracker is None:
        _calibration_tracker = CalibrationTracker(db_path=db_path)
    return _calibration_tracker
