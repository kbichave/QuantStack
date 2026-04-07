# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Slippage model enhancement — time-of-day multipliers, accuracy tracking, drift detection.

Provides adaptive slippage modelling that improves as the system observes real fills.
The time-of-day multiplier captures intraday liquidity regimes (morning spreads are
tighter than close spreads), while the accuracy tracker compares predicted vs realized
slippage to detect model drift over time.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection
from quantstack.execution.tca_ewma import resolve_time_bucket

# Time-of-day slippage multipliers reflecting intraday liquidity regimes.
# Morning has the tightest spreads but highest volatility at open, so 1.3.
# Midday is the calmest period: baseline 1.0.
# Afternoon sees volume pick up: 1.1.
# Close has MOC imbalances and wider spreads: 1.2.
_TOD_MULTIPLIERS: dict[str, float] = {
    "morning": 1.3,
    "midday": 1.0,
    "afternoon": 1.1,
    "close": 1.2,
}


def get_time_of_day_multiplier(time_bucket: str) -> float:
    """Return the slippage multiplier for a given intraday time bucket.

    Falls back to 1.0 for unknown buckets (defensive — should not happen
    with valid resolve_time_bucket output).
    """
    return _TOD_MULTIPLIERS.get(time_bucket, 1.0)


def classify_time_bucket(timestamp: datetime) -> str:
    """Classify a timestamp into an intraday liquidity bucket.

    Delegates to tca_ewma.resolve_time_bucket which handles timezone
    conversion to US/Eastern.
    """
    return resolve_time_bucket(timestamp)


def record_slippage_accuracy(
    conn: PgConnection,
    order_id: str,
    symbol: str,
    time_bucket: str,
    predicted_bps: float,
    realized_bps: float,
) -> None:
    """Record predicted vs realized slippage for model calibration.

    When predicted_bps is zero, ratio is stored as NULL to avoid division
    by zero.  This is expected for limit orders or zero-slippage fills.
    """
    ratio: float | None = None
    if predicted_bps != 0:
        ratio = realized_bps / predicted_bps

    conn.execute(
        """
        INSERT INTO slippage_accuracy
            (order_id, symbol, time_bucket, predicted_bps, realized_bps, ratio)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [order_id, symbol, time_bucket, predicted_bps, realized_bps, ratio],
    )


def check_slippage_drift(
    conn: PgConnection,
    symbol: str,
    lookback_count: int = 20,
) -> str | None:
    """Check whether the slippage model is drifting for a symbol.

    Queries the last `lookback_count` accuracy records (excluding rows
    where ratio is NULL) and computes the mean ratio.  If the mean falls
    outside [0.5, 2.0], returns an alert string describing the drift
    direction.  Returns None if the model is within tolerance or there
    is insufficient data.
    """
    rows = conn.execute(
        """
        SELECT ratio
        FROM slippage_accuracy
        WHERE symbol = ? AND ratio IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        [symbol, lookback_count],
    ).fetchall()

    if not rows:
        return None

    mean_ratio = sum(r[0] for r in rows) / len(rows)

    if mean_ratio < 0.5:
        msg = (
            f"Slippage drift alert for {symbol}: mean ratio {mean_ratio:.2f} "
            f"(model over-predicting, n={len(rows)})"
        )
        logger.warning(msg)
        return msg

    if mean_ratio > 2.0:
        msg = (
            f"Slippage drift alert for {symbol}: mean ratio {mean_ratio:.2f} "
            f"(model under-predicting, n={len(rows)})"
        )
        logger.warning(msg)
        return msg

    return None
