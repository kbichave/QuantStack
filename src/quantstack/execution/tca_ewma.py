# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
EWMA feedback loop for Transaction Cost Analysis.

After each fill, realized costs (spread + impact) are decomposed and fed into
an Exponentially Weighted Moving Average (EWMA) per (symbol, time_bucket).
Pre-trade forecasts query these EWMA values and apply a conservative multiplier
that decays as the sample count grows.

Time buckets reflect intraday liquidity regimes in US equity markets:
  - morning   (09:30-11:00 ET): highest volume, tightest spreads
  - midday    (11:00-14:00 ET): lower volume, wider spreads
  - afternoon (14:00-15:30 ET): volume picks up into close
  - close     (15:30-16:00 ET): MOC imbalances, wide spreads

Outside market hours maps to "close" as a conservative default because
close-bucket costs are typically the highest intraday.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from quantstack.db import PgConnection

_ET = ZoneInfo("America/New_York")

# EWMA smoothing factor: 0.1 gives ~90% weight to history, adapts slowly
# to avoid overreacting to single anomalous fills.
_ALPHA = 0.1

# Default decomposition ratio for single-leg fills where we cannot observe
# the spread/impact split directly.  60% impact / 40% spread is a standard
# approximation from empirical TCA literature (Almgren & Chriss, 2000).
_IMPACT_RATIO = 0.60
_SPREAD_RATIO = 0.40


def resolve_time_bucket(timestamp: datetime) -> str:
    """Map a timestamp to a US/Eastern intraday liquidity bucket.

    Args:
        timestamp: Fill timestamp.  If timezone-naive, assumed UTC.

    Returns:
        One of "morning", "midday", "afternoon", "close".
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    et_time = timestamp.astimezone(_ET).time()

    if time(9, 30) <= et_time < time(11, 0):
        return "morning"
    if time(11, 0) <= et_time < time(14, 0):
        return "midday"
    if time(14, 0) <= et_time < time(15, 30):
        return "afternoon"
    # 15:30-16:00 is explicitly "close"; outside market hours also defaults
    # to "close" as the conservative choice (highest-cost bucket).
    return "close"


def conservative_multiplier(sample_count: int) -> float:
    """Uncertainty multiplier that decays as sample count grows.

    At 0 samples the multiplier is 2.0 (maximum conservatism), linearly
    decaying to 1.0 at 50 samples and clamped there.

    Formula: max(1.0, 2.0 - sample_count / 50)
    """
    return max(1.0, 2.0 - sample_count / 50)


def update_ewma_after_fill(
    conn: PgConnection,
    order_id: str,
    symbol: str,
    fill_timestamp: datetime,
    arrival_price: float,
    fill_price: float,
    fill_quantity: int,
    adv: float,
) -> None:
    """Update EWMA cost parameters after a fill.

    Decomposes realized total cost into spread and impact components using
    a fixed 40/60 ratio (suitable for single-leg fills), then upserts into
    tca_parameters with EWMA smoothing.

    Args:
        conn: Active PgConnection (caller manages transaction).
        order_id: Order identifier (for audit trail, not stored in tca_parameters).
        symbol: Instrument ticker.
        fill_timestamp: When the fill occurred.  Naive timestamps assumed UTC.
        arrival_price: Decision price at order placement time.
        fill_price: Realized execution price.
        fill_quantity: Number of shares filled (unused today, reserved for
            volume-weighted EWMA in future).
        adv: Average daily volume (unused today, reserved for participation-rate
            impact models in future).
    """
    if arrival_price <= 0:
        return

    realized_total_bps = abs(fill_price - arrival_price) / arrival_price * 10_000
    realized_spread_bps = realized_total_bps * _SPREAD_RATIO
    realized_impact_bps = realized_total_bps * _IMPACT_RATIO

    bucket = resolve_time_bucket(fill_timestamp)

    conn.execute(
        """
        INSERT INTO tca_parameters
            (symbol, time_bucket, ewma_spread_bps, ewma_impact_bps,
             ewma_total_bps, sample_count, last_updated)
        VALUES (?, ?, ?, ?, ?, 1, NOW())
        ON CONFLICT (symbol, time_bucket) DO UPDATE SET
            ewma_spread_bps = ? * ? + (1.0 - ?) * tca_parameters.ewma_spread_bps,
            ewma_impact_bps = ? * ? + (1.0 - ?) * tca_parameters.ewma_impact_bps,
            ewma_total_bps  = ? * ? + (1.0 - ?) * tca_parameters.ewma_total_bps,
            sample_count    = tca_parameters.sample_count + 1,
            last_updated    = NOW()
        """,
        [
            # INSERT values
            symbol, bucket, realized_spread_bps, realized_impact_bps,
            realized_total_bps,
            # ON CONFLICT SET ewma_spread_bps
            _ALPHA, realized_spread_bps, _ALPHA,
            # ON CONFLICT SET ewma_impact_bps
            _ALPHA, realized_impact_bps, _ALPHA,
            # ON CONFLICT SET ewma_total_bps
            _ALPHA, realized_total_bps, _ALPHA,
        ],
    )


def get_ewma_forecast(
    conn: PgConnection,
    symbol: str,
    time_bucket: str,
) -> dict | None:
    """Retrieve EWMA cost forecast for a (symbol, time_bucket) pair.

    Returns None if no historical data exists.  Otherwise returns a dict with
    the raw EWMA values plus the conservative multiplier applied to total_bps.

    Returns:
        dict with keys: ewma_total_bps, ewma_spread_bps, ewma_impact_bps,
        sample_count, multiplier.  Or None if no row exists.
    """
    row = conn.execute(
        """
        SELECT ewma_spread_bps, ewma_impact_bps, ewma_total_bps, sample_count
        FROM tca_parameters
        WHERE symbol = ? AND time_bucket = ?
        """,
        [symbol, time_bucket],
    ).fetchone()

    if row is None:
        return None

    sample_ct = int(row[3])
    mult = conservative_multiplier(sample_ct)

    return {
        "ewma_spread_bps": float(row[0]),
        "ewma_impact_bps": float(row[1]),
        "ewma_total_bps": float(row[2]),
        "sample_count": sample_ct,
        "multiplier": mult,
    }
