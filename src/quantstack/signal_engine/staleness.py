"""Staleness checking for signal engine collectors.

Each collector calls check_freshness() before computing signals.
If the underlying data is too old, the collector should return {}
and let the synthesis engine redistribute weight.

Freshness is determined by querying data_metadata.last_timestamp.
The table column is (symbol, timeframe) where 'timeframe' doubles
as a data source identifier for non-OHLCV tables (e.g., "macro_indicators").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger

from quantstack.db import db_conn


# Staleness thresholds per data source type (calendar days).
# Price-derived uses 4 calendar days to cover 3-day weekends.
STALENESS_THRESHOLDS: dict[str, int] = {
    "ohlcv": 4,
    "options_chains": 3,
    "news_sentiment": 7,
    "company_overview": 90,
    "macro_indicators": 45,
    "insider_trades": 30,
    "short_interest": 14,
    "sector": 7,
    "events": 30,
    "ewf": 7,
}


def check_freshness(
    symbol: str,
    table: str,
    max_days: int,
) -> bool:
    """Check if the most recent data for symbol in table is within max_days of now.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        table: Data source identifier as stored in data_metadata.timeframe
               (e.g., "1d" for daily OHLCV, "macro_indicators" for macro data).
        max_days: Maximum acceptable age in calendar days.

    Returns:
        True if data is fresh enough to use. False if stale or missing.
    """
    with db_conn() as conn:
        row = conn.execute(
            "SELECT last_timestamp FROM data_metadata "
            "WHERE symbol = %s AND timeframe = %s",
            [symbol, table],
        ).fetchone()

    if row is None or row["last_timestamp"] is None:
        logger.warning(
            f"[staleness] {symbol}/{table}: no metadata row — treating as stale"
        )
        return False

    last_ts = row["last_timestamp"]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    if last_ts < cutoff:
        age_days = (datetime.now(timezone.utc) - last_ts).days
        logger.warning(
            f"[staleness] {symbol}/{table}: data is {age_days}d old "
            f"(threshold {max_days}d) — skipping"
        )
        return False

    return True
