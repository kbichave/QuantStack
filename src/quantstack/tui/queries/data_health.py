"""Data health and freshness queries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class DataFreshness:
    source: str
    symbol: str
    latest: datetime


def fetch_ohlcv_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {symbol: latest_timestamp} for daily OHLCV data."""
    try:
        conn.execute(
            "SELECT symbol, MAX(timestamp) FROM ohlcv "
            "WHERE timeframe = '1D' GROUP BY symbol"
        )
        return {r[0]: r[1] for r in conn.fetchall()}
    except Exception:
        logger.warning("fetch_ohlcv_freshness failed", exc_info=True)
        return {}


def fetch_news_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {symbol: latest_published_at} for news sentiment data."""
    try:
        conn.execute(
            "SELECT ticker, MAX(time_published) FROM news_sentiment GROUP BY ticker"
        )
        return {r[0]: r[1] for r in conn.fetchall() if r[0]}
    except Exception:
        logger.warning("fetch_news_freshness failed", exc_info=True)
        return {}


def fetch_sentiment_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {symbol: latest_timestamp} for sentiment scores (from news_sentiment)."""
    try:
        conn.execute(
            "SELECT ticker, MAX(time_published) FROM news_sentiment "
            "WHERE ticker_sentiment_score IS NOT NULL GROUP BY ticker"
        )
        return {r[0]: r[1] for r in conn.fetchall() if r[0]}
    except Exception:
        logger.warning("fetch_sentiment_freshness failed", exc_info=True)
        return {}


def fetch_options_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {symbol: latest_date} for options chain data."""
    try:
        conn.execute(
            "SELECT underlying, MAX(data_date) FROM options_chains GROUP BY underlying"
        )
        return {r[0]: r[1] for r in conn.fetchall() if r[0]}
    except Exception:
        logger.warning("fetch_options_freshness failed", exc_info=True)
        return {}


def fetch_insider_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {symbol: latest_date} for insider trades."""
    try:
        conn.execute(
            "SELECT ticker, MAX(transaction_date) FROM insider_trades GROUP BY ticker"
        )
        return {r[0]: r[1] for r in conn.fetchall() if r[0]}
    except Exception:
        logger.warning("fetch_insider_freshness failed", exc_info=True)
        return {}


def fetch_macro_freshness(conn: PgConnection) -> dict[str, datetime]:
    """Return {indicator: latest_date} for macro indicators."""
    try:
        conn.execute(
            "SELECT indicator, MAX(date) FROM macro_indicators GROUP BY indicator"
        )
        return {r[0]: r[1] for r in conn.fetchall() if r[0]}
    except Exception:
        logger.warning("fetch_macro_freshness failed", exc_info=True)
        return {}
