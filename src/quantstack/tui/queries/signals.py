"""Signal queries: active signals and signal brief details."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    position_size_pct: float
    generated_at: datetime
    factors: dict


@dataclass
class SignalBrief:
    symbol: str
    action: str
    confidence: float
    ml_score: float | None
    sentiment_score: float | None
    technical_score: float | None
    options_score: float | None
    macro_score: float | None
    risk_flags: list[str] = field(default_factory=list)
    collector_failures: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


def fetch_active_signals(conn: PgConnection) -> list[Signal]:
    """Return active signals ordered by confidence descending."""
    try:
        conn.execute(
            "SELECT symbol, action, confidence, position_size_pct, generated_at "
            "FROM signal_state ORDER BY confidence DESC"
        )
        return [
            Signal(
                symbol=r[0], action=r[1], confidence=float(r[2]),
                position_size_pct=float(r[3] or 0), generated_at=r[4],
                factors={},
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_active_signals failed", exc_info=True)
        return []


def fetch_signal_brief(conn: PgConnection, symbol: str) -> SignalBrief | None:
    """Return signal detail for a single symbol."""
    try:
        conn.execute(
            "SELECT symbol, action, confidence, position_size_pct, generated_at "
            "FROM signal_state WHERE symbol = %s",
            (symbol,),
        )
        row = conn.fetchone()
        if not row:
            return None
        return SignalBrief(
            symbol=row[0], action=row[1], confidence=float(row[2]),
            ml_score=None, sentiment_score=None, technical_score=None,
            options_score=None, macro_score=None,
            generated_at=row[4],
        )
    except Exception:
        logger.warning("fetch_signal_brief failed", exc_info=True)
        return None
