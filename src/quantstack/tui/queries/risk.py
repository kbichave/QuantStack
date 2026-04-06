"""Risk queries: snapshot, events, equity alerts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class RiskSnapshot:
    gross_exposure: float
    net_exposure: float
    concentration: float
    correlation: float
    sector_exposure: float
    var_1d: float
    max_drawdown: float
    snapshot_at: datetime


@dataclass
class RiskEvent:
    event_type: str
    symbol: str | None
    details: str
    created_at: datetime


@dataclass
class EquityAlert:
    alert_id: int
    alert_type: str
    status: str
    message: str
    created_at: datetime
    cleared_at: datetime | None


def fetch_risk_snapshot(conn: PgConnection) -> RiskSnapshot | None:
    """Return the latest risk snapshot. Maps actual DB columns to dataclass."""
    try:
        conn.execute(
            "SELECT gross_exposure, net_exposure, largest_position_pct, "
            "avg_pairwise_corr, 0, var_95, portfolio_dd_pct, snapshot_time "
            "FROM risk_snapshots ORDER BY snapshot_time DESC LIMIT 1"
        )
        row = conn.fetchone()
        if not row:
            return None
        return RiskSnapshot(
            gross_exposure=float(row[0] or 0), net_exposure=float(row[1] or 0),
            concentration=float(row[2] or 0), correlation=float(row[3] or 0),
            sector_exposure=float(row[4] or 0), var_1d=float(row[5] or 0),
            max_drawdown=float(row[6] or 0), snapshot_at=row[7],
        )
    except Exception:
        logger.warning("fetch_risk_snapshot failed", exc_info=True)
        return None


def fetch_risk_events(conn: PgConnection, days: int = 7) -> list[RiskEvent]:
    """Return recent risk-related decision events."""
    try:
        conn.execute(
            "SELECT event_type, symbol, COALESCE(output_summary, ''), created_at "
            "FROM decision_events "
            "WHERE event_type IN ('risk_rejection', 'risk_check', 'drawdown_alert') "
            "AND created_at >= NOW() - make_interval(days => %s) "
            "ORDER BY created_at DESC LIMIT 50",
            (days,),
        )
        return [
            RiskEvent(event_type=r[0], symbol=r[1], details=r[2] or "", created_at=r[3])
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_risk_events failed", exc_info=True)
        return []


def fetch_equity_alerts(conn: PgConnection) -> list[EquityAlert]:
    """Return equity alerts (entry signals) with status."""
    try:
        conn.execute(
            "SELECT id, action, status, "
            "COALESCE(symbol || ' ' || action || ' conf=' || confidence::text, ''), "
            "created_at, acted_at "
            "FROM equity_alerts ORDER BY created_at DESC LIMIT 20"
        )
        return [
            EquityAlert(
                alert_id=int(r[0]), alert_type=r[1] or "", status=r[2] or "",
                message=r[3] or "", created_at=r[4], cleared_at=r[5],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_equity_alerts failed", exc_info=True)
        return []
