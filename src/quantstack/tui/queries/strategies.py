"""Strategy pipeline queries."""
from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class StrategyCard:
    strategy_id: str
    name: str
    status: str
    symbol: str
    instrument_type: str
    time_horizon: str
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    fwd_trades: int
    fwd_pnl: float
    fwd_days: int
    fwd_required_days: int


@dataclass
class StrategyDetail:
    strategy_id: str
    name: str
    status: str
    symbol: str
    instrument_type: str | None
    time_horizon: str | None
    regime_affinity: str | None
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    total_trades: int
    fwd_trades: int
    fwd_pnl: float
    fwd_days: int
    entry_rules: list[str]
    exit_rules: list[str]


def _parse_json_obj(raw: str | dict | None) -> dict:
    """Safely parse JSONB that may be a raw string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _parse_json_list(raw: str | list | None) -> list[str]:
    """Parse JSONB that should be a list of strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(r) for r in raw]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(r) for r in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def fetch_strategy_detail(conn: PgConnection, strategy_id: str) -> StrategyDetail | None:
    """Return extended strategy detail for modal display."""
    try:
        conn.execute(
            "SELECT s.strategy_id, s.name, s.status, COALESCE(s.symbol, ''), "
            "s.instrument_type, s.time_horizon, s.regime_affinity, "
            "s.backtest_summary, s.entry_rules, s.exit_rules, "
            "COUNT(ct.id) AS total_trades, "
            "COUNT(CASE WHEN ct.closed_at >= s.updated_at THEN 1 END) AS fwd_trades, "
            "COALESCE(SUM(CASE WHEN ct.closed_at >= s.updated_at THEN ct.realized_pnl END), 0) AS fwd_pnl, "
            "COALESCE(EXTRACT(DAY FROM (NOW() - s.updated_at))::int, 0) AS fwd_days "
            "FROM strategies s "
            "LEFT JOIN closed_trades ct ON ct.strategy_id = s.strategy_id "
            "WHERE s.strategy_id = %s "
            "GROUP BY s.strategy_id, s.name, s.status, s.symbol, s.instrument_type, "
            "  s.time_horizon, s.regime_affinity, s.backtest_summary, "
            "  s.entry_rules, s.exit_rules, s.updated_at",
            (strategy_id,),
        )
        r = conn.fetchone()
        if not r:
            return None
        bt = _parse_json_obj(r[7])
        return StrategyDetail(
            strategy_id=r[0], name=r[1], status=r[2], symbol=r[3],
            instrument_type=r[4], time_horizon=r[5], regime_affinity=r[6],
            sharpe=bt.get("sharpe"),
            max_drawdown=bt.get("max_drawdown"),
            win_rate=bt.get("win_rate"),
            profit_factor=bt.get("profit_factor"),
            entry_rules=_parse_json_list(r[8]),
            exit_rules=_parse_json_list(r[9]),
            total_trades=int(r[10]),
            fwd_trades=int(r[11]), fwd_pnl=float(r[12]), fwd_days=int(r[13]),
        )
    except Exception:
        logger.warning("fetch_strategy_detail failed", exc_info=True)
        return None


def fetch_strategy_pipeline(conn: PgConnection) -> list[StrategyCard]:
    """Return all strategies ordered by lifecycle status priority."""
    try:
        conn.execute(
            "SELECT s.strategy_id, s.name, s.status, COALESCE(s.symbol, ''), "
            "COALESCE(s.instrument_type, 'equity'), "
            "COALESCE(s.time_horizon, 'swing'), "
            "s.backtest_summary, "
            "COUNT(ct.id) AS fwd_trades, "
            "COALESCE(SUM(ct.realized_pnl), 0) AS fwd_pnl, "
            "COALESCE(EXTRACT(DAY FROM (NOW() - s.updated_at))::int, 0) AS fwd_days "
            "FROM strategies s "
            "LEFT JOIN closed_trades ct ON ct.strategy_id = s.strategy_id "
            "  AND ct.closed_at >= s.updated_at "
            "GROUP BY s.strategy_id, s.name, s.status, s.symbol, s.instrument_type, "
            "  s.time_horizon, s.backtest_summary, s.updated_at "
            "ORDER BY CASE s.status "
            "  WHEN 'live' THEN 0 "
            "  WHEN 'forward_testing' THEN 1 "
            "  WHEN 'backtested' THEN 2 "
            "  WHEN 'draft' THEN 3 "
            "  WHEN 'retired' THEN 4 "
            "  ELSE 5 END, s.name"
        )
        results = []
        for r in conn.fetchall():
            bt = _parse_json_obj(r[6])
            results.append(StrategyCard(
                strategy_id=r[0], name=r[1], status=r[2], symbol=r[3],
                instrument_type=r[4], time_horizon=r[5],
                sharpe=bt.get("sharpe"),
                max_drawdown=bt.get("max_drawdown"),
                win_rate=bt.get("win_rate"),
                fwd_trades=int(r[7]), fwd_pnl=float(r[8]),
                fwd_days=int(r[9]), fwd_required_days=30,
            ))
        return results
    except Exception:
        logger.warning("fetch_strategy_pipeline failed", exc_info=True)
        return []
