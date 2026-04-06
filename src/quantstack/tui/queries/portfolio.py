"""Portfolio queries: equity summary, positions, closed trades, equity curve, benchmarks, PnL breakdowns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class EquitySummary:
    total_equity: float
    cash: float
    daily_pnl: float
    daily_return_pct: float
    high_water: float
    drawdown_pct: float


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    strategy_id: str
    holding_days: int


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    realized_pnl: float
    holding_days: int
    strategy_id: str
    exit_reason: str
    closed_at: datetime


@dataclass
class EquityPoint:
    date: date
    equity: float


@dataclass
class BenchmarkPoint:
    date: date
    symbol: str
    close: float
    daily_return_pct: float


@dataclass
class StrategyPnl:
    strategy_id: str
    strategy_name: str
    realized_pnl: float
    unrealized_pnl: float
    win_count: int
    loss_count: int
    sharpe: float | None


@dataclass
class SymbolPnl:
    symbol: str
    total_pnl: float


def fetch_equity_summary(conn: PgConnection) -> EquitySummary | None:
    """Return latest equity snapshot with drawdown from high-water mark."""
    try:
        conn.execute(
            "SELECT total_equity, cash, daily_pnl, daily_return_pct, "
            "high_water_mark, drawdown_pct "
            "FROM daily_equity ORDER BY date DESC LIMIT 1"
        )
        row = conn.fetchone()
        if not row:
            return None
        return EquitySummary(
            total_equity=float(row[0]), cash=float(row[1]),
            daily_pnl=float(row[2] or 0), daily_return_pct=float(row[3] or 0),
            high_water=float(row[4] or row[0]),
            drawdown_pct=float(row[5] or 0),
        )
    except Exception:
        logger.warning("fetch_equity_summary failed", exc_info=True)
        return None


def fetch_positions(conn: PgConnection) -> list[Position]:
    """Return open positions ordered by unrealized PnL descending."""
    try:
        conn.execute(
            "SELECT symbol, quantity, avg_cost, current_price, "
            "COALESCE(unrealized_pnl, 0), "
            "CASE WHEN avg_cost * quantity != 0 "
            "  THEN COALESCE(unrealized_pnl, 0) / (avg_cost * ABS(quantity)) * 100 "
            "  ELSE 0 END AS unrealized_pnl_pct, "
            "COALESCE(strategy_id, ''), "
            "COALESCE(EXTRACT(DAY FROM (NOW() - opened_at))::int, 0) AS holding_days "
            "FROM positions WHERE quantity != 0 ORDER BY unrealized_pnl DESC"
        )
        return [
            Position(
                symbol=r[0], quantity=float(r[1]), avg_cost=float(r[2]),
                current_price=float(r[3] or 0), unrealized_pnl=float(r[4]),
                unrealized_pnl_pct=float(r[5]), strategy_id=r[6] or "", holding_days=int(r[7]),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_positions failed", exc_info=True)
        return []


def fetch_closed_trades(conn: PgConnection, limit: int = 10) -> list[ClosedTrade]:
    """Return recent closed trades."""
    try:
        conn.execute(
            "SELECT symbol, side, realized_pnl, holding_days, strategy_id, "
            "exit_reason, closed_at FROM closed_trades ORDER BY closed_at DESC LIMIT %s",
            (limit,),
        )
        return [
            ClosedTrade(
                symbol=r[0], side=r[1], realized_pnl=float(r[2]),
                holding_days=int(r[3]), strategy_id=r[4], exit_reason=r[5], closed_at=r[6],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_closed_trades failed", exc_info=True)
        return []


def fetch_equity_curve(conn: PgConnection, days: int = 30) -> list[EquityPoint]:
    """Return equity curve for the last N days."""
    try:
        conn.execute(
            "SELECT date, total_equity FROM daily_equity "
            "ORDER BY date DESC LIMIT %s",
            (days,),
        )
        rows = conn.fetchall()
        return [EquityPoint(date=r[0], equity=float(r[1])) for r in reversed(rows)]
    except Exception:
        logger.warning("fetch_equity_curve failed", exc_info=True)
        return []


def fetch_benchmark(conn: PgConnection, symbol: str = "SPY", days: int = 30) -> list[BenchmarkPoint]:
    """Return benchmark daily data. Note: column is 'benchmark', not 'symbol'."""
    try:
        conn.execute(
            "SELECT date, benchmark, close_price, daily_return_pct "
            "FROM benchmark_daily WHERE benchmark = %s "
            "ORDER BY date DESC LIMIT %s",
            (symbol, days),
        )
        rows = conn.fetchall()
        return [
            BenchmarkPoint(date=r[0], symbol=r[1], close=float(r[2]), daily_return_pct=float(r[3] or 0))
            for r in reversed(rows)
        ]
    except Exception:
        logger.warning("fetch_benchmark failed", exc_info=True)
        return []


def fetch_pnl_by_strategy(conn: PgConnection) -> list[StrategyPnl]:
    """Return PnL breakdown per strategy."""
    try:
        conn.execute(
            "SELECT s.strategy_id, s.name, "
            "COALESCE(SUM(ct.realized_pnl), 0) AS realized, "
            "COALESCE(SUM(p.unrealized_pnl), 0) AS unrealized, "
            "COUNT(CASE WHEN ct.realized_pnl > 0 THEN 1 END) AS wins, "
            "COUNT(CASE WHEN ct.realized_pnl <= 0 THEN 1 END) AS losses, "
            "s.backtest_summary "
            "FROM strategies s "
            "LEFT JOIN closed_trades ct ON ct.strategy_id = s.strategy_id "
            "LEFT JOIN positions p ON p.strategy_id = s.strategy_id AND p.quantity != 0 "
            "GROUP BY s.strategy_id, s.name, s.backtest_summary "
            "ORDER BY realized DESC"
        )
        results = []
        for r in conn.fetchall():
            bt = {}
            if r[6]:
                import json
                try:
                    bt = json.loads(r[6]) if isinstance(r[6], str) else r[6]
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(StrategyPnl(
                strategy_id=r[0], strategy_name=r[1],
                realized_pnl=float(r[2]), unrealized_pnl=float(r[3]),
                win_count=int(r[4]), loss_count=int(r[5]),
                sharpe=bt.get("sharpe") if isinstance(bt, dict) else None,
            ))
        return results
    except Exception:
        logger.warning("fetch_pnl_by_strategy failed", exc_info=True)
        return []


def fetch_trade_decision(conn: PgConnection, symbol: str, entry_date: date) -> str | None:
    """Fetch entry decision reasoning from decision_events."""
    try:
        conn.execute(
            "SELECT reasoning FROM decision_events "
            "WHERE symbol = %s AND event_type = 'trade_entry' "
            "AND created_at::date = %s LIMIT 1",
            (symbol, entry_date),
        )
        row = conn.fetchone()
        return row[0] if row else None
    except Exception:
        logger.warning("fetch_trade_decision failed", exc_info=True)
        return None


def fetch_trade_reflection(conn: PgConnection, symbol: str, closed_at: datetime) -> str | None:
    """Fetch post-trade reflection if recorded."""
    try:
        conn.execute(
            "SELECT lesson FROM trade_reflections "
            "WHERE symbol = %s AND created_at::date = %s::date LIMIT 1",
            (symbol, closed_at),
        )
        row = conn.fetchone()
        return row[0] if row else None
    except Exception:
        logger.warning("fetch_trade_reflection failed", exc_info=True)
        return None


def fetch_pnl_by_symbol(conn: PgConnection) -> list[SymbolPnl]:
    """Return aggregated PnL per symbol."""
    try:
        conn.execute(
            "SELECT symbol, SUM(pnl) AS total_pnl FROM ("
            "  SELECT symbol, realized_pnl AS pnl FROM closed_trades "
            "  UNION ALL "
            "  SELECT symbol, unrealized_pnl AS pnl FROM positions WHERE quantity != 0"
            ") combined GROUP BY symbol ORDER BY total_pnl DESC"
        )
        return [
            SymbolPnl(symbol=r[0], total_pnl=float(r[1]))
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_pnl_by_symbol failed", exc_info=True)
        return []
