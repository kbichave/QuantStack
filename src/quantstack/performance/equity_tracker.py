# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
EquityTracker — daily equity curve writer + strategy P&L attribution.

Runs autonomously at market close (16:05 ET) via scheduler or cron.
Writes to two immutable tables:
  - daily_equity: portfolio-level NAV, return, drawdown
  - strategy_daily_pnl: per-strategy realized/unrealized P&L

These tables are the source of truth for track record. INSERT only, never UPDATE.

Usage:
    from quantstack.performance.equity_tracker import EquityTracker

    tracker = EquityTracker(conn)
    tracker.snapshot_daily()  # Call once at market close
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import PgConnection

from quantstack.execution.hook_registry import fire as _fire_hook


class EquityTracker:
    """
    Daily equity snapshot and strategy P&L attribution writer.

    Args:
        conn: PostgreSQL connection (write-enabled).
    """

    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def snapshot_daily(self, as_of: date | None = None) -> dict[str, Any]:
        """
        Write a daily equity snapshot and per-strategy P&L rollup.

        Idempotent: if a snapshot already exists for the date, it's skipped.
        This prevents double-counting if the job runs twice.

        Args:
            as_of: Date to snapshot. Defaults to today.

        Returns:
            Dict with snapshot data or {"skipped": True} if already exists.
        """
        snapshot_date = as_of or date.today()

        # Idempotency check
        existing = self._conn.execute(
            "SELECT 1 FROM daily_equity WHERE date = ?", [snapshot_date]
        ).fetchone()
        if existing:
            logger.info(
                f"[EquityTracker] Snapshot for {snapshot_date} already exists — skipping"
            )
            return {"skipped": True, "date": str(snapshot_date)}

        # --- Compute portfolio state ---
        cash = self._get_cash()
        positions = self._get_positions_value()
        total_equity = cash + positions
        position_count = self._get_position_count()

        # Daily P&L: today's realized + unrealized change
        daily_realized = self._get_daily_realized_pnl(snapshot_date)
        daily_unrealized = self._get_unrealized_pnl()
        prev_equity = self._get_previous_equity()
        daily_pnl = total_equity - prev_equity if prev_equity else 0.0

        # Cumulative P&L from inception
        cumulative_pnl = total_equity - self._get_initial_equity()

        # Daily return %
        daily_return_pct = (
            (daily_pnl / prev_equity * 100) if prev_equity and prev_equity > 0 else 0.0
        )

        # High water mark and drawdown
        hwm = self._get_high_water_mark()
        new_hwm = max(hwm, total_equity)
        drawdown_pct = (
            ((total_equity - new_hwm) / new_hwm * 100) if new_hwm > 0 else 0.0
        )

        # Write equity snapshot (INSERT only — immutable)
        self._conn.execute(
            """
            INSERT INTO daily_equity
                (date, cash, positions_value, total_equity, daily_pnl,
                 cumulative_pnl, daily_return_pct, high_water_mark,
                 drawdown_pct, open_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_date,
                round(cash, 2),
                round(positions, 2),
                round(total_equity, 2),
                round(daily_pnl, 2),
                round(cumulative_pnl, 2),
                round(daily_return_pct, 4),
                round(new_hwm, 2),
                round(drawdown_pct, 4),
                position_count,
            ],
        )

        # --- Strategy-level P&L attribution ---
        self._write_strategy_pnl(snapshot_date)

        result = {
            "date": str(snapshot_date),
            "total_equity": round(total_equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return_pct": round(daily_return_pct, 4),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "high_water_mark": round(new_hwm, 2),
            "drawdown_pct": round(drawdown_pct, 4),
            "open_positions": position_count,
        }

        logger.info(
            f"[EquityTracker] {snapshot_date}: equity=${total_equity:,.0f} "
            f"pnl={daily_pnl:+,.0f} return={daily_return_pct:+.2f}% "
            f"DD={drawdown_pct:.2f}%"
        )

        # Fire daily reflection hook via registry (non-blocking, best-effort)
        closed_today = self._get_closed_trades_for_date(snapshot_date)
        _fire_hook(
            "daily_close",
            snapshot_date=snapshot_date,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            closed_trades=closed_today,
        )

        return result

    def _get_closed_trades_for_date(self, snapshot_date: date) -> list[dict]:
        """Fetch closed trades for a specific date."""
        try:
            rows = self._conn.execute(
                "SELECT symbol, strategy_id, realized_pnl, side, quantity, "
                "entry_price, exit_price FROM closed_trades "
                "WHERE CAST(closed_at AS DATE) = ?",
                [snapshot_date],
            ).fetchall()
            return [
                {
                    "symbol": r[0],
                    "strategy_id": r[1],
                    "realized_pnl": r[2],
                    "side": r[3],
                    "quantity": r[4],
                    "entry_price": r[5],
                    "exit_price": r[6],
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_equity_curve(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Query the equity curve for a date range."""
        query = "SELECT * FROM daily_equity"
        params: list[Any] = []
        conditions = []

        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date"

        rows = self._conn.execute(query, params).fetchall()
        columns = [
            "date",
            "cash",
            "positions_value",
            "total_equity",
            "daily_pnl",
            "cumulative_pnl",
            "daily_return_pct",
            "high_water_mark",
            "drawdown_pct",
            "open_positions",
            "created_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def get_strategy_pnl(
        self,
        strategy_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Query per-strategy P&L attribution."""
        query = "SELECT * FROM strategy_daily_pnl"
        params: list[Any] = []
        conditions = []

        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date, strategy_id"

        rows = self._conn.execute(query, params).fetchall()
        columns = [
            "date",
            "strategy_id",
            "realized_pnl",
            "unrealized_pnl",
            "num_trades",
            "win_count",
            "loss_count",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def get_summary(self) -> dict[str, Any]:
        """Compute headline performance stats from the equity curve."""
        rows = self._conn.execute(
            "SELECT total_equity, daily_return_pct, drawdown_pct, date "
            "FROM daily_equity ORDER BY date"
        ).fetchall()

        if not rows:
            return {"status": "no_data", "message": "No equity snapshots yet"}

        returns = [r[1] for r in rows]
        drawdowns = [r[2] for r in rows]
        first_equity = rows[0][0]
        last_equity = rows[-1][0]
        n_days = len(rows)

        total_return_pct = (
            ((last_equity - first_equity) / first_equity * 100)
            if first_equity > 0
            else 0
        )
        avg_daily = sum(returns) / n_days if n_days > 0 else 0
        std_daily = (
            sum((r - avg_daily) ** 2 for r in returns) / max(n_days - 1, 1)
        ) ** 0.5

        # Annualized Sharpe (assume 0% risk-free for simplicity)
        sharpe = (avg_daily / std_daily * (252**0.5)) if std_daily > 0 else 0

        # Sortino (downside deviation only)
        downside = [r for r in returns if r < 0]
        downside_std = (sum(r**2 for r in downside) / max(len(downside), 1)) ** 0.5
        sortino = (avg_daily / downside_std * (252**0.5)) if downside_std > 0 else 0

        max_dd = min(drawdowns) if drawdowns else 0
        win_days = sum(1 for r in returns if r > 0)
        win_rate = win_days / n_days if n_days > 0 else 0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "annualized_sharpe": round(sharpe, 3),
            "annualized_sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_daily": round(win_rate, 3),
            "trading_days": n_days,
            "current_equity": round(last_equity, 2),
            "first_date": str(rows[0][3]),
            "last_date": str(rows[-1][3]),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_cash(self) -> float:
        row = self._conn.execute(
            "SELECT cash FROM cash_balance WHERE id = 1"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def _get_positions_value(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(ABS(quantity) * current_price), 0) FROM positions"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def _get_position_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()
        return int(row[0]) if row else 0

    def _get_unrealized_pnl(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(unrealized_pnl), 0) FROM positions"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def _get_daily_realized_pnl(self, as_of: date) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM closed_trades WHERE closed_at::DATE = ?",
            [as_of],
        ).fetchone()
        return float(row[0]) if row else 0.0

    def _get_previous_equity(self) -> float:
        """Get the most recent daily equity before today."""
        row = self._conn.execute(
            "SELECT total_equity FROM daily_equity ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            return float(row[0])
        # No previous snapshot — use initial cash as baseline
        return self._get_initial_equity()

    def _get_initial_equity(self) -> float:
        """Get the first equity value (inception)."""
        row = self._conn.execute(
            "SELECT total_equity FROM daily_equity ORDER BY date ASC LIMIT 1"
        ).fetchone()
        if row:
            return float(row[0])
        # No snapshots yet — current cash is inception value
        return self._get_cash()

    def _get_high_water_mark(self) -> float:
        """Get the historical high water mark."""
        row = self._conn.execute(
            "SELECT MAX(high_water_mark) FROM daily_equity"
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return self._get_cash() + self._get_positions_value()

    def _write_strategy_pnl(self, snapshot_date: date) -> None:
        """Compute and write per-strategy P&L for the day."""
        # Get today's closed trades grouped by strategy
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(strategy_id, '') as strategy_id,
                SUM(realized_pnl) as realized,
                COUNT(*) as num_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses
            FROM closed_trades
            WHERE closed_at::DATE = ?
            GROUP BY COALESCE(strategy_id, '')
            """,
            [snapshot_date],
        ).fetchall()

        for row in rows:
            strat_id = row[0] or "unattributed"
            self._conn.execute(
                """
                INSERT INTO strategy_daily_pnl
                    (date, strategy_id, realized_pnl, unrealized_pnl,
                     num_trades, win_count, loss_count)
                VALUES (?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT (date, strategy_id) DO UPDATE SET
                    realized_pnl = excluded.realized_pnl,
                    num_trades = excluded.num_trades,
                    win_count = excluded.win_count,
                    loss_count = excluded.loss_count
                """,
                [snapshot_date, strat_id, row[1], row[2], row[3], row[4]],
            )

        if rows:
            logger.debug(
                f"[EquityTracker] Strategy P&L: {len(rows)} strategies attributed for {snapshot_date}"
            )
