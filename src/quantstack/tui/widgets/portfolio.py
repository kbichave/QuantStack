"""Portfolio tab widgets — equity, positions, trades, PnL attribution, heatmap."""
from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget
from quantstack.tui.charts import daily_heatmap, equity_curve, horizontal_bar, sparkline


class EquitySummaryWidget(RefreshableWidget):
    """Equity headline: total, cash, exposure, daily PnL, drawdown."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_equity_summary
            return fetch_equity_summary(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No equity data available", style="dim"))
            return
        result = Text()
        result.append(f"Equity: ${data.total_equity:,.2f}  Cash: ${data.cash:,.2f}  ")
        exposure = ((data.total_equity - data.cash) / data.total_equity * 100) if data.total_equity else 0
        result.append(f"Exposure: {exposure:.1f}%  ")
        color = "green" if data.daily_pnl >= 0 else "red"
        result.append(f"Daily P&L: {data.daily_pnl:+,.2f} ({data.daily_return_pct:+.2f}%)", style=color)
        result.append(f"\nHigh Water: ${data.high_water:,.2f}  ")
        dd_color = "red" if data.drawdown_pct > 5 else ("yellow" if data.drawdown_pct > 2 else "green")
        result.append(f"Drawdown: -{data.drawdown_pct:.1f}%", style=dd_color)
        self.update(result)


class EquityCurveWidget(RefreshableWidget):
    """ASCII equity curve with benchmark comparison."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_benchmark, fetch_equity_curve
            return {"curve": fetch_equity_curve(conn), "benchmark": fetch_benchmark(conn)}

    def update_view(self, data: Any) -> None:
        if not data or not data.get("curve"):
            self.update(Text("No equity history available", style="dim"))
            return
        curve = data["curve"]
        values = [p.equity for p in curve]
        chart = equity_curve(values, width=60, height=5)
        result = Text(chart + "\n")
        result.append_text(sparkline(values, color="green"))
        bench = data.get("benchmark", [])
        if bench:
            result.append("  ")
            result.append_text(sparkline([b.close for b in bench], color="cyan"))
            port_ret = (values[-1] / values[0] - 1) * 100 if values[0] else 0
            bench_ret = (bench[-1].close / bench[0].close - 1) * 100 if bench[0].close else 0
            alpha = port_ret - bench_ret
            result.append(f"\nAlpha: {alpha:+.1f}% vs SPY")
        self.update(result)


class PositionsTableWidget(RefreshableWidget):
    """Open positions data table."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_positions
            return fetch_positions(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No open positions", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["Symbol", "Qty", "Entry", "Current", "P&L", "%", "Strategy", "Days"]:
            table.add_column(col, justify="right" if col not in ("Symbol", "Strategy") else "left")
        for p in data:
            color = "green" if p.unrealized_pnl >= 0 else "red"
            table.add_row(
                p.symbol, f"{p.quantity:.0f}", f"${p.avg_cost:.2f}", f"${p.current_price:.2f}",
                Text(f"${p.unrealized_pnl:+,.2f}", style=color),
                Text(f"{p.unrealized_pnl_pct:+.1f}%", style=color),
                p.strategy_id, str(p.holding_days),
            )
        self.update(table)


class ClosedTradesWidget(RefreshableWidget):
    """Recent closed trades table."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_closed_trades
            return fetch_closed_trades(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No closed trades", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["Date", "Symbol", "Side", "P&L", "Days", "Strategy", "Reason"]:
            table.add_column(col)
        for t in data:
            color = "green" if t.realized_pnl >= 0 else "red"
            dt = t.closed_at.strftime("%m/%d") if t.closed_at else "?"
            table.add_row(
                dt, t.symbol, t.side,
                Text(f"${t.realized_pnl:+,.2f}", style=color),
                str(t.holding_days), t.strategy_id, t.exit_reason or "",
            )
        self.update(table)


class PnlByStrategyWidget(RefreshableWidget):
    """P&L attribution by strategy."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_pnl_by_strategy
            return fetch_pnl_by_strategy(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No strategy P&L data", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["Strategy", "Realized", "Unrealized", "W/L", "Sharpe"]:
            table.add_column(col)
        for s in data:
            r_color = "green" if s.realized_pnl >= 0 else "red"
            u_color = "green" if s.unrealized_pnl >= 0 else "red"
            table.add_row(
                s.strategy_name,
                Text(f"${s.realized_pnl:+,.2f}", style=r_color),
                Text(f"${s.unrealized_pnl:+,.2f}", style=u_color),
                f"W:{s.win_count} L:{s.loss_count}",
                f"{s.sharpe:.2f}" if s.sharpe is not None else "N/A",
            )
        self.update(table)


class PnlBySymbolWidget(RefreshableWidget):
    """Per-symbol P&L with horizontal bars."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_pnl_by_symbol
            return fetch_pnl_by_symbol(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No symbol P&L data", style="dim"))
            return
        max_abs = max(abs(s.total_pnl) for s in data) or 1
        result = Text()
        for s in data:
            color = "green" if s.total_pnl >= 0 else "red"
            result.append(f"{s.symbol:6s} ")
            result.append_text(horizontal_bar(abs(s.total_pnl), max_abs, width=20, color=color))
            result.append(f" ${s.total_pnl:+,.2f}\n")
        self.update(result)


class DailyHeatmapWidget(RefreshableWidget):
    """Mon-Fri P&L heatmap."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-portfolio"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_equity_curve
            return fetch_equity_curve(conn)

    def update_view(self, data: Any) -> None:
        if not data or len(data) < 2:
            self.update(Text("No daily data for heatmap", style="dim"))
            return
        daily_pnl = []
        dates = []
        for i in range(1, len(data)):
            daily_pnl.append(data[i].equity - data[i - 1].equity)
            dates.append(data[i].date)
        self.update(daily_heatmap(daily_pnl, dates))
