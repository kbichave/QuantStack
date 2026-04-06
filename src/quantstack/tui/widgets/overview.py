"""Overview tab compact widgets — single-screen glance at all subsystems."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget
from quantstack.tui.charts import horizontal_bar


class ServicesCompact(RefreshableWidget):
    """Graph service status: R:UP T:UP S:UP with cycle counts."""

    REFRESH_TIER = "T1"
    TAB_ID = "tab-overview"
    ALWAYS_ON = True

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.system import fetch_graph_checkpoints
            return fetch_graph_checkpoints(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("R:? c#? ?s  T:? c#? ?s  S:? c#? ?s  Errors: 0", style="dim"))
            return
        by_name = {cp.graph_name: cp for cp in data}
        now = datetime.now()
        parts = []
        for short, full in [("R", "research"), ("T", "trading"), ("S", "supervisor")]:
            cp = by_name.get(full)
            if cp and cp.started_at:
                ago = int((now - cp.started_at.replace(tzinfo=None)).total_seconds())
                status = "UP" if ago < 120 else "DOWN"
                color = "green" if status == "UP" else "red"
                parts.append(Text(f"{short}:{status} c#{cp.cycle_number} {ago}s", style=color))
            else:
                parts.append(Text(f"{short}:? c#? ?s", style="dim"))
        result = Text("  ").join(parts)
        result.append("  Errors: 0")
        self.update(result)


class RiskCompact(RefreshableWidget):
    """Risk summary: exposure, drawdown, VaR, alerts, kill switch."""

    REFRESH_TIER = "T1"
    TAB_ID = "tab-overview"
    ALWAYS_ON = True

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.risk import fetch_risk_snapshot
            from quantstack.tui.queries.system import fetch_kill_switch
            return {"snapshot": fetch_risk_snapshot(conn), "halted": fetch_kill_switch(conn)}

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No risk data | Kill: ok", style="dim"))
            return
        snap = data.get("snapshot")
        halted = data.get("halted", False)
        kill_text = Text("HALT", style="bold red") if halted else Text("ok", style="green")
        result = Text()
        if snap:
            result.append(f"Exposure: {snap.gross_exposure:.0%}  DD: {snap.max_drawdown:+.1%}  ")
            result.append(f"VaR: ${snap.var_1d:,.0f}  ")
        else:
            result.append("No risk data  ")
        result.append("Kill: ")
        result.append_text(kill_text)
        self.update(result)


class PortfolioCompact(RefreshableWidget):
    """Equity summary and top 2 positions."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_equity_summary, fetch_positions
            return {"equity": fetch_equity_summary(conn), "positions": fetch_positions(conn)}

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("Equity: --", style="dim"))
            return
        eq = data.get("equity")
        positions = data.get("positions", [])
        result = Text()
        if eq:
            result.append(f"Equity: ${eq.total_equity:,.2f}  ")
            color = "green" if eq.daily_pnl >= 0 else "red"
            result.append(f"Today: {eq.daily_pnl:+,.2f}", style=color)
        else:
            result.append("Equity: --", style="dim")
        if positions:
            top = positions[:2]
            result.append("  Open: ")
            result.append("  ".join(f"{p.symbol} {p.unrealized_pnl_pct:+.1f}%" for p in top))
        self.update(result)


class TradesCompact(RefreshableWidget):
    """Last 3 closed trades."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.portfolio import fetch_closed_trades
            return fetch_closed_trades(conn, limit=3)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No recent trades", style="dim"))
            return
        parts = []
        for t in data[:3]:
            color = "green" if t.realized_pnl >= 0 else "red"
            parts.append(Text(f"{t.symbol} {t.realized_pnl:+,.2f} ({t.holding_days}d)", style=color))
        self.update(Text("  ").join(parts))


class StrategyCountsCompact(RefreshableWidget):
    """Strategy pipeline counts by status."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.strategies import fetch_strategy_pipeline
            return fetch_strategy_pipeline(conn)

    def update_view(self, data: Any) -> None:
        counts = {"draft": 0, "backtested": 0, "forward_testing": 0, "live": 0, "retired": 0}
        for s in (data or []):
            counts[s.status] = counts.get(s.status, 0) + 1
        text = (
            f"Draft:{counts['draft']} BT:{counts['backtested']} "
            f"FT:{counts['forward_testing']} Live:{counts['live']} Ret:{counts['retired']}"
        )
        self.update(Text(text))


class SignalsCompact(RefreshableWidget):
    """Top 3 signals by confidence."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.signals import fetch_active_signals
            return fetch_active_signals(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No active signals", style="dim"))
            return
        parts = []
        for s in data[:3]:
            color = {"BUY": "green", "SELL": "red"}.get(s.action, "yellow")
            parts.append(Text(f"{s.symbol} {s.action} {s.confidence:.0%}", style=color))
        self.update(Text("  ").join(parts))


class DataHealthCompact(RefreshableWidget):
    """Coverage bars for 4 key data types."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.data_health import (
                fetch_news_freshness,
                fetch_ohlcv_freshness,
                fetch_options_freshness,
                fetch_sentiment_freshness,
            )
            return {
                "ohlcv": len(fetch_ohlcv_freshness(conn)),
                "news": len(fetch_news_freshness(conn)),
                "sent": len(fetch_sentiment_freshness(conn)),
                "opts": len(fetch_options_freshness(conn)),
            }

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("Data health: no data", style="dim"))
            return
        total = max(data.get("ohlcv", 0), 1)
        result = Text()
        for label, key in [("OHLCV", "ohlcv"), ("News", "news"), ("Sent", "sent"), ("Opts", "opts")]:
            count = data.get(key, 0)
            result.append(f"{label} ")
            result.append_text(horizontal_bar(count, total, width=12))
            result.append("  ")
        self.update(result)


class ResearchCompact(RefreshableWidget):
    """Research activity counts."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import (
                fetch_breakthroughs,
                fetch_bugs,
                fetch_ml_experiments,
                fetch_research_queue,
                fetch_research_wip,
            )
            return {
                "wip": len(fetch_research_wip(conn)),
                "queue": len(fetch_research_queue(conn)),
                "ml": len(fetch_ml_experiments(conn)),
                "bugs": len(fetch_bugs(conn)),
                "breakthroughs": len(fetch_breakthroughs(conn)),
            }

    def update_view(self, data: Any) -> None:
        if not data:
            data = {}
        result = Text()
        result.append(f"WIP: {data.get('wip', 0)}  Queue: {data.get('queue', 0)}  ")
        result.append(f"ML: {data.get('ml', 0)}  ")
        bugs = data.get("bugs", 0)
        result.append(f"Bugs: {bugs}", style="red" if bugs > 0 else "")
        result.append(f"  Breakthroughs: {data.get('breakthroughs', 0)}")
        self.update(result)


class AgentActivityLine(RefreshableWidget):
    """One line per graph showing current agent and last tool call."""

    REFRESH_TIER = "T1"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.system import fetch_agent_events
            return fetch_agent_events(conn, limit=60)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("Research: idle  Trading: idle  Supervisor: idle", style="dim"))
            return
        now = datetime.now()
        by_graph: dict[str, Any] = {}
        for ev in data:
            if ev.graph_name not in by_graph:
                by_graph[ev.graph_name] = ev
        lines = []
        for name in ["research", "trading", "supervisor"]:
            ev = by_graph.get(name)
            if ev:
                ago = int((now - ev.created_at.replace(tzinfo=None)).total_seconds())
                lines.append(f"{name.title()}: {ev.agent_name} -> {ev.event_type} ({ago}s ago)")
            else:
                lines.append(f"{name.title()}: idle")
        self.update(Text("\n".join(lines)))


class DigestCompact(RefreshableWidget):
    """Daily digest summary after market close."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        return None

    def update_view(self, data: Any) -> None:
        now = datetime.now()
        if now.hour >= 17:
            self.update(Text("Daily digest available after market close", style="dim"))
        else:
            self.update(Text("Daily digest available after 17:00 ET", style="dim"))


class DecisionsCompact(RefreshableWidget):
    """Last 3 decision events."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.system import fetch_agent_events
            return fetch_agent_events(conn, limit=3)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No recent decisions", style="dim"))
            return
        lines = []
        for ev in data[:3]:
            ts = ev.created_at.strftime("%H:%M") if ev.created_at else "??:??"
            lines.append(f"{ts} {ev.agent_name} {ev.event_type} {ev.content[:60]}")
        self.update(Text("\n".join(lines)))
