"""Data & Signals tab widgets — calendar, data health matrix, signal engine."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from rich.table import Table
from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget
from quantstack.tui.charts import horizontal_bar


class MarketCalendarWidget(RefreshableWidget):
    """Upcoming market events: earnings."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-data-signals"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.calendar import fetch_earnings_calendar
            return {"earnings": fetch_earnings_calendar(conn)}

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No upcoming events", style="dim"))
            return
        earnings = data.get("earnings", [])
        if not earnings:
            self.update(Text("No upcoming events", style="dim"))
            return
        result = Text()
        for e in earnings[:15]:
            est = f" est={e.estimate:.2f}" if e.estimate is not None else ""
            result.append(f"{e.report_date} {e.symbol} earnings{est}\n", style="cyan")
        self.update(result)


class DataHealthMatrixWidget(RefreshableWidget):
    """Symbol x data-type freshness matrix with coverage bars."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-data-signals"

    STALENESS = {
        "ohlcv": timedelta(days=2),
        "news": timedelta(hours=24),
        "sentiment": timedelta(hours=24),
        "options": timedelta(days=1),
        "insider": timedelta(days=30),
        "macro": timedelta(days=7),
    }

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.data_health import (
                fetch_insider_freshness,
                fetch_macro_freshness,
                fetch_news_freshness,
                fetch_ohlcv_freshness,
                fetch_options_freshness,
                fetch_sentiment_freshness,
            )
            return {
                "ohlcv": fetch_ohlcv_freshness(conn),
                "news": fetch_news_freshness(conn),
                "sentiment": fetch_sentiment_freshness(conn),
                "options": fetch_options_freshness(conn),
                "insider": fetch_insider_freshness(conn),
                "macro": fetch_macro_freshness(conn),
            }

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No data health info", style="dim"))
            return
        today = date.today()
        now = datetime.now()
        all_symbols = set()
        for freshness in data.values():
            all_symbols.update(freshness.keys())
        if not all_symbols:
            self.update(Text("No symbols tracked", style="dim"))
            return
        total = len(all_symbols)
        result = Text()
        for dtype, threshold in self.STALENESS.items():
            freshness = data.get(dtype, {})
            fresh_count = 0
            for ts in freshness.values():
                if not ts:
                    continue
                if isinstance(ts, datetime):
                    age = now - ts.replace(tzinfo=None)
                else:
                    age = timedelta(days=(today - ts).days)
                if age < threshold:
                    fresh_count += 1
            result.append(f"{dtype:13s} ")
            result.append_text(horizontal_bar(fresh_count, total, width=15))
            result.append(f" {fresh_count}/{total}\n")
        self.update(result)


class SignalEngineWidget(RefreshableWidget):
    """Active signals with factor breakdown and collector health."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-data-signals"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.signals import fetch_active_signals
            return fetch_active_signals(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No active signals", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["Symbol", "Action", "Conf", "ML", "Sent", "Tech", "Opt", "Macro"]:
            table.add_column(col)
        for s in data[:20]:
            action_color = {"BUY": "green", "SELL": "red"}.get(s.action, "yellow")
            table.add_row(
                s.symbol,
                Text(s.action, style=action_color),
                f"{s.confidence:.0%}",
                f"{s.factors.get('ml', '-')}",
                f"{s.factors.get('sentiment', '-')}",
                f"{s.factors.get('technical', '-')}",
                f"{s.factors.get('options', '-')}",
                f"{s.factors.get('macro', '-')}",
            )
        self.update(table)
