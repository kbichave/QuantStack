"""Strategies tab widgets — pipeline kanban, promotion gates."""
from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget
from quantstack.tui.charts import progress_bar


class PipelineKanbanWidget(RefreshableWidget):
    """Kanban-style strategy pipeline: Draft -> Backtested -> Forward Testing -> Live -> Retired."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-strategies"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.strategies import fetch_strategy_pipeline
            return fetch_strategy_pipeline(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No strategies", style="dim"))
            return
        buckets: dict[str, list] = {
            "Draft": [], "Backtested": [], "Forward Testing": [], "Live": [], "Retired": [],
        }
        status_map = {
            "draft": "Draft", "backtested": "Backtested", "forward_testing": "Forward Testing",
            "live": "Live", "retired": "Retired",
        }
        for s in data:
            bucket = status_map.get(s.status, "Draft")
            buckets[bucket].append(s)
        table = Table(show_edge=False, box=None)
        for col in buckets:
            table.add_column(col, width=22)
        max_rows = max(len(v) for v in buckets.values()) if buckets else 0
        for i in range(max_rows):
            cells = []
            for col in buckets:
                items = buckets[col]
                if i < len(items):
                    s = items[i]
                    card = f"{s.name}\n{s.symbol}"
                    if s.sharpe is not None:
                        card += f" S:{s.sharpe:.1f}"
                    if s.status == "forward_testing":
                        card += f"\n{s.fwd_days}/{s.fwd_required_days}d"
                    cells.append(card)
                else:
                    cells.append("")
            table.add_row(*cells)
        self.update(table)


class PromotionGatesWidget(RefreshableWidget):
    """Display promotion gate criteria for each pipeline transition."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-strategies"

    def fetch_data(self) -> Any:
        return None

    def update_view(self, data: Any) -> None:
        result = Text()
        result.append("Promotion Gates\n", style="bold")
        result.append("Draft -> BT: ", style="cyan")
        result.append("Hypothesis defined, backtest complete\n")
        result.append("BT -> FT: ", style="cyan")
        result.append("Sharpe > 0.5, MaxDD < 15%, Win rate > 40%\n")
        result.append("FT -> Live: ", style="cyan")
        result.append("30d forward test, Sharpe > 1.0, > 10 trades, profitable\n")
        self.update(result)
