"""Research tab widgets — queue, ML experiments, discoveries, reflections, bugs."""
from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget


class ResearchQueueWidget(RefreshableWidget):
    """Active WIP and pending research queue."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-research"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import fetch_research_queue, fetch_research_wip
            return {"wip": fetch_research_wip(conn), "queue": fetch_research_queue(conn)}

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No research data", style="dim"))
            return
        result = Text()
        wip = data.get("wip", [])
        if wip:
            result.append("Work in Progress:\n", style="bold")
            for w in wip:
                result.append(f"  {w.symbol} ({w.domain}) — {w.agent_id} — {w.duration_minutes:.0f}m\n")
        else:
            result.append("No active research\n", style="dim")
        queue = data.get("queue", [])
        if queue:
            result.append("Queue:\n", style="bold")
            for q in queue[:10]:
                result.append(f"  [{q.priority}] {q.task_type}: {q.topic}\n")
        else:
            result.append("Queue empty\n", style="dim")
        self.update(result)


class MLExperimentsWidget(RefreshableWidget):
    """Recent ML experiments with concept drift alerts."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-research"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import fetch_concept_drift, fetch_ml_experiments
            return {"experiments": fetch_ml_experiments(conn), "drift": fetch_concept_drift(conn)}

    def update_view(self, data: Any) -> None:
        if not data or not data.get("experiments"):
            self.update(Text("No experiments recorded", style="dim"))
            return
        drift = data.get("drift", [])
        result = Text()
        if drift:
            result.append("Concept Drift Alerts:\n", style="bold red")
            for d in drift:
                result.append(f"  {d.symbol}: AUC {d.historical_auc:.3f} -> {d.recent_auc:.3f} (drift: {d.drift_magnitude:.3f})\n", style="yellow")
        table = Table(show_edge=False, box=None)
        for col in ["Date", "Model", "Symbol", "AUC", "Features", "Verdict"]:
            table.add_column(col)
        for e in data["experiments"]:
            v_color = "green" if e.verdict == "promoted" else ("red" if e.verdict == "rejected" else "")
            table.add_row(
                e.created_at.strftime("%m/%d") if e.created_at else "?",
                e.model_type, e.symbol,
                f"{e.test_auc:.3f}" if e.test_auc is not None else "N/A",
                str(e.feature_count),
                Text(e.verdict, style=v_color),
            )
        self.update(table)


class DiscoveriesWidget(RefreshableWidget):
    """Alpha research programs and breakthrough features."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-research"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import fetch_alpha_programs, fetch_breakthroughs
            return {"programs": fetch_alpha_programs(conn), "breakthroughs": fetch_breakthroughs(conn)}

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No discovery data", style="dim"))
            return
        result = Text()
        programs = data.get("programs", [])
        if programs:
            result.append("Alpha Programs:\n", style="bold")
            for p in programs:
                result.append(f"  {p.thesis} ({p.status}) exps={p.experiments_run}\n")
        breakthroughs = data.get("breakthroughs", [])
        if breakthroughs:
            result.append("Breakthroughs:\n", style="bold")
            for b in breakthroughs[:5]:
                result.append(f"  {b.feature_name}: importance {b.importance:.3f}\n")
        if not programs and not breakthroughs:
            result.append("No discoveries", style="dim")
        self.update(result)


class ReflectionsWidget(RefreshableWidget):
    """Trade reflections — lessons learned."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-research"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import fetch_reflections
            return fetch_reflections(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No reflections yet", style="dim"))
            return
        result = Text()
        for r in data[:10]:
            color = "green" if r.realized_pnl_pct >= 0 else "red"
            result.append(f"{r.symbol} ", style="bold")
            result.append(f"{r.realized_pnl_pct:+.1f}% ", style=color)
            result.append(f"{r.lesson}\n")
        self.update(result)


class BugStatusWidget(RefreshableWidget):
    """Self-healing / AutoResearchClaw bug status."""

    REFRESH_TIER = "T3"
    TAB_ID = "tab-research"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.research import fetch_bugs
            return fetch_bugs(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No bugs tracked", style="dim"))
            return
        result = Text()
        result.append(f"Open bugs: {len(data)}\n", style="bold")
        for b in data[:10]:
            style = "red" if b.status == "open" else "yellow"
            result.append(f"  [{b.status}] {b.tool_name}: {b.error_message[:60]}\n", style=style)
        self.update(result)
