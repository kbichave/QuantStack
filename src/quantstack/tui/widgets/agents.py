"""Agents tab widgets — graph activity, agent scorecard, agent roster."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget

_GRAPHS_DIR = Path(__file__).resolve().parents[2] / "graphs"

_GRAPH_CONFIGS = [
    ("Research", _GRAPHS_DIR / "research" / "config" / "agents.yaml"),
    ("Trading", _GRAPHS_DIR / "trading" / "config" / "agents.yaml"),
    ("Supervisor", _GRAPHS_DIR / "supervisor" / "config" / "agents.yaml"),
]


class AgentRosterWidget(Static):
    """Static roster of all agents from YAML configs, grouped by graph."""

    def on_mount(self) -> None:
        table = Table(show_edge=False, box=None, title="Agent Roster")
        table.add_column("Graph", style="bold", min_width=10)
        table.add_column("Agent", style="cyan", min_width=24)
        table.add_column("Role", min_width=28)
        table.add_column("Goal", max_width=80)

        for graph_name, config_path in _GRAPH_CONFIGS:
            try:
                agents = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                continue
            for agent_id, cfg in agents.items():
                role = cfg.get("role", "")
                goal = cfg.get("goal", "")
                if len(goal) > 100:
                    goal = goal[:97] + "..."
                table.add_row(graph_name, agent_id, role, goal)

        self.update(table)


class GraphActivityWidget(RefreshableWidget):
    """Three side-by-side panels showing current graph state."""

    REFRESH_TIER = "T1"
    TAB_ID = "tab-agents"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.agents import fetch_cycle_history, fetch_graph_activity
            return {
                "activity": fetch_graph_activity(conn),
                "history": fetch_cycle_history(conn),
            }

    def update_view(self, data: Any) -> None:
        if not data or not data.get("activity"):
            self.update(Text("No graph activity data", style="dim"))
            return
        now = datetime.now()
        result = Text()
        for g in data["activity"]:
            ago = int((now - g.cycle_started.replace(tzinfo=None)).total_seconds()) if g.cycle_started else 0
            result.append(f"{g.graph_name.title()}: ", style="bold")
            result.append(f"{g.current_agent} @ {g.current_node}")
            result.append(f"  c#{g.cycle_number} ({ago}s ago)  events: {g.event_count}\n")
        history = data.get("history", [])
        if history:
            result.append("\nRecent Cycles:\n", style="bold")
            for c in history[:3]:
                result.append(
                    f"  {c.graph_name} c#{c.cycle_number}: "
                    f"{c.duration_seconds:.0f}s, {c.primary_agent}, {c.tool_count} tools\n"
                )
        self.update(result)


class AgentScorecardWidget(RefreshableWidget):
    """Agent performance table with calibration and prompt versions."""

    REFRESH_TIER = "T4"
    TAB_ID = "tab-agents"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.agents import (
                fetch_agent_skills,
                fetch_calibration,
                fetch_prompt_versions,
            )
            return {
                "skills": fetch_agent_skills(conn),
                "calibration": fetch_calibration(conn),
                "prompts": fetch_prompt_versions(conn),
            }

    def update_view(self, data: Any) -> None:
        if not data or not data.get("skills"):
            self.update(Text("No agent data", style="dim"))
            return
        table = Table(show_edge=False, box=None, title="Agent Scorecard")
        for col in ["Agent", "Accuracy", "Win Rate", "Avg P&L", "IC", "Trend"]:
            table.add_column(col)
        for s in data["skills"]:
            trend_color = {"improving": "green", "declining": "red"}.get(s.trend, "")
            table.add_row(
                s.agent_name,
                f"{s.accuracy:.1%}" if s.accuracy is not None else "N/A",
                f"{s.win_rate:.1%}" if s.win_rate is not None else "N/A",
                f"${s.avg_pnl:+,.2f}" if s.avg_pnl is not None else "N/A",
                f"{s.information_coefficient:.3f}" if s.information_coefficient is not None else "N/A",
                Text(s.trend, style=trend_color),
            )
        cal = data.get("calibration", [])
        overconfident = [c for c in cal if c.is_overconfident]
        if overconfident:
            table.caption = f"Overconfident: {', '.join(c.agent_name for c in overconfident)}"
        self.update(table)
