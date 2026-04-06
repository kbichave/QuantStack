"""Agents tab container."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.agents import (
    AgentRosterWidget,
    AgentScorecardWidget,
    GraphActivityWidget,
)


class AgentsTab(ScrollableContainer):
    """Agents tab — agent roster, graph activity, and agent scorecard."""

    def compose(self) -> ComposeResult:
        yield AgentRosterWidget()
        yield GraphActivityWidget()
        yield AgentScorecardWidget()
