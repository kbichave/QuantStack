"""Overview tab container — single-screen compact summaries."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.overview import (
    AgentActivityLine,
    DataHealthCompact,
    DecisionsCompact,
    DigestCompact,
    PortfolioCompact,
    ResearchCompact,
    RiskCompact,
    ServicesCompact,
    SignalsCompact,
    StrategyCountsCompact,
    TradesCompact,
)


class OverviewTab(ScrollableContainer):
    """Overview tab — compact summaries from all subsystems."""

    def compose(self) -> ComposeResult:
        yield ServicesCompact(classes="overview-cell")
        yield RiskCompact(classes="overview-cell")
        yield PortfolioCompact(classes="overview-cell")
        yield TradesCompact(classes="overview-cell")
        yield StrategyCountsCompact(classes="overview-cell")
        yield SignalsCompact(classes="overview-cell")
        yield DataHealthCompact(classes="overview-cell")
        yield ResearchCompact(classes="overview-cell")
        yield AgentActivityLine(classes="full-width")
        yield DigestCompact(classes="full-width")
        yield DecisionsCompact(classes="full-width")
