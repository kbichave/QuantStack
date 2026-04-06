"""Portfolio tab container."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.portfolio import (
    ClosedTradesWidget,
    DailyHeatmapWidget,
    EquityCurveWidget,
    EquitySummaryWidget,
    PnlByStrategyWidget,
    PnlBySymbolWidget,
    PositionsTableWidget,
)


class PortfolioTab(ScrollableContainer):
    """Portfolio tab — equity, positions, trades, PnL, heatmap."""

    def compose(self) -> ComposeResult:
        yield EquitySummaryWidget()
        yield EquityCurveWidget()
        yield PositionsTableWidget()
        yield ClosedTradesWidget()
        yield PnlByStrategyWidget()
        yield PnlBySymbolWidget()
        yield DailyHeatmapWidget()
