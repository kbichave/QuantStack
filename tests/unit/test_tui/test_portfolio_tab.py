"""Tests for Portfolio tab widgets."""
from datetime import date, datetime

from quantstack.tui.queries.portfolio import (
    BenchmarkPoint,
    ClosedTrade,
    EquityPoint,
    EquitySummary,
    Position,
    StrategyPnl,
    SymbolPnl,
)
from quantstack.tui.widgets.portfolio import (
    ClosedTradesWidget,
    DailyHeatmapWidget,
    EquityCurveWidget,
    EquitySummaryWidget,
    PnlByStrategyWidget,
    PnlBySymbolWidget,
    PositionsTableWidget,
)


class TestEquitySummaryWidget:
    def test_renders_with_data(self):
        w = EquitySummaryWidget()
        w.update_view(EquitySummary(10234.56, 3456.78, 127.5, 1.26, 10500.0, 2.5))

    def test_handles_none(self):
        w = EquitySummaryWidget()
        w.update_view(None)


class TestEquityCurveWidget:
    def test_renders_with_data(self):
        w = EquityCurveWidget()
        curve = [EquityPoint(date(2026, 4, i), 10000 + i * 100) for i in range(1, 6)]
        bench = [BenchmarkPoint(date(2026, 4, i), "SPY", 500 + i * 5, 0.5) for i in range(1, 6)]
        w.update_view({"curve": curve, "benchmark": bench})

    def test_handles_empty(self):
        w = EquityCurveWidget()
        w.update_view({"curve": [], "benchmark": []})

    def test_handles_none(self):
        w = EquityCurveWidget()
        w.update_view(None)


class TestPositionsTableWidget:
    def test_renders_positions(self):
        w = PositionsTableWidget()
        w.update_view([
            Position("AAPL", 100, 150, 160, 1000, 6.67, "strat_1", 5),
        ])

    def test_handles_empty(self):
        w = PositionsTableWidget()
        w.update_view([])


class TestClosedTradesWidget:
    def test_renders_trades(self):
        w = ClosedTradesWidget()
        w.update_view([
            ClosedTrade("AAPL", "sell", 250.0, 3, "s1", "stop_loss", datetime.now()),
        ])

    def test_handles_empty(self):
        w = ClosedTradesWidget()
        w.update_view([])


class TestPnlByStrategyWidget:
    def test_renders_data(self):
        w = PnlByStrategyWidget()
        w.update_view([StrategyPnl("s1", "Momentum", 500, 200, 5, 2, 1.5)])

    def test_handles_empty(self):
        w = PnlByStrategyWidget()
        w.update_view([])


class TestPnlBySymbolWidget:
    def test_renders_bars(self):
        w = PnlBySymbolWidget()
        w.update_view([SymbolPnl("AAPL", 750), SymbolPnl("NVDA", -200)])

    def test_handles_empty(self):
        w = PnlBySymbolWidget()
        w.update_view([])


class TestDailyHeatmapWidget:
    def test_renders_heatmap(self):
        w = DailyHeatmapWidget()
        data = [EquityPoint(date(2026, 3, 25 + i), 10000 + i * 50) for i in range(5)]
        w.update_view(data)

    def test_handles_empty(self):
        w = DailyHeatmapWidget()
        w.update_view([])

    def test_handles_none(self):
        w = DailyHeatmapWidget()
        w.update_view(None)
