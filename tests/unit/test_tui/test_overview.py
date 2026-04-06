"""Tests for Overview tab compact widgets."""
from datetime import datetime
from unittest.mock import MagicMock, patch

from rich.text import Text

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


class TestServicesCompact:
    def test_handles_empty_data(self):
        w = ServicesCompact()
        w.update_view(None)

    def test_handles_checkpoint_list(self):
        from quantstack.tui.queries.system import GraphCheckpoint
        now = datetime.now()
        cps = [
            GraphCheckpoint("research", "node1", 9, now, 12.0),
            GraphCheckpoint("trading", "scan", 5, now, 28.0),
        ]
        w = ServicesCompact()
        w.update_view(cps)


class TestRiskCompact:
    def test_handles_none_data(self):
        w = RiskCompact()
        w.update_view(None)

    def test_renders_halt_when_killed(self):
        w = RiskCompact()
        w.update_view({"snapshot": None, "halted": True})

    def test_renders_ok_when_not_killed(self):
        w = RiskCompact()
        w.update_view({"snapshot": None, "halted": False})


class TestPortfolioCompact:
    def test_handles_none_data(self):
        w = PortfolioCompact()
        w.update_view(None)

    def test_renders_equity(self):
        from quantstack.tui.queries.portfolio import EquitySummary
        eq = EquitySummary(10234.0, 3456.0, 127.5, 1.26, 10500.0, 2.5)
        w = PortfolioCompact()
        w.update_view({"equity": eq, "positions": []})


class TestTradesCompact:
    def test_handles_empty(self):
        w = TradesCompact()
        w.update_view([])

    def test_handles_none(self):
        w = TradesCompact()
        w.update_view(None)


class TestStrategyCountsCompact:
    def test_handles_empty(self):
        w = StrategyCountsCompact()
        w.update_view([])

    def test_handles_none(self):
        w = StrategyCountsCompact()
        w.update_view(None)


class TestSignalsCompact:
    def test_handles_empty(self):
        w = SignalsCompact()
        w.update_view([])

    def test_handles_none(self):
        w = SignalsCompact()
        w.update_view(None)


class TestDataHealthCompact:
    def test_handles_none(self):
        w = DataHealthCompact()
        w.update_view(None)

    def test_renders_bars(self):
        w = DataHealthCompact()
        w.update_view({"ohlcv": 10, "news": 5, "sent": 3, "fund": 8})


class TestResearchCompact:
    def test_handles_none(self):
        w = ResearchCompact()
        w.update_view(None)

    def test_renders_counts(self):
        w = ResearchCompact()
        w.update_view({"wip": 2, "queue": 12, "ml": 47, "bugs": 0, "breakthroughs": 2})


class TestAgentActivityLine:
    def test_handles_empty(self):
        w = AgentActivityLine()
        w.update_view([])

    def test_handles_none(self):
        w = AgentActivityLine()
        w.update_view(None)


class TestDigestCompact:
    def test_renders_without_error(self):
        w = DigestCompact()
        w.update_view(None)


class TestDecisionsCompact:
    def test_handles_empty(self):
        w = DecisionsCompact()
        w.update_view([])

    def test_handles_none(self):
        w = DecisionsCompact()
        w.update_view(None)
