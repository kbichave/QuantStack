"""Tests for Strategies tab widgets."""
from quantstack.tui.queries.strategies import StrategyCard
from quantstack.tui.widgets.strategies import PipelineKanbanWidget, PromotionGatesWidget


class TestPipelineKanbanWidget:
    def test_renders_columns(self):
        w = PipelineKanbanWidget()
        cards = [
            StrategyCard("s1", "Mom AAPL", "live", "AAPL", "equity", "swing", 1.5, -0.08, 0.65, 10, 500, 30, 30),
            StrategyCard("s2", "MR QQQ", "draft", "QQQ", "equity", "swing", None, None, None, 0, 0, 0, 30),
            StrategyCard("s3", "FT NVDA", "forward_testing", "NVDA", "equity", "swing", 1.0, -0.05, 0.55, 5, 100, 15, 30),
        ]
        w.update_view(cards)

    def test_handles_empty(self):
        w = PipelineKanbanWidget()
        w.update_view([])

    def test_handles_none(self):
        w = PipelineKanbanWidget()
        w.update_view(None)


class TestPromotionGatesWidget:
    def test_renders_without_error(self):
        w = PromotionGatesWidget()
        w.update_view(None)
