"""Strategies tab container."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.strategies import PipelineKanbanWidget, PromotionGatesWidget


class StrategiesTab(ScrollableContainer):
    """Strategies tab — pipeline kanban and promotion gates."""

    def compose(self) -> ComposeResult:
        yield PipelineKanbanWidget()
        yield PromotionGatesWidget()
