"""Research tab container."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.research import (
    BugStatusWidget,
    DiscoveriesWidget,
    MLExperimentsWidget,
    ReflectionsWidget,
    ResearchQueueWidget,
)


class ResearchTab(ScrollableContainer):
    """Research tab — queue, ML experiments, discoveries, reflections, bugs."""

    def compose(self) -> ComposeResult:
        yield ResearchQueueWidget()
        yield MLExperimentsWidget()
        yield DiscoveriesWidget()
        yield ReflectionsWidget()
        yield BugStatusWidget()
