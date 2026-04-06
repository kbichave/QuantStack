from typing import Any

from loguru import logger
from rich.text import Text
from textual import work
from textual.widgets import Static


class RefreshableWidget(Static):
    """Base for widgets that fetch data from the DB in a background thread
    and update their rendering on the main thread.

    Subclasses must override:
        fetch_data() -> Any       # runs in thread via @work(thread=True)
        update_view(data) -> None # runs on main thread

    Class attributes subclasses can set:
        REFRESH_TIER: str   # "T1", "T2", "T3", or "T4" (default "T2")
        TAB_ID: str         # tab pane id this widget belongs to (default "tab-overview")
        ALWAYS_ON: bool     # if True, refreshes regardless of active tab
    """

    REFRESH_TIER: str = "T2"
    TAB_ID: str = "tab-overview"
    ALWAYS_ON: bool = False

    def on_mount(self) -> None:
        """Show loading placeholder and register with the scheduler."""
        self.update(Text("Loading...", style="dim"))
        try:
            scheduler = self.app.scheduler  # type: ignore[attr-defined]
            scheduler.register(self.REFRESH_TIER, self, self.TAB_ID, self.ALWAYS_ON)
        except (AttributeError, KeyError):
            pass
        self.refresh_data()

    def refresh_data(self) -> None:
        """Kick off a background fetch. Safe to call from any thread."""
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        try:
            data = self.fetch_data()
        except Exception:
            logger.opt(exception=True).warning(
                "RefreshableWidget fetch failed: {widget}",
                widget=self.__class__.__name__,
            )
            # Still call update_view with None so widget shows its empty state
            # instead of staying stuck on "Loading..." forever.
            self.app.call_from_thread(self.update_view, None)
            return
        self.app.call_from_thread(self.update_view, data)

    def fetch_data(self) -> Any:
        raise NotImplementedError

    def update_view(self, data: Any) -> None:
        raise NotImplementedError
