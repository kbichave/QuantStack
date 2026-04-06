import subprocess
from io import StringIO
from pathlib import Path

from loguru import logger
from rich.console import Console
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static, TabbedContent, TabPane

from quantstack.tui.refresh import TieredRefreshScheduler
from quantstack.tui.screens.agents import AgentsTab
from quantstack.tui.screens.data_signals import DataSignalsTab
from quantstack.tui.screens.overview import OverviewTab
from quantstack.tui.screens.portfolio import PortfolioTab
from quantstack.tui.screens.research import ResearchTab
from quantstack.tui.screens.strategies import StrategiesTab
from quantstack.tui.widgets.header import HeaderBar


class QuantStackApp(App):
    """Terminal dashboard for QuantStack autonomous trading system."""

    TITLE = "QUANTSTACK"
    CSS_PATH = "dashboard.tcss"
    ALLOW_SELECT = True

    BINDINGS = [
        Binding("1", "switch_tab('tab-overview')", "Overview"),
        Binding("2", "switch_tab('tab-portfolio')", "Portfolio"),
        Binding("3", "switch_tab('tab-strategies')", "Strategies"),
        Binding("4", "switch_tab('tab-data-signals')", "Data & Signals"),
        Binding("5", "switch_tab('tab-agents')", "Agents"),
        Binding("6", "switch_tab('tab-research')", "Research"),
        Binding("q", "quit", "Quit"),
        Binding("c", "copy_tab", "Copy"),
        Binding("r", "force_refresh", "Refresh"),
        Binding("question_mark", "toggle_help", "Help"),
        Binding("slash", "search", "Search"),
        Binding("j", "scroll_down", "Down"),
        Binding("k", "scroll_up", "Up"),
        Binding("enter", "drill_down", "Detail"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.scheduler = TieredRefreshScheduler()
        self._setup_logging()

    @staticmethod
    def _setup_logging() -> None:
        """Redirect loguru to a file so logs don't corrupt the TUI."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.remove()
        logger.add(log_dir / "tui.log", rotation="10 MB", retention="3 days", level="DEBUG")

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        with TabbedContent():
            with TabPane("Overview", id="tab-overview"):
                yield OverviewTab()
            with TabPane("Portfolio", id="tab-portfolio"):
                yield PortfolioTab()
            with TabPane("Strategies", id="tab-strategies"):
                yield StrategiesTab()
            with TabPane("Data & Signals", id="tab-data-signals"):
                yield DataSignalsTab()
            with TabPane("Agents", id="tab-agents"):
                yield AgentsTab()
            with TabPane("Research", id="tab-research"):
                yield ResearchTab()
        yield Static("q:Quit  1-6:Tabs  r:Refresh  c:Copy  ?:Help  /:Search", id="footer")

    def on_mount(self) -> None:
        self.scheduler.start(self)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane and event.pane.id:
            self.scheduler.refresh_tab(event.pane.id)

    def action_switch_tab(self, tab_id: str) -> None:
        tc = self.query_one(TabbedContent)
        tc.active = tab_id

    def action_force_refresh(self) -> None:
        self.scheduler.refresh_tab(self.scheduler.active_tab)

    def action_toggle_help(self) -> None:
        pass  # section 11

    def action_search(self) -> None:
        pass  # future

    def _pbcopy(self, text: str) -> None:
        """Copy text to macOS system clipboard via pbcopy."""
        try:
            subprocess.run(
                ["pbcopy"], input=text.encode("utf-8"), check=True, timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    @staticmethod
    def _render_to_text(obj: object) -> str:
        """Render a Rich renderable to plain text."""
        buf = StringIO()
        console = Console(file=buf, width=120, no_color=True, force_terminal=False)
        try:
            console.print(obj)
        except Exception:
            return str(obj)
        return buf.getvalue()

    def action_copy_tab(self) -> None:
        """Copy visible tab content to system clipboard via pbcopy."""
        tc = self.query_one(TabbedContent)
        active_pane = tc.get_pane(tc.active)
        lines = []
        for widget in active_pane.walk_children():
            content_obj = getattr(widget, "_Static__content", None)
            if content_obj is None:
                continue
            text = self._render_to_text(content_obj)
            if text.strip() and text.strip() != "Loading...":
                lines.append(text.strip())
        content = "\n\n".join(lines)
        if content:
            self._pbcopy(content)
            self.notify("Copied to clipboard", timeout=2)
        else:
            self.notify("Nothing to copy", timeout=2)

    def action_drill_down(self) -> None:
        pass  # section 11
