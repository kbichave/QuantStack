from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

    from quantstack.tui.base import RefreshableWidget


@dataclass
class _WidgetEntry:
    widget: RefreshableWidget
    tab_id: str
    always_on: bool = False


@dataclass
class TieredRefreshScheduler:
    """Coordinates data refresh across 4 tiers with staggered start times.

    Tier config:
        T1 = 5s   (header, kill switch, active agent)
        T2 = 15s  (positions, equity, signals)
        T3 = 60s  (strategies, calendar, research)
        T4 = 120s (ML experiments, calibration, benchmarks)

    Stagger offsets prevent all tiers from hitting the DB simultaneously at startup:
        T1 @ 0.0s, T2 @ 0.3s, T3 @ 0.6s, T4 @ 0.9s
    """

    TIERS: dict[str, float] = field(default_factory=lambda: {
        "T1": 5.0,
        "T2": 15.0,
        "T3": 60.0,
        "T4": 120.0,
    })

    STAGGER: dict[str, float] = field(default_factory=lambda: {
        "T1": 0.0,
        "T2": 0.3,
        "T3": 0.6,
        "T4": 0.9,
    })

    _db_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(5)
    )

    active_tab: str = "tab-overview"

    _registry: dict[str, list[_WidgetEntry]] = field(default_factory=lambda: {
        "T1": [],
        "T2": [],
        "T3": [],
        "T4": [],
    })

    def register(
        self,
        tier: str,
        widget: RefreshableWidget,
        tab_id: str,
        always_on: bool = False,
    ) -> None:
        """Register a widget for periodic refresh in the given tier."""
        self._registry[tier].append(_WidgetEntry(widget, tab_id, always_on))

    def get_refreshable_widgets(self, tier: str) -> list[RefreshableWidget]:
        """Return widgets that should refresh for the given tier right now."""
        return [
            entry.widget
            for entry in self._registry.get(tier, [])
            if entry.always_on or entry.tab_id == self.active_tab
        ]

    def _tick(self, tier: str) -> None:
        """Fire refresh for all eligible widgets in a tier."""
        for widget in self.get_refreshable_widgets(tier):
            if self._db_semaphore.acquire(blocking=False):
                try:
                    widget.refresh_data()
                finally:
                    self._db_semaphore.release()

    def start(self, app: App) -> None:
        """Start all tier timers on the given Textual app."""
        for tier, interval in self.TIERS.items():
            stagger = self.STAGGER[tier]
            app.set_timer(stagger, lambda t=tier, i=interval: app.set_interval(i, lambda t2=t: self._tick(t2)))

    def refresh_tab(self, tab_id: str) -> None:
        """Immediately refresh all widgets belonging to the given tab."""
        self.active_tab = tab_id
        for tier in self.TIERS:
            self._tick(tier)
