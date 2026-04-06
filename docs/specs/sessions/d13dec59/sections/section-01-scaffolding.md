# Section 1: Package Scaffolding & Textual App Shell

## Overview

This section creates the `src/quantstack/tui/` package — the foundation for the entire TUI dashboard. It delivers a working Textual App with 6 empty tabs, a header bar, the refresh scheduler, and the base class that every data-driven widget inherits from. Everything else in this plan depends on this section being complete first.

**Critical constraint:** `src/quantstack/dashboard/` already exists and contains `app.py` (FastAPI SSE dashboard on port 8421) and `events.py` (imported by `src/quantstack/graphs/agent_executor.py`). Do NOT touch that package. The TUI lives at `src/quantstack/tui/`.

---

## Tests (Write First)

All test files go under `tests/unit/test_tui/`. Create `tests/unit/test_tui/__init__.py` as an empty file.

### `tests/unit/test_tui/test_app.py`

```python
"""Tests for the Textual app shell."""
import pytest


class TestQuantStackApp:
    """QuantStackApp instantiation and composition."""

    def test_app_instantiates_without_error(self):
        """QuantStackApp() should not raise."""
        ...

    def test_compose_yields_header_tabbed_content_footer(self):
        """compose() must yield HeaderBar, TabbedContent with 6 TabPanes, and a footer Static."""
        ...

    def test_bindings_include_required_keys(self):
        """BINDINGS must include keys: 1-6, q, r, ?, /, j, k, enter."""
        ...

    def test_css_path_points_to_dashboard_tcss(self):
        """CSS_PATH must resolve to 'dashboard.tcss'."""
        ...

    def test_title_is_quantstack(self):
        """App TITLE must be 'QUANTSTACK'."""
        ...
```

### `tests/unit/test_tui/test_refresh.py`

```python
"""Tests for TieredRefreshScheduler."""
import threading
import pytest


class TestTieredRefreshScheduler:
    """Refresh tier configuration and behavior."""

    def test_has_four_tiers(self):
        """TIERS dict must contain exactly T1, T2, T3, T4."""
        ...

    def test_tier_intervals(self):
        """T1=5s, T2=15s, T3=60s, T4=120s."""
        ...

    def test_stagger_offsets(self):
        """Stagger offsets: T1=0.0, T2=0.3, T3=0.6, T4=0.9."""
        ...

    def test_active_tab_change_updates_query_set(self):
        """Switching active_tab changes which widget groups are refreshed."""
        ...

    def test_header_queries_fire_regardless_of_tab(self):
        """Always-on queries (header tier) fire on every T1 tick no matter which tab is active."""
        ...

    def test_db_semaphore_is_threading_semaphore(self):
        """_db_semaphore must be threading.Semaphore (not asyncio), because queries run in @work(thread=True)."""
        ...

    def test_db_semaphore_default_value_is_five(self):
        """Default semaphore permits = 5 (dashboard's share of the PG_POOL_MAX=20 pool)."""
        ...
```

### `tests/unit/test_tui/test_base.py`

```python
"""Tests for RefreshableWidget base class."""
import pytest


class TestRefreshableWidget:
    """Thread-to-UI data flow pattern."""

    def test_refresh_data_calls_fetch_in_thread(self):
        """refresh_data() must invoke fetch_data() inside a Textual @work(thread=True) worker."""
        ...

    def test_update_view_called_on_main_thread(self):
        """After fetch_data() returns, update_view(data) must be called via call_from_thread on the main thread."""
        ...

    def test_fetch_data_exception_does_not_crash_widget(self):
        """If fetch_data() raises, the widget must log the error and remain functional."""
        ...

    def test_subclass_can_override_fetch_and_update(self):
        """A subclass overriding fetch_data() and update_view() should work end-to-end."""
        ...
```

---

## Package Structure

Create the following files. The parent directory `src/quantstack/` already exists.

```
src/quantstack/tui/
├── __init__.py
├── __main__.py
├── app.py
├── base.py
├── refresh.py
├── charts.py              # empty placeholder — implemented in section 3
├── dashboard.tcss
├── widgets/
│   ├── __init__.py
│   ├── header.py
│   └── decisions.py       # empty placeholder — populated in section 4
├── queries/
│   ├── __init__.py
│   └── system.py          # empty placeholder — populated in section 2
└── screens/
    ├── __init__.py
    └── detail.py           # empty placeholder — populated in section 11
```

---

## Implementation Details

### `src/quantstack/tui/__init__.py`

Exports `QuantStackApp` for external use:

```python
from quantstack.tui.app import QuantStackApp

__all__ = ["QuantStackApp"]
```

### `src/quantstack/tui/__main__.py`

Entry point for `python -m quantstack.tui`:

```python
from quantstack.tui.app import QuantStackApp

def main() -> None:
    QuantStackApp().run()

if __name__ == "__main__":
    main()
```

### `src/quantstack/tui/app.py` — QuantStackApp

Subclasses `textual.app.App`. Key design points:

- **TITLE** = `"QUANTSTACK"`
- **CSS_PATH** = `"dashboard.tcss"` (Textual resolves relative to the module file)
- **BINDINGS** — keys 1-6 switch tabs, `q` quits, `r` forces refresh of the active tab, `?` toggles help, `/` opens search (future), `j`/`k` scroll, `enter` triggers drill-down on focused row
- **`compose()`** yields:
  1. `HeaderBar()` (docked top)
  2. `TabbedContent` containing 6 `TabPane`s: Overview, Portfolio, Strategies, Data & Signals, Agents, Research
  3. A `Footer()` or docked `Static` at bottom showing keybinding hints
- **`on_mount()`** creates a `TieredRefreshScheduler` and starts it
- **Tab switching** — handle `TabbedContent.TabActivated` message to tell the scheduler which tab is now active, triggering an immediate refresh of that tab's widgets

Tab pane IDs should be stable strings (e.g., `"tab-overview"`, `"tab-portfolio"`, ...) so the scheduler can map them to widget groups.

### `src/quantstack/tui/base.py` — RefreshableWidget

The central abstraction for background data loading. Every widget that queries the database inherits from this.

```python
from typing import Any
from textual.widgets import Static
from textual.worker import work

class RefreshableWidget(Static):
    """Base for widgets that fetch data from the DB in a background thread
    and update their rendering on the main thread.

    Subclasses must override:
        fetch_data() -> Any       # runs in thread via @work(thread=True)
        update_view(data) -> None # runs on main thread
    """

    def refresh_data(self) -> None:
        """Kick off a background fetch. Safe to call from any thread."""
        self._do_refresh()

    @work(thread=True)
    def _do_refresh(self) -> None:
        try:
            data = self.fetch_data()
        except Exception:
            # Log with loguru; do NOT re-raise — widget stays alive with stale data
            ...
            return
        self.app.call_from_thread(self.update_view, data)

    def fetch_data(self) -> Any:
        raise NotImplementedError

    def update_view(self, data: Any) -> None:
        raise NotImplementedError
```

**Why `@work(thread=True)` and not `asyncio`:** The project uses `psycopg2` (blocking driver) via `pg_conn()`. Running blocking I/O in an async event loop blocks all rendering. Textual's `@work(thread=True)` runs the function in a thread pool, then `call_from_thread()` safely marshals the result back to the main/async thread where widget mutations are allowed.

**Why `call_from_thread` and not `post_message`:** `call_from_thread` is simpler for the "fetch then render" pattern — no custom `Message` subclass needed per widget. The scheduler already limits concurrency via semaphore, so there is no thundering herd of `call_from_thread` calls.

### `src/quantstack/tui/refresh.py` — TieredRefreshScheduler

Manages staggered refresh intervals for all widgets. Key design:

```python
import threading
from dataclasses import dataclass, field

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
```

**Registration:** Widgets register themselves with a tier and a tab ID. The scheduler stores `dict[str, list[RefreshableWidget]]` keyed by tier name.

**Active tab filtering:** On each tick, the scheduler iterates the tier's widget list. It fires `widget.refresh_data()` only if the widget belongs to the currently active tab OR the widget is marked `always_on=True` (header widgets). This avoids wasting DB connections on invisible tabs.

**Tab switch:** When `active_tab` changes, the scheduler immediately fires all tiers for the new tab (one-shot) so the user sees data instantly.

**Semaphore usage:** Before calling `widget.refresh_data()`, acquire `_db_semaphore`. The widget's `fetch_data()` method (which calls `pg_conn()`) runs while holding a permit. Release on completion. This caps the dashboard at 5 concurrent DB connections, leaving 15 of the pool's 20 connections for the 3 graph services.

**Starting the scheduler:** The `start(app)` method uses `app.set_interval(tier_seconds, callback)` for each tier, with the stagger offset applied via `app.set_timer(stagger_seconds, lambda: app.set_interval(...))`.

### `src/quantstack/tui/widgets/header.py` — HeaderBar

Single-line status bar docked to the top of the app.

Renders: `QUANTSTACK  HH:MM:SS  | MODE | Kill: ok/HALTED | Regime: name (conf%) | AV: used/limit | Universe: N`

- Subclasses `Static` (not `RefreshableWidget` — it is simpler and refreshed directly by the scheduler calling a dedicated method)
- Uses Textual reactive attributes for each field (`kill_status`, `regime_text`, `av_count`, `trading_mode`)
- Color logic: Kill status `ok` = green, `HALTED` = red blinking. Mode `LIVE` = red background, `PAPER` = yellow, default = dim
- Refreshed on T1 (5s) via queries from `queries/system.py`

The header queries (kill switch, AV count, regime) are implemented as stubs in this section. Full query implementations come in section 2.

### `src/quantstack/tui/dashboard.tcss` — Textual CSS

Separate CSS file for layout and styling. Start with a minimal stylesheet:

```css
/* Global layout */
Screen {
    layout: vertical;
}

HeaderBar {
    dock: top;
    height: 1;
    background: $surface;
    color: $text;
}

TabbedContent {
    height: 1fr;
}

/* Footer */
#footer {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
}
```

Subsequent sections will add widget-specific styles. Keep all CSS in this single file for maintainability.

### Placeholder Files

These files are created empty (or with minimal `__init__.py` exports) so that imports resolve. They are populated by later sections:

- `src/quantstack/tui/charts.py` — section 3
- `src/quantstack/tui/widgets/decisions.py` — section 4
- `src/quantstack/tui/queries/system.py` — section 2 (stub with function signatures only in this section)
- `src/quantstack/tui/screens/detail.py` — section 11

---

## Dependency: pyproject.toml

Add `textual>=0.50` to the `dependencies` list in `pyproject.toml`. Place it after the existing `"exchange-calendars>=4.5.0"` line, in the `# DB` comment group or add a new `# TUI` comment:

```toml
    # TUI
    "textual>=0.50",
```

Also add `textual-dev>=1.0` to the `[project.optional-dependencies] dev` list for the developer console and snapshot testing.

---

## Entry Points

### `scripts/dashboard.py`

The existing 796-line Rich-based dashboard is deleted entirely. Replace with a thin wrapper:

```python
"""Launch the QuantStack TUI dashboard.

This replaces the original Rich Live dashboard. Run directly or via:
    python -m quantstack.tui
"""
from quantstack.tui import QuantStackApp

if __name__ == "__main__":
    QuantStackApp().run()
```

This change happens in this section because the app shell must be runnable end-to-end. The old code is removed in a single commit — no gradual migration needed since the v1 dashboard is a standalone script with no importers.

---

## Dependencies on Other Sections

- **Section 2 (Query Layer):** The header widget needs `queries/system.py` functions. In this section, stub those functions with signatures and docstrings only (returning hardcoded defaults). Section 2 provides the real implementations.
- **Section 3 (Charts):** `charts.py` is created as an empty placeholder. No widgets in this section use charts.
- **Sections 4-10:** All tab panes are created empty in `compose()`. Widget content is added by those sections.
- **Section 13 (Integration):** `status.sh` update and full integration tests are deferred. This section only creates the unit tests listed above.

---

## Verification Checklist

After implementation, the following must be true:

1. `python -m quantstack.tui` launches without errors (renders 6 empty tabs with a header bar)
2. Keys 1-6 switch tabs
3. Key `q` quits
4. `TieredRefreshScheduler` starts on mount with correct tier intervals
5. `RefreshableWidget` subclass can be instantiated and its `refresh_data()` triggers the fetch/update cycle (testable with a mock subclass)
6. All unit tests in `test_app.py`, `test_refresh.py`, `test_base.py` pass
7. No imports from `src/quantstack/dashboard/` — only `src/quantstack/tui/`
8. `textual>=0.50` is in `pyproject.toml` dependencies
