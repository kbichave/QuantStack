# Section 13: Integration, Testing, and Entry Points

## Overview

This is the final section. It assumes all prior sections (1-12) are complete and the `src/quantstack/tui/` package is fully functional. This section wires the TUI into the project's existing entry points, adds the `textual` dependency, implements Docker health-check fallback logic, and delivers the full test suite (unit + integration).

There are four deliverables:

1. Replace `scripts/dashboard.py` with a thin wrapper that imports from `quantstack.tui`
2. Update `status.sh` to invoke the new TUI
3. Add `textual>=0.50` to `pyproject.toml` dependencies
4. Write all test files (unit tests for queries, charts, refresh; integration tests via Textual pilot)

---

## Dependencies on Prior Sections

| Prerequisite | What it provides |
|---|---|
| Section 1 (Scaffolding) | `src/quantstack/tui/app.py` with `QuantStackApp`, `base.py` with `RefreshableWidget`, `refresh.py` with `TieredRefreshScheduler` |
| Section 2 (Query Layer) | `src/quantstack/tui/queries/*.py` — all 45 query functions returning typed dataclasses |
| Section 3 (Charts) | `src/quantstack/tui/charts.py` — sparkline, horizontal_bar, progress_bar, daily_heatmap, equity_curve |
| Sections 4-10 (Tab Widgets) | All widget classes in `src/quantstack/tui/widgets/` |
| Section 11 (Modals) | `src/quantstack/tui/screens/detail.py` and modal variants |
| Section 12 (DB Migrations) | `market_holidays` and `benchmark_daily` tables added to `src/quantstack/db.py` |

---

## Deliverable 1: Replace scripts/dashboard.py

### File: `scripts/dashboard.py`

Delete the entire existing 796-line Rich-based dashboard. Replace with a thin wrapper:

```python
#!/usr/bin/env python3
"""QuantStack terminal dashboard v2 — Textual TUI.

Usage:
    python scripts/dashboard.py
    python -m quantstack.tui
    ./status.sh
"""
from quantstack.tui.app import QuantStackApp


def main() -> None:
    QuantStackApp().run()


if __name__ == "__main__":
    main()
```

Key points:
- No `--watch` flag. Textual is always live; the flag is meaningless.
- No `argparse`. The old script accepted `--watch` and `--interval`; neither applies.
- No `sys.path` manipulation. The package is installed via `pip install -e .`.
- The import is `quantstack.tui.app`, NOT `quantstack.dashboard` (which is the existing FastAPI web dashboard and must remain untouched).

### File: `src/quantstack/tui/__main__.py`

This file should already exist from Section 1. Verify it contains:

```python
"""Allow `python -m quantstack.tui` to launch the dashboard."""
from quantstack.tui.app import QuantStackApp

QuantStackApp().run()
```

---

## Deliverable 2: Update status.sh

### File: `status.sh` (project root)

Replace the current content:

```bash
#!/usr/bin/env bash
# QuantStack — status dashboard (Textual TUI).
# Usage:
#   ./status.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 -m quantstack.tui
```

Changes from current version:
- Remove `"$@"` argument pass-through (no more `--watch`/`--interval` flags)
- Use `python3 -m quantstack.tui` instead of `python3 scripts/dashboard.py`
- Update usage comment to remove `--watch` and `--interval` references

---

## Deliverable 3: Add textual Dependency

### File: `pyproject.toml`

Add `textual>=0.50` to the main `dependencies` list, after the existing `rich` entry (or in the "Core libs" section if there is no explicit `rich` entry — Textual bundles Rich, but the project may already depend on Rich directly for other modules).

The dependency goes in the main `dependencies` array (not an optional extra) because the dashboard is a core operational tool, not an optional feature:

```toml
dependencies = [
    # ... existing entries ...
    "textual>=0.50",
]
```

Also add `textual-dev>=0.50` to the `dev` optional dependencies for the Textual devtools (console, snapshot testing):

```toml
[project.optional-dependencies]
dev = [
    # ... existing entries ...
    "textual-dev>=0.50",
]
```

And in the `[dependency-groups]` dev section:

```toml
[dependency-groups]
dev = [
    # ... existing entries ...
    "textual-dev>=0.50",
]
```

---

## Deliverable 4: Docker Health Check Fallback

### File: `src/quantstack/tui/queries/system.py`

The `fetch_docker_health()` query function (already created in Section 2) needs a fallback strategy for environments where `docker compose` is unavailable (e.g., inside Docker containers without socket access).

The function should implement this three-tier strategy:

1. **Try `docker compose ps --format json`** — parse JSON output, extract service name + status. This works in local dev where Docker CLI is on PATH.
2. **Fall back to TCP port probes** — if `docker compose` fails (FileNotFoundError, subprocess error), attempt TCP connections to known service ports:
   - PostgreSQL: port 5432
   - LangFuse: port 3100
   - Ollama: port 11434
   - Each probe uses `socket.create_connection()` with a 1-second timeout. Success = "running", timeout/refused = "down".
3. **Return "unknown"** — if neither method works (no Docker CLI, no network access), return a list of `ServiceHealth` dataclasses with `status="unknown"` for each service. Never raise an exception.

Implementation guidance (signature only):

```python
import json
import socket
import subprocess

@dataclass
class ServiceHealth:
    name: str
    status: str  # "running" | "down" | "unknown"
    port: int | None

KNOWN_SERVICES: list[tuple[str, int]] = [
    ("postgres", 5432),
    ("langfuse", 3100),
    ("ollama", 11434),
]

def fetch_docker_health() -> list[ServiceHealth]:
    """Check health of Docker Compose services.

    Strategy:
    1. docker compose ps --format json (local dev)
    2. TCP port probes (inside container / no Docker CLI)
    3. Return 'unknown' status (neither method available)
    """
    ...
```

The TCP probe helper:

```python
def _probe_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    ...
```

---

## Deliverable 5: Test Suite

### Unit Tests

All unit tests go in `tests/unit/test_tui/`. Create `__init__.py` in that directory.

#### File: `tests/unit/test_tui/__init__.py`

Empty file.

#### File: `tests/unit/test_tui/test_queries.py`

Patch `pg_conn()` at the query-function level (not the pool factory). Each test verifies:
- The function returns the correct dataclass type on success
- The function returns the documented default on exception (empty list, None, 0, False)
- The function uses `PgConnection.execute()` (not raw cursors)

Test stubs:

```python
"""Unit tests for TUI query layer."""
from unittest.mock import MagicMock, patch


class TestSystemQueries:
    """Tests for queries/system.py functions."""

    def test_fetch_kill_switch_returns_bool(self):
        """fetch_kill_switch returns a bool; default False on error."""
        ...

    def test_fetch_av_calls_returns_int(self):
        """fetch_av_calls returns an int; default 0 on error."""
        ...

    def test_fetch_regime_returns_dataclass(self):
        """fetch_regime returns a RegimeState dataclass with trend, vol, confidence."""
        ...

    def test_fetch_graph_checkpoints_returns_list(self):
        """fetch_graph_checkpoints returns list of GraphCheckpoint dataclasses."""
        ...

    def test_fetch_heartbeats_returns_list(self):
        """fetch_heartbeats returns list of Heartbeat dataclasses."""
        ...

    def test_fetch_agent_events_limited_and_ordered(self):
        """fetch_agent_events returns list of AgentEvent, LIMIT 60, ordered DESC."""
        ...


class TestPortfolioQueries:
    """Tests for queries/portfolio.py functions."""

    def test_fetch_equity_summary_returns_dataclass(self):
        """fetch_equity_summary returns EquitySummary dataclass."""
        ...

    def test_fetch_positions_ordered_by_pnl(self):
        """fetch_positions returns list of Position, ordered by unrealized_pnl DESC."""
        ...

    def test_fetch_closed_trades_limited(self):
        """fetch_closed_trades returns list of ClosedTrade, LIMIT 10."""
        ...

    def test_fetch_equity_curve_30_rows(self):
        """fetch_equity_curve returns list of EquityPoint (30 rows)."""
        ...

    def test_fetch_benchmark_30_rows(self):
        """fetch_benchmark returns list of BenchmarkPoint (30 rows for SPY)."""
        ...

    def test_fetch_pnl_by_strategy(self):
        """fetch_pnl_by_strategy returns list of StrategyPnl dataclasses."""
        ...

    def test_fetch_pnl_by_symbol(self):
        """fetch_pnl_by_symbol returns list of SymbolPnl dataclasses."""
        ...


class TestStrategyQueries:
    """Tests for queries/strategies.py functions."""

    def test_fetch_strategy_pipeline_returns_cards(self):
        """fetch_strategy_pipeline returns list of StrategyCard with fwd stats."""
        ...

    def test_strategies_ordered_by_status_priority(self):
        """Strategies ordered: live > forward_testing > backtested > draft > retired."""
        ...


class TestDataHealthQueries:
    """Tests for queries/data_health.py functions."""

    def test_fetch_ohlcv_freshness_returns_dict(self):
        """fetch_ohlcv_freshness returns dict[symbol, datetime]."""
        ...

    def test_fetch_news_freshness_returns_dict(self):
        """fetch_news_freshness returns dict[symbol, datetime]."""
        ...

    def test_freshness_returns_empty_dict_on_error(self):
        """Each freshness query returns empty dict on error."""
        ...

    def test_fetch_collector_health(self):
        """fetch_collector_health returns dict[collector_name, bool]."""
        ...


class TestSignalQueries:
    """Tests for queries/signals.py functions."""

    def test_fetch_active_signals_sorted(self):
        """fetch_active_signals returns list of Signal sorted by confidence DESC."""
        ...

    def test_fetch_signal_brief_parses_json(self):
        """fetch_signal_brief parses brief_json JSONB correctly."""
        ...


class TestRiskQueries:
    """Tests for queries/risk.py functions."""

    def test_fetch_risk_snapshot_returns_dataclass_or_none(self):
        """fetch_risk_snapshot returns RiskSnapshot or None if table empty."""
        ...

    def test_fetch_equity_alerts(self):
        """fetch_equity_alerts returns list of EquityAlert with status."""
        ...


class TestAllQueriesGracefulDegradation:
    """Every query function returns its default when pg_conn() raises."""

    def test_all_queries_return_default_on_error(self):
        """Iterate all public query functions, mock pg_conn to raise, verify default returned."""
        ...
```

#### File: `tests/unit/test_tui/test_charts.py`

```python
"""Unit tests for TUI chart renderers."""
from rich.text import Text


class TestSparkline:
    """Tests for sparkline() renderer."""

    def test_ascending_values_produce_ascending_blocks(self):
        """sparkline([1,2,3,4,5,6,7,8]) returns 8 ascending block characters."""
        ...

    def test_empty_input_returns_empty(self):
        """sparkline([]) returns empty string."""
        ...

    def test_uniform_values_produce_uniform_blocks(self):
        """sparkline with all same values returns all same block char."""
        ...

    def test_resamples_when_data_exceeds_width(self):
        """sparkline resamples when data length > width parameter."""
        ...

    def test_returns_rich_text_with_color(self):
        """sparkline returns Rich Text with the specified color."""
        ...


class TestHorizontalBar:
    """Tests for horizontal_bar() renderer."""

    def test_75_percent_fill(self):
        """horizontal_bar(75, 100, 20) renders ~15 filled + 5 empty + '75%'."""
        ...

    def test_zero_value(self):
        """horizontal_bar(0, 100, 20) renders all empty."""
        ...

    def test_zero_max_no_division_error(self):
        """horizontal_bar handles max_value=0 without ZeroDivisionError."""
        ...


class TestProgressBar:
    """Tests for progress_bar() renderer."""

    def test_high_progress_is_green(self):
        """progress_bar(80, 100) uses green color."""
        ...

    def test_medium_progress_is_yellow(self):
        """progress_bar(50, 100) uses yellow color."""
        ...

    def test_low_progress_is_red(self):
        """progress_bar(20, 100) uses red color."""
        ...


class TestDailyHeatmap:
    """Tests for daily_heatmap() renderer."""

    def test_renders_weekday_grid(self):
        """daily_heatmap renders a Mon-Fri grid."""
        ...

    def test_positive_green_negative_red(self):
        """Positive values green, negative values red."""
        ...

    def test_empty_input(self):
        """daily_heatmap handles empty input gracefully."""
        ...


class TestEquityCurve:
    """Tests for equity_curve() renderer."""

    def test_multiline_output_with_y_labels(self):
        """equity_curve renders multi-line output with Y-axis labels."""
        ...

    def test_single_data_point(self):
        """equity_curve handles single data point without error."""
        ...

    def test_flat_data(self):
        """equity_curve handles flat data (all same value)."""
        ...
```

#### File: `tests/unit/test_tui/test_refresh.py`

```python
"""Unit tests for TieredRefreshScheduler."""
import threading


class TestTieredRefreshScheduler:
    """Tests for refresh.py TieredRefreshScheduler."""

    def test_four_tiers_defined(self):
        """Scheduler defines 4 tiers: T1=5s, T2=15s, T3=60s, T4=120s."""
        ...

    def test_stagger_offsets(self):
        """Stagger offsets are 0.0, 0.3, 0.6, 0.9 seconds."""
        ...

    def test_tab_switch_updates_active_queries(self):
        """Changing active_tab updates which query sets fire."""
        ...

    def test_header_queries_always_fire(self):
        """Always-on queries (header/system) fire regardless of active tab."""
        ...

    def test_semaphore_is_threading_not_asyncio(self):
        """db_semaphore is threading.Semaphore, not asyncio.Semaphore."""
        ...
```

#### File: `tests/unit/test_tui/test_entry.py`

```python
"""Unit tests for TUI entry points."""
import ast
import importlib


class TestEntryPoints:
    """Verify entry point wiring."""

    def test_dashboard_script_imports_from_tui(self):
        """scripts/dashboard.py imports from quantstack.tui, not quantstack.dashboard.

        Parse the AST of the script to verify the import source without executing it.
        """
        ...

    def test_main_module_is_importable(self):
        """quantstack.tui.__main__ can be imported without error."""
        ...
```

### Integration Tests (Textual Pilot)

#### File: `tests/integration/test_tui_app.py`

These tests use Textual's `pilot` testing framework. All query modules are patched to return defaults so no database is required.

```python
"""Integration tests for TUI app using Textual pilot."""
import pytest
from unittest.mock import patch


@pytest.fixture
def patch_all_queries():
    """Patch every query module to return safe defaults.

    This fixture patches pg_conn to return a mock that yields a mock connection,
    ensuring no real DB access occurs. Individual query functions return their
    documented default values (empty lists, None, 0, False).
    """
    ...


class TestAppStartup:
    """App starts and renders without a database."""

    @pytest.mark.asyncio
    async def test_app_starts_without_db(self, patch_all_queries):
        """App renders all tabs with fallback content when DB is unavailable."""
        ...


class TestTabSwitching:
    """Tab navigation via keyboard."""

    @pytest.mark.asyncio
    async def test_switch_to_each_tab(self, patch_all_queries):
        """Keys 1-6 switch to the corresponding tab pane."""
        ...

    @pytest.mark.asyncio
    async def test_rapid_tab_switching(self, patch_all_queries):
        """Rapid sequential tab switches do not crash the app."""
        ...


class TestKeybindings:
    """Global keybinding tests."""

    @pytest.mark.asyncio
    async def test_q_quits(self, patch_all_queries):
        """Pressing q exits the application."""
        ...

    @pytest.mark.asyncio
    async def test_r_triggers_refresh(self, patch_all_queries):
        """Pressing r triggers an immediate data refresh."""
        ...

    @pytest.mark.asyncio
    async def test_question_mark_shows_help(self, patch_all_queries):
        """Pressing ? shows the help overlay."""
        ...


class TestModals:
    """Modal open/close behavior."""

    @pytest.mark.asyncio
    async def test_enter_opens_modal(self, patch_all_queries):
        """Pressing Enter on a selectable row opens a detail modal."""
        ...

    @pytest.mark.asyncio
    async def test_esc_closes_modal(self, patch_all_queries):
        """Pressing Esc dismisses an open modal."""
        ...
```

### What NOT to Test

- **Exact visual rendering** — pixel/character-level assertions break on terminal size changes and Textual version updates. Test data flow and widget state, not rendered output.
- **Live database queries** — all tests mock `pg_conn()`. Live DB tests belong in a separate integration suite that requires a running PostgreSQL instance.
- **Docker health checks** — subprocess calls to `docker compose` are environment-dependent. The fallback logic is simple enough to verify by code review.
- **CSS styling** — `dashboard.tcss` is visual polish. Test it manually.

---

## CSS Finalization

### File: `src/quantstack/tui/dashboard.tcss`

This file (created in Section 1) should be reviewed after all widgets are integrated. Key concerns:

- **Tab content area** should fill remaining vertical space after the header (1 line) and footer (1 line)
- **Grid layouts** in the Overview tab (Section 4) need `grid-size: 2` and `grid-gutter: 1`
- **Kanban columns** in the Strategies tab (Section 6) use `Horizontal` with equal-width children
- **Side-by-side panels** in the Agents tab (Section 8) use `Horizontal` with 3 equal-width children
- **Modal overlay** uses `background: rgba(0, 0, 0, 0.7)` for semi-transparency
- **Color scheme** should be consistent: green for positive/healthy, red for negative/error, yellow for warnings, cyan for informational

No specific CSS content is prescribed here — the visual tuning happens during integration when all widgets are rendered together.

---

## Implementation Checklist

1. Add `textual>=0.50` to `pyproject.toml` `dependencies` and `textual-dev>=0.50` to dev extras
2. Create `tests/unit/test_tui/__init__.py`
3. Write `tests/unit/test_tui/test_queries.py` — mock `pg_conn()`, test all query return types and error defaults
4. Write `tests/unit/test_tui/test_charts.py` — test all 5 chart renderers with known inputs
5. Write `tests/unit/test_tui/test_refresh.py` — test tier config, stagger, tab visibility, semaphore type
6. Write `tests/unit/test_tui/test_entry.py` — verify import paths and `__main__` importability
7. Write `tests/integration/test_tui_app.py` — Textual pilot tests for startup, tabs, keys, modals
8. Implement Docker health-check fallback in `src/quantstack/tui/queries/system.py` (three-tier: docker CLI, TCP probe, unknown)
9. Replace `scripts/dashboard.py` with the thin wrapper (delete all 796 lines of v1 code)
10. Update `status.sh` to call `python3 -m quantstack.tui`
11. Verify `src/quantstack/tui/__main__.py` exists and works
12. Run `uv run pytest tests/unit/test_tui/` — all unit tests pass
13. Run `uv run pytest tests/integration/test_tui_app.py` — all pilot tests pass
14. Manual smoke test: `python -m quantstack.tui` launches, tabs switch, widgets render with mocked or live data
15. Review `dashboard.tcss` for layout consistency across all 6 tabs
