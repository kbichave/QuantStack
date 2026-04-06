# Section 10: Risk Console

## Overview

This section builds the Risk Console — a set of three widgets that display real-time risk metrics, risk events (rejections, alerts, breaches), and equity alert status including kill switch state. The risk console appears in two places: as a compact `RiskCompact` widget on the Overview tab (Section 4), and as a full expanded view that can be embedded in the Portfolio tab or surfaced as its own scrollable section.

All widgets inherit from `RefreshableWidget` (Section 1) and consume query functions from `queries/risk.py` (Section 2). No chart renderers from Section 3 are required — this section uses Rich Tables and styled Text for display.

## Dependencies

- **Section 1 (Package Scaffolding):** `RefreshableWidget` base class, `src/quantstack/tui/base.py`.
- **Section 2 (Query Layer):** `queries/risk.py` provides `fetch_risk_snapshot`, `fetch_risk_events`, `fetch_equity_alerts` and their dataclasses (`RiskSnapshot`, `RiskEvent`, `EquityAlert`).
- No dependency on Section 3 (Charts), Section 12 (Migrations), or any other widget section.

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/tui/widgets/risk.py` | `RiskMetricsWidget`, `RiskEventsWidget`, `RiskAlertsWidget` |
| `tests/unit/test_tui/test_risk.py` | Unit tests for all three widgets |

## Tests (Write First)

### `tests/unit/test_tui/test_risk.py`

```python
"""Tests for risk console widgets."""
import pytest
from datetime import datetime


class TestRiskMetricsWidget:
    """RiskMetricsWidget renders risk snapshot values vs limits with color coding."""

    def test_renders_current_values_vs_limits(self):
        """Widget should display each metric with its current value and limit side by side."""
        ...

    def test_color_green_when_below_75_pct_of_limit(self):
        """Metrics below 75% of their limit should render green."""
        ...

    def test_color_yellow_when_between_75_and_100_pct(self):
        """Metrics at 75-100% of their limit should render yellow."""
        ...

    def test_color_red_when_above_100_pct(self):
        """Metrics exceeding their limit should render red."""
        ...

    def test_degrades_to_no_risk_data_when_snapshot_none(self):
        """When fetch_risk_snapshot returns None, widget shows 'No risk data available'."""
        ...

    def test_displays_all_seven_metrics(self):
        """Widget must show gross_exposure, net_exposure, concentration, correlation,
        sector_exposure, var_1d, max_drawdown."""
        ...

    def test_snapshot_timestamp_displayed(self):
        """The snapshot_at timestamp should be shown so operators know data freshness."""
        ...


class TestRiskEventsWidget:
    """RiskEventsWidget renders recent risk events from the last 7 days."""

    def test_renders_last_7_days_of_events(self):
        """Widget should show risk events from the past 7 days."""
        ...

    def test_displays_event_type_symbol_details_date(self):
        """Each event row should show event_type, symbol (if present), details, created_at."""
        ...

    def test_handles_empty_event_list(self):
        """When no risk events exist, widget shows a 'No risk events' message."""
        ...

    def test_event_types_color_coded(self):
        """risk_rejection=red, drawdown_alert=yellow, correlation_alert=cyan."""
        ...

    def test_null_symbol_renders_dash(self):
        """Events without a symbol (system-level alerts) show '-' in the symbol column."""
        ...


class TestRiskAlertsWidget:
    """RiskAlertsWidget renders active/cleared equity alerts and kill switch status."""

    def test_shows_active_vs_cleared_alerts(self):
        """Active alerts should be visually distinct from cleared alerts."""
        ...

    def test_kill_switch_status_rendered(self):
        """Kill switch status (ok/halted) must always be visible."""
        ...

    def test_handles_no_alerts(self):
        """When no equity alerts exist, widget shows 'No equity alerts'."""
        ...

    def test_cleared_alerts_show_cleared_at_timestamp(self):
        """Cleared alerts must display when they were cleared."""
        ...

    def test_active_alerts_styled_prominently(self):
        """Active alerts should use bold/red styling to draw attention."""
        ...
```

## Widget Design

### Widget Hierarchy

```
RiskConsole (Container — optional grouping wrapper)
├── RiskMetricsWidget        # Current risk snapshot vs limits
├── RiskEventsWidget         # Recent risk events (rejections, alerts, breaches)
└── RiskAlertsWidget         # Active/cleared equity alerts + kill switch status
```

All three widgets subclass `RefreshableWidget` from `src/quantstack/tui/base.py`. They follow the standard pattern: `fetch_data()` runs queries in a background thread via `pg_conn()`, `update_view()` renders Rich content on the main thread.

### `RiskMetricsWidget`

**Data source:** `fetch_risk_snapshot(conn)` from `queries/risk.py`. Returns `RiskSnapshot | None`.

**Refresh tier:** T3 (60s) — risk snapshots do not change frequently.

**Rendering:**

Renders a Rich `Table` with columns: Metric, Current, Limit, Status. Seven rows, one per metric:

| Metric | Current | Limit | Status |
|--------|---------|-------|--------|
| Gross Exposure | 37% | 100% | OK |
| Net Exposure | 25% | 80% | OK |
| Concentration | 22% | 30% | WARN |
| Correlation | 0.45 | 0.70 | OK |
| Sector Exposure | 40% | 50% | WARN |
| VaR (1d) | $245 | $500 | OK |
| Max Drawdown | -2.5% | -10% | OK |

**Limit constants** are defined at module level in `widgets/risk.py`:

```python
RISK_LIMITS: dict[str, float] = {
    "gross_exposure": 1.0,
    "net_exposure": 0.8,
    "concentration": 0.30,
    "correlation": 0.70,
    "sector_exposure": 0.50,
    "var_1d": 500.0,
    "max_drawdown": 0.10,
}
```

These limits mirror the risk gate configuration in `src/quantstack/execution/risk_gate.py`. If the risk gate limits are ever exposed as a queryable config, these constants should be replaced with a dynamic lookup. For now, static constants are acceptable because the risk gate limits rarely change and duplicating them avoids a coupling dependency on the execution layer.

**Color logic** applied per-row:

```python
def _status_color(current: float, limit: float) -> tuple[str, str]:
    """Return (status_label, color) based on how close current is to limit.

    Returns ('OK', 'green') if ratio < 0.75,
    ('WARN', 'yellow') if 0.75 <= ratio < 1.0,
    ('BREACH', 'red') if ratio >= 1.0.
    """
    ...
```

For `max_drawdown`, the comparison is inverted: a more negative drawdown is worse, so `ratio = abs(current) / abs(limit)`.

**Degradation:** When `fetch_risk_snapshot` returns None (empty table), `update_view` renders a single-line `Static` with text "No risk data available" in dim style.

### `RiskEventsWidget`

**Data source:** `fetch_risk_events(conn, days=7)` from `queries/risk.py`. Returns `list[RiskEvent]`.

**Refresh tier:** T2 (15s) — risk events are operationally important and should surface quickly.

**Rendering:**

Renders a Rich `Table` with columns: Time, Type, Symbol, Details. Events ordered by `created_at` descending (most recent first).

**Color coding by event type:**
- `risk_rejection` — red (a trade was blocked)
- `drawdown_alert` — yellow (drawdown threshold approached)
- `correlation_alert` — cyan (correlation limit warning)

**Symbol column:** If `symbol` is None (system-level event), render `—` (em-dash).

**Degradation:** Empty list renders "No risk events in last 7 days" in dim style.

### `RiskAlertsWidget`

**Data source:** `fetch_equity_alerts(conn)` from `queries/risk.py`. Returns `list[EquityAlert]`. Kill switch status comes from `fetch_kill_switch(conn)` in `queries/system.py` (already built in Section 2).

**Refresh tier:** T1 (5s) — kill switch and active alerts are the highest-priority operational signals.

**Rendering:**

Two sections within the widget:

1. **Kill Switch Status** — Single line at top: `Kill Switch: OK` (green) or `Kill Switch: HALTED` (bold red, blinking if terminal supports it). This is the most critical operational signal in the entire dashboard.

2. **Equity Alerts Table** — Rich `Table` with columns: ID, Type, Status, Message, Created, Cleared.
   - Active alerts: bold red text, status column shows "ACTIVE"
   - Cleared alerts: dim text, status column shows "CLEARED", `cleared_at` timestamp displayed
   - Active alerts sort first, then cleared alerts by `cleared_at` descending

**Degradation:** No alerts renders "No equity alerts" in dim style. Kill switch status always renders (falls back to "OK" per `fetch_kill_switch` default).

## Data Flow

The standard `RefreshableWidget` pattern applies to all three widgets:

```python
class RiskMetricsWidget(RefreshableWidget):
    """Current risk snapshot with limit comparison."""

    def fetch_data(self) -> RiskSnapshot | None:
        """Fetch latest risk snapshot. Runs in background thread."""
        from quantstack.db import pg_conn
        from quantstack.tui.queries.risk import fetch_risk_snapshot

        with pg_conn() as conn:
            return fetch_risk_snapshot(conn)

    def update_view(self, data: RiskSnapshot | None) -> None:
        """Render risk metrics table. Runs on main thread."""
        ...
```

`RiskAlertsWidget.fetch_data()` makes two query calls within the same `pg_conn()` block: `fetch_equity_alerts(conn)` and `fetch_kill_switch(conn)`.

## Refresh Tiers

| Widget | Tier | Interval | Rationale |
|--------|------|----------|-----------|
| `RiskMetricsWidget` | T3 | 60s | Snapshots update infrequently |
| `RiskEventsWidget` | T2 | 15s | Risk rejections need prompt visibility |
| `RiskAlertsWidget` | T1 | 5s | Kill switch is highest-priority signal |

## Integration with Other Tabs

The risk console widgets are designed to be composed into different tabs:

- **Overview tab (Section 4):** Uses `RiskCompact` (a separate compact widget already defined in Section 4) that shows a single-line summary: `Exposure: 37%  DD: -2.5%  VaR: $245  Alerts: 1  Kill: ok`. The compact widget calls the same query functions but renders a condensed view.

- **Standalone / Portfolio expanded:** The three full widgets from this section (`RiskMetricsWidget`, `RiskEventsWidget`, `RiskAlertsWidget`) can be composed into a `ScrollableContainer` and placed on the Portfolio tab below the positions table, or surfaced as a dedicated risk section.

The widgets are self-contained — they do not import from or depend on any other widget section.

## Implementation Checklist

1. Create `tests/unit/test_tui/test_risk.py` with the test stubs above.
2. Create `src/quantstack/tui/widgets/risk.py` with the `RISK_LIMITS` constant dict.
3. Implement `RiskMetricsWidget` — `fetch_data` calls `fetch_risk_snapshot`, `update_view` renders the metrics-vs-limits table with color logic.
4. Implement `RiskEventsWidget` — `fetch_data` calls `fetch_risk_events(conn, days=7)`, `update_view` renders color-coded event table.
5. Implement `RiskAlertsWidget` — `fetch_data` calls both `fetch_equity_alerts` and `fetch_kill_switch`, `update_view` renders kill switch line + alerts table.
6. Verify all three widgets degrade gracefully with None/empty data.
7. Run tests: `uv run pytest tests/unit/test_tui/test_risk.py -v`.

## Key Invariants

- Widgets never call `pg_conn()` outside of `fetch_data()`. All DB access is in the background thread.
- `update_view()` never blocks. It only renders Rich objects from already-fetched data.
- Kill switch status is always visible, even when other risk data is unavailable.
- Color thresholds (75%, 100%) are consistent across all metrics — no per-metric special cases except `max_drawdown` sign inversion.
- `RISK_LIMITS` are static constants, not queried from DB. This is intentional to avoid a circular dependency on the risk gate module and because these limits are operationally stable.
