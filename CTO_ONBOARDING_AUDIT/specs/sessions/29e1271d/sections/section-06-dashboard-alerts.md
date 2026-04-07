# Section 6: Dashboard Alert Integration

## Overview

This section adds system alert visibility to both the TUI (Textual) dashboard and the web (FastAPI) dashboard. It surfaces the `system_alerts` table (created in Section 1, populated by the tools in Section 2) through three integration points:

1. A new TUI widget on the Overview tab showing unresolved alerts, color-coded by severity
2. A new FastAPI REST endpoint (`/api/alerts`) for querying alerts
3. SSE stream extension so the web dashboard receives alert events in real-time
4. Event publishing integration so `emit_system_alert()` pushes events to the dashboard automatically

**Dependencies:** Section 1 (DB schema — `system_alerts` table must exist) and Section 2 (system alert tools — `emit_system_alert()` must exist in `src/quantstack/tools/functions/system_alerts.py`).

---

## Tests

Write all tests before implementation. Tests live in `tests/unit/` and `tests/integration/`.

### Unit Tests

File: `tests/unit/test_dashboard_alerts.py`

```python
# Test: alerts_widget query returns unresolved alerts sorted by severity then time
#   - Insert 3 alerts: info, critical, warning (in that order)
#   - Verify fetch returns critical first, then warning, then info
#   - Verify resolved alerts are excluded

# Test: alerts_widget color mapping: emergency=red, critical=red, warning=yellow, info=dim
#   - Call the color-mapping function with each severity level
#   - Assert: "emergency" -> "bold red"
#   - Assert: "critical" -> "red"
#   - Assert: "warning" -> "yellow"
#   - Assert: "info" -> "dim"

# Test: /api/alerts endpoint returns JSON list of alerts with correct schema
#   - Use FastAPI TestClient against the app
#   - Insert sample alerts into system_alerts
#   - GET /api/alerts -> verify response is a list of dicts
#   - Each dict has keys: id, category, severity, status, title, detail, created_at

# Test: /api/alerts filters by status parameter
#   - Insert alerts with status "open", "acknowledged", "resolved"
#   - GET /api/alerts?status=open -> only open alerts returned
#   - GET /api/alerts?status=resolved -> only resolved alerts returned

# Test: SSE stream includes system_alert event type
#   - Verify that publish_event() can be called with event_type="system_alert"
#   - Verify the SSE endpoint returns events with that type (mock DB fetch)
```

### Integration Tests

File: `tests/integration/test_dashboard_alerts_integration.py`

```python
# Test: TUI alerts widget renders with sample alert data (no crash)
#   - Instantiate AlertsCompact widget
#   - Call update_view() with a list of alert dicts
#   - Verify widget.renderable is not the "Loading..." placeholder

# Test: web dashboard /api/alerts returns alerts from system_alerts table
#   - Insert rows into system_alerts via db_conn()
#   - GET /api/alerts via TestClient
#   - Verify row count and field values match

# Test: emit_system_alert triggers dashboard event publication
#   - Patch publish_event
#   - Call emit_system_alert(category="test", severity="info", title="Test", detail="...")
#   - Assert publish_event was called with event_type="system_alert"
```

---

## Implementation Details

### 6.1 TUI Dashboard — Alerts Widget

**New file:** `src/quantstack/tui/widgets/alerts_widget.py`

Create a new widget that inherits from `RefreshableWidget` (defined in `src/quantstack/tui/base.py`). The existing widget pattern is:

- `REFRESH_TIER`: Set to `"T1"` (5-second refresh — alerts are highest priority)
- `TAB_ID`: Set to `"tab-overview"`
- `ALWAYS_ON`: Set to `True` (alerts should refresh regardless of which tab is active)
- `fetch_data()`: Query the DB, return a list of alert dicts
- `update_view(data)`: Render a Rich `Table` or `Text` with color-coded rows

**Query logic for `fetch_data()`:**

```sql
SELECT id, category, severity, status, title, created_at
FROM system_alerts
WHERE status != 'resolved'
ORDER BY
    CASE severity
        WHEN 'emergency' THEN 1
        WHEN 'critical' THEN 2
        WHEN 'warning' THEN 3
        WHEN 'info' THEN 4
    END,
    created_at DESC
LIMIT 20
```

Use `db_conn()` context manager from `quantstack.db` for the query.

**Severity color mapping (for Rich `Text` styling):**

| Severity | Rich Style |
|----------|-----------|
| emergency | `bold red` |
| critical | `red` |
| warning | `yellow` |
| info | `dim` |

**Rendering:** Show a compact table with columns: severity icon/color, title (truncated to ~60 chars), status, age (e.g., "2m ago", "1h ago"). If no alerts, show a dim "No active alerts" message.

**Registration in Overview tab:** Modify `src/quantstack/tui/screens/overview.py` to import and yield the new widget. Place it at the top of the `compose()` method so alerts appear first, above `ServicesCompact`:

```python
from quantstack.tui.widgets.alerts_widget import AlertsCompact

class OverviewTab(ScrollableContainer):
    def compose(self) -> ComposeResult:
        yield AlertsCompact(classes="full-width")  # NEW — alerts at top
        yield ServicesCompact(classes="overview-cell")
        # ... rest unchanged
```

Use `classes="full-width"` so the alerts banner spans the full width, matching the pattern used by `AgentActivityLine` and `DigestCompact`.

### 6.2 Web Dashboard — Alerts API Endpoint

**Modified file:** `src/quantstack/dashboard/app.py`

Add a new endpoint:

```python
@app.get("/api/alerts")
async def get_alerts(status: str = "open", limit: int = Query(default=20, le=100)) -> list[dict]:
    """Return recent system alerts for dashboard display."""
```

**Implementation notes:**

- Use `db_conn()` to query `system_alerts`, same pattern as `_fetch_recent_events()`
- Filter by `status` parameter (default "open" — show unresolved alerts)
- Order by severity DESC, created_at DESC
- Return fields: `id`, `category`, `severity`, `status`, `source`, `title`, `detail`, `metadata`, `created_at`, `acknowledged_at`, `resolved_at`
- Serialize `datetime` fields to ISO format strings, `metadata` as dict (it is JSONB in the DB)

Add a helper function `_fetch_alerts()` following the existing `_fetch_recent_events()` pattern — a private function that does the DB query, called by the endpoint handler.

### 6.3 Web Dashboard — SSE Stream Extension

**Modified file:** `src/quantstack/dashboard/app.py`

The existing SSE endpoint (`/api/stream`) polls `agent_events` for new rows. To include system alerts in the stream, extend the `event_generator()` inside `stream_events()` to also poll `system_alerts`:

- Track a `last_alert_id` alongside the existing `last_id`
- Each poll iteration: fetch new agent events AND new system alerts since `last_alert_id`
- Emit system alerts as SSE events with a distinct `event_type: "system_alert"` field so the frontend can distinguish them from agent events

Alternatively (simpler approach): since `emit_system_alert()` will call `publish_event()` (see 6.4 below), system alerts will automatically appear in the `agent_events` table with `event_type="system_alert"`. This means the existing SSE stream picks them up with zero changes to the stream endpoint itself. The frontend just needs to handle the new event type.

The simpler approach is preferred — it avoids dual-polling and keeps the SSE code unchanged.

### 6.4 Web Dashboard — Frontend Changes

**Modified file:** `src/quantstack/dashboard/app.py` (the `DASHBOARD_HTML` string)

Add handling for `system_alert` events in the frontend JavaScript:

- In `TYPE_CONFIG`, add: `system_alert: { icon: '!', label: 'ALERT' }`
- Add CSS for `.msg.system_alert`: red border-left, red-tinted background (matching `.msg.error` pattern)
- Optionally: add a collapsible alerts banner above the 2x2 grid that shows critical/emergency alerts as a persistent red bar. This is a frontend-only addition inside the HTML template.

### 6.5 Event Publishing Integration

**Modified file:** `src/quantstack/tools/functions/system_alerts.py` (created in Section 2)

At the end of `emit_system_alert()`, after the DB insert, call `publish_event()` to push the alert into the dashboard event stream:

```python
from quantstack.dashboard.events import publish_event

async def emit_system_alert(category, severity, title, detail, metadata=None) -> int:
    # ... existing DB insert logic ...
    
    # Publish to dashboard SSE stream (best-effort, never raises)
    publish_event(
        graph_name="system",
        node_name="alert_engine",
        event_type="system_alert",
        content=f"[{severity.upper()}] {title}: {detail[:200]}",
        metadata={"alert_id": alert_id, "category": category, "severity": severity, **(metadata or {})},
    )
    
    # TODO(kbichave): Add Discord webhook notification for CRITICAL/EMERGENCY alerts.
    # Trigger: when DISCORD_WEBHOOK_URL env var is set. See Phase 9 spec item 9.5 for
    # webhook patterns (rate limits, batching, embed formatting).
    
    return alert_id
```

The `publish_event()` function is already best-effort (wraps in try/except, never raises), so this integration cannot break alert creation even if the dashboard DB write fails.

---

## File Summary

| File | Action |
|------|--------|
| `src/quantstack/tui/widgets/alerts_widget.py` | **Create** — new `AlertsCompact` widget |
| `src/quantstack/tui/screens/overview.py` | **Modify** — add `AlertsCompact` to top of `compose()` |
| `src/quantstack/dashboard/app.py` | **Modify** — add `/api/alerts` endpoint, add `system_alert` to frontend JS `TYPE_CONFIG` and CSS |
| `src/quantstack/tools/functions/system_alerts.py` | **Modify** — add `publish_event()` call and Discord TODO in `emit_system_alert()` |
| `tests/unit/test_dashboard_alerts.py` | **Create** — unit tests |
| `tests/integration/test_dashboard_alerts_integration.py` | **Create** — integration tests |

---

## Key Design Decisions

**Why T1 refresh tier for the TUI widget:** System alerts are safety-critical. A 5-second refresh means operators see new alerts within seconds, matching the urgency of kill-switch and header status updates that are also T1.

**Why `ALWAYS_ON = True`:** Alerts must be visible regardless of which tab the operator is viewing. A critical risk breach alert hiding because the user is on the Research tab defeats the purpose.

**Why piggyback on `agent_events` via `publish_event()` instead of dual-polling:** The SSE stream already polls `agent_events`. Adding a second poll source (system_alerts table) doubles DB load on the stream endpoint and introduces ordering complexity. By writing alerts into the existing event stream, we get real-time delivery for free with zero changes to the SSE infrastructure.

**Why alerts at the top of the Overview tab:** Following dashboard UX conventions — alerts and warnings surface above normal operational data. The `full-width` CSS class ensures the alert banner spans both columns of the overview grid.
