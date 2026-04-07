# Section 02: System-Level Alert Lifecycle

## Overview

This section implements the system-level alert lifecycle for QuantStack: five LangChain tools for LLM-facing alert management, plus an internal `emit_system_alert()` helper for programmatic alert creation from deterministic code paths (risk gate, kill switch, corporate actions, factor exposure, EventBus ACK monitor).

System alerts are distinct from equity alerts. Equity alerts track ticker-level signals (entry, exit, thesis updates). System alerts track operational and infrastructure events: risk breaches, service failures, kill switch activations, data quality issues, performance degradation, factor drift, and ACK timeouts. They have a different schema, lifecycle, and set of consumers.

**Depends on:** section-01-db-schema (the `system_alerts` table must exist before any alert can be written).

**Blocks:** sections 03 (corporate actions), 04 (factor exposure), 05 (performance attribution), 06 (dashboard alerts), 07 (EventBus ACK) -- all of these emit system alerts and depend on this infrastructure being in place.

---

## Tests

All tests should be placed in `tests/unit/test_system_alerts.py` and `tests/integration/test_system_alerts_integration.py`.

### Unit Tests

```python
# tests/unit/test_system_alerts.py

# Test: create_system_alert returns alert ID and inserts row with status='open'
# Test: create_system_alert validates category against allowed values
#   Allowed categories: risk_breach, service_failure, kill_switch, data_quality,
#   performance_degradation, factor_drift, ack_timeout, thesis_review
# Test: create_system_alert validates severity against allowed values
#   Allowed severities: info, warning, critical, emergency
# Test: acknowledge_alert sets status='acknowledged', acknowledged_by, acknowledged_at
# Test: acknowledge_alert on already-acknowledged alert is idempotent
# Test: escalate_alert bumps severity one level (warning -> critical) and sets status='escalated'
# Test: escalate_alert on already-emergency severity doesn't go higher
# Test: resolve_alert sets status='resolved', resolution, resolved_at
# Test: resolve_alert on already-resolved alert is idempotent
# Test: query_system_alerts filters by severity correctly
# Test: query_system_alerts filters by status correctly
# Test: query_system_alerts filters by category correctly
# Test: query_system_alerts respects since_hours parameter
# Test: emit_system_alert (internal helper) writes same row as create_system_alert tool
```

### Integration Tests

```python
# tests/integration/test_system_alerts_integration.py

# Test: full lifecycle: create -> acknowledge -> resolve, verify all timestamps set
# Test: full lifecycle: create -> escalate -> resolve, verify severity bumped
# Test: all 5 tools registered in TOOL_REGISTRY
```

---

## Implementation

### File 1: LangChain Tools -- `src/quantstack/tools/langchain/system_alert_tools.py`

Five new `@tool`-decorated async functions. These are the LLM-facing interface -- supervisor graph agents (health_monitor, self_healer) use these tools to manage alert lifecycle.

```python
from langchain_core.tools import tool

ALLOWED_CATEGORIES = {
    "risk_breach", "service_failure", "kill_switch", "data_quality",
    "performance_degradation", "factor_drift", "ack_timeout", "thesis_review",
}

ALLOWED_SEVERITIES = {"info", "warning", "critical", "emergency"}

SEVERITY_ORDER = ["info", "warning", "critical", "emergency"]


@tool
async def create_system_alert(
    category: str, severity: str, title: str, detail: str,
    metadata: dict | None = None,
) -> str:
    """Create a new system-level alert. Returns alert ID.

    Args:
        category: One of risk_breach, service_failure, kill_switch, data_quality,
            performance_degradation, factor_drift, ack_timeout, thesis_review.
        severity: One of info, warning, critical, emergency.
        title: One-line summary of the alert.
        detail: Full context -- what happened, what state the system was in.
        metadata: Optional structured context (positions affected, thresholds, etc.).
    """
    # Validate category and severity against allowed values.
    # Insert row into system_alerts with status='open'.
    # Return the auto-generated alert ID as a string.


@tool
async def acknowledge_alert(alert_id: int, agent_name: str) -> str:
    """Mark alert as being investigated. Sets status to 'acknowledged'.

    Idempotent -- re-acknowledging an already-acknowledged alert is a no-op.
    Sets acknowledged_by and acknowledged_at on the row.
    """


@tool
async def escalate_alert(alert_id: int, reason: str) -> str:
    """Bump severity one level and set status to 'escalated'.

    Severity ladder: info -> warning -> critical -> emergency.
    If already at emergency, severity stays at emergency (ceiling).
    The reason is appended to the detail field for audit trail.
    """


@tool
async def resolve_alert(alert_id: int, resolution: str) -> str:
    """Close alert with resolution notes. Sets status to 'resolved'.

    Idempotent -- re-resolving an already-resolved alert is a no-op.
    Sets resolved_at timestamp and resolution text.
    """


@tool
async def query_system_alerts(
    severity: str | None = None,
    status: str | None = None,
    category: str | None = None,
    since_hours: int = 24,
) -> str:
    """Query system alerts with filters. Returns formatted alert list.

    All filters are optional. When omitted, that dimension is not filtered.
    Results are ordered by severity DESC, created_at DESC.
    The since_hours parameter limits results to alerts created within the
    specified window (default 24 hours).
    """
```

**DB access pattern:** Each tool uses `db_conn()` context manager for database access. All writes are single-row INSERT or UPDATE operations -- no transactions spanning multiple tables.

**Validation:** `create_system_alert` must raise a clear error if category or severity is not in the allowed set. Do not silently accept invalid values.

**Severity escalation logic in `escalate_alert`:** Use `SEVERITY_ORDER` list to find the current index and bump by one. If already at the last index (emergency), keep it there. This is a ceiling, not a cycle.

**Idempotency in `acknowledge_alert` and `resolve_alert`:** Check current status before updating. If already in the target state, return a message indicating no change was made. Do not error.

**Query formatting:** `query_system_alerts` returns a human-readable formatted string (not raw JSON) since the consumer is an LLM agent. Include: alert ID, severity, category, title, status, and age (e.g., "2h ago").

### File 2: Internal Helper -- `src/quantstack/tools/functions/system_alerts.py`

This is the programmatic entry point for non-LLM code to create system alerts. It shares the same `system_alerts` DB table but is called directly by Python code -- no LangChain tool wrapper.

```python
async def emit_system_alert(
    category: str,
    severity: str,
    title: str,
    detail: str,
    source: str = "system",
    metadata: dict | None = None,
) -> int:
    """Direct DB insert for system alerts from deterministic code paths.

    Used by: risk_gate, kill_switch, corporate_actions, factor_exposure,
    event_bus_monitor, and any other non-LLM code that needs to raise alerts.

    NOT a LangChain tool -- called directly by Python code.

    Args:
        category: Alert category (validated against ALLOWED_CATEGORIES).
        severity: Alert severity (validated against ALLOWED_SEVERITIES).
        title: One-line summary.
        detail: Full context.
        source: The module/graph that created the alert (for attribution).
        metadata: Optional JSONB payload.

    Returns:
        The auto-generated alert ID (BIGSERIAL).
    """
    # TODO(kbichave): Add Discord webhook notification for CRITICAL/EMERGENCY alerts.
    # Trigger: when DISCORD_WEBHOOK_URL env var is set. See Phase 9 spec item 9.5 for
    # webhook patterns (rate limits, batching, embed formatting).
```

**Shared validation:** Both the LangChain tools and the internal helper must validate against the same `ALLOWED_CATEGORIES` and `ALLOWED_SEVERITIES` sets. Extract these constants into a shared location (either the internal helper module and import into the LangChain tools, or a constants module). Do not duplicate the sets.

**Source field:** The `source` parameter lets callers identify which module created the alert. The LangChain tools should set `source` to the agent name (from the tool invocation context). The internal helper accepts it as an explicit argument.

### File 3: Tool Registry -- `src/quantstack/tools/registry.py`

Add all five tools to the registry using the `_try_import` pattern (consistent with how other tool modules are registered):

```python
_system_alert_tools = _try_import("quantstack.tools.langchain.system_alert_tools", [
    "create_system_alert", "acknowledge_alert", "escalate_alert",
    "resolve_alert", "query_system_alerts",
])
```

Then add `_system_alert_tools` to the merge loop at the bottom of the file alongside the other `_tools_dict` entries.

### File 4: Supervisor Agent YAML -- `src/quantstack/graphs/supervisor/config/agents.yaml`

Bind the system alert tools to the appropriate supervisor agents.

**health_monitor agent:** Add `query_system_alerts` and `create_system_alert` to its `tools` list. The health monitor needs to create alerts when it detects issues and query existing alerts to avoid duplicating active alerts.

**self_healer agent:** Add `acknowledge_alert`, `escalate_alert`, `resolve_alert`, and `query_system_alerts` to its `tools` list. The self-healer manages the lifecycle of alerts that the health monitor creates. Add `query_system_alerts` and `acknowledge_alert` to `always_loaded_tools` since they are used every cycle.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate system_alerts table | Not reusing equity alert_log | Different schema (no ticker/signal fields), different lifecycle (ack/escalate/resolve vs. status enum), different consumers (supervisor agents vs. trading agents). Combining them would require nullable columns and confusing queries. |
| Internal helper + LangChain tools | Two entry points to same table | Deterministic code (risk gate, kill switch) must create alerts without routing through an LLM. LLM agents need tool-decorated functions. Both write to the same table for unified querying. |
| Validation at creation time | Reject invalid category/severity | Catching bad values early prevents garbage data in the alerts table. Downstream consumers (dashboard, query tool) can trust the data. |
| Idempotent acknowledge/resolve | No-op on re-call | Multiple agents may attempt to acknowledge the same alert. Failing on re-acknowledge would create unnecessary error handling in agent logic. |
| Severity escalation ceiling | Cap at emergency | Prevents infinite escalation loops. An emergency alert cannot become "more emergency." |
| `_try_import` for registry | Graceful degradation | If the module fails to import (e.g., missing dependency during development), the rest of the tool registry still loads. Consistent with existing pattern. |

---

## Severity Escalation Ladder

```
info -> warning -> critical -> emergency (ceiling)
```

`escalate_alert` moves severity up one step. It never wraps around. If the alert is already at `emergency`, the severity stays at `emergency` and the escalation reason is still recorded in the detail field.

---

## Alert Status State Machine

```
open -> acknowledged -> resolved
open -> escalated -> resolved
open -> resolved (direct resolution without acknowledgment is valid)
acknowledged -> escalated -> resolved
```

Any status can transition to `resolved`. The `acknowledged` and `escalated` states are intermediate -- they indicate the alert is being worked on. There is no "reopen" transition; if a resolved issue recurs, a new alert is created.

---

## Downstream Consumers

Once this section is implemented, the following sections will use it:

- **section-03 (corporate actions):** Calls `emit_system_alert()` when M&A events are detected for held symbols (category: `thesis_review`, severity: `critical`). Also calls it after split auto-adjustments (category: `data_quality`, severity: `info`).
- **section-04 (factor exposure):** Calls `emit_system_alert()` when factor drift thresholds are breached (category: `factor_drift`).
- **section-06 (dashboard alerts):** Queries `system_alerts` table to render the TUI alerts widget and `/api/alerts` endpoint.
- **section-07 (EventBus ACK):** Calls `emit_system_alert()` when events miss their ACK deadline (category: `ack_timeout`).
