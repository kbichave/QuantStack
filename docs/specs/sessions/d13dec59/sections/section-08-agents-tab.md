# Section 8: Agents Tab

## Overview

Build the Agents tab, which provides real-time visibility into the three LangGraph StateGraphs (Research, Trading, Supervisor) and agent-level performance metrics. The tab contains two widgets: `GraphActivityWidget` (three side-by-side activity panels) and `AgentScorecardWidget` (performance table with calibration and prompt evolution tracking).

**Dependencies:** Section 01 (scaffolding — `RefreshableWidget`, `TieredRefreshScheduler`, app shell), Section 02 (query layer — `queries/agents.py`).

**Blocks:** Section 11 (drill-down modals — `AgentEventModal`), Section 13 (integration tests).

---

## File Inventory

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/tui/widgets/agents.py` | Create | `GraphActivityWidget`, `AgentScorecardWidget` |
| `src/quantstack/tui/queries/agents.py` | Create (if not done in Section 02) | Agent query functions and dataclasses |
| `src/quantstack/tui/app.py` | Modify | Wire widgets into the Agents `TabPane` |
| `src/quantstack/tui/dashboard.tcss` | Modify | Agents tab layout styles |
| `tests/unit/test_tui/test_agents.py` | Create | Unit tests (write first) |

---

## Tests (Write First)

File: `tests/unit/test_tui/test_agents.py`

```python
# tests/unit/test_tui/test_agents.py

# --- GraphActivityWidget tests ---

# Test: GraphActivityWidget renders 3 side-by-side panels
#   - Compose the widget, assert it yields 3 child containers (one per graph: Research, Trading, Supervisor)

# Test: each panel shows current agent + node
#   - Supply mock agent_events data with graph_name, agent_name, node_name fields
#   - Assert the rendered text contains the current agent and node for each graph

# Test: cycle progress bar renders based on checkpoint data
#   - Supply mock graph_checkpoints with node_index and total_nodes
#   - Assert a progress bar is rendered showing correct fraction

# Test: events show relative timestamps
#   - Supply events with timestamps 5s, 30s, 2m ago
#   - Assert rendered text shows "5s ago", "30s ago", "2m ago" (not raw datetimes)

# Test: cycle history shows last 3 cycles
#   - Supply 5 completed cycles in graph_checkpoints
#   - Assert only the 3 most recent are rendered, each showing duration, primary agent, tool count

# --- AgentScorecardWidget tests ---

# Test: AgentScorecardWidget renders table with correct columns
#   - Columns: Agent, Accuracy, Win Rate, Avg P&L, IC, Trend
#   - Supply mock agent_skills rows, verify all columns present

# Test: calibration section flags overconfident agents
#   - Supply calibration_records where stated_confidence >> actual win rate
#   - Assert the agent row is flagged (e.g., styled red or annotated)

# Test: prompt evolution section degrades if tables empty
#   - Supply empty prompt_versions data
#   - Assert widget renders without error, shows fallback text like "No prompt data"

# Test: handles empty agent_skills table
#   - Supply empty list for agent_skills
#   - Assert widget renders without crashing, shows "No agent data" or equivalent
```

---

## Query Layer (from Section 02)

The Agents tab depends on 5 queries defined in `src/quantstack/tui/queries/agents.py`. If Section 02 is already implemented, these will exist. If not, they must be created here.

### Dataclasses

```python
@dataclass
class AgentEvent:
    id: int
    graph_name: str       # 'research', 'trading', 'supervisor'
    agent_name: str
    node_name: str
    event_type: str       # 'tool_call', 'llm_message', 'error', etc.
    content: str
    created_at: datetime

@dataclass
class GraphCheckpoint:
    graph_name: str
    cycle_id: int
    node_name: str
    started_at: datetime
    duration_seconds: float
    tool_count: int
    primary_agent: str

@dataclass
class AgentSkill:
    agent_name: str
    accuracy: float | None
    win_rate: float | None
    avg_pnl: float | None
    information_coefficient: float | None
    trend: str | None        # 'improving', 'stable', 'declining'

@dataclass
class CalibrationRecord:
    agent_name: str
    stated_confidence: float
    actual_win_rate: float
    sample_size: int

@dataclass
class PromptVersion:
    agent_name: str
    version: int
    optimized_at: datetime
    active_candidates: int
```

### Query Functions

All functions accept a `PgConnection`, return typed results, and catch exceptions returning sensible defaults.

**`fetch_agent_events(conn, limit=60) -> list[AgentEvent]`**
- Query: `SELECT id, graph_name, agent_name, node_name, event_type, content, created_at FROM agent_events ORDER BY created_at DESC LIMIT %s`
- Default on error: `[]`
- Refresh tier: T1 (5s) -- this is the primary real-time feed

**`fetch_graph_checkpoints(conn, limit=15) -> list[GraphCheckpoint]`**
- Query: joins `graph_checkpoints` to compute per-cycle stats (duration, tool count, primary agent). Returns last 15 checkpoints (~5 per graph x 3 graphs).
- Default on error: `[]`
- Refresh tier: T3 (60s)

**`fetch_agent_skills(conn) -> list[AgentSkill]`**
- Query: `SELECT agent_name, accuracy, win_rate, avg_pnl, information_coefficient, trend FROM agent_skills ORDER BY agent_name`
- Default on error: `[]`
- Refresh tier: T4 (120s)

**`fetch_calibration_records(conn) -> list[CalibrationRecord]`**
- Query: `SELECT agent_name, stated_confidence, actual_win_rate, sample_size FROM calibration_records ORDER BY agent_name`
- Default on error: `[]`
- Refresh tier: T4 (120s)

**`fetch_prompt_versions(conn) -> list[PromptVersion]`**
- Query: `SELECT agent_name, version, optimized_at, active_candidates FROM prompt_versions WHERE is_active = true ORDER BY optimized_at DESC`
- Default on error: `[]`
- Refresh tier: T4 (120s)

### Error Handling Pattern

Every query function follows this structure:

```python
def fetch_agent_events(conn: PgConnection, limit: int = 60) -> list[AgentEvent]:
    """Fetch recent agent events across all graphs."""
    try:
        rows = conn.execute("SELECT ... LIMIT %s", (limit,)).fetchall()
        return [AgentEvent(*row) for row in rows]
    except Exception:
        logger.exception("fetch_agent_events failed")
        return []
```

Log the function name (not raw SQL). Return the default. Never re-raise.

---

## Widget Implementation

### AgentsTab Container

File: `src/quantstack/tui/widgets/agents.py`

The tab is a `ScrollableContainer` (or `VerticalScroll`) that composes the two widgets vertically:

```python
class AgentsTab(ScrollableContainer):
    """Agents tab: graph activity feeds + agent performance scorecard."""

    def compose(self) -> ComposeResult:
        yield GraphActivityWidget()
        yield AgentScorecardWidget()
```

### GraphActivityWidget

Subclasses `RefreshableWidget`. Renders three side-by-side bordered panels using a `Horizontal` container, one per graph (Research, Trading, Supervisor).

**Layout:** Three equal-width panels in a `Horizontal` container. Each panel is a bordered `Static` (or `Container` with a border CSS class).

**Data flow:**
- `fetch_data()` calls `fetch_agent_events(conn)` and `fetch_graph_checkpoints(conn)` inside a `pg_conn()` context manager. Returns both result lists.
- `update_view(data)` partitions events by `graph_name`, builds Rich renderables for each panel, and updates the three panel widgets.

**Per-panel content (top to bottom):**

1. **Header line:** `Research  [agent: quant_researcher @ hypothesis_gen]`
   - Current active agent and node, derived from the most recent event for that graph.

2. **Cycle progress bar:** Rendered using the `progress_bar()` chart helper from `charts.py`.
   - Node index / total nodes estimated from checkpoint durations. If no checkpoint data, show "No cycle data".

3. **Recent events (last N):** List of the most recent events for that graph.
   - Format: `2m ago  tool_call  fetch_fundamentals`
   - Timestamps rendered as relative (e.g., "5s ago", "2m ago", "1h ago"). Compute from `datetime.now(UTC) - event.created_at`.
   - Show up to 8 events per panel (fits a standard terminal height).

4. **Cycle history (last 3 completed cycles):**
   - Format: `Cycle #9  4m32s  quant_researcher  12 tools`
   - Derived from `graph_checkpoints` grouped by `cycle_id` for that graph.

**Relative timestamp helper:** Write a small utility function (private to the module or in a shared utils) that converts a timedelta to a human-readable string: `<60s` -> `Xs ago`, `<3600s` -> `Xm ago`, else `Xh ago`.

### AgentScorecardWidget

Subclasses `RefreshableWidget`. Renders a Rich `Table` from `agent_skills`, `calibration_records`, and `prompt_versions`.

**Data flow:**
- `fetch_data()` calls all three query functions (`fetch_agent_skills`, `fetch_calibration_records`, `fetch_prompt_versions`) inside one `pg_conn()` context.
- `update_view(data)` builds the scorecard table and optional subsections.

**Scorecard table columns:** Agent | Accuracy | Win Rate | Avg P&L | IC | Trend

- Trend column uses arrows or text: "improving" in green, "stable" in dim white, "declining" in red.
- Rows sorted by agent name.

**Calibration subsection:**
- Below the main table, render a "Calibration" header.
- For each agent in `calibration_records`, compare `stated_confidence` vs `actual_win_rate`.
- If `stated_confidence - actual_win_rate > 0.15` (15 percentage points), flag as overconfident. Render the agent name in red with a warning marker.
- If `sample_size < 10`, note "insufficient data" in dim text instead of flagging.
- If `calibration_records` is empty, show "No calibration data available" in dim text.

**Prompt evolution subsection:**
- Below calibration, render a "Prompt Evolution" header.
- For each agent in `prompt_versions`, show: `agent_name  v{version}  optimized {relative_date}  {active_candidates} candidates`.
- If `prompt_versions` is empty, show "No prompt optimization data" in dim text.
- This entire subsection degrades gracefully -- the widget renders fine without it.

**Empty state:** If `agent_skills` is empty, render a centered message: "No agent performance data available". Skip the calibration and prompt sections entirely.

---

## App Integration

In `src/quantstack/tui/app.py`, the Agents `TabPane` (tab index 4, key "4") should compose the `AgentsTab` container:

```python
with TabPane("Agents", id="agents-tab"):
    yield AgentsTab()
```

### Refresh Registration

Register widgets with the `TieredRefreshScheduler`:

| Widget | Tier | Interval | Tab-gated |
|--------|------|----------|-----------|
| `GraphActivityWidget` | T1 | 5s | Yes (agents tab) |
| `AgentScorecardWidget` | T4 | 120s | Yes (agents tab) |

The `GraphActivityWidget` also feeds the `AgentActivityLine` on the Overview tab (Section 04). The Overview's compact widget should share the same query results (fetched once, distributed to both consumers) or query independently at T2 (15s). This is an implementation detail -- either approach is acceptable as long as the DB semaphore is respected.

---

## CSS Additions

Add to `src/quantstack/tui/dashboard.tcss`:

```css
/* Agents tab layout */
#agents-tab GraphActivityWidget {
    height: auto;
    max-height: 70%;
}

#agents-tab GraphActivityWidget Horizontal {
    height: auto;
}

#agents-tab .graph-panel {
    width: 1fr;
    border: solid $primary;
    padding: 0 1;
    height: auto;
}

#agents-tab AgentScorecardWidget {
    height: auto;
    margin-top: 1;
}
```

The three graph panels should each take equal width (1fr). The `GraphActivityWidget` takes roughly the top 60-70% of the tab, with `AgentScorecardWidget` below.

---

## Drill-Down Integration (Section 11 dependency)

When implemented, pressing Enter on an event row in `GraphActivityWidget` should open an `AgentEventModal` (defined in Section 11) showing the full event content. For now, the widget should track a selected event index (cursor navigation with j/k or up/down arrows) but the modal push can be a no-op or commented stub until Section 11 is complete.

---

## Graceful Degradation

All widgets must handle missing or empty data without crashing:

- **No `agent_events` table or empty table:** GraphActivityWidget shows "No agent activity" in each panel.
- **No `graph_checkpoints` table or empty:** Progress bar and cycle history sections show "No cycle data".
- **No `agent_skills` table or empty:** AgentScorecardWidget shows "No agent performance data available".
- **No `calibration_records` or `prompt_versions` tables:** Respective subsections show dim fallback text. Widget does not error.

This is enforced by the query layer returning empty defaults on any exception, and the widget `update_view()` checking for empty data before rendering.
