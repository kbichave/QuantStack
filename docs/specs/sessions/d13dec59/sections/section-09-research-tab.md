# Section 9: Research Tab

## Overview

This section builds the Research tab, which contains five widgets: a research queue showing work-in-progress and pending tasks, an ML experiments table with concept drift alerts, a discoveries widget for alpha research programs and breakthrough features, a reflections widget for trade lessons learned, and a bug status widget for self-healing/AutoResearchClaw tracking. The tab lives inside the `ResearchTab` pane of the `TabbedContent` created in Section 1.

## Dependencies

- **Section 01 (Scaffolding):** `RefreshableWidget` base class, `TieredRefreshScheduler`, `QuantStackApp` with `TabbedContent`, `dashboard.tcss`
- **Section 02 (Query Layer):** Query functions from `queries/research.py` (queries 35-42)

No dependency on Section 03 (Charts) — this tab uses tables and text rendering, not chart widgets.

## Widget Hierarchy

```
ResearchTab (ScrollableContainer)
├── ResearchQueueWidget      # WIP + pending queue with priorities
├── MLExperimentsWidget      # Experiment table + concept drift alerts
├── DiscoveriesWidget        # Alpha programs + breakthrough features
├── ReflectionsWidget        # Trade reflections (lessons learned)
└── BugStatusWidget          # Self-healing / AutoResearchClaw status
```

All five widgets subclass `RefreshableWidget` from `src/quantstack/tui/base.py`. Each implements `fetch_data()` (runs in a background thread via `@work(thread=True)`) and `update_view()` (runs on the main thread to mutate widget state).

## File Locations

| File | Purpose |
|------|---------|
| `src/quantstack/tui/widgets/research.py` | All 5 widgets: `ResearchQueueWidget`, `MLExperimentsWidget`, `DiscoveriesWidget`, `ReflectionsWidget`, `BugStatusWidget` |
| `src/quantstack/tui/queries/research.py` | 8 query functions (research WIP, queue, ML experiments, alpha programs, breakthroughs, reflections, bugs, concept drift) |
| `tests/unit/test_tui/test_research.py` | Unit tests for all 5 widgets |

---

## Tests (Write First)

File: `tests/unit/test_tui/test_research.py`

```python
# tests/unit/test_tui/test_research.py

# --- ResearchQueueWidget ---
# Test: ResearchQueueWidget renders WIP items with computed duration from heartbeat_at
# Test: WIP section shows symbol, domain, agent, and duration
# Test: queue section sorted by priority DESC, shows priority badge + task_type + topic
# Test: handles empty WIP (renders "No active research")
# Test: handles empty queue (renders "Queue empty" or equivalent)

# --- MLExperimentsWidget ---
# Test: MLExperimentsWidget renders last 10 experiments in table form
# Test: table columns include date, model, symbol, AUC, Sharpe, features, verdict
# Test: concept drift alerts flag symbols where AUC dropped > 0.05 in last 14 days
# Test: drift alert renders as a distinct warning line above or below the table
# Test: handles empty ml_experiments table (renders "No experiments recorded")
# Test: handles missing AUC values (None) without crashing

# --- DiscoveriesWidget ---
# Test: DiscoveriesWidget renders alpha programs with investigation name and progress %
# Test: breakthrough_features section degrades gracefully if table is missing (no crash, shows fallback text)
# Test: alpha programs with 0 progress render correctly
# Test: handles empty alpha_research_program table

# --- ReflectionsWidget ---
# Test: ReflectionsWidget renders trade reflections with symbol, lesson, and P&L
# Test: reflections sorted by created_at DESC (most recent first)
# Test: handles empty trade_reflections table (renders "No reflections yet")

# --- BugStatusWidget ---
# Test: BugStatusWidget shows open vs resolved/fixed bug counts
# Test: open bugs listed with tool_name, loop_name, consecutive_errors count
# Test: handles empty bugs table (renders "No bugs tracked")
# Test: recently fixed bugs (last 7 days) shown separately from open bugs
```

Tests should patch `pg_conn()` at the query-function level (not the pool factory) to return a mock `PgConnection`. Every widget test should verify that empty/None data does not crash the widget.

---

## Queries

All queries live in `src/quantstack/tui/queries/research.py`. Each function accepts a `PgConnection` from the `pg_conn()` context manager, returns a typed dataclass or list of dataclasses, and returns a sensible default on any exception.

### Data Types

```python
@dataclass
class ResearchWIP:
    symbol: str
    domain: str        # 'investment', 'swing', 'options'
    agent: str
    heartbeat_at: datetime
    duration_minutes: float  # computed: now() - heartbeat_at

@dataclass
class ResearchTask:
    task_id: str
    task_type: str
    topic: str | None
    priority: int
    status: str        # 'pending', 'in_progress', 'done', 'failed'
    created_at: datetime

@dataclass
class MLExperiment:
    experiment_id: str
    created_at: datetime
    model_type: str
    symbol: str
    auc: float | None
    sharpe: float | None
    feature_count: int
    verdict: str | None

@dataclass
class DriftAlert:
    symbol: str
    recent_auc: float
    historical_auc: float
    drop: float        # historical - recent

@dataclass
class AlphaProgram:
    investigation_id: str
    name: str
    status: str
    progress_pct: float
    key_findings: str | None

@dataclass
class Breakthrough:
    feature_name: str
    importance_score: float
    first_seen: datetime

@dataclass
class TradeReflection:
    symbol: str
    strategy_id: str | None
    lesson: str
    pnl: float | None
    regime_at_entry: str | None
    created_at: datetime

@dataclass
class BugRecord:
    bug_id: str
    tool_name: str
    loop_name: str
    status: str        # 'open', 'in_progress', 'fixed', 'reverted', 'wont_fix'
    priority: int
    consecutive_errors: int
    error_message: str
    created_at: datetime
```

### Query Functions

```python
def fetch_research_wip(conn) -> list[ResearchWIP]:
    """Active research work-in-progress from research_wip table.

    Returns items with heartbeat_at within the last 30 minutes (stale items are
    considered dead). Duration computed as now() - heartbeat_at.
    Default: empty list on error.
    """

def fetch_research_queue(conn) -> list[ResearchTask]:
    """Pending research tasks from research_queue, ordered by priority DESC then created_at ASC.

    Filters to status IN ('pending', 'in_progress'). LIMIT 20.
    Default: empty list on error.
    """

def fetch_ml_experiments(conn) -> list[MLExperiment]:
    """Last 10 ML experiments from ml_experiments, ordered by created_at DESC.

    Default: empty list on error.
    """

def fetch_concept_drift(conn) -> list[DriftAlert]:
    """Detect concept drift by comparing recent AUC (last 14 days) vs historical average per symbol.

    Flags symbols where the AUC drop exceeds 0.05. Queries ml_experiments,
    groups by symbol, compares avg(auc) for last 14d vs avg(auc) for older rows.
    Default: empty list on error.
    """

def fetch_alpha_programs(conn) -> list[AlphaProgram]:
    """Active alpha research programs from alpha_research_program table.

    Ordered by status (active first), then created_at DESC.
    Default: empty list on error.
    """

def fetch_breakthroughs(conn) -> list[Breakthrough]:
    """Breakthrough features from breakthrough_features table, ordered by importance_score DESC.

    LIMIT 10. Returns empty list if the table does not exist (graceful degradation).
    Default: empty list on error.
    """

def fetch_reflections(conn) -> list[TradeReflection]:
    """Recent trade reflections from trade_reflections table, ordered by created_at DESC.

    LIMIT 15.
    Default: empty list on error.
    """

def fetch_bugs(conn) -> list[BugRecord]:
    """Bug records from bugs table. Returns all open bugs plus recently resolved (last 7 days).

    Ordered by: open first (by priority DESC), then resolved (by updated_at DESC).
    Default: empty list on error.
    """
```

---

## ResearchQueueWidget

### Purpose

Shows what the research pipeline is currently working on and what is queued up next. This gives visibility into the autonomous research loop's workload.

### Rendering

Two sections rendered as Rich `Panel`s:

**WIP section** — titled "Work In Progress". A Rich `Table` with columns: Symbol, Domain, Agent, Duration. Duration is computed as `now() - heartbeat_at` and displayed in human-friendly format (e.g., "12m", "1h 23m"). Rows with duration > 15 minutes should be colored yellow (may be stuck). Items with heartbeat older than 30 minutes are excluded by the query (considered dead).

**Queue section** — titled "Pending Queue". A Rich `Table` with columns: Priority, Type, Topic. Priority rendered as a badge: P1 = red bold, P2 = yellow, P3-P5 = dim white. Task type is the `task_type` field (e.g., `bug_fix`, `ml_arch_search`, `strategy_hypothesis`). Topic is the human-readable description.

If WIP is empty, render "No active research" in dim text. If queue is empty, render "Queue empty" in dim text.

### Refresh Tier

T2 (15-second interval). Research WIP changes frequently during active research cycles.

---

## MLExperimentsWidget

### Purpose

Displays recent ML experiment results and flags potential concept drift where model performance is degrading.

### Rendering

**Experiments table** — Rich `Table` with columns: Date, Model, Symbol, AUC, Sharpe, Features, Verdict. Date formatted as `MM-DD HH:MM`. AUC and Sharpe formatted to 3 decimal places. Verdict column colored: green for "promoted"/"accepted", red for "rejected", yellow for "pending"/"inconclusive". Feature count is an integer.

**Concept drift alerts** — Rendered above the experiments table as a distinct alert section. Each drift alert renders as a styled warning line: `"DRIFT: {symbol} AUC dropped {drop:.3f} (from {historical:.3f} to {recent:.3f})"` in bold yellow/red. If no drift alerts, this section is not rendered (no placeholder needed).

If `ml_experiments` is empty, render "No experiments recorded" in dim text in place of the table.

### Refresh Tier

T3 (60-second interval). Experiments run infrequently.

---

## DiscoveriesWidget

### Purpose

Shows alpha research programs (longer-running investigations) and any breakthrough features the system has identified.

### Rendering

**Alpha programs section** — Rich `Table` with columns: Investigation, Status, Progress, Findings. Progress rendered as a percentage string (e.g., "45%"). If the investigation has `key_findings`, show a truncated one-line summary (first 80 characters). Status colored: green for "complete", yellow for "active"/"in_progress", dim for "paused"/"abandoned".

**Breakthrough features section** — Below the alpha programs table. Rendered as a simple list: feature name + importance score formatted to 3 decimals. If the `breakthrough_features` table does not exist (the query returns empty list after catching the relation-does-not-exist error), render "Breakthrough tracking not available" in dim text instead of crashing.

If `alpha_research_program` is empty, render "No active research programs" in dim text.

### Refresh Tier

T4 (120-second interval). Research programs evolve slowly.

---

## ReflectionsWidget

### Purpose

Displays trade reflections — lessons learned from closed trades. These are generated by the trade reflector agent after each trade closes.

### Rendering

Rich `Table` with columns: Date, Symbol, Lesson, P&L, Regime. Date formatted as `MM-DD`. Lesson column is the core insight (truncated to 60 characters in the table; full text available in drill-down). P&L colored green if positive, red if negative. Regime shows the market regime at entry time.

If no reflections exist, render "No reflections yet" in dim text.

### Refresh Tier

T3 (60-second interval). Reflections are created after trade closes.

---

## BugStatusWidget

### Purpose

Shows the self-healing system's status. Every tool failure recorded by `record_tool_error()` lands in the `bugs` table. The AutoResearchClaw pipeline picks up `bug_fix` tasks from `research_queue` and attempts automated patches.

### Rendering

Two sections:

**Open bugs** — Rich `Table` with columns: Tool, Loop, Errors, Priority, Status. "Errors" is the `consecutive_errors` count. Priority badge same style as ResearchQueueWidget (P1 red, P2 yellow, etc.). Status colored: red for "open", yellow for "in_progress".

**Recently resolved** — Bugs with status IN ('fixed', 'reverted', 'wont_fix') and updated within last 7 days. Same table structure but with dimmer styling and the status column showing the resolution.

Summary line at the top: `"Bugs: {open_count} open, {resolved_count} resolved (7d)"` — green if 0 open, yellow if 1-2 open, red if 3+ open.

If no bugs tracked, render "No bugs tracked" in dim green text (good news).

### Refresh Tier

T2 (15-second interval). Bug status matters during active trading.

---

## Tab Composition

The `ResearchTab` is a `ScrollableContainer` that composes the five widgets vertically:

```python
class ResearchTab(ScrollableContainer):
    """Research tab: queue, ML experiments, discoveries, reflections, bugs."""

    def compose(self) -> ComposeResult:
        yield ResearchQueueWidget()
        yield MLExperimentsWidget()
        yield DiscoveriesWidget()
        yield ReflectionsWidget()
        yield BugStatusWidget()
```

This tab is mounted into TabPane index 5 (the 5th tab, 0-indexed) in the `QuantStackApp.compose()` method from Section 1. The tab is activated by pressing key `5` (per the keybinding scheme).

---

## Underlying Database Tables

The queries in this section read from the following existing tables (all created by `db.py` migrations — no new tables are needed for this section):

- **`research_wip`** — columns: `symbol`, `domain`, `agent` (implicit from context), `heartbeat_at`. Domain is constrained to `('investment', 'swing', 'options')`.
- **`research_queue`** — columns: `task_id`, `task_type`, `topic`, `priority`, `status`, `created_at`. Status lifecycle: `pending` -> `in_progress` -> `done`/`failed`. Index on `(status, priority DESC, created_at)`.
- **`ml_experiments`** — columns: `experiment_id`, `created_at`, `model_type`, `symbol`, `auc`, `sharpe`, `feature_count` (derived from features JSONB), `verdict`. Index on `(symbol, created_at DESC)`.
- **`alpha_research_program`** — columns: `investigation_id`, `created_at`, `status`, `progress_pct` (derived), `key_findings` (derived from findings or summary field).
- **`breakthrough_features`** — columns: `feature_name`, `importance_score`, `first_seen`. May not exist in all environments.
- **`trade_reflections`** — columns: `id`, `symbol`, `strategy_id`, `lesson` (derived from reflection text), `pnl` (derived from outcome), `regime_at_entry`, `created_at`. Index on `(regime_at_entry, symbol)`.
- **`bugs`** — columns: `bug_id`, `tool_name`, `loop_name`, `error_message`, `error_fingerprint`, `stack_trace`, `status`, `priority`, `consecutive_errors`. Status lifecycle: `open` -> `in_progress` -> `fixed`/`reverted`/`wont_fix`.

When writing queries, inspect the actual column names in `db.py` migration functions. The dataclass field names above are semantic — the query must map actual column names to dataclass fields (e.g., `research_wip` may not have an explicit `agent` column; check the migration and adapt accordingly).

---

## CSS Styling Notes

Add rules to `src/quantstack/tui/dashboard.tcss` for:
- `ResearchQueueWidget` — WIP and Queue panels side-by-side if terminal width > 120, stacked otherwise
- `MLExperimentsWidget` — drift alerts section should have a distinct background (e.g., `background: $warning-darken-3`) to draw attention
- `BugStatusWidget` — open bugs section should have a red-tinted border when bugs exist, green border when clean
- All widgets should have a small vertical margin between them for visual separation

---

## Implementation Checklist

1. Write tests in `tests/unit/test_tui/test_research.py`
2. Implement query functions in `src/quantstack/tui/queries/research.py` with dataclass return types
3. Implement `ResearchQueueWidget` in `src/quantstack/tui/widgets/research.py`
4. Implement `MLExperimentsWidget` in the same file
5. Implement `DiscoveriesWidget` in the same file
6. Implement `ReflectionsWidget` in the same file
7. Implement `BugStatusWidget` in the same file
8. Implement `ResearchTab` container in the same file
9. Add CSS rules to `src/quantstack/tui/dashboard.tcss`
10. Wire `ResearchTab` into the appropriate `TabPane` in `app.py`
11. Verify all tests pass with mocked queries
