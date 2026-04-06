# Implementation Plan — QuantStack Dashboard v2

## What We're Building

A Textual-based terminal dashboard that replaces the existing Rich-based `scripts/dashboard.py` (796 lines, single-screen). The new dashboard lives at `src/quantstack/tui/` as a separate package from the existing `src/quantstack/dashboard/` (which contains a FastAPI web dashboard and `events.py` used by the agent executor). It provides 6 tabs, 13 sections, 45 database queries on tiered refresh intervals, tabbed navigation, drill-down modals, and Unicode chart rendering. It is strictly read-only.

**Critical constraint:** `src/quantstack/dashboard/` already exists with `app.py` (FastAPI SSE dashboard on port 8421) and `events.py` (`publish_event()` imported by `src/quantstack/graphs/agent_executor.py`). The TUI dashboard MUST NOT overwrite this package. It lives at `src/quantstack/tui/` instead.

## Why This Architecture

The existing dashboard is a monolithic script using Rich's `Live` display. It works but crams 8 sections onto one screen with no interactivity beyond scrolling. The owner needs to answer 6 priority-ordered questions (Is anything on fire? Am I making money? What's the system doing? etc.) which maps naturally to tabs.

Textual (built on Rich) adds tabs, keybindings, scrollable panels, modal overlays, and async workers — all in a pure terminal app that runs over SSH with no browser dependency. The dashboard stays terminal-native, matching the autonomous trading system's ethos.

DB access uses `@work(thread=True)` wrappers around the existing `pg_conn()` psycopg2 pattern, avoiding an asyncpg migration. Workers return data via Textual's message system — thread workers post custom messages or use `App.call_from_thread()` to safely update widget state from background threads.

---

## Section 1: Package Scaffolding & Textual App Shell

### Goal
Create the `src/quantstack/tui/` package with a working Textual App that renders an empty 6-tab layout with header, footer, and the core refresh/data-flow patterns.

### Package Structure

```
src/quantstack/tui/
├── __init__.py
├── __main__.py         # python -m quantstack.tui
├── app.py              # QuantStackApp(App) — tabs, keybindings, refresh scheduler
├── base.py             # RefreshableWidget base class (thread->UI data flow pattern)
├── widgets/
│   ├── __init__.py
│   ├── header.py       # HeaderBar widget (kill switch, regime, AV, mode)
│   └── decisions.py    # Decision log / audit trail widget
├── queries/
│   ├── __init__.py
│   └── system.py       # Kill switch, regime, AV count queries
├── screens/
│   ├── __init__.py
│   └── detail.py       # DetailModal(ModalScreen) base
├── refresh.py          # TieredRefreshScheduler
├── charts.py           # Unicode rendering utilities (sparklines, bars, heatmap)
└── dashboard.tcss      # Textual CSS (separate file, not inline)
```

### App Class Design

`QuantStackApp` subclasses `textual.app.App`. It defines:

- `TITLE = "QUANTSTACK"`
- `CSS_PATH = "dashboard.tcss"` (separate file for maintainability)
- `BINDINGS` for keys 1-6 (tab switch), q (quit), r (force refresh), ? (help), / (search), j/k (scroll)
- `compose()` yields: docked `HeaderBar` at top, `TabbedContent` with 6 `TabPane`s, docked footer `Static` at bottom
- `on_mount()` initializes the `TieredRefreshScheduler`

### RefreshableWidget Base Class

All data-driven widgets inherit from `RefreshableWidget`, which codifies the thread->UI data flow:

```python
class RefreshableWidget(Static):
    """Base for widgets that refresh from DB queries in background threads.

    Subclasses override fetch_data() (runs in thread) and update_view() (runs in main thread).
    """

    def refresh_data(self) -> None:
        """Trigger a background data fetch. Calls fetch_data() in a thread worker,
        then update_view() on the main thread via call_from_thread()."""
        ...

    def fetch_data(self) -> Any:
        """Override: run queries using pg_conn(). Runs in a background thread."""
        ...

    def update_view(self, data: Any) -> None:
        """Override: update widget rendering with fetched data. Runs on main thread."""
        ...
```

This pattern is used by every widget in sections 4-10. It handles the `@work(thread=True)` -> `App.call_from_thread()` bridge that is required because psycopg2 is blocking and Textual widgets can only be mutated from the main thread.

### TieredRefreshScheduler

A coordinator class that manages 4 refresh tiers via `set_interval()`:

```python
class TieredRefreshScheduler:
    """Manages staggered data refresh across 4 tiers.

    Uses threading.Semaphore (not asyncio.Semaphore) to limit concurrent
    DB queries, since all queries run in @work(thread=True) workers.
    """

    TIERS: dict[str, float]  # tier_name -> interval_seconds
    # T1=5s, T2=15s, T3=60s, T4=120s

    _db_semaphore: threading.Semaphore  # limits concurrent DB connections (default 5)
```

It tracks the active tab and only fires queries for visible sections + always-on header queries. On tab switch (via `TabbedContent.TabActivated` message), it immediately fires the new tab's queries.

Staggered start: T1 at 0.0s, T2 at 0.3s offset, T3 at 0.6s, T4 at 0.9s.

Connection pool note: the actual pool default is 20 (`PG_POOL_MAX`), not 10. The dashboard runs alongside 3 graph services sharing the same pool. The `threading.Semaphore(5)` limits the dashboard's share of pool connections.

### Entry Points

- `__main__.py` imports and runs `QuantStackApp().run()`
- `scripts/dashboard.py` becomes a thin wrapper: `from quantstack.tui.app import QuantStackApp; QuantStackApp().run()`
- The old v1 code in `scripts/dashboard.py` is deleted entirely

### HeaderBar Widget

Renders the single-line header:
```
QUANTSTACK  16:48:37  | LIVE | Kill: ok | Regime: trending_up (82%) | AV: 393/25000 | Universe: 20
```

Subclasses `Static`, updates via reactive attributes (kill_status, regime, av_count, trading_mode). Refreshed on T1 (5s).

---

## Section 2: Query Layer

### Goal
Create a typed query module that encapsulates all 45 database queries, returning dataclasses instead of raw tuples.

### Design

Each `queries/*.py` file contains functions that:
1. Accept a `PgConnection` (from `pg_conn()` context manager)
2. Use `PgConnection.execute()` / `.fetchall()` (NOT raw cursors — `PgConnection` provides retry logic and transaction management)
3. Return a typed dataclass or list of dataclasses
4. Return a sensible default on any exception (empty list, None, zero)

The queries module does NOT manage connections — callers (widget `fetch_data()` methods) acquire connections via `pg_conn()` context managers.

### Porting from v1

Many queries already exist in `scripts/dashboard.py` (the v1 code). The following should be ported (adapted to return dataclasses):
- Kill switch, AV calls, regime, docker health (queries 1-4)
- Positions, closed trades, equity summary (queries 8-11)
- Strategy list with status breakdown (query 16, partial)
- Research queue, WIP, ML experiments, bugs (queries 35-37, 41)
- Agent events grouped by team (query 30)
- Decision events (query from decisions section)
- Data freshness (OHLCV only — v1 does this; v2 extends to 7 data types)

New queries (not in v1): per-data-type freshness (17-23), collector health (24), signals (25-26), calendar (27-29), agent skills/calibration (32-34), alpha programs (38-39), concept drift (42), risk (43-45), equity curve/benchmark (12-13), P&L attribution (14-15).

### Query Files

**`queries/system.py`** — kill switch, AV calls, regime, docker health, graph checkpoints, heartbeats, agent events (queries 1-7)

**`queries/portfolio.py`** — equity, cash, positions, closed trades, equity curve 30d, benchmark, P&L by strategy, P&L by symbol (queries 8-15)

**`queries/strategies.py`** — strategy pipeline with forward test stats (query 16)

**`queries/data_health.py`** — per-data-type freshness (OHLCV, news, sentiment, fundamentals, options, insider, macro), collector health (queries 17-24)

**`queries/signals.py`** — active signals, signal brief detail (queries 25-26)

**`queries/calendar.py`** — earnings 90d, market holidays, macro events (queries 27-29)

**`queries/agents.py`** — agent events by graph, cycle history, agent skills, calibration, prompt versions (queries 30-34)

**`queries/research.py`** — research WIP, queue, ML experiments, alpha programs, breakthroughs, reflections, bugs, concept drift (queries 35-42)

**`queries/risk.py`** — risk snapshot, equity alerts, risk rejections (queries 43-45)

### Result Types

Define dataclasses in each query file for their return types. Examples:

```python
@dataclass
class EquitySummary:
    total_equity: float
    cash: float
    daily_pnl: float
    daily_return_pct: float
    high_water: float
    drawdown_pct: float

@dataclass
class StrategyCard:
    strategy_id: str
    name: str
    status: str  # draft, backtested, forward_testing, live, retired
    symbol: str
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    fwd_trades: int
    fwd_pnl: float
    fwd_days: int
    fwd_required_days: int
```

### Error Handling Pattern

Every query function wraps its body in try/except, logs the error (with query name, not raw SQL), and returns the default value. This preserves the v1 graceful degradation pattern.

---

## Section 3: Unicode Chart Renderers

### Goal
Build reusable chart rendering functions in `charts.py` that return Rich `Text` objects for embedding in Textual widgets.

### Functions

**`sparkline(data, width, color) -> Text`**
Renders a series as Unicode block characters (▁▂▃▄▅▆▇█). Normalizes to 0-7 range. Resamples if data length exceeds width. Returns Rich `Text` with specified color.

**`horizontal_bar(value, max_value, width, color) -> Text`**
Renders a single horizontal bar: `████████░░░░ 75%`. Uses `█` for filled and `░` for empty. Shows percentage label.

**`progress_bar(current, total, width) -> Text`**
Renders `████████░░ 80%` style progress bar. Green if >70%, yellow if >40%, red otherwise.

**`daily_heatmap(daily_values, dates) -> Table`**
Renders a Mon-Fri grid of daily P&L values. Each cell colored green (positive) or red (negative), intensity proportional to magnitude. Uses Rich `Table` with no borders, styled cells.

**`equity_curve(values, width, height) -> str`**
Renders a multi-line ASCII chart using box-drawing characters (╭╮╯╰─│). Height in terminal rows (default 5). Width auto-fits. Shows Y-axis labels (min, max) and X-axis (first, last dates).

---

## Section 4: Overview Tab Widgets

### Goal
Build the Overview tab — a single-screen glance that shows compact summaries from all other tabs.

### Layout (fits 24-line terminal)

The Overview tab uses a grid layout with 2 columns:

| Left Column | Right Column |
|-------------|-------------|
| Services compact (3 lines) | Risk compact (3 lines) |
| Portfolio compact (3 lines) | Recent trades (3 lines) |
| Strategy counts (2 lines) | Top signals (2 lines) |
| Data health bars (4 lines) | Research compact (3 lines) |

Below the grid: Agent activity (1 line per graph), Daily digest (2 lines).

### Widgets

Each compact widget subclasses `Static` and renders a Rich `Panel` or `Table`. They receive data via method calls from the refresh scheduler and re-render.

**`ServicesCompact`** — One line per graph: `R:UP c#9 170s  T:UP c#5 28s  S:UP c#3 45s  Errors: 0`

**`RiskCompact`** — `Exposure: 37%  DD: -2.5%  VaR: $245  Alerts: 1  Kill: ok`

**`PortfolioCompact`** — `Equity: $10,234  Today: +$127.50  Open: AAPL +2.1%, NVDA +1.7%`

**`TradesCompact`** — Last 3 closed trades, one line each

**`StrategyCountsCompact`** — `Draft:3 BT:6 FT:2 Live:0 Ret:4  Next promotion: AAPL inv (18d)`

**`SignalsCompact`** — Top 3 signals: `NVDA BUY 87%  AAPL HOLD 72%  TSLA SELL 68%`

**`DataHealthCompact`** — Coverage bars for each data type (4 most important)

**`ResearchCompact`** — `WIP: 2  Queue: 12  ML: 47  Bugs: 0  Breakthroughs: 2`

**`AgentActivityLine`** — One per graph showing current agent + last tool call

**`DigestCompact`** — Shows daily digest after 17:00 ET (from `loop_iteration_context`)

---

## Section 5: Portfolio Tab

### Goal
Build the full Portfolio tab with equity curve, positions, trades, P&L attribution, and daily heatmap.

### Widget Hierarchy

```
PortfolioTab (ScrollableContainer)
├── EquitySummaryWidget      # 2 lines: equity, cash, exposure, drawdown, Sharpe, Sortino
├── EquityCurveWidget        # 6 lines: ASCII chart of last 30 days + benchmark comparison
├── PositionsTableWidget     # DataTable: symbol, qty, entry, current, P&L, %, strategy, days
├── ClosedTradesWidget       # DataTable: last 10 closed trades
├── PnlByStrategyWidget      # Table: strategy, realized, unrealized, win/loss, sharpe
├── PnlBySymbolWidget        # Horizontal bars per symbol
└── DailyHeatmapWidget       # Mon-Fri grid, last 30 trading days
```

### EquityCurveWidget

Queries `daily_equity` (last 30 rows) and `benchmark_daily` (last 30 rows for SPY). Renders two overlaid sparklines (portfolio and benchmark) with a label showing alpha. Uses the `equity_curve()` chart renderer for the multi-line version.

### PositionsTableWidget

Uses Textual's `DataTable` widget (sortable columns, scrollable). Columns: Symbol, Qty, Entry, Current, P&L, %, Strategy, Days. Rows colored green/red by P&L sign. Supports Enter for drill-down.

### DailyHeatmapWidget

Queries `daily_equity` last 30 rows, groups by weekday, renders using `daily_heatmap()` from charts.py.

---

## Section 6: Strategies Tab

### Goal
Build the Kanban-style strategy pipeline visualization.

### Widget Hierarchy

```
StrategiesTab (ScrollableContainer)
├── PipelineKanbanWidget     # 5 columns: Draft, Backtested, Forward Testing, Live, Retired
├── PromotionGatesWidget     # Static text showing gate criteria
└── StrategyDetailWidget     # Expandable detail for selected strategy
```

### PipelineKanbanWidget

Renders 5 side-by-side columns using Rich `Columns`. Each column is a `Panel` with strategy cards inside. Cards show: name, symbol, Sharpe, MaxDD, win rate. Forward testing cards also show progress bar (days / required).

Color coding: green = meeting promotion gates, yellow = borderline, red = failing.

The widget tracks a "selected" card index for cursor navigation. Enter triggers drill-down modal.

### Strategy Query

Single query joining `strategies` with aggregated `closed_trades` to compute forward testing stats (trade count, P&L, win rate since `updated_at`).

---

## Section 7: Data & Signals Tab

### Goal
Build the data health matrix, market calendar, and signal engine dashboard.

### Widget Hierarchy

```
DataSignalsTab (ScrollableContainer)
├── MarketCalendarWidget     # Upcoming events: holidays, earnings, FOMC, macro
├── DataHealthMatrixWidget   # Symbol x data-type matrix + coverage bars
└── SignalEngineWidget       # Top signals + collector health + expandable briefs
```

### DataHealthMatrixWidget

The most query-intensive widget: 7 freshness queries (one per data type) + 1 collector health query.

Renders a `Table` with symbols as rows, data types as columns. Each cell shows age + checkmark/cross based on staleness thresholds:
- OHLCV: stale after 2 trading days
- News/Sentiment: stale after 24h
- Fundamentals: stale after 90d
- Options: stale after 1 trading day
- Insider: stale after 30d
- Macro: stale after 7d

Below the matrix: coverage summary bars per data type using `horizontal_bar()`.

### MarketCalendarWidget

Combines data from 3 sources:
1. `market_holidays` table (hardcoded US holidays, seeded at startup)
2. `earnings_calendar` (next 90 days for universe symbols)
3. `macro_indicators` (FOMC dates, CPI/PPI releases)

Renders chronologically with color coding: red = market closed, yellow = early close, cyan = earnings, white = macro.

### SignalEngineWidget

Queries `signal_state` and parses `brief_json` JSONB. Renders a table of top signals sorted by confidence with per-factor columns (ML, Sentiment, Technical, Options, Macro). Enter on a row shows a curated signal brief in a drill-down modal.

Collector health: iterates `brief_json` across symbols, counts `collector_failures`. Shows checkmark/cross per collector name.

---

## Section 8: Agents Tab

### Goal
Build graph activity feeds and agent performance scorecard.

### Widget Hierarchy

```
AgentsTab (ScrollableContainer)
├── GraphActivityWidget      # 3 side-by-side panels (Research, Trading, Supervisor)
└── AgentScorecardWidget     # Performance table with calibration
```

### GraphActivityWidget

Three bordered panels side-by-side (using `Horizontal` container). Each panel shows:
- Current active agent + node
- Cycle progress bar (node index / total nodes, estimated from graph_checkpoints duration)
- Last N events (tool calls, LLM messages) with relative timestamps
- Cycle history (last 3 cycles: duration, primary agent, tool count)

Events come from `agent_events` (T1, 5s refresh). Cycle data from `graph_checkpoints` (T3, 60s).

### AgentScorecardWidget

Table from `agent_skills` and `calibration_records`. Columns: Agent, Accuracy, Win Rate, Avg P&L, IC, Trend.

Calibration section: compares stated_confidence vs actual win rate per agent. Flags overconfident agents.

Prompt evolution section (from `prompt_versions`): shows last optimization date, active candidates count. Degrades gracefully if tables are empty.

---

## Section 9: Research Tab

### Goal
Build the research queue, ML lab, discoveries, reflections, and self-healing status.

### Widget Hierarchy

```
ResearchTab (ScrollableContainer)
├── ResearchQueueWidget      # WIP + pending queue with priorities
├── MLExperimentsWidget      # Experiment table + concept drift alerts
├── DiscoveriesWidget        # Alpha programs + breakthrough features
├── ReflectionsWidget        # Trade reflections (lessons learned)
└── BugStatusWidget          # Self-healing / AutoResearchClaw status
```

### ResearchQueueWidget

Two sections:
1. **WIP** — from `research_wip`: symbol, domain, agent, duration (calculated from heartbeat_at)
2. **Queue** — from `research_queue` ORDER BY priority: priority badge, task_type, topic

### MLExperimentsWidget

Table from `ml_experiments` (last 10): date, model, symbol, AUC, Sharpe, features, verdict.

Concept drift detection: compare recent AUC per symbol vs historical average. Flag symbols where AUC dropped >0.05 in last 14 days. Show as alert line.

### DiscoveriesWidget

From `alpha_research_program`: investigation name, progress (% complete), key findings summary.

From `breakthrough_features` (if table exists): feature name, importance score. Degrades if table missing.

---

## Section 10: Risk Console

### Goal
Build risk metrics display with alerts and event history.

### Widget Hierarchy

```
RiskWidget (can appear on Overview compact + Portfolio expanded)
├── RiskMetricsWidget        # Current risk snapshot vs limits
├── RiskEventsWidget         # Recent risk events (rejections, alerts, breaches)
└── RiskAlertsWidget         # Active/cleared equity alerts + kill switch status
```

### RiskMetricsWidget

From `risk_snapshots` (latest row): gross/net exposure, concentration, correlation, sector, VaR, max drawdown. Each metric shows current value, limit, and status (OK/warning/breach).

Color logic: green if <75% of limit, yellow if 75-100%, red if >100%.

Degrades to "No risk data available" if `risk_snapshots` table is empty.

### RiskEventsWidget

From `decision_events` WHERE event_type IN ('risk_rejection', 'drawdown_alert', 'correlation_alert'). Last 7 days. Shows date, type, details.

### RiskAlertsWidget

From `equity_alerts` + `alert_updates`: active vs cleared alerts. Plus kill switch status from `system_state`.

---

## Section 11: Drill-Down Modals

### Goal
Implement `ModalScreen` overlays for detail views.

### Base Design

`DetailModal(ModalScreen)` provides:
- Semi-transparent overlay
- Bordered container with title
- ESC to dismiss
- Scrollable content area

### Modal Variants

**PositionDetailModal** — title: symbol name. Content: entry date, entry price, current price, P&L ($, %), strategy name, regime at entry, stop level, target level. Queried from `positions` + `strategies`.

**StrategyDetailModal** — title: strategy name. Content: type, horizon, regime affinity, entry/exit rules (from strategy record), backtest metrics (Sharpe, MaxDD, trades, win rate), forward test stats.

**SignalDetailModal** — title: symbol. Content: action, confidence, top 5 contributing factors with values, risk flags (upcoming events), signal expiry. Parsed from `signal_state.brief_json`.

**TradeDetailModal** — title: symbol + date. Content: side, entry/exit prices, P&L, holding days, strategy, exit reason, decision reasoning (from `decision_events`), reflection (from `trade_reflections` if exists).

**AgentEventModal** — title: agent name + timestamp. Content: full event content text, tool name, graph name, node name.

---

## Section 12: Database Migrations

### Goal
Add new tables needed by the dashboard.

### market_holidays

```python
@dataclass
class MarketHoliday:
    date: date
    name: str
    market_status: str  # 'closed' | 'early_close'
    close_time: time | None  # None for full close
    exchange: str  # default 'NYSE'
```

Created in `db.py`'s `run_migrations_pg()` with `CREATE TABLE IF NOT EXISTS`. Seeded with US market holidays for current year + next year using `INSERT ... ON CONFLICT DO NOTHING` for idempotency. Includes: New Year's, MLK Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas. Early closes: day before Independence Day (13:00), Black Friday (13:00), Christmas Eve (13:00).

### benchmark_daily

```python
@dataclass
class BenchmarkDaily:
    date: date
    symbol: str  # default 'SPY'
    close: float
    daily_return_pct: float
```

Created in migrations. Populated by the data acquisition pipeline — extract SPY daily closes from existing `ohlcv` table and compute returns. A migration helper function seeds historical data from `ohlcv` WHERE symbol='SPY' AND timeframe='1D'.

---

## Section 13: Integration, Testing, and Entry Points

### scripts/dashboard.py Replacement

Replace the 796-line script with a thin wrapper that imports from `quantstack.tui`. The old v1 code is deleted entirely.

### status.sh Update

Update `status.sh` to call `python -m quantstack.tui` instead of `python scripts/dashboard.py --watch`. Remove `--watch` flag handling (Textual is always live).

### pyproject.toml

Add `textual>=0.50` to dependencies.

### Docker Health Check Fallback

The v1 dashboard runs `docker compose ps` via subprocess to check service health. This works locally but fails inside Docker containers (no Docker socket). The v2 queries should:
1. Try `docker compose ps --format json` first (local dev)
2. Fall back to TCP port checks for known services (PostgreSQL:5432, LangFuse:3100) if Docker CLI unavailable
3. Return "unknown" status (not error) if neither method works

### Testing Strategy

**Unit tests** in `tests/unit/test_tui/`:

- `test_queries.py` — Patch `pg_conn()` to return a mock `PgConnection`, verify each query function returns correct dataclass type, handles exceptions gracefully (returns default). Patch at the query-function level, not the pool factory.
- `test_charts.py` — Test sparkline, horizontal_bar, progress_bar, daily_heatmap, equity_curve with known inputs
- `test_refresh.py` — Test tier intervals, tab-visibility filtering, stagger offsets

**Integration tests** using Textual's `pilot` framework in `tests/integration/test_tui_app.py`:

- App starts without DB (graceful degradation) — patch all query modules to return defaults
- Tab switching (`pilot.press("1")` through `"6"`)
- Keybindings work (q quits, r triggers refresh)
- Modal opens on Enter, closes on Esc

**What NOT to test:**
- Exact visual rendering (too brittle)
- Live DB queries (require running PostgreSQL)
- Docker health checks

### Decisions Widget

The spec includes a decision log / audit trail section. It appears on the Overview tab (compact: last 3 decisions) and the Research tab (expanded: full decision history with reasoning).

`widgets/decisions.py` renders from `decision_events` table: timestamp, agent_name, action (ENTER/SKIP/TIGHTEN/REJECT/APPROVE), symbol, confidence, and one-line reasoning summary. Enter on a row opens the TradeDetailModal with full decision reasoning chain.

---

## Dependencies Between Sections

```
Section 1 (Package + App Shell)
  ↓
Section 2 (Query Layer)    Section 3 (Charts)
  ↓                          ↓
Section 4-10 (Tab Widgets — can be built in parallel once 1-3 are done)
  ↓
Section 11 (Drill-down Modals — needs widgets to trigger them)
  ↓
Section 12 (DB Migrations — can be done early but tested with widgets)
  ↓
Section 13 (Integration + Testing)
```

Sections 4-10 are independent and can be built in any order or in parallel. Each section is self-contained with its own widgets and queries.
