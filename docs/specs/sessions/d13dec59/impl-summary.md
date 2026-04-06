# Implementation Summary

## What Was Implemented

### Section 1: Scaffolding
TUI package at `src/quantstack/tui/` with `QuantStackApp` (Textual App), `RefreshableWidget` base class, `TieredRefreshScheduler` (4 tiers: 5s/15s/60s/120s), `HeaderBar` widget, CSS layout, and `__main__.py` entry point. Added `textual>=0.50` dependency.

### Section 2: Query Layer
9 query modules in `src/quantstack/tui/queries/` with 45+ functions returning typed dataclasses. Modules: system, portfolio, strategies, data_health, signals, calendar, agents, research, risk. All functions accept `PgConnection`, catch exceptions, return safe defaults.

### Section 3: Charts
5 pure rendering functions in `charts.py`: sparkline (block chars), horizontal_bar, progress_bar, daily_heatmap, equity_curve. All handle edge cases (empty, zero range, NaN).

### Section 4: Overview Tab
11 compact widgets providing a dashboard summary: ServicesCompact, RiskCompact, PortfolioCompact, TradesCompact, StrategyCountsCompact, SignalsCompact, DataHealthCompact, ResearchCompact, AgentActivityLine, DigestCompact, DecisionsCompact. 2-column grid layout.

### Section 5: Portfolio Tab
7 widgets: EquitySummaryWidget, EquityCurveWidget, PositionsTableWidget, ClosedTradesWidget, PnlByStrategyWidget, PnlBySymbolWidget, DailyHeatmapWidget.

### Section 6: Strategies Tab
PipelineKanbanWidget (lifecycle columns: draft/backtested/forward_testing/live/retired), PromotionGatesWidget.

### Section 7: Data & Signals Tab
MarketCalendarWidget, DataHealthMatrixWidget, SignalEngineWidget.

### Section 8: Agents Tab
GraphActivityWidget, AgentScorecardWidget (with overconfidence flagging via calibration records).

### Section 9: Research Tab
ResearchQueueWidget, MLExperimentsWidget (with concept drift alerts), DiscoveriesWidget, ReflectionsWidget, BugStatusWidget.

### Section 10: Risk Console
RiskMetricsWidget (color-coded against RISK_LIMITS thresholds), RiskEventsWidget, RiskAlertsWidget.

### Section 11: Drill-Down Modals
`DetailModal` base class (ModalScreen with ESC dismiss, scrollable content). 5 variants: PositionDetailModal, StrategyDetailModal (async DB fetch), SignalDetailModal (async DB fetch), TradeDetailModal (async decision/reflection fetch), AgentEventModal. Added `fetch_strategy_detail`, `fetch_trade_decision`, `fetch_trade_reflection` queries.

### Section 12: DB Migrations
`market_holidays` table with `_compute_us_holidays()` seeding function and `_migrate_market_holidays_pg()` registered in `run_migrations_pg()`.

### Section 13: Integration
Updated `status.sh` to use `python3 -m quantstack.tui`. Added Docker health check with 3-tier fallback (docker compose CLI, TCP port probes, unknown). Entry point tests verifying import paths.

## Key Technical Decisions

- **`from textual import work`** not `from textual.worker import work` — the decorator is re-exported at the `textual` package level.
- **`threading.Semaphore(5)`** for DB connection cap — queries run in `@work(thread=True)` workers, not async.
- **`benchmark` column** in `benchmark_daily` table (not `symbol`), `close_price` (not `close`).
- **JSONB parsing** via `_parse_json()` helper — `register_default_jsonb(loads=lambda x: x)` returns raw strings.
- **Autouse conftest fixture** patches `Static.update` for all widget unit tests — avoids `NoActiveAppError` without running a full Textual app.
- **Modals receive data directly** where possible (Position, ClosedTrade, GraphActivity) to avoid extra DB calls. Only StrategyDetailModal and SignalDetailModal do async DB fetches on mount.

## Known Issues / Remaining TODOs

- `test_coordination.py::TestEventBus::test_publish_and_poll` has an intermittent ordering-dependent error (pre-existing, unrelated to TUI).
- Integration pilot tests (`tests/integration/test_tui_app.py`) not created — would require async test infrastructure and full app mounting. Manual smoke testing recommended.
- Widget Enter-key handlers for triggering modals are not wired in tab widgets yet — modal classes exist and are tested, but the `on_key` / `on_data_table_row_selected` handlers in widgets need wiring during final integration.

## Test Results

```
233 passed (TUI unit tests)
489 passed, 1 pre-existing error (full unit suite)
```

## Files Created or Modified

### Created (src/quantstack/tui/)
- `__init__.py`, `__main__.py`, `app.py`, `base.py`, `refresh.py`, `charts.py`, `dashboard.tcss`
- `widgets/__init__.py`, `widgets/header.py`, `widgets/overview.py`, `widgets/portfolio.py`, `widgets/strategies.py`, `widgets/data_signals.py`, `widgets/agents.py`, `widgets/research.py`, `widgets/risk.py`
- `screens/__init__.py`, `screens/overview.py`, `screens/portfolio.py`, `screens/strategies.py`, `screens/data_signals.py`, `screens/agents.py`, `screens/research.py`, `screens/risk.py`, `screens/detail.py`
- `queries/__init__.py`, `queries/system.py`, `queries/portfolio.py`, `queries/strategies.py`, `queries/data_health.py`, `queries/signals.py`, `queries/calendar.py`, `queries/agents.py`, `queries/research.py`, `queries/risk.py`

### Created (tests/)
- `tests/unit/test_tui/__init__.py`, `conftest.py`
- `tests/unit/test_tui/test_app.py`, `test_base.py`, `test_refresh.py`, `test_queries.py`, `test_charts.py`, `test_migrations.py`
- `tests/unit/test_tui/test_overview.py`, `test_portfolio_tab.py`, `test_strategies_tab.py`, `test_data_signals_tab.py`, `test_agents_tab.py`, `test_research_tab.py`, `test_risk_tab.py`
- `tests/unit/test_tui/test_modals.py`, `test_entry.py`

### Modified
- `scripts/dashboard.py` — replaced 796-line Rich dashboard with thin TUI wrapper
- `status.sh` — updated to call `python3 -m quantstack.tui`
- `src/quantstack/db.py` — added market_holidays migration
- `pyproject.toml` — added textual>=0.50, textual-dev>=1.0
