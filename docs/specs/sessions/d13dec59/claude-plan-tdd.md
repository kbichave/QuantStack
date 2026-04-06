# TDD Plan — QuantStack Dashboard v2

Testing framework: **pytest** with **pytest-asyncio** (existing project conventions). Textual integration tests use the **pilot** framework. Test files in `tests/unit/test_tui/` and `tests/integration/`.

---

## Section 1: Package Scaffolding & Textual App Shell

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_app.py

# Test: QuantStackApp can be instantiated without errors
# Test: QuantStackApp.compose() yields HeaderBar, TabbedContent with 6 TabPanes, footer
# Test: BINDINGS includes keys 1-6, q, r, ?, /, j, k, Enter
# Test: CSS_PATH points to dashboard.tcss

# tests/unit/test_tui/test_refresh.py

# Test: TieredRefreshScheduler has 4 tiers (T1=5, T2=15, T3=60, T4=120)
# Test: stagger offsets are 0.0, 0.3, 0.6, 0.9
# Test: changing active_tab updates which queries fire
# Test: always-on queries (header) fire regardless of active tab
# Test: db_semaphore is threading.Semaphore (not asyncio.Semaphore)

# tests/unit/test_tui/test_base.py

# Test: RefreshableWidget.refresh_data() calls fetch_data() in a thread
# Test: RefreshableWidget.update_view() is called on main thread after fetch
# Test: RefreshableWidget handles fetch_data() exceptions without crashing
```

---

## Section 2: Query Layer

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_queries.py

# -- System queries --
# Test: fetch_kill_switch returns bool, default False on error
# Test: fetch_av_calls returns int, default 0 on error
# Test: fetch_regime returns RegimeState dataclass with trend/vol/confidence
# Test: fetch_graph_checkpoints returns list of GraphCheckpoint dataclasses
# Test: fetch_heartbeats returns list of Heartbeat dataclasses
# Test: fetch_agent_events returns list of AgentEvent, LIMIT 60, ordered DESC

# -- Portfolio queries --
# Test: fetch_equity_summary returns EquitySummary dataclass
# Test: fetch_positions returns list of Position dataclasses, ordered by unrealized_pnl DESC
# Test: fetch_closed_trades returns list of ClosedTrade, LIMIT 10
# Test: fetch_equity_curve returns list of EquityPoint (30 rows)
# Test: fetch_benchmark returns list of BenchmarkPoint (30 rows for SPY)
# Test: fetch_pnl_by_strategy returns list of StrategyPnl dataclasses
# Test: fetch_pnl_by_symbol returns list of SymbolPnl dataclasses

# -- Strategy queries --
# Test: fetch_strategy_pipeline returns list of StrategyCard with fwd_trades/fwd_pnl computed
# Test: strategies are ordered by status priority (live > forward > backtested > draft > retired)

# -- Data health queries --
# Test: fetch_ohlcv_freshness returns dict[symbol, datetime]
# Test: fetch_news_freshness returns dict[symbol, datetime]
# Test: each freshness query returns empty dict on error
# Test: fetch_collector_health returns dict[collector_name, bool]

# -- Signal queries --
# Test: fetch_active_signals returns list of Signal sorted by confidence DESC
# Test: fetch_signal_brief parses brief_json JSONB correctly

# -- Risk queries --
# Test: fetch_risk_snapshot returns RiskSnapshot or None if table empty
# Test: fetch_equity_alerts returns list of EquityAlert with status

# -- All queries --
# Test: every query function returns its default value when pg_conn() raises
# Test: every query function uses PgConnection.execute() (not raw cursors)
```

---

## Section 3: Unicode Chart Renderers

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_charts.py

# Test: sparkline([1,2,3,4,5,6,7,8]) returns 8 block characters ascending
# Test: sparkline([]) returns empty string
# Test: sparkline with all same values returns all same block char
# Test: sparkline resamples when data exceeds width
# Test: sparkline returns Rich Text with specified color

# Test: horizontal_bar(75, 100, 20) renders ~15 filled + 5 empty + "75%"
# Test: horizontal_bar(0, 100, 20) renders all empty
# Test: horizontal_bar handles max_value=0 without division error

# Test: progress_bar(80, 100) renders green
# Test: progress_bar(50, 100) renders yellow
# Test: progress_bar(20, 100) renders red

# Test: daily_heatmap renders Mon-Fri grid
# Test: daily_heatmap colors positive green, negative red
# Test: daily_heatmap handles empty input

# Test: equity_curve renders multi-line output with Y-axis labels
# Test: equity_curve handles single data point
# Test: equity_curve handles flat data (all same value)
```

---

## Section 4: Overview Tab Widgets

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_overview.py

# Test: ServicesCompact renders graph status with cycle numbers
# Test: RiskCompact renders exposure, drawdown, VaR, alert count
# Test: PortfolioCompact renders equity, daily pnl, top positions
# Test: TradesCompact renders last 3 trades with P&L
# Test: StrategyCountsCompact renders counts per status
# Test: SignalsCompact renders top 3 signals
# Test: DataHealthCompact renders coverage bars
# Test: ResearchCompact renders WIP/queue/ML/bug counts
# Test: DigestCompact renders digest after 17:00, empty before
# Test: all compact widgets handle empty/None data without crashing
```

---

## Section 5: Portfolio Tab

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_portfolio.py

# Test: EquitySummaryWidget renders equity, cash, exposure, drawdown
# Test: EquityCurveWidget renders sparkline from equity data
# Test: EquityCurveWidget shows alpha vs benchmark
# Test: PositionsTableWidget renders DataTable with correct columns
# Test: PositionsTableWidget colors green for positive, red for negative P&L
# Test: ClosedTradesWidget renders last 10 trades
# Test: PnlByStrategyWidget renders per-strategy attribution
# Test: PnlBySymbolWidget renders horizontal bars
# Test: DailyHeatmapWidget renders weekday grid from daily_equity data
# Test: all widgets handle empty data (no positions, no trades)
```

---

## Section 6: Strategies Tab

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_strategies.py

# Test: PipelineKanbanWidget renders 5 columns (Draft, Backtested, Forward Testing, Live, Retired)
# Test: strategy cards show name, symbol, Sharpe, MaxDD
# Test: forward testing cards show progress bar (days / required)
# Test: color coding: green for meeting gates, yellow for borderline, red for failing
# Test: selected card index updates on navigation
# Test: Enter on selected card triggers drill-down
# Test: PromotionGatesWidget renders gate criteria text
# Test: handles empty strategy list (all columns empty)
```

---

## Section 7: Data & Signals Tab

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_data_signals.py

# Test: DataHealthMatrixWidget renders symbol x data-type table
# Test: cells show checkmark for fresh, cross for stale
# Test: staleness thresholds: OHLCV=2d, News=24h, Fundamentals=90d, etc.
# Test: coverage bars show correct percentages per data type

# Test: MarketCalendarWidget renders events chronologically
# Test: holiday entries colored red (closed) or yellow (early close)
# Test: earnings entries colored cyan
# Test: handles empty calendar gracefully

# Test: SignalEngineWidget renders signals sorted by confidence
# Test: per-factor columns show ML, Sentiment, Technical, Options, Macro
# Test: collector health shows checkmark/cross per collector
# Test: Enter on signal row triggers drill-down
```

---

## Section 8: Agents Tab

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_agents.py

# Test: GraphActivityWidget renders 3 side-by-side panels
# Test: each panel shows current agent + node
# Test: cycle progress bar renders based on checkpoint data
# Test: events show relative timestamps
# Test: cycle history shows last 3 cycles

# Test: AgentScorecardWidget renders table with correct columns
# Test: calibration section flags overconfident agents
# Test: prompt evolution section degrades if tables empty
# Test: handles empty agent_skills table
```

---

## Section 9: Research Tab

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_research.py

# Test: ResearchQueueWidget renders WIP with duration
# Test: queue sorted by priority, shows priority badge
# Test: MLExperimentsWidget renders last 10 experiments
# Test: concept drift alerts flag AUC drops > 0.05 in 14d
# Test: DiscoveriesWidget renders alpha programs with progress
# Test: breakthrough_features section degrades if table missing
# Test: ReflectionsWidget renders lessons with P&L
# Test: BugStatusWidget shows open/resolved counts
```

---

## Section 10: Risk Console

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_risk.py

# Test: RiskMetricsWidget renders current values vs limits
# Test: color logic: green <75%, yellow 75-100%, red >100%
# Test: degrades to "No risk data" if risk_snapshots empty
# Test: RiskEventsWidget renders last 7 days of risk events
# Test: RiskAlertsWidget shows active vs cleared alerts
# Test: kill switch status rendered correctly
```

---

## Section 11: Drill-Down Modals

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_modals.py

# Test: DetailModal renders with semi-transparent overlay
# Test: ESC dismisses the modal
# Test: scrollable content area works for long content

# Test: PositionDetailModal shows entry price, P&L, strategy, stop/target
# Test: StrategyDetailModal shows backtest + forward test metrics
# Test: SignalDetailModal shows top 5 factors and risk flags
# Test: TradeDetailModal shows decision reasoning + reflection
# Test: AgentEventModal shows full event content
# Test: all modals handle missing/null fields gracefully
```

---

## Section 12: Database Migrations

### Tests (write BEFORE implementing)

```python
# tests/unit/test_tui/test_migrations.py

# Test: market_holidays table created with correct schema (date PK, name, status, close_time, exchange)
# Test: market_holidays seeding is idempotent (ON CONFLICT DO NOTHING)
# Test: seeded holidays include all major US holidays for current + next year
# Test: early closes have correct close_time (13:00)

# Test: benchmark_daily table created with correct schema (date+symbol PK, close, return)
# Test: benchmark_daily seeding from ohlcv WHERE symbol='SPY' produces correct returns
```

---

## Section 13: Integration, Testing, and Entry Points

### Tests (write BEFORE implementing)

```python
# tests/integration/test_tui_app.py (Textual pilot tests)

# Test: app starts and renders without DB connection (all widgets show fallback)
# Test: tab switching via keys 1-6
# Test: q key quits the app
# Test: r key triggers refresh
# Test: ? key shows help overlay
# Test: Enter opens modal, Esc closes it
# Test: app handles rapid tab switching without crashing

# tests/unit/test_tui/test_entry.py

# Test: scripts/dashboard.py imports from quantstack.tui (not quantstack.dashboard)
# Test: python -m quantstack.tui entry point works
```
