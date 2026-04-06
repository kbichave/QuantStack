<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-scaffolding
section-02-query-layer
section-03-charts
section-04-overview-tab
section-05-portfolio-tab
section-06-strategies-tab
section-07-data-signals-tab
section-08-agents-tab
section-09-research-tab
section-10-risk-console
section-11-drill-down-modals
section-12-db-migrations
section-13-integration
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-scaffolding | - | all | No |
| section-02-query-layer | 01 | 04-10, 13 | Yes |
| section-03-charts | 01 | 04-10 | Yes |
| section-04-overview-tab | 02, 03 | 13 | Yes |
| section-05-portfolio-tab | 02, 03 | 11, 13 | Yes |
| section-06-strategies-tab | 02, 03 | 11, 13 | Yes |
| section-07-data-signals-tab | 02, 03, 12 | 11, 13 | Yes |
| section-08-agents-tab | 02 | 11, 13 | Yes |
| section-09-research-tab | 02 | 11, 13 | Yes |
| section-10-risk-console | 02 | 11, 13 | Yes |
| section-11-drill-down-modals | 04-10 | 13 | No |
| section-12-db-migrations | - | 07 | Yes |
| section-13-integration | 01-12 | - | No |

## Execution Order

1. **Batch 1:** section-01-scaffolding (foundation — everything depends on this)
2. **Batch 2:** section-02-query-layer, section-03-charts, section-12-db-migrations (parallel — all independent after 01)
3. **Batch 3:** section-04-overview-tab, section-05-portfolio-tab, section-06-strategies-tab, section-07-data-signals-tab, section-08-agents-tab, section-09-research-tab, section-10-risk-console (parallel — all independent, need 02+03)
4. **Batch 4:** section-11-drill-down-modals (needs widgets from batch 3)
5. **Batch 5:** section-13-integration (final — needs everything)

## Section Summaries

### section-01-scaffolding
Create `src/quantstack/tui/` package, Textual App shell with 6 empty tabs, RefreshableWidget base class, TieredRefreshScheduler, HeaderBar widget, dashboard.tcss, entry points. Add textual dependency to pyproject.toml.

### section-02-query-layer
All 45 database query functions in `queries/` subpackage. Typed dataclass returns, PgConnection API, error handling with defaults. Port existing v1 queries where possible.

### section-03-charts
Unicode chart renderers: sparkline, horizontal_bar, progress_bar, daily_heatmap, equity_curve. Returns Rich Text/Table objects.

### section-04-overview-tab
Overview tab with compact summary widgets: ServicesCompact, RiskCompact, PortfolioCompact, TradesCompact, StrategyCountsCompact, SignalsCompact, DataHealthCompact, ResearchCompact, AgentActivityLine, DigestCompact, DecisionsCompact.

### section-05-portfolio-tab
Full Portfolio tab: EquitySummaryWidget, EquityCurveWidget, PositionsTableWidget, ClosedTradesWidget, PnlByStrategyWidget, PnlBySymbolWidget, DailyHeatmapWidget.

### section-06-strategies-tab
Strategies tab: PipelineKanbanWidget (5-column Kanban), PromotionGatesWidget, StrategyDetailWidget with cursor navigation.

### section-07-data-signals-tab
Data & Signals tab: MarketCalendarWidget, DataHealthMatrixWidget (symbol x data-type freshness), SignalEngineWidget with collector health.

### section-08-agents-tab
Agents tab: GraphActivityWidget (3 side-by-side panels), AgentScorecardWidget with calibration.

### section-09-research-tab
Research tab: ResearchQueueWidget, MLExperimentsWidget with drift alerts, DiscoveriesWidget, ReflectionsWidget, BugStatusWidget, DecisionsWidget (expanded).

### section-10-risk-console
Risk console: RiskMetricsWidget (values vs limits), RiskEventsWidget, RiskAlertsWidget. Appears on Overview (compact) and Portfolio (expanded).

### section-11-drill-down-modals
ModalScreen overlays: DetailModal base, PositionDetailModal, StrategyDetailModal, SignalDetailModal, TradeDetailModal, AgentEventModal. Curated summaries, ESC to dismiss.

### section-12-db-migrations
New tables in db.py migrations: market_holidays (with idempotent seeding), benchmark_daily (with historical seed from ohlcv SPY data).

### section-13-integration
Replace scripts/dashboard.py with thin wrapper, update status.sh, add Docker health fallback, unit tests (queries, charts, refresh), Textual pilot integration tests, CSS finalization.
