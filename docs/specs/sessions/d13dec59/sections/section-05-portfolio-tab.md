# Section 5: Portfolio Tab

## Overview

Build the full Portfolio tab for the QuantStack TUI dashboard. This tab answers the question "Am I making money?" with an equity curve, open positions table, closed trades, P&L attribution by strategy and symbol, and a daily heatmap.

**File to create:** `src/quantstack/tui/widgets/portfolio.py`

**Dependencies (must be completed first):**

- Section 1 (Package Scaffolding) — provides `RefreshableWidget` base class in `src/quantstack/tui/base.py`, the `QuantStackApp` shell, and `TieredRefreshScheduler` in `src/quantstack/tui/refresh.py`
- Section 2 (Query Layer) — provides `src/quantstack/tui/queries/portfolio.py` with `fetch_equity_summary`, `fetch_positions`, `fetch_closed_trades`, `fetch_equity_curve`, `fetch_benchmark`, `fetch_pnl_by_strategy`, `fetch_pnl_by_symbol`
- Section 3 (Charts) — provides `sparkline()`, `equity_curve()`, `horizontal_bar()`, `daily_heatmap()` from `src/quantstack/tui/charts.py`

---

## Tests (Write First)

**File:** `tests/unit/test_tui/test_portfolio.py`

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

Each test should:

1. Instantiate the widget in isolation (no live DB).
2. Call `update_view()` directly with mock data (dataclass instances from `queries/portfolio.py`).
3. Assert the widget's rendered content contains expected strings or structural elements.
4. For the empty-data case, pass empty lists or a zeroed-out `EquitySummary` and verify no exception is raised.

Use `unittest.mock.patch` on `pg_conn` at the query-function level when testing `fetch_data()` paths. For widget rendering tests, bypass `fetch_data()` entirely and call `update_view()` with constructed dataclass instances.

---

## Widget Hierarchy

The Portfolio tab is a `ScrollableContainer` registered as a `TabPane` in the main app. It contains 7 widgets stacked vertically:

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

---

## Widget Specifications

### EquitySummaryWidget

Subclasses `RefreshableWidget`. Refresh tier: T2 (15s).

**`fetch_data()`** calls `fetch_equity_summary(conn)` from `queries/portfolio.py`, which returns an `EquitySummary` dataclass:

```python
@dataclass
class EquitySummary:
    total_equity: float
    cash: float
    daily_pnl: float
    daily_return_pct: float
    high_water: float
    drawdown_pct: float
```

**`update_view(data)`** renders two lines inside a Rich `Text`:

```
Equity: $10,234.56  Cash: $3,456.78  Exposure: 66.2%  Daily P&L: +$127.50 (+1.26%)
High Water: $10,500.00  Drawdown: -2.5%  Sharpe: 1.42  Sortino: 2.01
```

- Daily P&L colored green if positive, red if negative.
- Drawdown colored red if below -5%, yellow if below -2%, green otherwise.
- Sharpe and Sortino are computed from the equity curve data (fetched separately by `EquityCurveWidget`); if not available, display "N/A".
- On empty data: render "No equity data available".

### EquityCurveWidget

Subclasses `RefreshableWidget`. Refresh tier: T3 (60s).

**`fetch_data()`** calls both `fetch_equity_curve(conn)` (returns list of `EquityPoint` — last 30 rows from `daily_equity`) and `fetch_benchmark(conn)` (returns list of `BenchmarkPoint` — last 30 rows for SPY from `benchmark_daily`).

**`update_view(data)`** renders:

1. A multi-line ASCII equity curve using `equity_curve(values, width, height=5)` from `charts.py`. Width auto-fits to the widget's content width.
2. Below the chart: two overlaid sparklines — portfolio (green) and benchmark (cyan) — using `sparkline()` from `charts.py`.
3. An alpha label: `Alpha: +2.3% vs SPY` (portfolio cumulative return minus benchmark cumulative return over the 30-day window).

- If benchmark data is unavailable, render only the portfolio curve without comparison.
- If equity data is empty, render "No equity history available".

### PositionsTableWidget

Subclasses `RefreshableWidget`. Refresh tier: T2 (15s).

**`fetch_data()`** calls `fetch_positions(conn)`, which returns a list of `Position` dataclasses ordered by `unrealized_pnl DESC`.

**`update_view(data)`** renders a Textual `DataTable` with sortable columns:

| Column | Content |
|--------|---------|
| Symbol | Ticker symbol |
| Qty | Share count |
| Entry | Entry price |
| Current | Current market price |
| P&L | Unrealized P&L in dollars |
| % | Unrealized P&L as percentage |
| Strategy | Strategy name |
| Days | Days held (since entry date) |

Row styling:
- Entire row styled green if P&L is positive, red if negative.
- Use Textual `DataTable`'s built-in styling via `add_row()` with Rich `Text` objects for colored cells.

Interaction:
- Pressing Enter on a selected row triggers the `PositionDetailModal` (Section 11) with that position's symbol.
- The widget should post a custom Textual message (e.g., `PositionSelected(symbol)`) that the app routes to the modal screen.

Empty state: show "No open positions" as a single centered row.

### ClosedTradesWidget

Subclasses `RefreshableWidget`. Refresh tier: T3 (60s).

**`fetch_data()`** calls `fetch_closed_trades(conn)`, which returns a list of `ClosedTrade` dataclasses (LIMIT 10, most recent first).

**`update_view(data)`** renders a `DataTable` with columns: Date, Symbol, Side, Entry, Exit, P&L, %, Days, Strategy, Exit Reason.

Row styling: green for profitable trades, red for losses.

Pressing Enter on a row triggers the `TradeDetailModal` (Section 11).

Empty state: "No closed trades".

### PnlByStrategyWidget

Subclasses `RefreshableWidget`. Refresh tier: T3 (60s).

**`fetch_data()`** calls `fetch_pnl_by_strategy(conn)`, which returns a list of `StrategyPnl` dataclasses.

**`update_view(data)`** renders a Rich `Table` (not DataTable — no interactivity needed) with columns: Strategy, Realized P&L, Unrealized P&L, Win/Loss, Sharpe.

- Realized and unrealized P&L cells colored by sign.
- Win/Loss rendered as "W:12 L:3" format.
- Sorted by total P&L (realized + unrealized) descending.

Empty state: "No strategy P&L data".

### PnlBySymbolWidget

Subclasses `RefreshableWidget`. Refresh tier: T3 (60s).

**`fetch_data()`** calls `fetch_pnl_by_symbol(conn)`, which returns a list of `SymbolPnl` dataclasses.

**`update_view(data)`** renders one horizontal bar per symbol using `horizontal_bar()` from `charts.py`:

```
AAPL  ████████████████░░░░  +$1,234 (+3.2%)
NVDA  ██████████░░░░░░░░░░   +$567 (+1.8%)
TSLA  ███░░░░░░░░░░░░░░░░░    -$234 (-2.1%)
```

- Bar color: green for positive, red for negative P&L.
- `max_value` for scaling: the absolute value of the largest P&L across all symbols.

Empty state: "No symbol P&L data".

### DailyHeatmapWidget

Subclasses `RefreshableWidget`. Refresh tier: T4 (120s).

**`fetch_data()`** calls `fetch_equity_curve(conn)` (reuses the same query as `EquityCurveWidget` — last 30 rows of `daily_equity`). Extracts daily P&L values and dates.

**`update_view(data)`** renders a Mon-Fri grid using `daily_heatmap(daily_values, dates)` from `charts.py`. Each cell represents one trading day, colored green (profit) or red (loss) with intensity proportional to magnitude.

Empty state: "No daily data for heatmap".

---

## Query Functions Used

All queries are defined in `src/quantstack/tui/queries/portfolio.py` (Section 2). The Portfolio tab uses these 7 query functions:

| Function | Returns | Refresh Tier |
|----------|---------|-------------|
| `fetch_equity_summary(conn)` | `EquitySummary` | T2 (15s) |
| `fetch_positions(conn)` | `list[Position]` | T2 (15s) |
| `fetch_closed_trades(conn)` | `list[ClosedTrade]` | T3 (60s) |
| `fetch_equity_curve(conn)` | `list[EquityPoint]` | T3 (60s) |
| `fetch_benchmark(conn)` | `list[BenchmarkPoint]` | T3 (60s) |
| `fetch_pnl_by_strategy(conn)` | `list[StrategyPnl]` | T3 (60s) |
| `fetch_pnl_by_symbol(conn)` | `list[SymbolPnl]` | T3 (60s) |

Each function accepts a `PgConnection` (from `pg_conn()` context manager), uses `PgConnection.execute()` / `.fetchall()`, and returns a sensible default (empty list, zeroed dataclass) on any exception.

---

## Chart Functions Used

From `src/quantstack/tui/charts.py` (Section 3):

- **`sparkline(data, width, color)`** — used by `EquityCurveWidget` for portfolio and benchmark sparklines
- **`equity_curve(values, width, height)`** — used by `EquityCurveWidget` for the multi-line ASCII chart
- **`horizontal_bar(value, max_value, width, color)`** — used by `PnlBySymbolWidget` for per-symbol bars
- **`daily_heatmap(daily_values, dates)`** — used by `DailyHeatmapWidget` for the Mon-Fri grid

---

## Refresh Tier Assignments

Register these widgets with the `TieredRefreshScheduler` (from Section 1's `src/quantstack/tui/refresh.py`):

- **T2 (15s):** `EquitySummaryWidget`, `PositionsTableWidget`
- **T3 (60s):** `EquityCurveWidget`, `ClosedTradesWidget`, `PnlByStrategyWidget`, `PnlBySymbolWidget`
- **T4 (120s):** `DailyHeatmapWidget`

These widgets only refresh when the Portfolio tab is active (the scheduler's tab-visibility filtering handles this). On tab switch to Portfolio, all widgets fire an immediate refresh.

---

## Tab Registration

In `src/quantstack/tui/app.py`, the Portfolio tab is registered as the second `TabPane` (key "2") inside `TabbedContent.compose()`:

```python
with TabPane("Portfolio", id="portfolio"):
    yield ScrollableContainer(
        EquitySummaryWidget(),
        EquityCurveWidget(),
        PositionsTableWidget(),
        ClosedTradesWidget(),
        PnlByStrategyWidget(),
        PnlBySymbolWidget(),
        DailyHeatmapWidget(),
    )
```

---

## Interaction with Drill-Down Modals (Section 11)

Two widgets in this tab support drill-down:

1. **PositionsTableWidget** — Enter on a row opens `PositionDetailModal` showing entry date, entry price, current price, P&L, strategy name, regime at entry, stop/target levels.
2. **ClosedTradesWidget** — Enter on a row opens `TradeDetailModal` showing side, entry/exit prices, P&L, holding days, strategy, exit reason, decision reasoning, trade reflection.

The implementation pattern: each widget posts a custom `Message` subclass (e.g., `PositionSelected`, `TradeSelected`) carrying the row's identifier. The `QuantStackApp` handles these messages and pushes the corresponding `ModalScreen`.

---

## CSS Styling

Add to `src/quantstack/tui/dashboard.tcss`:

- `PositionsTableWidget` and `ClosedTradesWidget` DataTables should have a max height to avoid dominating the scroll area (e.g., `max-height: 12` rows visible).
- `EquityCurveWidget` fixed at `height: 8` (5 chart rows + labels + sparklines).
- `PnlBySymbolWidget` and `DailyHeatmapWidget` sized to content.
- All widgets within the Portfolio tab get a small vertical margin for visual separation.

---

## Error Handling

Every widget's `fetch_data()` is wrapped by the `RefreshableWidget` base class error handling (Section 1). If a query fails:

- The widget retains its last successfully rendered state (no flicker).
- If no data has ever been fetched, the widget shows its empty-state message.
- Errors are logged with the widget name and query function name (not raw SQL).

This matches the v1 graceful degradation pattern where individual widget failures do not crash the dashboard.
