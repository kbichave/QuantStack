# Section 7: Data & Signals Tab

## Overview

This section builds the Data & Signals tab, which contains three widgets: a market calendar showing upcoming events, a data health matrix showing freshness of each data type per symbol, and a signal engine dashboard showing top signals with collector health. The tab lives inside the `DataSignalsTab` pane of the `TabbedContent` created in Section 1.

## Dependencies

- **Section 01 (Scaffolding):** `RefreshableWidget` base class, `TieredRefreshScheduler`, `QuantStackApp` with `TabbedContent`, `dashboard.tcss`
- **Section 02 (Query Layer):** Query functions from `queries/data_health.py`, `queries/signals.py`, and `queries/calendar.py`
- **Section 03 (Charts):** `horizontal_bar()` renderer for coverage bars
- **Section 12 (DB Migrations):** `market_holidays` table must exist before `MarketCalendarWidget` can query it

## Widget Hierarchy

```
DataSignalsTab (ScrollableContainer)
├── MarketCalendarWidget     # Upcoming events: holidays, earnings, FOMC, macro
├── DataHealthMatrixWidget   # Symbol x data-type matrix + coverage bars
└── SignalEngineWidget       # Top signals + collector health + expandable briefs
```

All three widgets subclass `RefreshableWidget` from `src/quantstack/tui/base.py`. Each implements `fetch_data()` (runs in a background thread via `@work(thread=True)`) and `update_view()` (runs on the main thread to mutate widget state).

## File Locations

| File | Purpose |
|------|---------|
| `src/quantstack/tui/widgets/data_signals.py` | All 3 widgets: `MarketCalendarWidget`, `DataHealthMatrixWidget`, `SignalEngineWidget` |
| `src/quantstack/tui/queries/data_health.py` | 8 query functions (7 freshness + 1 collector health) |
| `src/quantstack/tui/queries/signals.py` | `fetch_active_signals()`, `fetch_signal_brief()` |
| `src/quantstack/tui/queries/calendar.py` | `fetch_earnings_calendar()`, `fetch_market_holidays()`, `fetch_macro_events()` |
| `tests/unit/test_tui/test_data_signals.py` | Unit tests for all 3 widgets |

---

## Tests (Write First)

File: `tests/unit/test_tui/test_data_signals.py`

```python
# tests/unit/test_tui/test_data_signals.py

# --- DataHealthMatrixWidget ---
# Test: DataHealthMatrixWidget renders symbol x data-type table
# Test: cells show checkmark for fresh, cross for stale
# Test: staleness thresholds: OHLCV=2d, News=24h, Fundamentals=90d, Options=1d, Insider=30d, Macro=7d, Sentiment=24h
# Test: coverage bars show correct percentages per data type
# Test: handles empty freshness data (all symbols missing)
# Test: handles partial data (some data types present, others missing for a symbol)

# --- MarketCalendarWidget ---
# Test: MarketCalendarWidget renders events chronologically
# Test: holiday entries colored red (closed) or yellow (early close)
# Test: earnings entries colored cyan
# Test: macro entries colored white
# Test: handles empty calendar gracefully (renders "No upcoming events" or equivalent)
# Test: events from all 3 sources are merged and sorted by date

# --- SignalEngineWidget ---
# Test: SignalEngineWidget renders signals sorted by confidence DESC
# Test: per-factor columns show ML, Sentiment, Technical, Options, Macro
# Test: collector health shows checkmark/cross per collector
# Test: Enter on signal row triggers drill-down (posts message or pushes SignalDetailModal)
# Test: handles empty signal_state table (renders "No active signals")
# Test: handles malformed brief_json gracefully (skip row, log warning)
```

Tests should patch `pg_conn()` at the query-function level (not the pool factory) to return a mock `PgConnection`. Every widget test should verify that empty/None data does not crash the widget.

---

## DataHealthMatrixWidget

### Purpose

Displays a symbol-by-data-type matrix showing how fresh each data source is for each tracked symbol. This is the most query-intensive widget in the dashboard: 7 freshness queries (one per data type) plus 1 collector health query.

### Queries Used

From `queries/data_health.py`:

- `fetch_ohlcv_freshness(conn) -> dict[str, datetime]` — maps symbol to last OHLCV timestamp
- `fetch_news_freshness(conn) -> dict[str, datetime]` — maps symbol to last news timestamp
- `fetch_sentiment_freshness(conn) -> dict[str, datetime]`
- `fetch_fundamentals_freshness(conn) -> dict[str, datetime]`
- `fetch_options_freshness(conn) -> dict[str, datetime]`
- `fetch_insider_freshness(conn) -> dict[str, datetime]`
- `fetch_macro_freshness(conn) -> dict[str, datetime]`
- `fetch_collector_health(conn) -> dict[str, bool]` — maps collector name to healthy/unhealthy

All freshness queries return `dict[str, datetime]` mapping symbol to the most recent data timestamp for that type. On error, each returns an empty dict.

### Staleness Thresholds

These are the thresholds for marking data as stale (cross) vs fresh (checkmark):

| Data Type | Stale After |
|-----------|-------------|
| OHLCV | 2 trading days |
| News | 24 hours |
| Sentiment | 24 hours |
| Fundamentals | 90 days |
| Options | 1 trading day |
| Insider | 30 days |
| Macro | 7 days |

Define these as a module-level constant dict, e.g. `STALENESS_THRESHOLDS: dict[str, timedelta]`.

### Rendering

Renders a Rich `Table`:
- **Rows:** one per symbol in the trading universe
- **Columns:** Symbol, OHLCV, News, Sentiment, Fundamentals, Options, Insider, Macro
- **Cell content:** age string (e.g., "2h", "3d") plus a colored checkmark or cross. Green checkmark if within threshold, red cross if stale or missing.

Below the matrix, render coverage summary bars (one per data type) using the `horizontal_bar()` function from `charts.py`. Coverage = (number of symbols with fresh data) / (total symbols).

### Refresh Tier

T3 (60-second interval). Data freshness changes slowly, no need for rapid refresh.

---

## MarketCalendarWidget

### Purpose

Shows upcoming market events in chronological order: holidays, earnings, and macro events.

### Queries Used

From `queries/calendar.py`:

- `fetch_market_holidays(conn) -> list[MarketHoliday]` — from the `market_holidays` table (created in Section 12). Returns holidays for the next 90 days.
- `fetch_earnings_calendar(conn) -> list[EarningsEvent]` — from `earnings_calendar` table, next 90 days, filtered to universe symbols.
- `fetch_macro_events(conn) -> list[MacroEvent]` — from `macro_indicators` table (FOMC dates, CPI/PPI releases), next 90 days.

### Data Types

```python
@dataclass
class MarketHoliday:
    date: date
    name: str
    market_status: str  # 'closed' | 'early_close'
    close_time: time | None

@dataclass
class EarningsEvent:
    date: date
    symbol: str
    timing: str  # 'BMO' | 'AMC' | 'unknown'

@dataclass
class MacroEvent:
    date: date
    name: str
    prior_value: float | None
```

### Rendering

Merge all three event lists, sort by date, and render as a Rich `Table` with columns: Date, Type, Description.

Color coding:
- **Red:** market closed (holiday with `market_status='closed'`)
- **Yellow:** early close (holiday with `market_status='early_close'`)
- **Cyan:** earnings event
- **White:** macro event (FOMC, CPI, PPI)

If no upcoming events exist, render a single line: "No upcoming events in the next 90 days."

### Refresh Tier

T4 (120-second interval). Calendar data changes rarely.

---

## SignalEngineWidget

### Purpose

Displays the signal engine's output: top signals ranked by confidence, per-factor breakdowns, and collector health status.

### Queries Used

From `queries/signals.py`:

- `fetch_active_signals(conn) -> list[Signal]` — from `signal_state` table, ordered by confidence DESC.
- `fetch_signal_brief(conn, symbol: str) -> dict | None` — parses `brief_json` JSONB from `signal_state` for a specific symbol. Returns the parsed dict or None.

### Data Types

```python
@dataclass
class Signal:
    symbol: str
    action: str        # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    ml_score: float | None
    sentiment_score: float | None
    technical_score: float | None
    options_score: float | None
    macro_score: float | None
    updated_at: datetime
```

### Rendering

**Signal table:** Rich `Table` with columns: Symbol, Action, Confidence, ML, Sentiment, Technical, Options, Macro. Sorted by confidence DESC. Action column colored: green for BUY, red for SELL, yellow for HOLD. Confidence rendered as percentage.

**Collector health section:** Below the signal table. Iterates `brief_json` across all symbols, counts `collector_failures` entries. Renders one line per collector: name + checkmark (healthy) or cross (has failures). This gives visibility into which data pipelines are broken.

**Drill-down:** When the user presses Enter on a signal row, the widget should push a `SignalDetailModal` (defined in Section 11) showing the full curated signal brief with top 5 contributing factors, risk flags, and signal expiry.

### Refresh Tier

T2 (15-second interval). Signals update after each research cycle.

---

## Tab Composition

The `DataSignalsTab` is a `ScrollableContainer` that composes the three widgets vertically:

```python
class DataSignalsTab(ScrollableContainer):
    """Data & Signals tab: calendar, data health matrix, signal engine."""

    def compose(self) -> ComposeResult:
        yield MarketCalendarWidget()
        yield DataHealthMatrixWidget()
        yield SignalEngineWidget()
```

This tab is mounted into TabPane index 4 (the 4th tab, 0-indexed) in the `QuantStackApp.compose()` method from Section 1. The tab is activated by pressing key `4` (per the keybinding scheme).

---

## CSS Styling Notes

Add rules to `src/quantstack/tui/dashboard.tcss` for:
- `DataHealthMatrixWidget` — table should not exceed terminal width; use `overflow-x: auto` if needed
- `MarketCalendarWidget` — compact height, max ~8 lines before scrolling
- `SignalEngineWidget` — signal table gets remaining vertical space
- Coverage bars section below the matrix should have a top margin separator

---

## Implementation Checklist

1. Write tests in `tests/unit/test_tui/test_data_signals.py`
2. Implement `MarketCalendarWidget` in `src/quantstack/tui/widgets/data_signals.py`
3. Implement `DataHealthMatrixWidget` in the same file
4. Implement `SignalEngineWidget` in the same file
5. Implement `DataSignalsTab` container in the same file
6. Add CSS rules to `src/quantstack/tui/dashboard.tcss`
7. Wire `DataSignalsTab` into the appropriate `TabPane` in `app.py`
8. Verify all tests pass with mocked queries
