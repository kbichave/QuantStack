# Section 06: Strategies Tab

## Overview

Build the Strategies tab with a Kanban-style pipeline visualization, promotion gate criteria display, and cursor-navigable strategy detail. This tab answers: "Where are my strategies in the pipeline, and which ones are ready for promotion?"

**Dependencies:** Section 01 (scaffolding, `RefreshableWidget`, app shell), Section 02 (query layer — `queries/strategies.py`), Section 03 (charts — `progress_bar()`).

**Blocks:** Section 11 (drill-down modals — `StrategyDetailModal` triggered from this tab), Section 13 (integration).

**Files to create:**
- `src/quantstack/tui/widgets/strategies.py` — `PipelineKanbanWidget`, `PromotionGatesWidget`, `StrategyDetailWidget`
- `tests/unit/test_tui/test_strategies.py` — all test stubs below

**Files to modify:**
- `src/quantstack/tui/app.py` — mount `StrategiesTab` inside TabPane #3
- `src/quantstack/tui/dashboard.tcss` — layout rules for kanban columns and strategy cards

---

## Tests (Write First)

```python
# tests/unit/test_tui/test_strategies.py

"""Tests for Strategies tab widgets.

All tests patch pg_conn() to return mock data so no live DB is needed.
Widget rendering tests instantiate widgets directly and call update_view()
with known data, then assert on the rendered output.
"""

# --- PipelineKanbanWidget ---

# Test: PipelineKanbanWidget renders 5 columns (Draft, Backtested, Forward Testing, Live, Retired)
#   - Provide a list of StrategyCard objects spanning all 5 statuses
#   - Assert the rendered output contains all 5 column headers

# Test: strategy cards show name, symbol, Sharpe, MaxDD
#   - Provide a StrategyCard with known values
#   - Assert rendered text includes the name, symbol, formatted Sharpe, formatted MaxDD

# Test: forward testing cards show progress bar (days / required)
#   - Provide a StrategyCard with status='forward_testing', fwd_days=12, fwd_required_days=30
#   - Assert rendered output includes a progress bar (from charts.progress_bar) and "12/30" text

# Test: color coding: green for meeting gates, yellow for borderline, red for failing
#   - Provide cards with Sharpe > 1.0 (green), Sharpe 0.5-1.0 (yellow), Sharpe < 0.5 (red)
#   - Assert Rich style includes correct color for each card

# Test: selected card index updates on navigation
#   - Simulate key_down / key_up events on the widget
#   - Assert selected_index changes accordingly and stays within bounds

# Test: Enter on selected card triggers drill-down
#   - Set selected_index to a valid card, simulate Enter
#   - Assert a message/event is posted to open StrategyDetailModal

# Test: handles empty strategy list (all columns empty)
#   - Provide an empty list of StrategyCard
#   - Assert widget renders without error, shows placeholder text like "No strategies"

# --- PromotionGatesWidget ---

# Test: PromotionGatesWidget renders gate criteria text
#   - Assert rendered output contains the gate criteria for each transition
#     (draft->backtested, backtested->forward_testing, forward_testing->live)
```

---

## Data Model

The query layer (Section 02) provides `StrategyCard` from `queries/strategies.py`. For reference, the dataclass definition:

```python
@dataclass
class StrategyCard:
    strategy_id: str
    name: str
    status: str  # 'draft' | 'backtested' | 'forward_testing' | 'live' | 'retired'
    symbol: str
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    fwd_trades: int
    fwd_pnl: float
    fwd_days: int
    fwd_required_days: int
```

This comes from a single query joining `strategies` with aggregated `closed_trades`:

- `strategies` table columns used: `strategy_id`, `name`, `status`, `backtest_summary` (JSONB containing `sharpe_ratio`, `max_drawdown_pct`, `win_rate`), `updated_at`, `holding_period_days`.
- `closed_trades` table: joined on `strategy_id`, filtered to trades opened after the strategy's `updated_at` (i.e., forward testing period). Aggregates: `COUNT(*)` as `fwd_trades`, `SUM(realized_pnl)` as `fwd_pnl`, `COUNT(DISTINCT DATE(closed_at))` for win rate calc.
- `fwd_days` = `EXTRACT(DAY FROM NOW() - strategies.updated_at)` (days since strategy entered current status).
- `fwd_required_days` = derived from `holding_period_days * 6` (minimum 30 days observation window, configurable).

Results are ordered by status priority: `live` > `forward_testing` > `backtested` > `draft` > `retired`.

---

## Widget Hierarchy

```
StrategiesTab (ScrollableContainer)
├── PipelineKanbanWidget     # 5 columns: Draft, Backtested, Forward Testing, Live, Retired
├── PromotionGatesWidget     # Static text showing gate criteria
└── StrategyDetailWidget     # Expandable detail for currently selected strategy
```

All three are `RefreshableWidget` subclasses (from Section 01 `base.py`), refreshed on **T3 (60s)** since strategy pipeline data is slow-changing.

---

## PipelineKanbanWidget

The primary widget. Renders 5 side-by-side columns using Rich `Columns` (or Textual `Horizontal` container). Each column is a Rich `Panel` titled with the status name and a count badge.

### Column Layout

Each column panel contains vertically stacked strategy cards. A card is a bordered Rich `Panel` (or styled text block) showing:

```
 AAPL inv_momentum
 Sharpe: 1.24  MaxDD: -8.2%
 WR: 62%
```

For `forward_testing` cards, append a progress bar row:

```
 AAPL inv_momentum
 Sharpe: 1.24  MaxDD: -8.2%
 WR: 62%  FT: 18/30d
 ████████████░░░░░░ 60%
```

The progress bar is rendered using `charts.progress_bar(fwd_days, fwd_required_days, width=18)` from Section 03.

### Color Coding

Card border/title color is determined by promotion gate health:

| Condition | Color | Meaning |
|-----------|-------|---------|
| Sharpe >= 1.0 AND MaxDD >= -15% AND win_rate >= 55% | green | Meeting all promotion gates |
| Any metric within 20% of gate threshold | yellow | Borderline — close to promotion or demotion |
| Any metric failing gate threshold by >20% | red | Failing gates |

For `None` metric values (e.g., draft strategies with no backtest), use dim/grey styling — no color judgment.

### Cursor Navigation

The widget maintains a `selected_index: int` reactive attribute tracking which card is currently highlighted (across all columns, flattened left-to-right, top-to-bottom). Navigation:

- **Up/Down arrows** or **j/k**: move selection within a column
- **Left/Right arrows** or **h/l**: move selection between columns (jump to same row index or last card in that column)
- **Enter**: post a message to open `StrategyDetailModal` (Section 11) for the selected strategy
- **Escape**: clear selection

The selected card gets a highlighted border (bright white or inverse style) to distinguish it from unselected cards.

### fetch_data() Implementation

```python
def fetch_data(self) -> list[StrategyCard]:
    """Query strategies pipeline. Runs in background thread."""
    # Uses pg_conn() context manager
    # Calls queries.strategies.fetch_strategy_pipeline(conn)
    # Returns list[StrategyCard]
```

### update_view() Implementation

```python
def update_view(self, data: list[StrategyCard]) -> None:
    """Bucket cards by status, render 5 columns. Runs on main thread."""
    # Group cards into 5 buckets by status
    # Build Rich Columns with one Panel per bucket
    # Apply color coding per card
    # Render progress bars for forward_testing cards
    # Preserve selected_index if still valid
```

---

## PromotionGatesWidget

A simple `Static` widget (not refreshable — content is hardcoded gate criteria). Renders a compact reference of what each promotion requires:

```
Promotion Gates:
  draft -> backtested:     Backtest completed, Sharpe > 0.5, MaxDD > -25%
  backtested -> forward:   Sharpe > 0.8, MaxDD > -20%, Win Rate > 50%
  forward -> live:         30+ days FT, 10+ trades, Sharpe > 1.0, MaxDD > -15%, WR > 55%
  Demotion:                3 consecutive losing weeks OR MaxDD > -20% in live
```

This is static text styled with dim colors. It provides context for interpreting the Kanban card colors. The gate values should be defined as constants at the top of the widget module so they stay in sync with the color-coding logic.

---

## StrategyDetailWidget

An expandable detail panel that shows more information about the currently selected strategy from the Kanban. It listens for selection changes from `PipelineKanbanWidget` and renders:

- Strategy name, ID, status, symbol
- Time horizon, instrument type, regime affinity
- Backtest summary: Sharpe, MaxDD, win rate, total trades, profit factor (from `backtest_summary` JSONB)
- Forward test summary: days elapsed, trades taken, cumulative P&L, win rate (computed from `closed_trades` aggregation)
- Entry/exit rules summary (from `entry_rules` / `exit_rules` JSONB — first 2-3 rules, truncated)

This widget uses the same `StrategyCard` data already fetched by the Kanban widget (no additional query). For the extended fields (entry/exit rules, regime affinity), the `fetch_data()` in the Kanban widget should include them in the `StrategyCard` or the detail widget can issue a targeted query for the selected strategy only.

Design decision: start with what `StrategyCard` already provides. If the extended fields are needed, add an optional `fetch_strategy_detail(conn, strategy_id) -> StrategyDetail` query to `queries/strategies.py` that fetches the full row. This query fires only on selection change (not on timer), keeping DB load minimal.

---

## CSS (dashboard.tcss additions)

```css
/* Strategies tab layout */
StrategiesTab {
    layout: vertical;
}

PipelineKanbanWidget {
    height: auto;
    max-height: 80%;
}

PromotionGatesWidget {
    height: auto;
    max-height: 4;
    color: $text-muted;
}

StrategyDetailWidget {
    height: auto;
    max-height: 20%;
    border: solid $accent;
}
```

The exact CSS selectors and values should be adjusted during implementation to fit the terminal dimensions. The Kanban columns should divide available width equally (each ~20% of terminal width).

---

## Refresh Tier

All widgets on this tab use **T3 (60s)** refresh. Strategy pipeline data changes infrequently (promotions happen at most daily). The 60s interval provides timely updates without unnecessary DB load.

On tab activation (user switches to Strategies tab), the scheduler fires an immediate refresh per the `TieredRefreshScheduler` behavior defined in Section 01.

---

## Integration Points

- **Section 02 (Query Layer):** `queries/strategies.py` must provide `fetch_strategy_pipeline(conn) -> list[StrategyCard]`. This section consumes that function.
- **Section 03 (Charts):** `charts.progress_bar()` is used for forward testing progress display.
- **Section 11 (Drill-Down Modals):** `StrategyDetailModal` is opened when user presses Enter on a selected card. This section posts the message; Section 11 handles the modal rendering. The message should carry the `strategy_id` so the modal can query full details.
- **Section 04 (Overview Tab):** `StrategyCountsCompact` on the Overview tab shows aggregated counts from the same query. The query is shared but the rendering is independent.
