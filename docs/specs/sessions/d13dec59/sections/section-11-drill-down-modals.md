# Section 11: Drill-Down Modals

## Overview

Implement `ModalScreen` overlays that provide detailed views when users press Enter on rows/cards in the tab widgets. Each modal is a self-contained detail view with curated content, scrollable body, and ESC-to-dismiss behavior. This section provides the base `DetailModal` class and five specialized modal variants.

**Dependencies:** Section 01 (scaffolding — `QuantStackApp`, `screens/` package), Section 02 (query layer — several query functions used for detail fetches), Sections 04-10 (the tab widgets that trigger modals via Enter key).

**Blocks:** Section 13 (integration testing — modal open/close is tested there).

**Files to create:**
- `src/quantstack/tui/screens/detail.py` — `DetailModal` base class, `PositionDetailModal`, `StrategyDetailModal`, `SignalDetailModal`, `TradeDetailModal`, `AgentEventModal`
- `tests/unit/test_tui/test_modals.py` — all test stubs below

**Files to modify:**
- `src/quantstack/tui/app.py` — register modal screen classes so widgets can push them via `app.push_screen()`
- `src/quantstack/tui/dashboard.tcss` — overlay styling, modal container dimensions, scrollable content area
- Tab widget files (from Sections 05-10) — wire Enter key handlers to push the appropriate modal. Each widget that supports drill-down should post a message or call `app.push_screen(ModalClass(id=...))`.

---

## Tests (Write First)

```python
# tests/unit/test_tui/test_modals.py

"""Tests for drill-down modal screens.

All tests patch pg_conn() to return mock data. Modal tests instantiate the
modal with known data/IDs and assert on rendered content. Textual pilot
framework is used for interaction tests (ESC dismiss, scroll).
"""

# --- DetailModal (base) ---

# Test: DetailModal renders with semi-transparent overlay
#   - Instantiate DetailModal with a title and body content
#   - Assert the screen has an overlay layer (CSS class or background alpha)

# Test: ESC dismisses the modal
#   - Mount a DetailModal via app.push_screen()
#   - Simulate pilot.press("escape")
#   - Assert the modal is no longer in the screen stack

# Test: scrollable content area works for long content
#   - Provide body content exceeding terminal height (e.g., 100 lines)
#   - Assert the content area is a ScrollableContainer (or VerticalScroll)
#   - Simulate scroll down, assert viewport changes

# --- PositionDetailModal ---

# Test: PositionDetailModal shows entry price, current price, P&L, strategy, stop/target
#   - Provide a position_id referencing mock Position data
#   - Assert rendered text includes entry_price, current_price, pnl ($, %),
#     strategy_name, stop_level, target_level

# Test: PositionDetailModal handles missing stop/target gracefully
#   - Provide mock data where stop_level and target_level are None
#   - Assert widget renders without error, shows "—" or "Not set" for those fields

# --- StrategyDetailModal ---

# Test: StrategyDetailModal shows backtest + forward test metrics
#   - Provide a strategy_id referencing mock StrategyCard + extended detail
#   - Assert rendered text includes Sharpe, MaxDD, win_rate (backtest section)
#   - Assert rendered text includes fwd_days, fwd_trades, fwd_pnl (forward test section)

# Test: StrategyDetailModal shows entry/exit rules
#   - Provide mock strategy with entry_rules and exit_rules JSONB
#   - Assert the rendered modal includes summarized rule text

# Test: StrategyDetailModal handles draft strategy (no backtest data)
#   - Provide mock data where all metric fields are None
#   - Assert renders without error, shows "No backtest data" or equivalent

# --- SignalDetailModal ---

# Test: SignalDetailModal shows top 5 contributing factors and risk flags
#   - Provide a symbol referencing mock signal_state with brief_json
#   - Assert rendered text includes action, confidence, and 5 factor rows
#     (ML, Sentiment, Technical, Options, Macro) with values

# Test: SignalDetailModal shows signal expiry and upcoming events
#   - Provide mock brief_json with risk_flags (e.g., earnings in 3 days)
#   - Assert rendered text includes the risk flag and expiry info

# Test: SignalDetailModal handles partial brief_json (missing collectors)
#   - Provide brief_json with only 2 of 5 factors populated
#   - Assert renders without error, missing factors show "N/A" or are omitted

# --- TradeDetailModal ---

# Test: TradeDetailModal shows decision reasoning + reflection
#   - Provide trade data with entry/exit prices, P&L, strategy, exit_reason
#   - Assert rendered text includes all trade fields
#   - Provide mock decision_events row with reasoning text
#   - Assert reasoning is displayed
#   - Provide mock trade_reflections row
#   - Assert reflection/lesson is displayed

# Test: TradeDetailModal handles trade with no reflection
#   - Provide mock data where trade_reflections query returns None
#   - Assert renders without error, reflection section shows "No reflection recorded"

# --- AgentEventModal ---

# Test: AgentEventModal shows full event content
#   - Provide an agent event with agent_name, timestamp, content, tool_name,
#     graph_name, node_name
#   - Assert all fields appear in the rendered output

# --- Cross-cutting ---

# Test: all modals handle missing/null fields gracefully
#   - For each modal class, provide minimal data (most fields None)
#   - Assert no exceptions raised, widget renders placeholder text
```

---

## Base Class: DetailModal

`DetailModal` is a `ModalScreen` subclass in `src/quantstack/tui/screens/detail.py`. It provides the shared chrome and behavior for all drill-down views.

### Structure

```python
class DetailModal(ModalScreen):
    """Base modal screen for drill-down detail views.

    Provides: semi-transparent overlay, bordered container with title,
    scrollable content area, ESC to dismiss.

    Subclasses override compose_content() to yield their specific widgets.
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, title: str = "Detail") -> None:
        """Accept a title for the modal header."""
        ...

    def compose(self) -> ComposeResult:
        """Yield the modal layout: overlay + bordered container + scrollable body.

        The container is a Vertical with:
        - Static title bar
        - VerticalScroll containing results of compose_content()
        - Static footer hint ("ESC to close")
        """
        ...

    def compose_content(self) -> ComposeResult:
        """Override in subclasses to yield detail content widgets."""
        ...
```

### CSS Styling

The modal overlay uses Textual's built-in `ModalScreen` overlay behavior. Additional CSS in `dashboard.tcss`:

```css
DetailModal {
    align: center middle;
}

DetailModal > #modal-container {
    width: 80%;
    max-width: 100;
    height: 80%;
    max-height: 40;
    border: thick $accent;
    background: $surface;
}

DetailModal > #modal-container > #modal-title {
    dock: top;
    height: 1;
    background: $accent;
    color: $text;
    text-align: center;
}

DetailModal > #modal-container > #modal-body {
    overflow-y: auto;
}

DetailModal > #modal-container > #modal-footer {
    dock: bottom;
    height: 1;
    color: $text-muted;
    text-align: center;
}
```

The exact dimensions should be tuned during implementation. The modal should be large enough to show useful detail but leave visible tab content around the edges to maintain spatial context.

---

## Modal Variants

### PositionDetailModal

**Triggered from:** `PositionsTableWidget` (Section 05) when user presses Enter on a position row.

**Input:** `symbol: str` (used to query position details).

**Content layout:**

```
                    AAPL — Position Detail
    ─────────────────────────────────────────────
    Side:            LONG
    Entry Date:      2026-03-15
    Entry Price:     $172.50
    Current Price:   $178.30
    Quantity:        50

    P&L:             +$290.00 (+3.36%)
    Strategy:        aapl_inv_momentum
    Regime at Entry: trending_up (82%)

    Stop Level:      $165.00 (-4.35%)
    Target Level:    $190.00 (+10.14%)
    Days Held:       19

                      ESC to close
```

**Data source:** The position data is already available in the `Position` dataclass from `queries/portfolio.py` (Section 02). The modal receives either the full `Position` object or a `symbol` identifier and queries `fetch_positions()` to find the matching row. Additional fields (regime at entry, stop/target) come from joining `positions` with `strategies` and `decision_events`.

Design decision: pass the `Position` dataclass directly from the widget rather than re-querying. This avoids an extra DB call and the data is already fresh (refreshed on T2). For extended fields (regime at entry, stop/target), add an optional `fetch_position_detail(conn, symbol) -> PositionDetail` query that fires once on modal open if those fields are needed beyond what `Position` provides.

### StrategyDetailModal

**Triggered from:** `PipelineKanbanWidget` (Section 06) when user presses Enter on a strategy card.

**Input:** `strategy_id: str`.

**Content layout:**

```
              AAPL inv_momentum — Strategy Detail
    ─────────────────────────────────────────────
    Status:          forward_testing
    Symbol:          AAPL
    Type:            swing_momentum
    Horizon:         5-15 days
    Regime Affinity: trending_up

    ── Backtest Results ──
    Sharpe:          1.24
    Max Drawdown:    -8.2%
    Win Rate:        62%
    Total Trades:    47
    Profit Factor:   1.85

    ── Forward Test ──
    Days Elapsed:    18 / 30
    Trades Taken:    6
    Cumulative P&L:  +$340.00
    Win Rate:        66.7%

    ── Entry Rules (summary) ──
    • RSI(14) < 35 AND price > SMA(50)
    • Volume > 1.5x 20d average

    ── Exit Rules (summary) ──
    • Trailing stop at 2x ATR
    • Target: 3:1 reward/risk

                      ESC to close
```

**Data source:** Uses `StrategyCard` data already fetched by the Kanban widget for the core metrics. For extended fields (entry/exit rules, regime affinity, profit factor), uses `fetch_strategy_detail(conn, strategy_id) -> StrategyDetail` from `queries/strategies.py`. This detail query fetches the full `strategies` row including `entry_rules` and `exit_rules` JSONB columns, and is issued once on modal open (not on timer).

The `StrategyDetail` dataclass extends `StrategyCard` with:
```python
@dataclass
class StrategyDetail:
    """Extended strategy info for modal display. Superset of StrategyCard."""
    # All StrategyCard fields, plus:
    strategy_type: str | None
    horizon: str | None
    regime_affinity: str | None
    entry_rules: list[str]  # parsed from JSONB
    exit_rules: list[str]   # parsed from JSONB
    profit_factor: float | None
    total_trades: int
```

### SignalDetailModal

**Triggered from:** `SignalEngineWidget` (Section 07) when user presses Enter on a signal row.

**Input:** `symbol: str`.

**Content layout:**

```
                  NVDA — Signal Detail
    ─────────────────────────────────────────────
    Action:          BUY
    Confidence:      87%
    Generated:       2026-04-03 14:30

    ── Contributing Factors ──
    ML Prediction:        0.82  (bullish)
    Sentiment Score:      0.71  (positive)
    Technical Score:      0.90  (strong uptrend)
    Options Flow:         0.65  (moderately bullish)
    Macro Alignment:      0.78  (favorable)

    ── Risk Flags ──
    ⚠ Earnings in 5 days (2026-04-08)
    ⚠ VIX elevated (22.5)

    Signal Expiry:   2026-04-04 09:30

                      ESC to close
```

**Data source:** `fetch_signal_brief(conn, symbol) -> SignalBrief` from `queries/signals.py` (Section 02). This query reads `signal_state` and parses the `brief_json` JSONB column. The JSONB structure contains per-collector scores, risk flags, and metadata.

Graceful degradation: if a collector failed (indicated by `collector_failures` in the JSONB), that factor row shows "N/A" or "Collector error" instead of a numeric score. If `brief_json` is NULL or unparseable, the modal shows "No signal brief available".

### TradeDetailModal

**Triggered from:** `ClosedTradesWidget` (Section 05) when user presses Enter on a trade row. Also from `DecisionsWidget` (Section 09 / Overview).

**Input:** `trade_id: str` (or `symbol + closed_at` composite key).

**Content layout:**

```
              AAPL — Trade Detail (2026-03-28)
    ─────────────────────────────────────────────
    Side:            LONG
    Entry Price:     $168.20   (2026-03-10)
    Exit Price:      $175.80   (2026-03-28)
    Quantity:        50
    P&L:             +$380.00  (+4.52%)
    Holding Days:    18
    Strategy:        aapl_inv_momentum
    Exit Reason:     target_reached

    ── Decision Reasoning ──
    Entry: "RSI crossed above 30 with volume confirmation.
    Regime trending_up with 82% confidence. Signal brief
    showed 4/5 collectors bullish."

    ── Trade Reflection ──
    "Good entry timing. Held through a 2-day pullback that
    tested conviction. Target hit cleanly. Consider tighter
    trailing stop for similar setups — left ~1.5% on table."

                      ESC to close
```

**Data source:** Trade data comes from the `ClosedTrade` dataclass (already fetched by `ClosedTradesWidget`). Pass it directly to avoid re-querying.

For decision reasoning: query `decision_events` WHERE `symbol = X AND event_type = 'trade_entry' AND created_at` near the trade entry date. This is a targeted query fired once on modal open.

For trade reflection: query `trade_reflections` WHERE `trade_id = X` (or match on symbol + close date). Returns the reflection text if one exists, or None.

Add to `queries/portfolio.py` (or a new `queries/decisions.py`):
```python
def fetch_trade_decision(conn, symbol: str, entry_date: date) -> str | None:
    """Fetch entry decision reasoning from decision_events."""
    ...

def fetch_trade_reflection(conn, trade_id: str) -> str | None:
    """Fetch post-trade reflection if recorded."""
    ...
```

### AgentEventModal

**Triggered from:** `GraphActivityWidget` (Section 08) when user presses Enter on an event row.

**Input:** An `AgentEvent` dataclass (passed directly from the widget).

**Content layout:**

```
            research_agent — Event Detail
    ─────────────────────────────────────────────
    Timestamp:   2026-04-03 16:48:37
    Graph:       research
    Node:        analyze_fundamentals
    Tool:        fetch_fundamentals_data

    ── Content ──
    Analyzing NVDA fundamentals for Q1 2026...
    Revenue growth: +38% YoY, beating consensus by 4.2%.
    Operating margin expanded 200bps to 62.1%.
    Forward PE at 35x vs sector median 28x.
    Initiating position sizing analysis.

                      ESC to close
```

**Data source:** The `AgentEvent` dataclass from `queries/agents.py` (Section 02) is passed directly — no additional query needed. The `content` field contains the full event text which may be multi-line and lengthy, hence the scrollable container.

---

## Triggering Modals from Widgets

Each tab widget that supports drill-down needs to wire its Enter key handler to push the correct modal. The pattern:

```python
# In any widget that supports drill-down (e.g., PositionsTableWidget)

def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
    """Handle Enter on a table row — open detail modal."""
    row_data = self._get_row_data(event.row_key)
    self.app.push_screen(PositionDetailModal(position=row_data))
```

For `PipelineKanbanWidget` which uses cursor navigation instead of `DataTable`:

```python
def on_key(self, event: Key) -> None:
    if event.key == "enter" and self._selected_card is not None:
        self.app.push_screen(
            StrategyDetailModal(strategy_id=self._selected_card.strategy_id)
        )
```

The modal classes accept their data/identifiers in `__init__()` and perform any additional queries in `on_mount()` (using `@work(thread=True)` for DB calls, then updating the content area on the main thread).

---

## Data Fetching in Modals

Modals that need additional data beyond what the triggering widget already has (e.g., `StrategyDetailModal` needing entry/exit rules, `TradeDetailModal` needing decision reasoning) follow this pattern:

```python
class StrategyDetailModal(DetailModal):
    """Modal showing full strategy detail."""

    def __init__(self, strategy_id: str) -> None:
        super().__init__(title="Strategy Detail")
        self._strategy_id = strategy_id
        self._detail: StrategyDetail | None = None

    def compose_content(self) -> ComposeResult:
        """Yield a loading placeholder; replaced after data fetch."""
        yield Static("Loading...", id="detail-content")

    def on_mount(self) -> None:
        """Kick off background data fetch."""
        self._fetch_detail()

    @work(thread=True)
    def _fetch_detail(self) -> None:
        """Fetch full strategy detail from DB."""
        # with pg_conn() as conn:
        #     self._detail = fetch_strategy_detail(conn, self._strategy_id)
        # self.app.call_from_thread(self._render_detail)

    def _render_detail(self) -> None:
        """Replace loading placeholder with rendered detail content."""
        # self.query_one("#detail-content").update(...)
        ...
```

This pattern ensures:
1. The modal appears immediately with a "Loading..." placeholder (no lag).
2. The DB query runs in a background thread (psycopg2 is blocking).
3. The UI update happens on the main thread via `call_from_thread()`.
4. If the query fails, the modal shows an error message instead of crashing.

---

## Refresh Tier

Modals do not participate in the `TieredRefreshScheduler`. They fetch data once on open and display it statically. The data shown in the modal is a point-in-time snapshot. If the user wants fresh data, they close and re-open the modal.

This is intentional: modals are transient detail views, not live dashboards. Adding refresh to modals would complicate the UX (content shifting while reading) and add unnecessary DB load.

---

## CSS (dashboard.tcss additions)

```css
/* Modal overlay and container */
DetailModal {
    align: center middle;
}

#modal-container {
    width: 80%;
    max-width: 100;
    height: 80%;
    max-height: 40;
    border: thick $accent;
    background: $surface;
    padding: 1 2;
}

#modal-title {
    dock: top;
    height: 1;
    width: 100%;
    background: $accent;
    color: $text;
    text-align: center;
    text-style: bold;
}

#modal-body {
    overflow-y: auto;
    padding: 1 2;
}

#modal-footer {
    dock: bottom;
    height: 1;
    width: 100%;
    color: $text-muted;
    text-align: center;
}
```

---

## Integration Points

- **Section 01 (Scaffolding):** The `screens/` package and `screens/__init__.py` are created in Section 01. This section populates `screens/detail.py` with the actual modal classes.
- **Section 02 (Query Layer):** `fetch_signal_brief()`, `fetch_strategy_detail()`, and the new `fetch_trade_decision()` / `fetch_trade_reflection()` queries are consumed here. If `fetch_strategy_detail()` and the trade decision/reflection queries don't exist yet, they should be added to `queries/strategies.py` and `queries/portfolio.py` respectively.
- **Sections 05-10 (Tab Widgets):** Each widget that supports Enter-to-drill-down pushes the appropriate modal. The modal classes must be importable from `screens.detail`. The triggering widgets are: `PositionsTableWidget` (05), `ClosedTradesWidget` (05), `PipelineKanbanWidget` (06), `SignalEngineWidget` (07), `GraphActivityWidget` (08), `DecisionsWidget` (09/04).
- **Section 13 (Integration):** The pilot test "Enter opens modal, Esc closes it" validates the full flow. This section provides the modal side of that interaction.
