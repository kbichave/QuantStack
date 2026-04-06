# Section 4: Overview Tab Widgets

## Purpose

Build the Overview tab — the first thing the user sees when the dashboard launches. It provides a single-screen glance (fits a 24-line terminal) that surfaces compact summaries from every other tab: services, risk, portfolio, trades, strategies, signals, data health, research, agent activity, and daily digest. No scrolling required for the core view.

This tab answers the owner's #1 question: "Is anything on fire?" within 2 seconds of opening the dashboard.

## Dependencies

- **Section 01 (Scaffolding):** `RefreshableWidget` base class in `src/quantstack/tui/base.py`, `TieredRefreshScheduler` in `src/quantstack/tui/refresh.py`, `QuantStackApp` with `TabbedContent` in `src/quantstack/tui/app.py`, and `dashboard.tcss` for layout.
- **Section 02 (Query Layer):** Query functions in `src/quantstack/tui/queries/` that return typed dataclasses — specifically `system.py`, `portfolio.py`, `strategies.py`, `signals.py`, `risk.py`, `research.py`, `data_health.py`, `agents.py`.
- **Section 03 (Charts):** `horizontal_bar()` from `src/quantstack/tui/charts.py` (used by `DataHealthCompact` for coverage bars).

## Files to Create

- `src/quantstack/tui/widgets/overview.py` — All compact widgets for this tab
- `src/quantstack/tui/screens/overview.py` — The `OverviewTab` container that composes widgets into the 2-column grid
- `tests/unit/test_tui/test_overview.py` — Unit tests

## Files to Modify

- `src/quantstack/tui/app.py` — Import and mount `OverviewTab` as the first `TabPane` in `compose()`
- `src/quantstack/tui/dashboard.tcss` — Grid layout rules for the overview tab

---

## Tests (Write First)

These tests go in `tests/unit/test_tui/test_overview.py`. Each widget is tested in isolation by constructing it, calling its `update_view()` method with known data, and asserting on the rendered output.

```python
# tests/unit/test_tui/test_overview.py

import pytest

# -- ServicesCompact --
# Test: renders graph status lines with cycle numbers and last-seen durations
#   Input: list of GraphCheckpoint dataclasses for research, trading, supervisor
#   Expected output contains: "R:UP", "T:UP", "S:UP", cycle numbers, "Errors: 0"
# Test: renders "DOWN" when a graph's last heartbeat exceeds 120s
# Test: handles empty checkpoint list (all graphs show "?")

# -- RiskCompact --
# Test: renders exposure, drawdown, VaR, alert count, kill switch status
#   Input: RiskSnapshot dataclass + kill_switch bool
#   Expected output contains: "Exposure:", "DD:", "VaR:", "Alerts:", "Kill: ok"
# Test: renders "Kill: HALT" when kill switch is True
# Test: handles None risk snapshot (renders "No risk data")

# -- PortfolioCompact --
# Test: renders equity, daily P&L, and top 2 open positions
#   Input: EquitySummary dataclass + list of Position dataclasses
#   Expected output contains: "Equity: $10,234", "Today: +$127.50"
# Test: formats negative P&L with minus sign and red styling
# Test: handles zero positions (renders equity only, no position list)

# -- TradesCompact --
# Test: renders last 3 closed trades, one line each with symbol, P&L, date
#   Input: list of 5 ClosedTrade dataclasses (should only render first 3)
# Test: handles empty trade list (renders "No recent trades")

# -- StrategyCountsCompact --
# Test: renders counts per status bucket (Draft, BT, FT, Live, Retired)
#   Input: list of StrategyCard dataclasses with mixed statuses
#   Expected output contains: "Draft:3 BT:6 FT:2 Live:0 Ret:4"
# Test: shows next promotion candidate (strategy closest to completing forward test)
# Test: handles empty strategy list (all counts zero)

# -- SignalsCompact --
# Test: renders top 3 signals with symbol, action, confidence
#   Input: list of Signal dataclasses sorted by confidence DESC
#   Expected output contains: "NVDA BUY 87%", "AAPL HOLD 72%"
# Test: handles fewer than 3 signals
# Test: handles empty signal list (renders "No active signals")

# -- DataHealthCompact --
# Test: renders coverage bars for 4 most important data types
#   Input: dict of freshness data per data type
#   Uses horizontal_bar() from charts.py
# Test: handles empty freshness data (renders 0% bars)

# -- ResearchCompact --
# Test: renders WIP, queue, ML experiment, bug, and breakthrough counts
#   Input: integers for each metric
#   Expected output contains: "WIP: 2  Queue: 12  ML: 47  Bugs: 0  Breakthroughs: 2"
# Test: handles all-zero counts

# -- AgentActivityLine --
# Test: renders one line per graph showing current agent and last tool call
#   Input: list of AgentEvent dataclasses grouped by graph
# Test: handles no recent events (renders "idle" per graph)

# -- DigestCompact --
# Test: renders daily digest text after 17:00 ET
# Test: renders empty/placeholder before 17:00 ET
# Test: handles missing digest data

# -- DecisionsCompact --
# Test: renders last 3 decision events with timestamp, agent, action, symbol
#   Input: list of decision event records
# Test: handles empty decision list

# -- All compact widgets --
# Test: every widget handles None/empty data passed to update_view() without raising
# Test: every widget produces a renderable (Rich Text, Panel, or Table)
```

---

## Widget Design

All compact widgets live in `src/quantstack/tui/widgets/overview.py`. Each one follows the same pattern:

1. Subclass `RefreshableWidget` (from `src/quantstack/tui/base.py`)
2. Override `fetch_data()` — acquires a connection via `pg_conn()`, calls the appropriate query function from `src/quantstack/tui/queries/`, returns the typed dataclass result
3. Override `update_view(data)` — receives the dataclass, renders a Rich `Text`, `Panel`, or `Table`, and calls `self.update()` to push the renderable to the screen

The `RefreshableWidget` base handles the thread-to-UI bridge: `fetch_data()` runs in a `@work(thread=True)` worker, then `update_view()` is called on the main thread via `App.call_from_thread()`.

### ServicesCompact

Displays one line showing all three graph statuses with cycle counts and error tally.

```
R:UP c#9 170s  T:UP c#5 28s  S:UP c#3 45s  Errors: 0
```

- **Refresh tier:** T1 (5 seconds)
- **Query:** `fetch_graph_checkpoints()` from `queries/system.py` — returns list of `GraphCheckpoint` dataclasses
- **Logic:** For each graph (research, trading, supervisor), show UP/DOWN based on whether last heartbeat is within 120s of now. Show cycle number and seconds since last checkpoint. Sum error counts across all graphs.
- **Degradation:** If query returns empty list, render all graphs as `?:? c#? ?s`.

### RiskCompact

Displays risk metrics and kill switch status on a single line.

```
Exposure: 37%  DD: -2.5%  VaR: $245  Alerts: 1  Kill: ok
```

- **Refresh tier:** T2 (15 seconds)
- **Queries:** `fetch_risk_snapshot()` from `queries/risk.py` + `fetch_kill_switch()` from `queries/system.py`
- **Logic:** Pull latest row from `risk_snapshots`. Format exposure as percentage, drawdown as signed percentage, VaR as dollar amount. Count active equity alerts. Kill switch shows "ok" (green) or "HALT" (red bold).
- **Degradation:** If `risk_snapshots` is empty, render "No risk data | Kill: ok/HALT".

### PortfolioCompact

Displays equity summary and top open positions.

```
Equity: $10,234  Today: +$127.50  Open: AAPL +2.1%, NVDA +1.7%
```

- **Refresh tier:** T2 (15 seconds)
- **Queries:** `fetch_equity_summary()` + `fetch_positions()` from `queries/portfolio.py`
- **Logic:** Format equity with dollar sign and commas. Daily P&L with sign prefix, green if positive, red if negative. Show top 2 positions by absolute unrealized P&L percentage.
- **Degradation:** If no equity data, render "Equity: --". If no positions, omit the "Open:" section.

### TradesCompact

Displays last 3 closed trades.

```
AAPL +$45.20 (3d)  NVDA -$12.50 (1d)  TSLA +$89.00 (5d)
```

- **Refresh tier:** T3 (60 seconds)
- **Query:** `fetch_closed_trades()` from `queries/portfolio.py` — returns list of `ClosedTrade`, LIMIT 10
- **Logic:** Take first 3 results. Render symbol, signed P&L (green/red), holding period in days.
- **Degradation:** If no trades, render "No recent trades".

### StrategyCountsCompact

Displays strategy pipeline counts and next promotion candidate.

```
Draft:3 BT:6 FT:2 Live:0 Ret:4  Next promotion: AAPL inv (18d)
```

- **Refresh tier:** T3 (60 seconds)
- **Query:** `fetch_strategy_pipeline()` from `queries/strategies.py` — returns list of `StrategyCard`
- **Logic:** Group by `status` field, count each. For "Next promotion", find the forward_testing strategy with the highest `fwd_days / fwd_required_days` ratio (closest to completing).
- **Degradation:** If empty, render "Draft:0 BT:0 FT:0 Live:0 Ret:0".

### SignalsCompact

Displays top 3 active signals by confidence.

```
NVDA BUY 87%  AAPL HOLD 72%  TSLA SELL 68%
```

- **Refresh tier:** T2 (15 seconds)
- **Query:** `fetch_active_signals()` from `queries/signals.py` — returns list of `Signal` sorted by confidence DESC
- **Logic:** Take first 3. Color code action: green for BUY, yellow for HOLD, red for SELL. Confidence as integer percentage.
- **Degradation:** If empty, render "No active signals".

### DataHealthCompact

Displays coverage bars for the 4 most important data types.

```
OHLCV ████████████░░ 85%  News ██████░░░░░░░░ 42%
Sent  █████████░░░░░ 64%  Fund ████████████░░ 92%
```

- **Refresh tier:** T3 (60 seconds)
- **Queries:** `fetch_ohlcv_freshness()`, `fetch_news_freshness()`, `fetch_sentiment_freshness()`, `fetch_fundamentals_freshness()` from `queries/data_health.py`
- **Logic:** For each data type, compute coverage as (symbols with fresh data / total universe symbols). Render using `horizontal_bar()` from `charts.py`. Show the 4 types with most user value: OHLCV, News, Sentiment, Fundamentals.
- **Degradation:** If freshness data empty, render 0% bars.

### ResearchCompact

Displays research activity counts on a single line.

```
WIP: 2  Queue: 12  ML: 47  Bugs: 0  Breakthroughs: 2
```

- **Refresh tier:** T3 (60 seconds)
- **Queries:** `fetch_research_wip()`, `fetch_research_queue()`, `fetch_ml_experiments()`, `fetch_bugs()`, `fetch_breakthroughs()` from `queries/research.py`
- **Logic:** Each query returns a list; render the count. Color "Bugs" red if > 0.
- **Degradation:** If any query fails, show 0 for that metric.

### AgentActivityLine

One line per graph showing current agent and last tool call.

```
Research: alpha_discovery → scan_universe (12s ago)
Trading: risk_monitor → check_positions (3s ago)
Supervisor: health_check → verify_heartbeats (45s ago)
```

- **Refresh tier:** T1 (5 seconds)
- **Query:** `fetch_agent_events()` from `queries/agents.py` — returns list of `AgentEvent` ordered DESC
- **Logic:** Group by `graph_name`, take the most recent event per graph. Show agent name, arrow, tool/action name, and relative time since event.
- **Degradation:** If no events for a graph, render "idle".

### DigestCompact

Shows the daily digest summary after market close.

```
Daily: +$127.50 (+1.2%) | 3 trades closed | Risk: nominal | Next: earnings AAPL
```

- **Refresh tier:** T4 (120 seconds)
- **Logic:** Only renders content after 17:00 ET. Before that, renders "Daily digest available after market close". Data comes from `loop_iteration_context` or a dedicated digest query. Shows daily P&L, trade count, risk status, and upcoming events.
- **Degradation:** If no digest data after 17:00, render "Digest pending...".

### DecisionsCompact

Shows last 3 decision events (audit trail).

```
16:45 risk_monitor REJECT TSLA 45% "volatility exceeds threshold"
16:32 entry_scanner ENTER AAPL 82% "momentum breakout confirmed"
16:20 position_mgr TIGHTEN NVDA 71% "trailing stop adjustment"
```

- **Refresh tier:** T2 (15 seconds)
- **Query:** Fetch from `decision_events` table, ordered by timestamp DESC, LIMIT 3
- **Logic:** Render timestamp (HH:MM), agent name, action (ENTER/SKIP/TIGHTEN/REJECT/APPROVE), symbol, confidence, and one-line reasoning truncated to fit.
- **Degradation:** If no decisions, render "No recent decisions".

---

## Tab Layout

The `OverviewTab` container composes widgets into a 2-column grid that fits a 24-line terminal (subtracting header and footer leaves ~20 usable lines).

### Container: `src/quantstack/tui/screens/overview.py`

```python
class OverviewTab(ScrollableContainer):
    """Overview tab — single-screen compact summaries from all subsystems.

    Layout: 2-column grid with compact widgets, plus full-width rows at the bottom
    for agent activity and daily digest.
    """

    def compose(self) -> ComposeResult:
        """Yields compact widgets arranged in a 2-column grid.

        Left column: ServicesCompact, PortfolioCompact, StrategyCountsCompact, DataHealthCompact
        Right column: RiskCompact, TradesCompact, SignalsCompact, ResearchCompact
        Below grid: AgentActivityLine (one per graph), DigestCompact, DecisionsCompact
        """
        ...
```

### CSS Layout: `src/quantstack/tui/dashboard.tcss`

The overview grid uses Textual's CSS grid:

```css
OverviewTab {
    layout: grid;
    grid-size: 2;
    grid-gutter: 1;
}

OverviewTab .full-width {
    column-span: 2;
}
```

The agent activity lines, digest, and decisions sections span both columns (full-width rows below the grid).

---

## Refresh Tier Assignments

| Widget | Tier | Interval | Rationale |
|--------|------|----------|-----------|
| ServicesCompact | T1 | 5s | Graph health is critical — detect outages fast |
| AgentActivityLine | T1 | 5s | Shows what agents are doing right now |
| RiskCompact | T2 | 15s | Risk changes with positions, not per-second |
| PortfolioCompact | T2 | 15s | Equity and P&L update on trade events |
| SignalsCompact | T2 | 15s | Signals update on collector runs |
| DecisionsCompact | T2 | 15s | Decisions happen during trading cycles |
| TradesCompact | T3 | 60s | Closed trades are infrequent |
| StrategyCountsCompact | T3 | 60s | Strategy pipeline changes slowly |
| DataHealthCompact | T3 | 60s | Data freshness changes on collector schedule |
| ResearchCompact | T3 | 60s | Research queue changes on cycle boundaries |
| DigestCompact | T4 | 120s | Digest is computed once daily |

---

## Error Handling

Every widget must render a meaningful fallback when data is unavailable:

- If `fetch_data()` raises any exception, `RefreshableWidget` catches it (per the base class contract) and passes `None` to `update_view()`.
- Each widget's `update_view()` checks for `None` input and renders a degraded but informative state (e.g., "No risk data" instead of crashing or showing stale data with no indication).
- No widget should ever display a traceback or raw exception message. Log errors via the standard logger; show a user-friendly fallback in the UI.

---

## Implementation Notes

- All widgets use Rich renderables (`Text`, `Panel`, `Table`) for styling. They do not use raw string concatenation for colored output.
- Dollar amounts use `${:,.2f}` formatting. Percentages use `{:.1f}%` or `{:.0f}%` depending on precision needs.
- Color conventions: green = positive/healthy, red = negative/critical, yellow = warning/borderline, cyan = informational, dim = secondary data.
- The overview tab is Tab 1 (keybinding `1`). It is the default active tab on app launch.
- The `TieredRefreshScheduler` fires all overview widgets' queries immediately on mount (since it's the default tab), then on their respective tier intervals.
