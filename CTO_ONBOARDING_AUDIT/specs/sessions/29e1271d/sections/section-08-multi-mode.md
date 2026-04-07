# Section 8: Multi-Mode 24/7 Operation

## Overview

QuantStack currently stops the trading graph entirely outside of market hours. This section adds a four-mode operating system that keeps the trading graph running in a monitor-only capacity during extended hours (active stop losses, position reviews, exit execution) while blocking all new entries via a hard risk gate check. Research and supervisor graphs adjust their cycle intervals based on mode.

This section depends on:
- **Section 01 (DB Schema)** — no new tables, but the `TradingState` schema must include `cycle_attribution` before the monitor-only subgraph can route through the attribution node.
- **Section 04 (Factor Exposure)** — factor computation runs in monitor-only mode during extended hours.
- **Section 05 (Performance Attribution)** — the attribution node must exist for the monitor-only subgraph path.

---

## Tests First

All tests live in `tests/unit/` and `tests/integration/`. Testing framework is pytest with existing fixtures (`trading_ctx`, `paper_broker`, `kill_switch`, `portfolio`, `risk_state`).

### Unit Tests

```python
# tests/unit/test_operating_mode.py

# Test: get_operating_mode returns MARKET at 10:00 ET Monday
# Test: get_operating_mode returns EXTENDED at 17:00 ET Tuesday
# Test: get_operating_mode returns EXTENDED at 05:00 ET Wednesday
# Test: get_operating_mode returns OVERNIGHT at 22:00 ET Thursday
# Test: get_operating_mode returns WEEKEND at 14:00 ET Saturday
# Test: get_operating_mode handles NYSE holidays (market closed on holiday = weekend mode)
# Test: get_cycle_interval returns 300 for trading in MARKET mode
# Test: get_cycle_interval returns 300 for trading in EXTENDED mode (was None)
# Test: get_cycle_interval returns None for trading in OVERNIGHT mode
# Test: get_cycle_interval returns 120 for research in OVERNIGHT mode (heavy research)
```

For `get_operating_mode` tests, mock `datetime.now()` to fixed ET timestamps and verify the returned enum value. For holiday handling, mock or patch the holiday calendar to include a known date.

```python
# tests/unit/test_risk_gate_trading_window.py

# Test: risk gate _check_trading_window rejects new long entry in EXTENDED mode
# Test: risk gate _check_trading_window rejects new short entry in EXTENDED mode
# Test: risk gate _check_trading_window allows exit (sell to close long) in EXTENDED mode
# Test: risk gate _check_trading_window allows cover (buy to close short) in EXTENDED mode
# Test: risk gate _check_trading_window passes all orders in MARKET mode
# Test: risk gate checks absolute exposure change, not order side
```

The risk gate tests must verify that the check examines whether `abs(position_quantity)` would increase (reject) or decrease (allow). A sell order on a long position is an exit (allowed). A sell order with no existing position is a new short (rejected). A buy order on a short position is a cover (allowed). A buy order with no position is a new long (rejected).

### Integration Tests

```python
# tests/integration/test_trading_graph_modes.py

# Test: trading graph conditional edge routes to monitor-only subgraph in EXTENDED mode
# Test: trading graph runs full pipeline in MARKET mode
# Test: monitor-only subgraph visits: safety_check -> position_review -> execute_exits -> reflect -> attribution
# Test: monitor-only subgraph does NOT visit: plan_day, entry_scan, execute_entries
```

For integration tests, mock `get_operating_mode()` to return the desired mode and run the trading graph with a traced execution. Verify which nodes were visited by inspecting the LangGraph trace or state transitions.

---

## Implementation Details

### 8.1 Mode Detection — `OperatingMode` Enum and `get_operating_mode()`

**File:** `src/quantstack/runners/__init__.py`

Add a new enum and detection function:

```python
class OperatingMode(str, Enum):
    MARKET = "market"           # 9:30-16:00 ET Mon-Fri
    EXTENDED = "extended"       # 16:00-20:00 ET, 04:00-09:30 ET Mon-Fri
    OVERNIGHT = "overnight"     # 20:00-04:00 ET Mon-Fri
    WEEKEND = "weekend"         # Sat-Sun all day
```

`get_operating_mode() -> OperatingMode`:
- Get current time in US/Eastern timezone.
- Check day of week: Saturday/Sunday returns `WEEKEND`.
- Check NYSE holiday calendar: if today is a holiday, return `WEEKEND`.
- Check time windows in order: `MARKET` (09:30-16:00), `EXTENDED` (04:00-09:30 and 16:00-20:00), else `OVERNIGHT`.

For NYSE holidays, use `pandas_market_calendars` if already a dependency, otherwise maintain a minimal hardcoded list of known holidays for the current year, updated annually. The holiday check prevents the system from attempting market-hours behavior on days like MLK Day or Independence Day.

### 8.2 Graph Behavior Matrix

| Graph | Market | Extended | Overnight | Weekend |
|-------|--------|----------|-----------|---------|
| Trading | Full pipeline (300s) | Monitor-only (300s) | Off (None) | Off (None) |
| Research | Normal (120s) | Normal (180s) | Heavy (120s) | Heavy (300s) |
| Supervisor | Normal (300s) | Normal (300s) | Normal (300s) | Normal (300s) |

"Monitor-only" means the trading graph executes: `data_refresh` -> `safety_check` -> `position_review` -> `execute_exits` -> `reflect` -> `attribution` -> END. It skips: `plan_day`, `entry_scan`, `execute_entries`.

Key behavioral change: the trading graph now runs during extended hours (previously stopped entirely). This keeps stop losses and trailing stops active for open positions. No new entries are opened.

### 8.3 Trading Graph Conditional Edge — Monitor-Only Routing

**File:** `src/quantstack/graphs/trading/graph.py`

Add a conditional edge after the `safety_check` node that calls `get_operating_mode()`:

```python
def route_after_safety_check(state: TradingState) -> str:
    """Route to full pipeline or monitor-only based on operating mode."""
    ...
```

- If mode is `MARKET`: return the existing next-node name (continues to the full pipeline including `plan_day` and `entry_scan`).
- If mode is `EXTENDED`: return a node name that routes to the monitor-only path (`position_review` -> `execute_exits` -> `reflect` -> `attribution` -> END).
- If mode is `OVERNIGHT` or `WEEKEND`: the runner interval is `None`, so the graph should not be invoked at all. But as a safety net, if the graph somehow runs, route to monitor-only.

Follow the existing conditional edge pattern already used in the trading graph (e.g., existing `route_after_safety_check` or similar router functions). The new router replaces or wraps the existing routing logic at that decision point.

The monitor-only subgraph is not a separate `StateGraph` — it is a path through the same graph defined by conditional edges. The nodes `position_review`, `execute_exits`, `reflect`, and `attribution` already exist (or will exist after Section 05). The conditional edge simply skips the entry-side nodes.

### 8.4 Risk Gate Hard Block — `_check_trading_window()`

**File:** `src/quantstack/execution/risk_gate.py`

Add a new check method at the top of the risk gate's validation chain:

```python
def _check_trading_window(self) -> RiskDecision:
    """Reject exposure-increasing orders outside market hours.
    
    Logic:
    1. Call get_operating_mode()
    2. If MARKET: PASS (all orders allowed)
    3. If EXTENDED/OVERNIGHT/WEEKEND:
       - Look up current position for the symbol
       - Compute what abs(position_quantity) would be after the order
       - If abs(new_qty) > abs(current_qty): REJECT (exposure increasing)
       - If abs(new_qty) <= abs(current_qty): PASS (exposure decreasing or flat)
    
    Rejection reasons:
    - EXTENDED: 'Extended hours - no new entries'
    - OVERNIGHT/WEEKEND: 'Market closed'
    """
```

This is the **hardest block** in the system — it runs at the risk gate level, which every order must pass through regardless of which graph node submitted it. Even if an agent somehow generates an entry signal during extended hours (e.g., a bug in graph routing), the risk gate will reject it.

The check must examine absolute exposure, not order side:
- A **sell** order when `position_qty > 0` is closing (allowed) — `abs(new_qty) < abs(current_qty)`.
- A **sell** order when `position_qty == 0` is opening a new short (rejected) — `abs(new_qty) > abs(current_qty)`.
- A **buy** order when `position_qty < 0` is covering a short (allowed) — `abs(new_qty) < abs(current_qty)`.
- A **buy** order when `position_qty == 0` is opening a new long (rejected) — `abs(new_qty) > abs(current_qty)`.

Wire this check into the existing validation chain in `risk_gate.py`. It should be the first check evaluated (fail fast before computing position sizing, factor limits, etc.).

### 8.5 Runner Interval Updates

**File:** `src/quantstack/runners/__init__.py`

Update `get_cycle_interval()` to use `OperatingMode`:

```python
INTERVALS = {
    "trading": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,
        OperatingMode.OVERNIGHT: None,   # graph does not run
        OperatingMode.WEEKEND: None,     # graph does not run
    },
    "research": {
        OperatingMode.MARKET: 120,
        OperatingMode.EXTENDED: 180,
        OperatingMode.OVERNIGHT: 120,    # heavy research mode
        OperatingMode.WEEKEND: 300,      # heavy research, longer interval
    },
    "supervisor": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,
        OperatingMode.OVERNIGHT: 300,
        OperatingMode.WEEKEND: 300,
    },
}
```

When `get_cycle_interval()` returns `None`, the runner sleeps and re-checks mode periodically (e.g., every 60 seconds) until the interval becomes non-None. The existing runner loop already handles `None` intervals by skipping the cycle — verify this behavior and add the re-check if it currently stops entirely.

---

## Edge Cases and Failure Modes

**Mode transition mid-cycle:** If the mode changes from MARKET to EXTENDED while a trading graph cycle is in progress, the cycle completes normally (it was started during market hours). The conditional edge is evaluated at the point in the graph where routing happens. If the mode flips between `safety_check` and the router evaluation, the router will see `EXTENDED` and route to monitor-only for the remainder of that cycle. This is correct behavior — finishing the current cycle's entries is fine, but the risk gate will reject any new entries submitted after the transition.

**NYSE holiday detection:** If the holiday calendar is stale or missing a holiday, the system will treat the day as a regular weekday. The risk gate hard block provides a safety net: even if mode detection is wrong, Alpaca will reject orders on closed markets, and the risk gate's position-based check prevents exposure increases during any non-MARKET mode.

**Timezone handling:** All mode detection must use `US/Eastern` (or `America/New_York`) timezone, not UTC. The `zoneinfo` standard library module (Python 3.9+) should be used for timezone conversion. Do not use `pytz` unless it is already a project dependency.

**Runner startup during off-hours:** When the system starts during overnight/weekend, the trading runner should not crash. It should detect the mode, see `None` interval, and enter a sleep-and-recheck loop. Log the mode at startup so operators can verify correct detection.

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/runners/__init__.py` | Add `OperatingMode` enum, `get_operating_mode()`, update `get_cycle_interval()` with mode-aware intervals |
| `src/quantstack/graphs/trading/graph.py` | Add conditional edge router for monitor-only mode after `safety_check` |
| `src/quantstack/execution/risk_gate.py` | Add `_check_trading_window()` as first check in validation chain |
| `tests/unit/test_operating_mode.py` | New: unit tests for mode detection and cycle intervals |
| `tests/unit/test_risk_gate_trading_window.py` | New: unit tests for trading window check in risk gate |
| `tests/integration/test_trading_graph_modes.py` | New: integration tests for graph routing in different modes |
