# Section 12: Multi-Mode Operation

## Overview

The system must operate differently depending on the time of day and day of the week. During market hours, trading is the priority. During extended hours, only position monitoring runs (no new entries). Overnight and on weekends, the Research Graph runs heavy compute (ML training, autoresearch, community intel scans) while trading is idle.

This section adds a `ScheduleMode` enum, mode-aware graph routing for all three graphs, and timezone-aware scheduling anchored to `America/New_York`.

## Current State

Significant groundwork already exists:

- **`src/quantstack/runners/__init__.py`** already defines `OperatingMode` (an enum with `MARKET`, `EXTENDED`, `OVERNIGHT`, `WEEKEND` values), `get_operating_mode()` (timezone-aware, uses `America/New_York`), and per-graph cycle intervals via the `INTERVALS` dict.
- **Trading Graph** (`src/quantstack/graphs/trading/graph.py`) already has mode-aware routing: `_safety_check_router` routes to `"monitor_only"` when mode is not `MARKET`, and `_exits_router` skips entry pipeline in monitor-only mode.
- **Research Graph** (`src/quantstack/graphs/research/graph.py`) has no mode awareness. It runs the same pipeline regardless of time.
- **Supervisor Graph** (`src/quantstack/graphs/supervisor/graph.py`) has no mode awareness. It runs a linear 6-node pipeline at all times.

The plan calls for a `ScheduleMode` enum, but the existing `OperatingMode` enum in `runners/__init__.py` already covers the same semantics (`MARKET_HOURS` = `MARKET`, `EXTENDED_HOURS` = `EXTENDED`, `OVERNIGHT_WEEKEND` = `OVERNIGHT` + `WEEKEND`). The implementation should use the existing `OperatingMode` rather than creating a duplicate enum.

## Dependencies

- **Phase 2 complete**: This section is in Phase 3, Batch 5. All Phase 1 and Phase 2 sections must be implemented first.
- **Blocks Section 13**: The autoresearch overnight loop (Section 13) depends on the overnight routing implemented here.

## Tests First

All tests go in `tests/graphs/test_multi_mode.py`.

```python
# tests/graphs/test_multi_mode.py

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quantstack.runners import OperatingMode, get_operating_mode

ET = ZoneInfo("America/New_York")


class TestModeDetection:
    """Verify OperatingMode is resolved correctly for all time windows."""

    def test_market_hours_weekday(self):
        """9:30-16:00 ET on a weekday → MARKET."""
        dt = datetime(2026, 4, 7, 10, 0, tzinfo=ET)  # Tuesday 10:00 ET
        assert get_operating_mode(dt) == OperatingMode.MARKET

    def test_extended_hours_morning(self):
        """04:00-09:30 ET weekday → EXTENDED."""
        dt = datetime(2026, 4, 7, 7, 0, tzinfo=ET)  # Tuesday 7:00 ET
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_extended_hours_evening(self):
        """16:00-20:00 ET weekday → EXTENDED."""
        dt = datetime(2026, 4, 7, 18, 0, tzinfo=ET)  # Tuesday 18:00 ET
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_overnight(self):
        """20:00-04:00 ET → OVERNIGHT."""
        dt = datetime(2026, 4, 7, 22, 0, tzinfo=ET)  # Tuesday 22:00 ET
        assert get_operating_mode(dt) == OperatingMode.OVERNIGHT

    def test_weekend_saturday(self):
        """Saturday → WEEKEND."""
        dt = datetime(2026, 4, 11, 12, 0, tzinfo=ET)  # Saturday
        assert get_operating_mode(dt) == OperatingMode.WEEKEND

    def test_weekend_sunday(self):
        """Sunday → WEEKEND."""
        dt = datetime(2026, 4, 12, 12, 0, tzinfo=ET)  # Sunday
        assert get_operating_mode(dt) == OperatingMode.WEEKEND

    def test_timezone_aware_not_utc(self):
        """Mode detection uses America/New_York, not UTC.

        14:00 UTC on a weekday = 10:00 ET = MARKET.
        A naive UTC-based check might think this is extended hours.
        """
        from datetime import timezone
        dt = datetime(2026, 4, 7, 14, 0, tzinfo=timezone.utc)
        assert get_operating_mode(dt) == OperatingMode.MARKET

    def test_dst_transition_spring_forward(self):
        """EST→EDT transition (March second Sunday).

        After spring forward, 10:00 ET is still MARKET regardless of
        whether the system was previously running on EST offsets.
        """
        dt = datetime(2026, 3, 9, 10, 0, tzinfo=ET)  # Monday after spring forward
        assert get_operating_mode(dt) == OperatingMode.MARKET

    def test_early_close_day(self):
        """Early close days (e.g., day before Thanksgiving at 13:00 ET close).

        The existing OperatingMode uses fixed 16:00 close. If exchange_calendars
        integration is added for early closes, this test verifies correctness.
        For now, documents the known limitation.
        """
        # Day before Thanksgiving 2026 is Nov 25 (Wednesday)
        # At 14:00 ET, market would actually be closed on an early close day
        # Current implementation returns MARKET (known limitation — uses fixed windows)
        dt = datetime(2026, 11, 25, 14, 0, tzinfo=ET)
        mode = get_operating_mode(dt)
        # Document current behavior; update assertion when early close support added
        assert mode == OperatingMode.MARKET  # Known limitation

    def test_holiday_treated_as_weekend(self):
        """NYSE holidays are treated as WEEKEND mode."""
        # 2026-01-01 is New Year's Day (Thursday)
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=ET)
        assert get_operating_mode(dt) == OperatingMode.WEEKEND


class TestTradingGraphRouting:
    """Verify Trading Graph routes correctly based on operating mode."""

    def test_market_hours_runs_full_pipeline(self):
        """During MARKET mode, safety_check routes to 'continue' (full pipeline)."""
        # Verified by _safety_check_router returning "continue" when mode is MARKET
        # and no halt condition exists.

    def test_extended_hours_monitoring_only(self):
        """During EXTENDED mode, routes to position_review (exits only, no new entries)."""
        # Verified by _safety_check_router returning "monitor_only" when mode != MARKET.

    def test_exits_router_skips_entries_outside_market(self):
        """In monitor-only mode, execute_exits routes to reflect (not merge_parallel)."""
        # Verified by _exits_router returning "monitor_only" when operating_mode != "market".


class TestResearchGraphRouting:
    """Verify Research Graph routes based on operating mode."""

    def test_market_hours_light_research(self):
        """During MARKET mode, Research Graph runs light pipeline only.

        Light = skip ML experiments and heavy backtest. Run hypothesis generation
        and signal validation only.
        """

    def test_overnight_heavy_compute(self):
        """During OVERNIGHT/WEEKEND, Research Graph routes to heavy compute.

        Heavy = full pipeline including ML experiments, autoresearch (Section 13),
        and community intel scans.
        """

    def test_extended_hours_eod_processing(self):
        """During EXTENDED mode, Research Graph runs EOD sync and earnings processing."""


class TestSupervisorGraphRouting:
    """Verify Supervisor Graph adjusts behavior by mode."""

    def test_market_hours_monitoring(self):
        """During MARKET, Supervisor runs health checks and monitors."""

    def test_overnight_strategy_lifecycle(self):
        """During OVERNIGHT/WEEKEND, Supervisor runs strategy lifecycle management."""
```

## Implementation Details

### 1. Enhance `get_operating_mode` with `exchange_calendars` for Early Closes

**File:** `src/quantstack/runners/__init__.py`

The current implementation uses hardcoded `NYSE_HOLIDAYS` and fixed time windows. Early close days (e.g., day before Thanksgiving, Christmas Eve) are not handled — at 13:01 ET on those days, the system still thinks the market is open.

Add an optional `exchange_calendars` integration to detect early closes. The `exchange_calendars` library is already a dependency (used in `src/quantstack/execution/compliance/calendar.py` and `src/quantstack/core/core/calendar.py`).

The enhancement:
- Use `exchange_calendars.get_calendar("XNYS")` to get the NYSE calendar
- On trading days, check `session_close` time — if the current time is past the actual close (which may be 13:00 ET on early close days), return `EXTENDED` instead of `MARKET`
- Cache the calendar instance at module level (it is expensive to construct)
- Fall back to the existing hardcoded logic if `exchange_calendars` is unavailable

This is a refinement of the existing function, not a rewrite. The hardcoded `NYSE_HOLIDAYS` set can remain as a fast-path check; `exchange_calendars` adds precision for early closes.

### 2. Add Mode-Aware Routing to Research Graph

**File:** `src/quantstack/graphs/research/graph.py`

The Research Graph currently runs the same pipeline regardless of time. Add a mode-aware entry router after `context_load`:

- **MARKET mode**: Run a lightweight pipeline — `context_load` -> `domain_selection` -> `hypothesis_generation` -> `hypothesis_critique` -> `knowledge_update` -> END. Skip `signal_validation`, `backtest_validation`, and `ml_experiment` (these are expensive). The goal is to generate and critique hypotheses during market hours without consuming heavy LLM/compute budget that trading needs.
- **EXTENDED mode**: Run EOD processing — the full existing pipeline is appropriate since trading is monitoring-only.
- **OVERNIGHT / WEEKEND mode**: Run the full pipeline plus route to the autoresearch node (Section 13 will add this node). For now, add a conditional edge placeholder that routes to the full pipeline. When Section 13 lands, it will add the autoresearch node and update this routing.

Implementation approach:
- Add a `_research_mode_router` function that checks `get_operating_mode()` and returns `"light"`, `"full"`, or `"heavy"` 
- Insert a conditional edge after `hypothesis_critique` that, in light mode, skips directly to `strategy_registration` (bypassing validation/backtest/ML)
- In full and heavy modes, continue through the existing pipeline

The router function:

```python
def _research_mode_router(state: ResearchState) -> str:
    """Route research pipeline based on operating mode.

    Light (market hours): skip expensive validation/ML, register hypotheses only.
    Full (extended): run complete pipeline.
    Heavy (overnight/weekend): run complete pipeline (autoresearch added by Section 13).
    """
    from quantstack.runners import OperatingMode, get_operating_mode

    mode = get_operating_mode()
    if mode == OperatingMode.MARKET:
        return "light"
    return "full"
```

Integrate this into the existing `route_after_hypothesis` / `route_after_hypothesis_fanout` routing functions in `src/quantstack/graphs/research/nodes.py` rather than adding a separate conditional edge. The hypothesis critique router already decides between looping back and proceeding forward — extend it to also check operating mode when deciding which forward path to take.

### 3. Add Mode-Aware Routing to Supervisor Graph

**File:** `src/quantstack/graphs/supervisor/graph.py`

The Supervisor Graph runs a linear 6-node pipeline. Add mode-aware behavior:

- **MARKET mode**: Run the full pipeline (health_check -> diagnose_issues -> execute_recovery -> strategy_pipeline -> strategy_lifecycle -> scheduled_tasks -> eod_data_sync). This is the current behavior — no change needed.
- **EXTENDED mode**: Same as market but skip `eod_data_sync` if it's the morning extended window (04:00-09:30). Only run `eod_data_sync` in the evening extended window (16:00-20:00).
- **OVERNIGHT / WEEKEND mode**: Focus on `strategy_lifecycle` (promotions, demotions, retirements) and `scheduled_tasks` (community intel scans, model retraining triggers). Skip `execute_recovery` (no trading to recover). Health checks still run.

Implementation approach:
- Add a conditional edge after `execute_recovery` that checks operating mode
- In OVERNIGHT/WEEKEND, skip `strategy_pipeline` (no active trading pipeline to check) and go directly to `strategy_lifecycle`
- This is a minimal change — one conditional edge replacing a direct edge

The router function:

```python
def _supervisor_mode_router(state: SupervisorState) -> str:
    """Route supervisor pipeline based on operating mode."""
    from quantstack.runners import OperatingMode, get_operating_mode

    mode = get_operating_mode()
    if mode in (OperatingMode.OVERNIGHT, OperatingMode.WEEKEND):
        return "skip_pipeline"
    return "full"
```

### 4. Propagate Operating Mode Through State

**File:** `src/quantstack/graphs/state.py`

The `TradingState` already has an `operating_mode: str` field. Verify that `ResearchState` and `SupervisorState` also have this field (or add it). Each graph's entry node should set `operating_mode` from `get_operating_mode().value` so that downstream nodes and routers can read it from state without re-calling `get_operating_mode()`.

This avoids a subtle bug: if a graph cycle takes several minutes and crosses a mode boundary (e.g., starts at 15:58 MARKET, finishes at 16:02 EXTENDED), each node calling `get_operating_mode()` independently could get a different answer. Setting the mode once at cycle start and propagating through state ensures consistency within a single cycle.

### 5. Update Cycle Intervals

**File:** `src/quantstack/runners/__init__.py`

The `INTERVALS` dict already defines per-graph, per-mode cycle intervals. Verify these match the design intent:

| Graph | MARKET | EXTENDED | OVERNIGHT | WEEKEND |
|-------|--------|----------|-----------|---------|
| Trading | 300s (5 min) | 300s (5 min) | None (paused) | None (paused) |
| Research | 120s (2 min) | 180s (3 min) | 120s (2 min) | 300s (5 min) |
| Supervisor | 300s (5 min) | 300s (5 min) | 300s (5 min) | 300s (5 min) |

The current values look correct. Trading is paused overnight/weekend (returning `None`). Research runs at all times with shorter intervals when compute is cheap (overnight). Supervisor runs at constant 5-min intervals.

One consideration: the `None` interval for Trading during overnight/weekend means the runner must handle `None` gracefully (skip the cycle entirely, not sleep forever). Verify this in the runner loop.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/runners/__init__.py` | Optional: add `exchange_calendars` early-close detection |
| `src/quantstack/graphs/research/graph.py` | Add mode-aware routing after hypothesis critique |
| `src/quantstack/graphs/research/nodes.py` | Extend `route_after_hypothesis` with mode check |
| `src/quantstack/graphs/supervisor/graph.py` | Add conditional edge after `execute_recovery` |
| `src/quantstack/graphs/state.py` | Verify `operating_mode` field on `ResearchState` and `SupervisorState` |
| `tests/graphs/test_multi_mode.py` | New test file |

## Edge Cases

- **Mode boundary crossing mid-cycle**: Set mode once at cycle start, propagate through state. All nodes within a cycle see the same mode.
- **DST transitions**: The existing `get_operating_mode()` already handles DST correctly because it uses `ZoneInfo("America/New_York")` which accounts for EST/EDT automatically. No additional work needed.
- **Early close days**: The hardcoded `NYSE_HOLIDAYS` only covers full-day closures. Early closes (13:00 ET) on days like the day before Thanksgiving are not detected. The `exchange_calendars` integration addresses this but is an enhancement, not a blocker.
- **Holiday calendar staleness**: The hardcoded `NYSE_HOLIDAYS` set only covers 2025-2027. After 2027, it will miss holidays. The `exchange_calendars` library provides a long-term solution. Add a TODO with a trigger condition: "migrate fully to exchange_calendars when the 2027 boundary approaches."
- **Runner handling of `None` interval**: When `get_cycle_interval()` returns `None`, the runner must skip that graph's cycle entirely. Verify the runner loop does not interpret `None` as 0 (immediate re-run) or raise an error.

## Rollback

Revert the conditional edges in Research and Supervisor graphs to their original direct edges. The Trading Graph's existing mode-aware routing (already in production) is unaffected. The `OperatingMode` enum and `get_operating_mode()` are used by other modules and should not be reverted.
