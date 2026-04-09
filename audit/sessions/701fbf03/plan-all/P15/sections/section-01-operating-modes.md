# Section 01: Operating Modes

## Objective

Define the four operating modes that govern what the system does at any given time of day. This is the foundation for scheduler integration (Section 02) and mode-aware resource allocation.

## Files to Create

### `src/quantstack/config/operating_modes.py`

Defines the `OperatingMode` enum and mode-detection logic.

## Implementation Details

### OperatingMode Enum

```python
class OperatingMode(str, Enum):
    MARKET_HOURS = "market_hours"          # 09:30-16:00 ET Mon-Fri
    EXTENDED_HOURS = "extended_hours"      # 16:00-20:00, 04:00-09:30 ET Mon-Fri
    OVERNIGHT_WEEKEND = "overnight_weekend"  # All other times
    CRYPTO_FUTURES = "crypto_futures"      # 24/7, conditional on P12 availability
```

### Mode Detection Function

`detect_current_mode() -> OperatingMode`:
- Uses system clock converted to US/Eastern timezone
- Checks market calendar for holidays (use `exchange_calendars` or a static holiday list)
- Returns the appropriate mode based on current time and day of week
- `CRYPTO_FUTURES` is only returned if `ENABLE_CRYPTO_FUTURES=true` env var is set AND no other equity mode is active — otherwise it overlays on top of other modes

### Mode Configuration Dataclass

`ModeConfig` for each mode specifying:
- `graphs_active: list[str]` — which LangGraph services run (research, trading, supervisor)
- `trading_enabled: bool` — whether new equity positions can be opened
- `research_compute_priority: float` — 0.0 to 1.0, resource share for research
- `position_monitoring: bool` — whether existing positions are monitored
- `transition_hooks: list[str]` — hook names to call on entry/exit of this mode

### Default Mode Configs

| Mode | Trading | Research Priority | Monitoring | Active Graphs |
|------|---------|-------------------|------------|---------------|
| MARKET_HOURS | True | 0.2 | True | trading, supervisor, research (lightweight) |
| EXTENDED_HOURS | False | 0.3 | True | supervisor |
| OVERNIGHT_WEEKEND | False | 1.0 | False | research, supervisor |
| CRYPTO_FUTURES | True (crypto only) | 0.1 | True | trading, supervisor |

### Mode Transition Detection

`get_mode_transition(previous: OperatingMode, current: OperatingMode) -> ModeTransition | None`:
- Returns a `ModeTransition` dataclass with `from_mode`, `to_mode`, `transition_hooks`
- Returns `None` if no transition occurred
- Transition hooks are named strings resolved by the scheduler (Section 02)

### Key Design Decisions

- Modes are mutually exclusive for equity trading; `CRYPTO_FUTURES` can overlay
- Holiday detection uses a static list updated annually plus an env var `EXTRA_MARKET_HOLIDAYS` for ad-hoc closures
- All times are ET (US/Eastern) since that is the US equity market timezone
- Mode detection is a pure function of clock time — no database reads, no network calls

## Test Requirements

- `tests/unit/config/test_operating_modes.py`:
  - Test mode detection at boundary times (09:29 vs 09:30, 15:59 vs 16:00)
  - Test weekend detection (Saturday, Sunday)
  - Test holiday detection (known US holidays)
  - Test `CRYPTO_FUTURES` overlay behavior with and without env var
  - Test mode transition detection returns correct hooks
  - Test that mode detection is timezone-correct when system TZ is not ET

## Acceptance Criteria

1. `detect_current_mode()` returns the correct mode for any given US/Eastern time
2. Mode configs are immutable dataclasses, not dicts
3. Holiday handling covers NYSE holidays for the current year
4. No network calls or DB reads in mode detection (pure function of clock + config)
5. All boundary conditions tested (market open/close exact second)
