# Section 02: Scheduler Integration

## Objective

Extend the existing `scripts/scheduler.py` to be mode-aware, activating and deactivating graphs per operating mode, allocating resources, and firing transition hooks at mode boundaries.

**Depends on:** section-01-operating-modes

## Files to Modify

### `scripts/scheduler.py`

Add mode-aware scheduling on top of the existing cron-based job system.

### `src/quantstack/config/operating_modes.py`

May need minor additions for transition hook registration if not fully covered in Section 01.

## Files to Create

### `src/quantstack/autonomous/mode_manager.py`

Runtime mode manager that tracks current mode, detects transitions, and executes hooks.

## Implementation Details

### ModeManager Class

`ModeManager`:
- Holds `current_mode: OperatingMode` state
- `check_and_transition() -> ModeTransition | None`: called on a 1-minute interval, detects if mode changed, fires hooks if so
- `_execute_transition_hooks(transition: ModeTransition)`: runs named hooks in order
- Logs every mode transition with timestamp and hook results

### Transition Hooks

Named hooks mapped to callables:

| Transition | Hook Name | Action |
|------------|-----------|--------|
| MARKET_HOURS → EXTENDED_HOURS | `eod_reconciliation` | Run position reconciliation (Section 05) |
| EXTENDED_HOURS → OVERNIGHT_WEEKEND | `data_sync` | Trigger overnight data refresh |
| OVERNIGHT_WEEKEND → EXTENDED_HOURS | `pre_market_prep` | Load fresh signals, check earnings calendar |
| EXTENDED_HOURS → MARKET_HOURS | `market_open_ready` | Verify all systems healthy, positions reconciled |
| Any → CRYPTO_FUTURES | `crypto_activate` | Enable crypto trading graphs |

### Scheduler Changes

Add to the existing `BlockingScheduler`:
- A 60-second interval job: `mode_check_job` that calls `ModeManager.check_and_transition()`
- Mode-conditional job activation: existing jobs (data_refresh, pnl_attribution, etc.) only run if the current mode allows them
- New overnight jobs: ML training trigger, community intel scan, strategy lifecycle review

### Resource Allocation

`allocate_resources(mode: OperatingMode)`:
- Sets `RESEARCH_COMPUTE_SHARE` env var based on mode config
- Research graph reads this to throttle concurrent hypothesis evaluations
- During MARKET_HOURS: max 1 concurrent research task
- During OVERNIGHT_WEEKEND: unlimited (up to system capacity)

### Graph Activation/Deactivation

- Do NOT start/stop Docker containers — that is too slow and fragile
- Instead, use a `GRAPH_ACTIVE_{name}` flag that each graph's main loop checks before executing a cycle
- `activate_graphs(mode: OperatingMode)`: sets flags based on `ModeConfig.graphs_active`
- Graphs that are deactivated complete their current cycle then sleep until reactivated

## Test Requirements

- `tests/unit/test_mode_manager.py`:
  - Mock clock progression through a full 24h cycle, verify all transitions fire
  - Verify hooks execute in order
  - Verify graph activation flags match mode config
  - Verify resource allocation values per mode
  - Test that a failed hook logs the error but does not block the transition
  - Test idempotency: calling `check_and_transition()` twice in the same mode does nothing

## Acceptance Criteria

1. Mode transitions happen within 60 seconds of the actual boundary time
2. All transition hooks are logged with success/failure status
3. Existing scheduler jobs continue to work unchanged
4. Graph activation flags are set atomically (no partial state)
5. A failed transition hook does not crash the scheduler or block other hooks
6. Mode manager state survives scheduler restart (reads current mode from clock, not persisted state)
