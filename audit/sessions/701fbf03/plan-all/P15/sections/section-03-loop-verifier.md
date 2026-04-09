# Section 03: Loop Verifier

## Objective

Build a verification framework that checks whether each of the 5 closed feedback loops is actually closing — detecting broken feedback before it causes silent degradation.

## Files to Create

### `src/quantstack/autonomous/loop_verifier.py`

Core loop verification logic.

## Files to Modify

### `src/quantstack/autonomous/__init__.py`

Export `LoopVerifier` and `LoopHealth`.

## Implementation Details

### Loop Definitions

Each loop is defined as a `FeedbackLoopSpec` dataclass:

```python
@dataclass(frozen=True)
class FeedbackLoopSpec:
    name: str
    description: str
    trigger_query: str       # SQL to detect trigger events in last N hours
    behavior_change_query: str  # SQL to detect resulting behavior change
    max_staleness_hours: float  # alert if loop hasn't closed in this many hours
```

### The 5 Loops

1. **trade_outcome_to_research** — Trade loss triggers research queue entry with priority boost
   - Trigger: `SELECT ... FROM trade_outcomes WHERE pnl < 0 AND created_at > NOW() - INTERVAL '48 hours'`
   - Behavior change: corresponding entry in `research_queue` with `priority > default`
   - Max staleness: 48h

2. **realized_cost_to_cost_model** — TCA feedback updates cost estimates
   - Trigger: `SELECT ... FROM tca_results WHERE created_at > NOW() - INTERVAL '48 hours'`
   - Behavior change: `cost_model_updates` table has new entry
   - Max staleness: 48h

3. **ic_degradation_to_signal_weight** — IC drops below 0.02 triggers weight floor
   - Trigger: `SELECT ... FROM signal_ic WHERE ic < 0.02 AND measured_at > NOW() - INTERVAL '48 hours'`
   - Behavior change: `signal_weights` table shows weight at minimum for that collector
   - Max staleness: 48h

4. **live_perf_to_strategy_demotion** — 3 consecutive losing weeks triggers demotion
   - Trigger: `SELECT ... FROM strategy_weekly_pnl WHERE consecutive_losses >= 3`
   - Behavior change: strategy status changed to `paper_only` or `demoted`
   - Max staleness: 168h (weekly cycle)

5. **agent_quality_to_prompt** — Agent win rate < 40% triggers few-shot injection
   - Trigger: `SELECT ... FROM agent_metrics WHERE win_rate < 0.40 AND window = 'last_5_cycles'`
   - Behavior change: agent config updated with new few-shot examples
   - Max staleness: 48h

### LoopVerifier Class

```python
class LoopVerifier:
    def __init__(self, loop_specs: list[FeedbackLoopSpec] | None = None): ...
    def check_loop(self, loop_name: str) -> LoopHealth: ...
    def check_all(self) -> dict[str, LoopHealth]: ...
    def get_broken_loops(self) -> list[str]: ...
```

### LoopHealth Dataclass

```python
@dataclass
class LoopHealth:
    loop_name: str
    status: Literal["healthy", "stale", "broken", "no_trigger"]
    last_triggered: datetime | None
    last_behavior_change: datetime | None
    details: str
```

Status logic:
- `no_trigger`: no trigger events in the staleness window (loop not tested, but not broken)
- `healthy`: trigger found AND behavior change found within staleness window
- `stale`: trigger found but behavior change is 24-48h old
- `broken`: trigger found but no behavior change within `max_staleness_hours`

### Daily Verification

`run_daily_verification() -> dict[str, LoopHealth]`:
- Called by supervisor graph or scheduler
- Checks all 5 loops
- Returns health dict
- Logs results at INFO level; logs WARNING for stale, ERROR for broken

## Test Requirements

- `tests/unit/autonomous/test_loop_verifier.py`:
  - Mock DB queries to simulate each loop status (healthy, stale, broken, no_trigger)
  - Verify correct status classification for each case
  - Test that `get_broken_loops()` returns only broken loop names
  - Test with missing tables (graceful degradation, not crash)
  - Test that `no_trigger` is not classified as broken

## Acceptance Criteria

1. All 5 feedback loops have explicit trigger and behavior-change queries
2. `check_all()` completes in under 5 seconds (simple DB queries)
3. Status classification matches the documented thresholds exactly
4. `no_trigger` is distinct from `broken` — absence of trigger events is not a failure
5. All DB access uses `db_conn()` context manager
6. Loop specs are configurable (can add new loops without code changes to the verifier class)
