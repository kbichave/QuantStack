# Section 10: Burn-In Protocol

## Objective

Define the 7-day unattended validation protocol that must pass before the system is declared autonomous. This section integrates all prior P15 subsystems into a formal go/no-go gate.

**Depends on:** section-02-scheduler-integration, section-03-loop-verifier, section-04-authority-matrix, section-05-reconciler, section-06-health-dashboard

## Files to Create

### `src/quantstack/autonomous/burn_in.py`

Burn-in validation logic and go-live checklist.

### `scripts/burn_in_status.sh`

CLI tool to check current burn-in progress.

## Implementation Details

### BurnInValidator Class

```python
class BurnInValidator:
    def __init__(
        self,
        loop_verifier: LoopVerifier,
        reconciler: PositionReconciler,
        dashboard: HealthDashboard,
        burn_in_days: int = 7,
    ): ...

    async def check_progress(self) -> BurnInStatus: ...
    async def is_complete(self) -> bool: ...
    def get_remaining_criteria(self) -> list[str]: ...
```

### BurnInStatus Dataclass

```python
@dataclass
class BurnInStatus:
    start_date: date
    current_day: int  # day N of burn-in
    target_days: int  # 7
    criteria: list[BurnInCriterion]
    overall_pass: bool
    blockers: list[str]  # criteria not yet met

@dataclass
class BurnInCriterion:
    name: str
    description: str
    met: bool
    first_met_at: datetime | None
    details: str
```

### The 5 Burn-In Criteria

1. **All 5 feedback loops closed at least once**
   - Query: each loop in `LoopVerifier` has `status != "no_trigger"` at least once during burn-in
   - Tracked via: `loop_verifier.check_all()` daily results stored in `burn_in_log`

2. **No kill switch triggers from bugs**
   - Real risk triggers (drawdown, stop loss) are acceptable
   - Bug-triggered kills (uncaught exception, DB error) are NOT acceptable
   - Query: `kill_switch_events` table, check `trigger_reason` classification
   - Zero bug-triggered kills for 7 consecutive days

3. **Reconciliation matches within tolerance for 7 consecutive days**
   - Query: `reconciliation_log` table, `all_clear = true` for every reconciliation in the window
   - Any single mismatch resets the consecutive-day counter

4. **Weekly report generates successfully**
   - At least 1 weekly report generated without error during burn-in
   - Query: check report generation log

5. **At least 1 strategy promoted AND 1 strategy managed through lifecycle**
   - Query: `strategies` table, lifecycle changes during burn-in window
   - "Managed through lifecycle" = any strategy that moved between stages (promoted, demoted, or retired)

### Burn-In Log

```sql
CREATE TABLE IF NOT EXISTS burn_in_log (
    id SERIAL PRIMARY KEY,
    burn_in_start DATE NOT NULL,
    check_date DATE NOT NULL,
    criterion_name TEXT NOT NULL,
    met BOOLEAN NOT NULL,
    details TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Daily Check

`run_daily_burn_in_check()`:
- Called by scheduler at 23:00 ET daily during burn-in
- Evaluates all 5 criteria
- Inserts results into `burn_in_log`
- Logs progress at INFO level
- If all 5 criteria met AND current_day >= 7: mark burn-in complete

### Go-Live Checklist

Separate from the automated criteria, a manual checklist stored as a config:

```python
GO_LIVE_CHECKLIST = [
    "All feature flags enabled for production features",
    "Kill switch tested (trigger + recovery)",
    "Backup restore tested",
    "Discord alerts verified",
    "Mode transitions tested (market → extended → overnight → market)",
]
```

`get_checklist_status() -> list[ChecklistItem]`:
- Reads from `go_live_checklist` DB table where items are marked complete by human operator
- Returns status of each item

### burn_in_status.sh

```bash
#!/usr/bin/env bash
python -c "
import asyncio
from quantstack.autonomous.burn_in import BurnInValidator, create_default_validator
validator = create_default_validator()
status = asyncio.run(validator.check_progress())
for c in status.criteria:
    mark = '✓' if c.met else '✗'
    print(f'  {mark} {c.name}: {c.details}')
print(f'Day {status.current_day}/{status.target_days}')
print(f'Overall: {\"PASS\" if status.overall_pass else \"IN PROGRESS\"}')"
```

## Test Requirements

- `tests/unit/autonomous/test_burn_in.py`:
  - Mock all sub-components, test criteria evaluation for each
  - Test that a single failed criterion blocks overall_pass
  - Test consecutive-day reset for reconciliation criterion
  - Test that bug-triggered kills are distinguished from risk-triggered kills
  - Test day counting (burn-in started 3 days ago → current_day = 3)
  - Test completion detection (all criteria met + day >= 7)

## Acceptance Criteria

1. All 5 criteria are independently evaluated and reported
2. Burn-in cannot complete in fewer than 7 days even if all criteria are met immediately
3. A single reconciliation mismatch resets the consecutive-day counter (not the burn-in clock)
4. Bug-triggered vs risk-triggered kill switches are correctly classified
5. `burn_in_status.sh` provides clear, human-readable progress output
6. Go-live checklist requires explicit human sign-off (not auto-completable)
7. All state persisted in DB (survives restart)
