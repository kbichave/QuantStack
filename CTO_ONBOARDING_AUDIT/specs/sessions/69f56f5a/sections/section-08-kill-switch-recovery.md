# Section 8: Kill Switch Auto-Recovery

## Background

The kill switch (`src/quantstack/execution/kill_switch.py`, ~455 lines) is a thread-safe singleton that halts all trading when triggered. It persists state via a sentinel file at `~/.quantstack/KILL_SWITCH_ACTIVE` for cross-process durability. Four auto-trigger conditions exist in the `AutoTriggerMonitor` class: consecutive broker API failures, SPY halt, 3-day rolling drawdown, and model drift. Currently, ALL conditions require manual `reset(reset_by=...)` — there is no auto-recovery and no escalation notification beyond the initial trigger log.

The current `reset()` signature is:

```python
def reset(self, reset_by: str = "manual") -> None
```

This section adds three capabilities:
1. **AutoRecoveryManager** — automatic reset for broker failures only (other conditions carry signal about model/market state and must remain manual)
2. **KillSwitchEscalationManager** — tiered notification escalation for all kill switch types
3. **Sizing ramp-back** — gradual return to full position sizing after auto-reset

## Dependencies

- **Section 01 (env-validation):** The `validate_environment()` function should validate any new env vars introduced here (e.g., `MAX_AUTO_RESETS_PER_DAY`). Ensure section 01 is complete or stub the vars.
- No database migration dependency — state is stored in the existing `system_state` table using key-value pairs.
- No EventBus or pub/sub dependency — both managers poll kill switch state directly via the singleton API.

## Tests First

All tests go in `tests/unit/test_kill_switch_recovery.py`. Fixtures needed: `mock_broker` (simulates Alpaca broker health check responses), `mock_discord` (captures Discord webhook payloads), `env_override` (context manager for setting/unsetting env vars).

### AutoRecoveryManager Tests

```python
# tests/unit/test_kill_switch_recovery.py

# --- AutoRecoveryManager ---
# Test: manager does nothing when kill switch is not active
# Test: manager does nothing when kill switch triggered by drawdown (not broker failure)
# Test: manager does nothing when kill switch triggered by drift
# Test: manager does nothing when kill switch triggered by SPY halt
# Test: manager initiates investigation at 5 min after broker failure trigger
# Test: manager auto-resets at 15 min when broker is responsive
# Test: manager does NOT auto-reset at 15 min when broker still down
# Test: manager sets sizing_scalar=0.5 on auto-reset
# Test: manager respects MAX_AUTO_RESETS_PER_DAY=2 (third reset blocked)
# Test: manager backs off when reset followed by immediate re-trigger
```

Key assertion patterns:
- "Does nothing" tests verify that `reset()` is never called on the kill switch.
- The 5-min investigation test should mock `datetime.now()` and verify a broker health check call is made.
- The MAX_AUTO_RESETS test writes two reset events to `system_state` (key: `auto_resets_{YYYY-MM-DD}`) and asserts the third attempt is blocked.
- The back-off test triggers reset, then immediately re-triggers the kill switch, and asserts the manager does not attempt another reset within a cooldown window.

### Sizing Ramp-back Tests

```python
# --- Sizing ramp-back ---
# Test: sizing_scalar starts at 0.5 after auto-reset
# Test: sizing_scalar ramps 0.5 -> 0.75 -> 1.0 over 3 successful cycles
# Test: sizing_scalar resets to 0.5 if kill switch re-triggers during ramp
# Test: sizing_scalar key removed from system_state after reaching 1.0
```

These tests interact with the `system_state` table (mock or in-memory dict). The ramp sequence is: cycle 1 success -> 0.75, cycle 2 success -> 1.0, cycle 3 success -> key removed. The scalar multiplies final position size at the same integration point where `FORWARD_TESTING_SIZE_SCALAR` is applied.

### KillSwitchEscalationManager Tests

```python
# --- KillSwitchEscalationManager ---
# Test: escalation sends Discord at 0 min
# Test: escalation sends email at 4 hours (or enhanced Discord if email not configured)
# Test: escalation sends emergency Discord at 24 hours
# Test: escalation does not send duplicate notifications for same tier
# Test: escalation resets tier tracking when kill switch is cleared
```

The escalation manager stores `last_escalation_tier` in `system_state` to prevent duplicate sends. Mock the Discord client and email sender to capture payloads and verify content includes: trigger reason, open positions, unrealized P&L.

### Reset Signature Tests

```python
# --- reset() signature ---
# Test: reset() works with reason provided (audit trail logged)
# Test: reset() works without reason (backward compatible, warning logged)
# Test: reset event written to kill_switch_events table
```

The `reason` parameter must be optional to avoid breaking existing callers. When omitted, a warning is logged encouraging callers to provide context.

## Implementation Details

### 8.1 AutoRecoveryManager

Add a new class, either in `src/quantstack/execution/kill_switch.py` or a dedicated `src/quantstack/execution/kill_switch_recovery.py` module.

```python
class AutoRecoveryManager:
    """Manages tiered recovery for broker failure kill switch triggers.

    Only broker failures qualify for auto-recovery. All other kill switch
    conditions (drawdown, drift, SPY halt) require manual reset because
    they carry signal about model or market state.

    Recovery timeline:
      0 min  - Kill switch triggers, Discord alert fires
      5 min  - Auto-investigate: call broker health check endpoint
      15 min - If broker responsive AND trigger was broker failure: auto-reset
               with sizing_scalar=0.5
      15 min - If broker still down: do NOT reset, continue escalation

    Safety: MAX_AUTO_RESETS_PER_DAY = 2. Third trigger stays halted.
    """
    MAX_AUTO_RESETS_PER_DAY = 2
```

The manager runs inside the supervisor graph cycle (every 5 minutes). It does not spawn its own background thread. Each supervisor cycle, the manager:

1. Checks if the kill switch is active via `get_kill_switch().is_active()`.
2. If not active, no-op (also clears any internal tracking state).
3. If active, reads the trigger reason from `get_kill_switch().status().reason`.
4. If the reason is NOT a broker failure pattern, no-op (leave for manual reset).
5. If it IS a broker failure, check elapsed time since `triggered_at`:
   - Less than 5 min: do nothing (too early).
   - 5-15 min: run broker health check (call Alpaca account endpoint). Log result.
   - At or past 15 min AND broker is responsive: check daily reset count. If under cap, call `reset(reset_by="auto_recovery", reason="Broker failure auto-recovery: broker responsive after 15min")` and set `sizing_scalar=0.5` in `system_state`.
   - At or past 15 min AND broker still down: do nothing (escalation manager handles notifications).

**Broker failure detection:** The manager must distinguish broker failures from other trigger reasons. The `AutoTriggerMonitor` sets the reason string when triggering. Match on a known prefix or enum value (e.g., reason starts with `"Consecutive broker"` or similar). Document the exact string contract between `AutoTriggerMonitor.record_broker_result()` and `AutoRecoveryManager`.

**Daily reset tracking:** Store in `system_state` table with key `auto_resets_{YYYY-MM-DD}` and an integer value. Increment on each auto-reset. Check before resetting.

**Back-off on immediate re-trigger:** After auto-reset, if the kill switch re-triggers within 5 minutes (i.e., `triggered_at` is within 5 min of the last reset), the manager should NOT attempt another auto-reset for that trigger. This prevents a flapping loop where the broker is intermittently failing.

### 8.2 KillSwitchEscalationManager

```python
class KillSwitchEscalationManager:
    """Sends progressively urgent notifications while the kill switch is active.

    Escalation schedule:
      0 min  -> Discord: trigger reason + open positions + unrealized P&L
      4 hours -> Email (or enhanced Discord if email not configured)
      24 hours -> Emergency Discord: all-caps header, full position summary,
                  daily P&L, recommended actions

    Stores last_escalation_tier in system_state to avoid duplicate sends.
    Resets when kill switch is cleared.
    """
```

The escalation manager also runs inside the supervisor graph cycle. Each cycle:

1. If kill switch is NOT active: clear `last_escalation_tier` from `system_state` if it exists. Return.
2. If kill switch IS active: compute elapsed time since `triggered_at`.
3. Check `last_escalation_tier` from `system_state`.
4. If elapsed >= 24h and tier < 3: send emergency Discord notification, set tier = 3.
5. If elapsed >= 4h and tier < 2: send email (or enhanced Discord), set tier = 2.
6. If tier < 1: send initial Discord notification, set tier = 1.

The tier 1 (0 min) notification fires on the first supervisor cycle after the kill switch triggers. Content should include:
- Trigger reason
- Current open positions (query from portfolio state)
- Unrealized P&L
- Whether auto-recovery is eligible (broker failure vs. other)

The tier 2 (4h) email should include everything from tier 1 plus:
- Time elapsed since trigger
- Any auto-recovery attempts and their outcomes
- Explicit instructions: "Log in and investigate. Run `switch.reset(reset_by='manual', reason='...')` after confirming root cause."

If email is not configured (no SMTP credentials or no email address set), send an enhanced Discord message noting: "Email escalation would fire here but is not configured."

The tier 3 (24h) emergency Discord uses an all-caps header and includes full position summary, daily P&L, and recommended actions.

### 8.3 Sizing Ramp-back

After any auto-reset from broker failure, the system does not return to full position sizing immediately. The ramp-back mechanism:

**State keys in `system_state` table:**
- `kill_switch_sizing_scalar` — current scalar value (float). Values: 0.5, 0.75, 1.0.
- `successful_cycles_since_reset` — integer counter of successful trading cycles since last auto-reset.

**Ramp schedule:**
- Auto-reset sets `sizing_scalar = 0.5` and `successful_cycles_since_reset = 0`.
- After each successful trading cycle (no errors, positions managed correctly): increment counter.
- Counter reaches 1: set scalar to 0.75.
- Counter reaches 2: set scalar to 1.0.
- Counter reaches 3: remove both keys from `system_state` (fully recovered).

**If the kill switch re-triggers during ramp:** Reset scalar back to 0.5 and counter to 0. The ramp starts over.

**Integration point in trading graph:** Wherever position size is calculated and `FORWARD_TESTING_SIZE_SCALAR` is applied, also read `kill_switch_sizing_scalar` from `system_state` and multiply. If the key does not exist, the scalar is implicitly 1.0 (no effect). This is a single query to `system_state` per position sizing calculation.

### 8.4 Reset Reason Logging

Modify the existing `reset()` method in `KillSwitch`:

```python
def reset(self, reset_by: str = "manual", reason: str = "") -> None:
    """Reset kill switch. Reason is logged for audit trail.

    Args:
        reset_by: Who/what initiated the reset (e.g., "manual", "auto_recovery")
        reason: Why the reset is happening. Logged for audit. Warning emitted if empty.
    """
```

Implementation notes:
- `reason` defaults to empty string for backward compatibility — all existing callers that only pass `reset_by` continue to work.
- If `reason` is empty, log a WARNING: "Kill switch reset by '{reset_by}' without reason. Provide a reason for audit trail."
- Write a reset event to the `system_state` table with a timestamped key (e.g., `kill_switch_reset_{ISO_TIMESTAMP}`) containing a JSON blob: `{"reset_by": ..., "reason": ..., "triggered_at": ..., "duration_seconds": ...}`.
- Alternatively, if a dedicated `kill_switch_events` table is preferred, create it. But using `system_state` avoids needing a migration and keeps the implementation self-contained within this section.

### Key Files to Create or Modify

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/execution/kill_switch.py` | Modify | Add `reason` parameter to `reset()`. Add `AutoRecoveryManager` and `KillSwitchEscalationManager` classes (or put them in a new module). |
| `src/quantstack/execution/kill_switch_recovery.py` | Create (optional) | If the classes are large enough to warrant a separate module. Import into `kill_switch.py` or supervisor nodes. |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Add recovery and escalation manager checks to the supervisor cycle. Instantiate both managers and call their `check()` methods each cycle. |
| `src/quantstack/coordination/daily_digest.py` | Modify | Add kill switch event formatting to the daily digest (reset events, escalation events). |
| `tests/unit/test_kill_switch_recovery.py` | Create | All tests listed above. |

### Risks and Mitigations

**Thread safety:** The `AutoRecoveryManager` must only call `reset()` through the existing KillSwitch API. It must not directly modify `_active`, `_status`, or the sentinel file. The `reset()` method already holds a `Lock()` internally, so concurrent access from the supervisor graph and any other thread is safe.

**Race condition — reset then immediate re-trigger:** After `reset()` clears the kill switch, the `AutoTriggerMonitor` may immediately re-trigger if the underlying broker failures continue. The recovery manager must detect this pattern (reset happened very recently + triggered again within 5 min) and back off rather than attempting another reset. This is the flapping protection described in 8.1.

**Escalation notification failures:** If Discord is unreachable when an escalation fires, log the failure but do not retry aggressively. The next supervisor cycle (5 min later) will attempt again since the tier tracking only advances on successful send.

**system_state table coupling:** Both managers read and write to `system_state`. Ensure key names are namespaced (prefix with `kill_switch_`) to avoid collisions with other subsystems that use the same table.
