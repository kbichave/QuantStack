# Section 11: Kill Switch Auto-Recovery & Log Aggregation

## Overview

The kill switch (`src/quantstack/execution/kill_switch.py`) currently requires a manual `reset()` call after every trigger. For 24/7 unattended operation, transient conditions (broker disconnect, brief API outage) should auto-recover after a cooldown, while permanent conditions (daily loss limit, consecutive failures) must stay triggered until a human intervenes.

Separately, container logs currently go only to Docker's local json-file driver. The Fluent-bit, Loki, and Grafana services are defined in `docker-compose.yml` and config files already exist, but the pipeline needs verification, error-rate alert rules need an email escalation contact point, and the kill switch must integrate with the email alerting system from Section 8.

## Dependencies

- **Section 8 (Email Alerting):** This section calls `send_alert()` from the alerting module created in Section 8 (`src/quantstack/alerting/email_sender.py`). That module must exist before this section can be implemented. If implementing in isolation, stub `send_alert()` as a no-op that logs to stderr.

## Tests First

All tests go in a single file. They validate the recovery classification logic, auto-reset behavior, email integration, and escalation timing.

```python
# tests/execution/test_kill_switch_recovery.py

"""
Tests for tiered kill switch recovery.

Covers:
  - Transient trigger reasons auto-reset after cooldown
  - Permanent trigger reasons do NOT auto-reset
  - CRITICAL email sent immediately on any trigger
  - Escalation email sent after 4 hours if still active
  - Trigger reason classification for each AutoTriggerMonitor condition
"""

# Test: transient trigger (broker_disconnect) auto-resets after 30-min cooldown
#   - Trigger kill switch with reason containing "broker" keyword
#   - Advance time past 30-minute cooldown
#   - Call recovery_check() — assert kill switch is no longer active
#   - Assert reset_by == "auto_recovery"

# Test: permanent trigger (daily_loss_limit) does NOT auto-reset
#   - Trigger kill switch with reason containing "daily" or "loss limit"
#   - Advance time past 30 minutes, 1 hour, 4 hours
#   - Call recovery_check() at each interval — assert kill switch remains active

# Test: CRITICAL email sent immediately on trigger
#   - Mock send_alert from alerting module
#   - Trigger kill switch with any reason
#   - Assert send_alert called once with level=CRITICAL
#   - Assert email body contains trigger reason and timestamp

# Test: escalation email sent after 4 hours if not reset
#   - Trigger kill switch with permanent reason
#   - Advance time to 4 hours + 1 minute
#   - Call recovery_check()
#   - Assert send_alert called with escalation message
#   - Assert escalation email is distinct from initial alert (different subject/body)

# Test: trigger reason classification is correct for each AutoTriggerMonitor condition
#   - "consecutive broker API failures" -> transient
#   - "broker" in reason -> transient
#   - "data staleness" or "stale" in reason -> transient
#   - "daily loss" or "loss limit" in reason -> permanent
#   - "consecutive failures" (non-broker, e.g. tool failures) -> permanent
#   - "model drift" in reason -> permanent
#   - "market-wide circuit breaker" -> permanent
#   - Unknown/unclassified reason -> permanent (safe default)
```

## Implementation Details

### Part 1: Trigger Reason Classification

Add a classification function to `kill_switch.py` that categorizes trigger reasons as transient or permanent. The classification drives whether auto-recovery is allowed.

**File:** `src/quantstack/execution/kill_switch.py`

**Design:**

```python
class TriggerSeverity(str, Enum):
    TRANSIENT = "transient"   # Auto-recovery allowed after cooldown
    PERMANENT = "permanent"   # Manual reset required

def classify_trigger(reason: str) -> TriggerSeverity:
    """
    Classify a kill switch trigger reason as transient or permanent.

    Transient conditions resolve on their own (broker reconnects, data refreshes).
    Permanent conditions require human judgment (loss limits, model drift).

    Unknown reasons default to PERMANENT — fail safe, not fail open.
    """
    ...
```

Classification rules based on the four `AutoTriggerMonitor` conditions:

| AutoTriggerMonitor Condition | Reason Pattern | Classification | Rationale |
|------------------------------|---------------|----------------|-----------|
| Consecutive broker failures | `"broker"` in reason | TRANSIENT | Broker outages are typically brief (minutes) |
| Market-wide circuit breaker | `"market-wide"` or `"SPY halted"` | PERMANENT | Market structure event, needs human review |
| Rolling 3-day drawdown | `"rolling drawdown"` or `"daily loss"` or `"loss limit"` | PERMANENT | Capital preservation, human must assess |
| Model drift | `"model drift"` or `"drift"` | PERMANENT | Strategy integrity compromised |

Additional patterns:
- `"data staleness"` or `"stale"` -> TRANSIENT (data feeds recover)
- `"API outage"` or `"timeout"` -> TRANSIENT
- Any unrecognized reason -> PERMANENT (conservative default)

The classification uses case-insensitive substring matching. It does not use regex — the patterns are simple keywords and the trigger reasons are generated by our own code, not user input.

### Part 2: Recovery Logic in KillSwitch

Add recovery state tracking and a `recovery_check()` method to the `KillSwitch` class.

**File:** `src/quantstack/execution/kill_switch.py`

**New fields on `KillSwitchStatus`:**

```python
class KillSwitchStatus(BaseModel):
    active: bool = False
    triggered_at: datetime | None = None
    reason: str | None = None
    reset_at: datetime | None = None
    reset_by: str | None = None
    severity: TriggerSeverity | None = None          # NEW
    escalation_sent_at: datetime | None = None       # NEW
```

**New constants:**

```python
TRANSIENT_COOLDOWN = timedelta(minutes=30)
ESCALATION_DELAY = timedelta(hours=4)
```

**Changes to `trigger()` method:**

After the existing trigger logic (writing sentinel, logging, publishing EventBus event, closing positions), add:

1. Classify the trigger reason using `classify_trigger(reason)`.
2. Store the severity on `_status.severity`.
3. Write severity to the sentinel file so it survives restarts.
4. Send a CRITICAL email alert immediately via `send_alert()` from the Section 8 alerting module.

The email alert call should be wrapped in a try/except that logs on failure but does not prevent the trigger from completing. Email is best-effort — the kill switch must activate even if email is down.

**New `recovery_check()` method:**

```python
def recovery_check(self) -> None:
    """
    Check if an active kill switch can be auto-recovered.

    Called periodically (e.g., every 5 minutes) by the Supervisor Graph
    or a scheduler job.

    For TRANSIENT triggers: auto-reset after TRANSIENT_COOLDOWN (30 min).
    For PERMANENT triggers: send escalation email after ESCALATION_DELAY (4 hr).
    """
    ...
```

Logic:
1. If not active, return immediately.
2. Compute `elapsed = now - triggered_at`.
3. If `severity == TRANSIENT` and `elapsed >= TRANSIENT_COOLDOWN`:
   - Call `self.reset(reset_by="auto_recovery", reason=f"Transient condition auto-recovered after {TRANSIENT_COOLDOWN}")`.
   - Send an INFO-level email confirming auto-recovery.
4. If `severity == PERMANENT` and `elapsed >= ESCALATION_DELAY` and `escalation_sent_at is None`:
   - Send a WARNING-level escalation email: "Kill switch has been active for 4+ hours. Reason: {reason}. Manual reset required."
   - Set `_status.escalation_sent_at = now` to prevent repeat escalation emails.

### Part 3: Wire Recovery Check into Supervisor Graph

The `recovery_check()` method needs to be called periodically. The Supervisor Graph already runs health monitoring loops. Add a call to `get_kill_switch().recovery_check()` in the Supervisor's health check node or cycle start.

**File:** `src/quantstack/graphs/supervisor/nodes.py`

This is a single-line addition in the appropriate health-check or cycle-start function:

```python
get_kill_switch().recovery_check()
```

No new node is needed. The Supervisor already runs on a schedule — piggyback on its existing cycle.

### Part 4: Log Aggregation Verification and Alert Wiring

The Fluent-bit, Loki, and Grafana infrastructure is already defined in `docker-compose.yml` and the config files exist:

- `config/fluent-bit/fluent-bit.conf` — Tails Docker container logs, ships to Loki
- `config/fluent-bit/parsers.conf` — JSON parsing for Docker log format
- `config/loki/loki-config.yaml` — Loki storage with 30-day retention
- `config/grafana/provisioning/alerting/alerts.yaml` — Alert rules including error-spike and CRITICAL-log detection
- `config/grafana/provisioning/datasources/datasources.yaml` — Loki as datasource

**What already works:** The pipeline (Fluent-bit -> Loki -> Grafana) is wired. Alert rules for error-spike (>5 errors/5min) and CRITICAL log detection already exist. Discord webhook is configured as a contact point.

**What needs to be added:**

1. **Email contact point in Grafana.** Add a Gmail SMTP contact point to `config/grafana/provisioning/alerting/alerts.yaml` for escalation alerts. This uses the same Gmail app password as Section 8's email alerting module (env var `GMAIL_APP_PASSWORD`).

2. **Error rate alert rule refinement.** The existing `error-spike` rule triggers at >5 errors/5min. Add a second tier: >10 errors/5min triggers CRITICAL severity (routed to email). The existing rule stays at WARNING (routed to Discord only).

**File:** `config/grafana/provisioning/alerting/alerts.yaml`

Add to the `contactPoints` section:

```yaml
  - orgId: 1
    name: Email
    receivers:
      - uid: gmail-smtp
        type: email
        settings:
          addresses: ${ALERT_RECIPIENT_EMAIL}
          singleEmail: true
```

Add to the `rules` section:

```yaml
      - uid: error-rate-critical
        title: Error rate critical (>10/5min)
        condition: A
        data:
          - refId: A
            datasourceUid: loki
            model:
              expr: 'rate({job=~".*-graph"} |= "ERROR" [5m]) > 10'
              instant: true
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Critical error rate: >10 errors/5min in graph services"
```

Add a routing policy that sends critical-severity alerts to both Discord and Email:

```yaml
policies:
  - orgId: 1
    receiver: Discord
    group_wait: 30s
    group_interval: 5m
    repeat_interval: 4h
    routes:
      - receiver: Email
        matchers:
          - severity = critical
        continue: true   # Also send to Discord
```

**Grafana SMTP configuration:** Grafana needs SMTP configured at the server level to send emails. Add environment variables to the `grafana` service in `docker-compose.yml`:

```yaml
  grafana:
    environment:
      - GF_SMTP_ENABLED=true
      - GF_SMTP_HOST=smtp.gmail.com:587
      - GF_SMTP_USER=${GMAIL_SENDER_EMAIL:-}
      - GF_SMTP_PASSWORD=${GMAIL_APP_PASSWORD:-}
      - GF_SMTP_FROM_ADDRESS=${GMAIL_SENDER_EMAIL:-quantstack@gmail.com}
      - GF_SMTP_FROM_NAME=QuantStack Alerts
```

### Part 5: Sentinel File Update

The sentinel file format needs to include the severity classification so that recovery logic works across process restarts.

**File:** `src/quantstack/execution/kill_switch.py`

Update `_write_sentinel()` to write severity:

```
triggered_at=2026-04-07T10:30:00
reason=Auto-trigger: 3 consecutive broker API failures
severity=transient
```

Update `_load_from_file()` to read severity, defaulting to `PERMANENT` if missing (backward compatibility with old sentinel files).

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/execution/kill_switch.py` | Modify | Add `TriggerSeverity` enum, `classify_trigger()`, recovery fields on `KillSwitchStatus`, `recovery_check()` method, email integration in `trigger()`, sentinel file severity |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Add `recovery_check()` call in health/cycle-start node |
| `config/grafana/provisioning/alerting/alerts.yaml` | Modify | Add Email contact point, critical error-rate rule, routing policy |
| `docker-compose.yml` | Modify | Add SMTP env vars to Grafana service |
| `tests/execution/test_kill_switch_recovery.py` | Create | Recovery classification and auto-reset tests |

## Edge Cases and Failure Modes

- **Email module unavailable:** If Section 8 is not yet implemented, the `send_alert()` import will fail. Guard with a try/except at module level that replaces `send_alert` with a stderr logger fallback. This ensures the kill switch functions independently of the alerting system.
- **Sentinel file from old format:** Missing `severity` field in sentinel. `_load_from_file()` defaults to `PERMANENT` -- safe, never auto-resets a trigger that shouldn't be.
- **Clock skew in Docker:** `recovery_check()` uses `datetime.now()` which is host-clock-dependent. Docker containers share the host clock, so this is not a concern for single-host deployment.
- **Rapid re-trigger after auto-recovery:** A transient condition that triggers, auto-recovers, and immediately re-triggers suggests the root cause persists. Consider adding a counter: if the same transient condition triggers 3 times within 2 hours, escalate to PERMANENT. This is a Phase 4 enhancement -- not in scope for this section, but document it as a TODO with context.
- **Escalation email sent repeatedly:** The `escalation_sent_at` field prevents duplicates within a single process lifetime. If the process restarts, `escalation_sent_at` is not persisted to the sentinel file (only `severity` is). To prevent post-restart duplicate escalation, also write `escalation_sent_at` to the sentinel file.
