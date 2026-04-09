# Section 07: Alerting and Reports

## Objective

Wire the health dashboard to Discord webhook alerts for critical events and build an automated weekly performance report.

**Depends on:** section-06-health-dashboard

## Files to Create

### `src/quantstack/autonomous/alerting.py`

Discord webhook alerting with deduplication.

### `src/quantstack/autonomous/weekly_report.py`

Automated weekly report generation.

## Files to Modify

### `scripts/scheduler.py`

Add weekly report job (Sunday 20:00 ET) and periodic alert-check job.

## Implementation Details

### AlertManager Class

```python
class AlertManager:
    def __init__(self, webhook_url: str | None = None, cooldown_minutes: int = 30): ...

    async def check_and_alert(self, health: SystemHealth) -> list[Alert]: ...
    def _should_alert(self, alert_type: str) -> bool: ...
    async def _send_discord(self, alert: Alert) -> bool: ...
```

### Alert Triggers

| Condition | Severity | Cooldown |
|-----------|----------|----------|
| Kill switch triggered | CRITICAL | 0 (always send) |
| Drawdown > 5% from peak | CRITICAL | 60 min |
| Agent win rate < 30% for 5 cycles | WARNING | 24 hours |
| Data staleness > threshold | WARNING | 4 hours |
| Reconciliation mismatch | ERROR | 30 min |
| Feedback loop broken (>48h) | WARNING | 12 hours |
| Authority ceiling escalation | INFO | 30 min |

### Deduplication

- Track `last_alert_sent: dict[str, datetime]` per alert type
- Do NOT re-send if cooldown has not elapsed
- Persist last-sent times in DB (survive restart):

```sql
CREATE TABLE IF NOT EXISTS alert_state (
    alert_type TEXT PRIMARY KEY,
    last_sent TIMESTAMPTZ,
    last_message TEXT
);
```

### Discord Webhook Format

```python
payload = {
    "embeds": [{
        "title": f"[{severity}] {alert_type}",
        "description": details,
        "color": color_map[severity],  # RED=critical, ORANGE=error, YELLOW=warning
        "timestamp": datetime.utcnow().isoformat(),
        "fields": [{"name": k, "value": str(v), "inline": True} for k, v in context.items()],
    }]
}
```

### Env Var

- `DISCORD_WEBHOOK_URL`: if not set, alerts log locally only (no crash)

### WeeklyReport Class

```python
class WeeklyReport:
    def __init__(self, dashboard: HealthDashboard): ...

    async def generate(self) -> ReportData: ...
    async def send_to_discord(self, report: ReportData) -> bool: ...
    def format_text(self, report: ReportData) -> str: ...
```

### Report Contents

Generated every Sunday 20:00 ET:

1. **Performance Metrics** (1-week window):
   - Weekly return, Sharpe ratio, max drawdown, Calmar ratio, Sortino ratio
   - Compared to SPY benchmark

2. **Alpha Decomposition**:
   - Signal alpha: return from signal-driven entries
   - Execution alpha: return from execution optimization
   - Timing alpha: return from entry/exit timing vs benchmark
   - Risk management: return preserved by risk gate rejections

3. **Top 3 Winners / Top 3 Losers**:
   - Symbol, strategy, P&L, entry/exit reasoning

4. **Research Velocity**:
   - Hypotheses generated, validated, rejected
   - Strategies promoted, demoted, retired

5. **System Health Summary**:
   - Uptime percentage
   - Error rate (errors / total operations)
   - Feedback loop health summary
   - Reconciliation status

### Report Data Sources

- Performance: `trade_outcomes` table, `portfolio_snapshots` table
- Alpha decomposition: `tca_results` table, `signal_attribution` table
- Winners/losers: `trade_outcomes` sorted by P&L
- Research: `research_queue` table, `strategies` table lifecycle changes
- System: health dashboard aggregation

## Test Requirements

- `tests/unit/autonomous/test_alerting.py`:
  - Test each alert trigger fires when condition met
  - Test cooldown deduplication (same alert within cooldown → not sent)
  - Test cooldown expiry (same alert after cooldown → sent again)
  - Test missing webhook URL → log only, no crash
  - Test Discord payload format matches expected schema

- `tests/unit/autonomous/test_weekly_report.py`:
  - Mock data → verify report generates without error
  - Test with empty trade history (new system) → report still generates
  - Test text formatting produces readable output

## Acceptance Criteria

1. Critical alerts (kill switch, severe drawdown) are never deduplicated (cooldown = 0)
2. Missing `DISCORD_WEBHOOK_URL` degrades gracefully to local logging
3. Weekly report generates even with no trades (shows zero values, not errors)
4. Alert state persists across restarts via DB
5. Discord message format renders correctly in Discord (embeds with color coding)
6. Report is both machine-readable (ReportData) and human-readable (formatted text)
