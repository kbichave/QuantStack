---
name: get-alerts
description: View, filter, and manage equity/investment alerts — entry signals, exit signals, thesis updates, and status lifecycle.
user_invocable: true
---

# /get-alerts — Equity & Investment Alert Dashboard

## Purpose

View and manage equity alerts created by the research loop. Alerts are entry
opportunities with full thesis, pricing, and risk analysis. The trading loop
monitors them for exit conditions. You use this skill to review, act on, or
dismiss alerts.

**Cadence:** On demand. Run before `/review` or `/trade` to see pending opportunities.

## Arguments

| Usage | What it does |
|-------|-------------|
| `/get-alerts` | Show all pending + watching alerts |
| `/get-alerts AAPL` | Filter by symbol |
| `/get-alerts investment` | Filter by time horizon |
| `/get-alerts swing` | Filter by time horizon |
| `/get-alerts acted` | Show alerts you've acted on |
| `/get-alerts expired` | Show expired/closed alerts |
| `/get-alerts detail 12` | Full history for alert #12 |

## Workflow

### Step 0: Parse Arguments

Determine the view mode from the user's input:
- No args or `all` -> fetch pending + watching (default dashboard)
- A ticker symbol (all caps, 1-5 chars) -> filter by symbol
- `investment` / `swing` / `position` -> filter by time_horizon
- `pending` / `watching` / `acted` / `expired` / `skipped` -> filter by status
- `detail N` -> single alert deep-dive with full timeline

### Step 1: System Check

```python
get_system_status()  # confirm MCP is operational
```

### Step 2: Fetch Alerts

**Default dashboard:**
```python
pending = get_equity_alerts(status="pending", include_updates=True, include_exit_signals=True)
watching = get_equity_alerts(status="watching", include_updates=True, include_exit_signals=True)
alerts = pending["alerts"] + watching["alerts"]
```

**Filtered view:**
```python
alerts = get_equity_alerts(symbol="AAPL", include_updates=True, include_exit_signals=True)
# or
alerts = get_equity_alerts(time_horizon="investment", include_updates=True, include_exit_signals=True)
# or
alerts = get_equity_alerts(status="acted", include_updates=True, include_exit_signals=True)
```

**Detail view:**
```python
result = get_equity_alerts(alert_id=N, include_updates=True, include_exit_signals=True)
alert = result["alerts"][0]
```

### Step 3: Display Summary Table

```
## Active Alerts — {date}

| ID | Symbol | Action | Horizon | Confidence | Entry | Stop | Target | R:R | Status | Urgency | Age |
|----|--------|--------|---------|------------|-------|------|--------|-----|--------|---------|-----|
| 12 | AAPL   | buy    | invest  | 82%        | $175  | $155 | $210   | 1.8 | watching | this_week | 5d |
| 15 | NVDA   | buy    | swing   | 74%        | $890  | $860 | $950   | 2.0 | pending  | today    | 1d |
```

For each alert, show a one-line thesis summary (first 100 chars of `thesis` field).

### Step 4: Display Exit Signals (if any)

If any alert has unacknowledged exit signals, highlight them:

```
## Exit Signals (requires attention)

| Alert | Symbol | Signal | Severity | Headline | Recommended |
|-------|--------|--------|----------|----------|-------------|
| 12    | AAPL   | fundamental_deterioration | warning | F-Score dropped 8 to 5 | tighten_stop |
```

For `critical` or `auto_close` severity, display prominently with reasoning.

### Step 5: Display Timeline (detail view only)

For `/get-alerts detail N`, show the full narrative:

```
## Alert #12: AAPL — Buy (Investment)

**Thesis:** {full thesis text}
**Key Risks:** {key_risks}
**Strategy:** {strategy_name} ({strategy_id})
**Catalyst:** {catalyst}
**Regime at Entry:** {regime}

### Fundamentals
| Metric | Value |
|--------|-------|
| Piotroski F-Score | 8 |
| FCF Yield | 5.2% |
| P/E | 18.2 |
| Analyst Consensus | buy |

### Price Levels
| Level | Price | Distance |
|-------|-------|----------|
| Current | $182 | — |
| Entry | $175 | -3.8% |
| Stop | $155 | -14.8% |
| Target | $210 | +15.4% |
| R:R | 1.8:1 | — |

### Timeline

| Date | Type | Thesis | Commentary |
|------|------|--------|------------|
| Mar 22 | price_update | intact | Holding above 50d MA, volume confirming... |
| Mar 20 | fundamental_update | intact | Q4 earnings beat, revenue +12% YoY... |
| Mar 18 | thesis_check | intact | Initial alert. Strong quality metrics... |

### Exit Signals

| Date | Type | Severity | Headline | Recommendation |
|------|------|----------|----------|----------------|
| (none yet) |
```

### Step 6: Interactive Actions

After displaying, prompt the user with available actions:

```
## Actions

- `act 12` — Mark alert #12 as acted (you entered the position)
- `watch 12` — Move to watching (monitoring but not entered yet)
- `skip 12` — Skip this alert (not interested)
- `expire 12` — Expire this alert (opportunity passed)
- `note 12 <text>` — Add a personal note to the alert
```

Execute the chosen action:

| Command | MCP Call |
|---------|----------|
| `act ID` | `update_alert_status(alert_id=ID, status="acted", status_reason="User confirmed entry")` |
| `watch ID` | `update_alert_status(alert_id=ID, status="watching", status_reason="Monitoring thesis")` |
| `skip ID` | `update_alert_status(alert_id=ID, status="skipped", status_reason="User skipped")` |
| `expire ID` | `update_alert_status(alert_id=ID, status="expired", status_reason="Opportunity passed")` |
| `note ID text` | `add_alert_update(alert_id=ID, update_type="user_note", commentary=text)` |

### Step 7: Update Memory

If any alerts were acted on, expired, or had significant exit signals:
- Note in `.claude/memory/session_handoffs.md`
- If acted, add entry to `.claude/memory/trade_journal.md` under `## Equity Alerts`

## Notes

- Alerts are created by the research loop, not the trading loop
- The trading loop monitors price/regime and writes exit signals
- The research loop writes fundamental updates and thesis checks
- This skill is read-heavy — safe to run anytime without side effects (unless you take actions in Step 6)
- If no alerts exist, suggest running the research loop: `./scripts/start_research_loop.sh investment`
