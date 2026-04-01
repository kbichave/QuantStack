# Deployment Guide

---

## Starting and stopping

The system has one entry point:

```bash
./start.sh           # start everything
./report.sh          # performance summary (can run while system is live)
tmux kill-session -t quantstack-loops   # stop everything
```

`start.sh` is idempotent — re-running it kills any existing session and starts fresh. State is preserved in PostgreSQL; re-starting the loops does not lose positions or fills.

---

## tmux windows

| Window | Process | Restart interval |
|--------|---------|-----------------|
| `trading` | `prompts/trading_loop.md \| claude` | Every 5 min |
| `research` | `prompts/research_loop.md \| claude` | Every 2 min |
| `supervisor` | `quantstack.coordination.supervisor_main` | Continuous |
| `scheduler` | `scripts/scheduler.py` | Continuous |

```bash
# Attach
tmux attach -t quantstack-loops

# Switch windows
Ctrl-b 0   # trading
Ctrl-b 1   # research
Ctrl-b 2   # supervisor
Ctrl-b 3   # scheduler

# Detach without stopping
Ctrl-b d
```

Logs go to `data/logs/{window}.log` (appended, never truncated during a run).

---

## Stateless loops

Each `claude` invocation is a completely fresh session — no `--continue` flag. All inter-iteration state lives in PostgreSQL:

| Table | Purpose |
|-------|---------|
| `loop_iteration_context` | Key-value store per loop (replaces in-session `state[]` dict) |
| `loop_heartbeats` | Iteration metadata; supervisor reads these to detect stalls |
| `positions`, `fills` | Trade state |
| `strategies` | Strategy registry |
| `bugs` | Tool error tracking and auto-patch lifecycle |
| `system_state` | Global key-value (credit regime, AV quota counter, kill switch) |

If a loop crashes mid-iteration, the next invocation picks up from DB — no lost state.

---

## Scheduled jobs

The scheduler (`scripts/scheduler.py`) runs these cron jobs:

| Job | Schedule (ET) | Description |
|-----|--------------|-------------|
| Strategy lifecycle | Sunday 18:00 | Promote/retire strategies by performance |
| Memory compaction | Sunday 17:00 | Trim oversized `.claude/memory/` files |
| Credit regime check | Daily 16:05 | Revalidate HYG/LQD regime from EOD data |
| AV quota reset | Daily 00:01 | Reset Alpha Vantage daily call counter |

Memory compaction also runs at `start.sh` launch time, so files are trimmed before the first iteration.

---

## Data paths

All runtime data lives in PostgreSQL (`TRADER_PG_URL`). File-system paths:

```
data/logs/          # loop output logs
reports/            # AutoResearchClaw output artifacts
reports/autoresclaw/YYYY-MM-DD/<task_id>/
  ├── task_prompt.md
  └── fix_summary.md   # written by ARC for bug_fix tasks
.claude/memory/     # persistent memory files (gitignored)
  ├── workshop_lessons.md
  ├── trade_journal.md
  ├── session_handoffs.md
  └── ...
```

Sentinel files for the kill switch and daily halt:

```
~/.quantstack/KILL_SWITCH_ACTIVE
~/.quantstack/DAILY_HALT_ACTIVE
```

---

## Environment variables

Full annotated list in `.env.example`. Summary:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADER_PG_URL` | — | **Required.** PostgreSQL DSN |
| `ALPACA_API_KEY` | — | **Required.** Alpaca data + execution |
| `ALPACA_SECRET_KEY` | — | **Required.** Alpaca secret |
| `ALPACA_PAPER` | `true` | Paper mode toggle |
| `ALPHA_VANTAGE_API_KEY` | — | **Required.** Primary market data source |
| `AV_DAILY_CALL_LIMIT` | `450` | AV premium plan allows 500/day; 450 leaves 10% buffer |
| `USE_REAL_TRADING` | `false` | Master live trading switch |
| `FORWARD_TESTING_SIZE_SCALAR` | `0.5` | Position size scalar for `forward_testing` strategies |
| `USE_FORWARD_TESTING_FOR_ENTRIES` | `true` | Allow `forward_testing` strategies to place trades |
| `GROQ_API_KEY` | — | Sentiment collector (optional) |
| `AUTORESCLAW_CMD` | `researchclaw` | AutoResearchClaw CLI path override |
| `AUTORESCLAW_TIMEOUT` | `3600` | Max seconds per ARC task |

---

## Alpha Vantage quota management

QuantStack tracks AV calls per day in `system_state` (`av_daily_calls_{YYYY-MM-DD}`). Calls have three priority tiers:

| Priority | Shed threshold | Examples |
|----------|---------------|---------|
| `critical` | Never shed | Options Greeks for open positions, EOD OHLCV for held symbols |
| `normal` | Daily usage > 80% | Signal brief for watchlist, news sentiment |
| `low` | Daily usage > 50% | ML feature refresh for non-watchlist, fundamentals |

When AV is unavailable or quota-exhausted, OHLCV falls back to Alpaca automatically.

---

## Self-healing pipeline

Tool errors flow through a fully automated pipeline:

```
tool raises exception
        ↓
record_tool_error()  →  bugs table (upsert, dedup by fingerprint)
        ↓ (after 3 consecutive failures)
research_queue insert (priority=9, bug_fix task)
        ↓
supervisor bug-fix watcher (polls every 60s)
        ↓
autoresclaw_runner.py --task-id <id>
        ↓
ARC edits src/ directly, writes fix_summary.md
        ↓
_apply_bug_fix(): syntax check → protected-file check → commit
        ↓
_update_bug_status(): bugs → fixed, research_queue → done
        ↓
_restart_loops_after_fix(): tmux send-keys C-c + restart
```

Protected files (`risk_gate.py`, `kill_switch.py`, `db.py`) are never auto-patched. Low-confidence or human-review-flagged fixes are reverted and noted in `session_handoffs.md`.

---

## Utility scripts

Located in `scripts/`.

### `scheduler.py`

The APScheduler process. Run automatically by `start.sh` in the `scheduler` tmux window. Can also be run standalone for testing:

```bash
python scripts/scheduler.py --dry-run
```

### `autoresclaw_runner.py`

AutoResearchClaw dispatcher. Normally invoked by the supervisor, but can be run manually:

```bash
# Process top 3 pending tasks
python scripts/autoresclaw_runner.py

# Dry run (print prompts, no execution)
python scripts/autoresclaw_runner.py --dry-run

# Run a specific task by ID
python scripts/autoresclaw_runner.py --task-id <uuid>
```

### `log_decision.py`

Append a manual decision to the audit trail:

```bash
python scripts/log_decision.py \
  --symbol AAPL \
  --action BUY \
  --reasoning "Breaking out of 6-month consolidation" \
  --confidence 0.75
```

---

## Running tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only (no network, fast)
uv run pytest tests/unit/ -v

# With coverage
uv run pytest tests/ --cov=src/quantstack
```
