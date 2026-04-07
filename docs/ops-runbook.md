# QuantStack Ops Runbook

Quick reference for debugging, monitoring, and operating the autonomous trading system.
Written for future Claude sessions and agents that start with zero context.

---

## Architecture at a Glance

```
start.sh
  |
  +--> tmux: quantstack-loops
         |
         +-- research   : claude session every 2min (prompts/research_loop.md)
         +-- trading    : claude session every 5min (prompts/trading_loop.md)
         +-- supervisor : monitors heartbeats, kills orphans
         +-- scheduler  : APScheduler daemon (scripts/scheduler.py)
```

Each loop is a FRESH `claude` CLI invocation. No `--continue`. State survives via PostgreSQL and `.claude/memory/` files.

---

## Strategy Lifecycle

```
draft ──[scheduler: strategy_pipeline_10m]──> backtested ──[research loop: Step 2e / strategy-rd]──> forward_testing
                                                                                                  |
                                                                                                  +──> retired (rejected)

forward_testing ──[AutoPromoter: min 21d, 15 trades, Sharpe>0.5]──> live

live ──[monthly lifecycle: degradation check]──> retired
```

| Status | Meaning | Set By |
|--------|---------|--------|
| `draft` | Registered, no backtest yet | Research loop, AlphaDiscoveryEngine |
| `backtested` | backtest_summary populated | `strategy_pipeline` scheduler job |
| `forward_testing` | Paper trading, drift baseline created | Research loop Step 2e (strategy-rd agent) |
| `live` | Active trading | AutoPromoter (21+ days forward testing) |
| `retired` | End of life | Monthly lifecycle, strategy-rd REJECT |
| `failed` | Deprecated status | Legacy |
| `design_only` | Options/special strategies not yet backtestable | Manual |
| `paper` / `paper_testing` / `designed` | Legacy statuses from old system | Should be migrated |

### Promotion Thresholds (strategy_lifecycle.py)

```
_MIN_OOS_SHARPE      = 0.5    # walk-forward OOS Sharpe
_MAX_OVERFIT_RATIO   = 2.0    # IS/OOS ratio
_FORWARD_TEST_DAYS   = 30     # minimum paper trading duration
_MIN_LIVE_SHARPE     = 0.3    # live performance threshold
```

AutoPromoter (forward_testing -> live):
```
min_forward_test_days    = 21
min_forward_test_trades  = 15
min_live_sharpe          = 0.5
max_degradation_vs_bt    = 0.40
min_win_rate             = 0.40
max_max_drawdown         = 0.08
max_concurrent_live      = 8
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/scheduler.py` | All deterministic scheduled jobs (no LLM) |
| `scripts/acquire_historical_data.py` | Data refresh (12 phases, AV + options + macro) |
| `src/quantstack/autonomous/strategy_lifecycle.py` | Weekly/monthly lifecycle + pipeline pass |
| `src/quantstack/coordination/auto_promoter.py` | forward_testing -> live promotion |
| `src/quantstack/execution/risk_gate.py` | Hard risk limits (NEVER modify) |
| `src/quantstack/mcp/tools/_impl.py` | Core tool implementations (backtest, strategy CRUD) |
| `src/quantstack/strategies/signal_generator.py` | Rule engine: entry/exit signal generation |
| `src/quantstack/db.py` | DB pool, schema, migrations, `pg_conn()` / `db_conn()` |
| `prompts/research_loop.md` | Research loop prompt (BLITZ, Step 2e promotion) |
| `prompts/trading_loop.md` | Trading loop prompt |
| `prompts/reference/python_toolkit.md` | All available Python imports |
| `.claude/agents/*.md` | Agent definitions (strategy-rd, quant-researcher, etc.) |

---

## Diagnostic Queries

Run any of these via `python3 -c "..."` or `python3 << 'EOF'`.

### Strategy Status Breakdown

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("SELECT status, COUNT(*) FROM strategies GROUP BY status ORDER BY COUNT(*) DESC").fetchall()
for r in rows: print(f"  {r[0]}: {r[1]}")
conn.close()
```

### Loop Health (are loops running?)

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("""
    SELECT loop_name, iteration, status, started_at, finished_at,
           EXTRACT(EPOCH FROM (NOW() - started_at))/60 as minutes_ago
    FROM loop_heartbeats
    ORDER BY started_at DESC LIMIT 15
""").fetchall()
for r in rows:
    print(f"  {r[0]:20} | iter={r[1]:3} | {r[2]:10} | {r[5]:.0f}min ago")
conn.close()
```

### Stuck/Orphaned Loops

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("""
    SELECT loop_name, iteration, started_at
    FROM loop_heartbeats
    WHERE status = 'running' AND finished_at IS NULL
      AND started_at < NOW() - INTERVAL '30 minutes'
""").fetchall()
print(f"Orphaned runs: {len(rows)}")
for r in rows: print(f"  {r[0]} iter={r[1]} started={r[2]}")
conn.close()
```

### Draft Strategies Missing Data (why pipeline skips them)

```python
from quantstack.db import open_db
import json
conn = open_db()
rows = conn.execute("SELECT strategy_id, name, symbol, parameters, entry_rules FROM strategies WHERE status='draft'").fetchall()
for r in rows:
    sid, name, sym, params, entry = r
    issues = []
    if sid is None: issues.append("NULL strategy_id")
    if sym is None: issues.append("NULL symbol")
    if params is None: issues.append("NULL parameters")
    if entry is None: issues.append("NULL entry_rules")
    elif isinstance(entry, str):
        try:
            parsed = json.loads(entry) if isinstance(entry, str) else entry
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                issues.append("plain-text rules (not dict format)")
        except: issues.append("invalid JSON entry_rules")
    if issues:
        print(f"  {sid or name}: {issues}")
conn.close()
```

### Backtested Strategies Awaiting Promotion

```python
from quantstack.db import open_db
import json
conn = open_db()
rows = conn.execute("""
    SELECT strategy_id, name, symbol, backtest_summary, updated_at
    FROM strategies WHERE status = 'backtested' AND symbol IS NOT NULL
    ORDER BY updated_at ASC
""").fetchall()
print(f"Awaiting promotion review: {len(rows)}")
for r in rows:
    sid, name, sym, bs, updated = r
    sharpe = 0
    if bs:
        bs_dict = json.loads(bs) if isinstance(bs, str) else bs
        sharpe = bs_dict.get("sharpe_ratio", 0) if bs_dict else 0
    print(f"  {sid[:30]:32} | {sym:6} | Sharpe={sharpe:.2f} | updated={updated}")
conn.close()
```

### Pipeline Job History

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("""
    SELECT iteration, status, started_at, finished_at, symbols_processed, errors
    FROM loop_heartbeats
    WHERE loop_name = 'strategy_pipeline'
    ORDER BY started_at DESC LIMIT 10
""").fetchall()
for r in rows:
    print(f"  iter={r[0]} | {r[1]:10} | processed={r[4]} errors={r[5]} | {r[2]}")
conn.close()
```

### Open Positions

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("SELECT symbol, quantity, avg_cost, side, current_price, unrealized_pnl FROM positions").fetchall()
for r in rows: print(f"  {r[0]:6} | qty={r[1]} | cost={r[2]} | side={r[3]} | price={r[4]} | pnl={r[5]}")
conn.close()
```

### Recent Events (cross-loop communication)

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("""
    SELECT event_type, source_loop, payload, created_at
    FROM loop_events
    ORDER BY created_at DESC LIMIT 10
""").fetchall()
for r in rows: print(f"  {r[0]:30} | {r[1]:15} | {r[3]}")
conn.close()
```

### System State (kill switch, credit regime, etc.)

```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("SELECT key, value, updated_at FROM system_state ORDER BY key").fetchall()
for r in rows: print(f"  {r[0]:40} = {str(r[1])[:60]:60} | {r[2]}")
conn.close()
```

---

## Common Failure Modes

### 1. Strategies stuck as `draft`

**Symptom:** `strategy_pipeline` runs but strategies don't move to `backtested`.

**Check:**
```python
# Run the "Draft Strategies Missing Data" query above
```

**Root causes:**
| Cause | Fix |
|-------|-----|
| `symbol IS NULL` | Research loop must set symbol when registering |
| `strategy_id IS NULL` | Orphaned row. Delete: `DELETE FROM strategies WHERE strategy_id IS NULL` |
| `parameters IS NULL` | Bug in `_impl.py` handled (uses `or {}`). Should backtest now. |
| Plain-text entry_rules (e.g. `"FCF_yield > 3%"`) | Invalid format. Delete and let research re-register with dict rules. |
| Double-encoded JSON | Bug in `_impl.py` handled (double json.loads). Should parse now. |
| Options strategy (exit_rules is dict not list) | Mark as `design_only`. Needs options_engine. |

### 2. Strategies stuck as `backtested` (not promoting to forward_testing)

**Symptom:** 56 backtested strategies, none moving.

**Root causes:**
| Cause | Fix |
|-------|-----|
| Research loop not running | Check heartbeats. If last heartbeat > 10min ago, loops are down. Check tmux. |
| Research loop running but Step 2e not executing | Step 2e is in `prompts/research_loop.md`. The Claude session may deprioritize it. Check the research loop tmux pane output for "Strategy Promotion Review". |
| Walk-forward fails for all candidates | Check if `walkforward_service.run_walkforward()` works: test with one strategy manually. |
| All candidates rejected by strategy-rd | Check `workshop_lessons.md` for REJECT verdicts. Lower the bar or improve strategy quality. |

**Manual promotion test (one-shot):**
```bash
# Run backtest + walk-forward on one strategy to verify the pipeline works end-to-end
python3 -c "
import asyncio
from quantstack.core.backtesting.walkforward_service import run_walkforward
result = asyncio.run(run_walkforward(strategy_id='SOME_STRATEGY_ID', symbol='SPY', n_splits=5, test_size=63, min_train_size=126, use_purged_cv=True))
print(result)
"
```

### 3. Loops not running (tmux shows no output)

**Symptom:** `status.sh` says tmux running but no heartbeats for 10+ minutes.

**Debug:**
```bash
tmux attach -t quantstack-loops    # Check each pane
# Ctrl-B then number (0-3) to switch panes
```

**Common causes:**
- Claude CLI rate-limited (check for 429 errors in pane output)
- API key expired or missing
- DB connection pool exhausted (check for "pool timeout" in logs)
- Python import error breaking the session (check for tracebacks)

**Fix:** Kill and restart:
```bash
tmux kill-session -t quantstack-loops
./start.sh
```

### 4. Backtest returns 0 trades

**Symptom:** Strategy backtests successfully but `total_trades: 0, sharpe: 0.0`.

**Root causes:**
- Entry rules reference indicators not in the DataFrame (check warnings for "indicator X is not in df.columns")
- All rules return False because data doesn't match conditions
- Missing enrichment: `hmm_regime`, `hmm_stability`, `gex`, `institutional_accumulation` are not standard indicators

**Debug:**
```python
# Check which indicators a strategy needs vs what's available
from quantstack.strategies.signal_generator import enrich_with_indicators
import pandas as pd
from quantstack.data.pg_storage import PgDataStore
store = PgDataStore()
df = store.read_ohlcv("SPY", "1D", limit=500)
df = enrich_with_indicators(df, {})
print("Available indicators:", sorted(df.columns.tolist()))
```

### 5. `'NoneType' object does not support item assignment` in backtest

**Cause:** Strategy's `parameters` column is NULL in DB. Fixed in `_impl.py` (uses `or {}`).

**If it recurs:** The research loop is registering strategies without parameters. Check `register_strategy_impl()` call.

### 6. `can only concatenate list (not "dict") to list` in backtest

**Cause:** Options strategy exit_rules stored as a dict instead of a list. The standard backtest engine expects `[rule1, rule2, ...]` not `{legs: [...]}`.

**Fix:** Mark as `design_only` — options strategies need the options_engine, not the standard equity engine.

### 7. Signal generator warnings: `indicator 'X' is not in df.columns`

**Common missing indicators and why:**
| Indicator | Why Missing | Resolution |
|-----------|-------------|-----------|
| `hmm_regime` | HMM model not trained/loaded | Train HMM or remove from rules |
| `hmm_stability` | Same as above | Same |
| `gex` | Gamma exposure not computed (needs options chain) | Needs enrichment pipeline |
| `institutional_accumulation` | Requires external data | Needs enrichment pipeline |
| `earnings_surprise` | Fundamental data not in OHLCV | Needs enrichment pipeline |
| `guidance_raise` | Same | Same |
| Empty string `''` | Malformed rule with `"indicator": ""` | Delete strategy, re-register |

---

## Scheduler Jobs

| Label | Schedule | Function | Purpose |
|-------|----------|----------|---------|
| `strategy_pipeline_10m` | `*/10` always | `run_strategy_pipeline` | Backtest drafts -> backtested |
| `av_intraday_5min_5m` | `*/5 9-16 Mon-Fri` | `run_av_intraday_5min` | 5-min OHLCV bars |
| `intraday_quote_refresh` | `*/15 9-16 Mon-Fri` | `run_intraday_quote_refresh` | Position quotes via Alpaca |
| `credit_regime_intraday_2h` | `10,12,14 Mon-Fri` | `run_credit_regime_revalidation` | Intraday credit check |
| `data_refresh_08:00` | `8:00 Mon-Fri` | `run_data_refresh` | Full 12-phase data refresh |
| `eod_data_refresh_16:30` | `16:30 Mon-Fri` | `run_eod_data_refresh` | EOD close + options/news |
| `daily_attribution_16:10` | `16:10 Mon-Fri` | `run_daily_attribution` | Equity snapshot + P&L |
| `credit_regime_eod_16:45` | `16:45 Mon-Fri` | `run_credit_regime_revalidation` | EOD credit check |
| `memory_compaction_sun17:00` | `17:00 Sun+Wed` | `run_memory_compaction` | Trim memory files |
| `strategy_lifecycle_weekly` | `18:00 Sun` | `run_strategy_lifecycle_weekly` | Gap analysis + template generation |
| `community_intel_weekly` | `19:00 Sun` | `run_community_intel_weekly` | Reddit/GitHub/arXiv scan |
| `autoresclaw_weekly` | `20:00 Sun` | `run_autoresclaw_weekly` | Deep research tasks |
| `strategy_lifecycle_monthly` | `9:00 1st` | `run_strategy_lifecycle_monthly` | Validate live, retire degraded |

Run any job manually:
```bash
python3 scripts/scheduler.py --run-now strategy_pipeline
python3 scripts/scheduler.py --dry-run   # see schedule without running
```

---

## DB Schema Quick Reference

### strategies

```sql
strategy_id         TEXT PRIMARY KEY
name                TEXT UNIQUE NOT NULL
symbol              TEXT              -- ticker (backfilled from name)
status              TEXT DEFAULT 'draft'
parameters          JSONB NOT NULL    -- indicator settings
entry_rules         JSONB NOT NULL    -- [{indicator, condition, value, type}]
exit_rules          JSONB NOT NULL    -- same format
risk_params         JSONB             -- {stop_loss_atr, take_profit_atr}
backtest_summary    JSONB             -- populated by run_backtest
walkforward_summary JSONB             -- populated by run_walkforward
regime_affinity     JSONB             -- {regime: confidence} or [regime, ...]
instrument_type     TEXT DEFAULT 'equity'  -- equity | options | multi_leg
time_horizon        TEXT DEFAULT 'swing'   -- intraday | swing | position | investment
holding_period_days INTEGER DEFAULT 5
source              TEXT DEFAULT 'manual'  -- manual | generated | evolved | decoded
created_at, updated_at  TIMESTAMPTZ
```

Indices: `strategies_status_idx`, `strategies_symbol_idx`

### Entry/exit rule format

```json
[
  {"indicator": "rsi_14", "condition": "below", "value": 30, "type": "prerequisite"},
  {"indicator": "adx", "condition": "above", "value": 25, "type": "confirmation"},
  {"indicator": "close", "condition": "within_pct", "value": "sma_200", "pct_range": 3.0}
]
```

Supported conditions: `above`, `below`, `greater_than`, `less_than`, `crosses_above`, `crosses_below`, `within_pct`, `between`, `equals`, `in`, `not_in`.

Values can be numeric (`30`) or column references (`"sma_200"`, `"sma_50"`).

### loop_heartbeats

```sql
loop_name       TEXT NOT NULL
iteration       INTEGER NOT NULL
started_at      TIMESTAMPTZ NOT NULL
finished_at     TIMESTAMPTZ
symbols_processed  INTEGER DEFAULT 0
errors          INTEGER DEFAULT 0
status          TEXT DEFAULT 'running'  -- running | completed | failed | orphaned
PRIMARY KEY (loop_name, iteration)
```

### Key tables for debugging

| Table | When to check |
|-------|---------------|
| `strategies` | Strategy not progressing, missing data |
| `loop_heartbeats` | Loops not running, stale sessions |
| `system_state` | Kill switch, credit regime, AV counter |
| `positions` | Open position issues |
| `equity_alerts` | Trading loop entry signals |
| `research_wip` | Duplicate research prevention |
| `bugs` | Auto-detected tool failures |
| `research_queue` | Pending AutoResearchClaw tasks |
| `loop_events` | Cross-loop communication |
| `loop_iteration_context` | Per-loop key-value state |

---

## Known Bugs Fixed in This Session (2026-04-01)

1. **`_impl.py` NULL parameters**: `strat.get("parameters", {})` returns `None` when column is NULL. Fixed: `strat.get("parameters") or {}`.

2. **`_impl.py` double-encoded JSON**: Some strategies have `'"[{...}]"'` (JSON string wrapping JSON). Fixed: after `json.loads`, if result is still a string, parse again.

3. **`signal_generator.py` `within_pct` + column reference**: `float("sma_200")` crashed. Fixed: moved `_resolve()` before `within_pct` check, made it use `pct_range`/`tolerance` parameter.

4. **`signal_generator.py` `in`/`not_in` with list values**: No handler for list membership in general path. Fixed: added `series.isin(value)` before `_resolve`.

5. **`signal_generator.py` `credit_regime` DB read**: Used `conn.cursor()` which doesn't exist on `PgConnection`. Fixed: use `conn.execute()` directly.

---

## Recovery Procedures

### Full restart
```bash
tmux kill-session -t quantstack-loops 2>/dev/null
./start.sh
```

### Drain backtested strategies manually (if research loop isn't promoting)
```bash
python3 scripts/scheduler.py --run-now strategy_pipeline
# Then check: strategies should move draft -> backtested
# For backtested -> forward_testing, research loop Step 2e must run
```

### Force-promote a specific strategy
```python
from quantstack.db import open_db
conn = open_db()
conn.execute("UPDATE strategies SET status='forward_testing', updated_at=NOW() WHERE strategy_id=%s", ['STRATEGY_ID_HERE'])
conn.commit()
conn.close()
```

### Check what the trading loop would pick up
```python
from quantstack.db import open_db
conn = open_db()
rows = conn.execute("SELECT strategy_id, name, symbol FROM strategies WHERE status IN ('forward_testing', 'live')").fetchall()
for r in rows: print(f"  {r[0]:30} | {r[1]:30} | {r[2]}")
conn.close()
```

---

## Credential Rotation

Rotate credentials periodically (quarterly minimum). Steps:

1. **Generate new credentials** at the provider (Alpaca, Alpha Vantage, etc.)
2. **Update `.env`** with the new values — keep the old values commented above as backup
3. **Restart services**: `./stop.sh && ./start.sh`
4. **Verify**: `./status.sh` — all services should be healthy
5. **Revoke old credentials** at the provider once confirmed working

**Critical order**: update `.env` *before* revoking old keys. If new keys fail validation, you can revert instantly.

**Env var validation**: `start.sh` validates all required env vars on boot. If a rotated key is malformed or missing, the system will refuse to start with a clear error message identifying which variable failed.

| Credential | Provider | Rotation URL |
|------------|----------|--------------|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Alpaca | alpaca.markets → Paper Trading → API Keys |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | alphavantage.co → My Account |
| `POLYGON_API_KEY` | Polygon | polygon.io → Dashboard → API Keys |
| `GROQ_API_KEY` | Groq | console.groq.com → API Keys |

---

## Database Backup & Restore

### Backup

Automated backups run via `scripts/backup.sh`. The script uses `pg_dump` in custom format (compressed, supports selective restore), verifies dump integrity with `pg_restore --list`, and prunes backups older than 30 days. An `flock`-based lock prevents concurrent runs.

```bash
# Manual backup
./scripts/backup.sh

# Scheduled (add to crontab or scheduler)
# 0 2 * * * /path/to/scripts/backup.sh >> /var/log/quantstack_backup.log 2>&1
```

Backups are stored in the `quantstack-backups` Docker volume, mounted at `/data/quantstack/backups` inside the postgres container.

### Full Restore from pg_dump

```bash
# Stop all services
./stop.sh

# Restore from most recent dump
pg_restore --dbname=quantstack --clean --if-exists \
    /data/quantstack/backups/quantstack_YYYY-MM-DD.dump

# Verify key tables
psql quantstack -c "SELECT 'positions' AS tbl, COUNT(*) FROM positions
    UNION ALL SELECT 'orders', COUNT(*) FROM orders
    UNION ALL SELECT 'strategies', COUNT(*) FROM strategies;"

# Restart
./start.sh
```

### Point-in-Time Recovery (PITR)

Use PITR when you need to restore to a specific moment (e.g., right before a bad trade or data corruption). Requires WAL archiving to be enabled in `postgresql.conf` (`archive_mode = on`).

```bash
# Stop postgres only
docker-compose stop postgres

# Restore base backup
pg_restore --dbname=quantstack --clean --if-exists \
    /data/quantstack/backups/quantstack_YYYY-MM-DD.dump

# Create recovery signal and configure recovery target
touch /var/lib/postgresql/data/recovery.signal

# Set in postgresql.auto.conf:
# restore_command = 'cp /data/quantstack/wal_archive/%f %p'
# recovery_target_time = 'YYYY-MM-DD HH:MM:SS UTC'
# recovery_target_action = 'promote'

# Start postgres — it will replay WAL up to the target time
docker-compose start postgres
```

After PITR completes, verify data integrity with the key tables query above, then restart all services with `./start.sh`.
