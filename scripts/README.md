# scripts/

Operational scripts for QuantStack. No business logic — all trading logic is in `src/quantstack/`.

## Autonomous Loops

**Use the Ralph Wiggum skill in Claude Code to start loops:**

```
/ralph-loop prompts/research_loop.md    # Strategy discovery, ML training, optimization
/ralph-loop prompts/trading_loop.md     # Position monitoring, entry scanning, execution
/cancel-ralph                            # Stop active loop
```

Each runs in its own tmux session managed by the Ralph skill.

## Scheduler

| Script | Purpose | Usage |
|--------|---------|-------|
| `scheduler.py` | Cron daemon — triggers Claude Code sessions at market times (09:15, 12:30, 15:45, 17:00 Fri) | `python scripts/scheduler.py [--dry-run]` |

## Data

| Script | Purpose | Usage |
|--------|---------|-------|
| `acquire_historical_data.py` | CLI wrapper around `quantstack.data.AcquisitionPipeline` | `python scripts/acquire_historical_data.py [--symbols SPY QQQ]` |

Bootstrap: `quantstack-bootstrap [--symbols SPY QQQ]` (registered entry point, see `src/quantstack/flows/bootstrap.py`).

## Hooks (in `src/quantstack/hooks/`)

PostToolUse hooks registered as CLI entry points:
- `quantstack-log-decision` — appends trades to trade_journal.md (after `execute_trade`)
- `quantstack-validate-brief` — warns on low signal quality (after `run_analysis`)

## Docker

| Script | Purpose |
|--------|---------|
| `docker-entrypoint.sh` | Container entry point — API server or shell |
