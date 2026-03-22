# scripts/

Operational scripts for QuantStack. No business logic — all trading logic is in `src/quantpod/`.

## Loops

| Script | Purpose | Usage |
|--------|---------|-------|
| `start_research_loop.sh` | Research loop — strategy discovery, ML training, optimization | `FORCE_LOOPS=1 ./scripts/start_research_loop.sh` |
| `start_trading_loop.sh` | Trading loop — position monitoring, entry scanning, execution | `FORCE_LOOPS=1 ./scripts/start_trading_loop.sh` |

Both are **PAUSED** until P&L attribution is built. Each runs in its own tmux session (`quantstack-research` / `quantstack-trading`).

## Scheduler

| Script | Purpose | Usage |
|--------|---------|-------|
| `scheduler.py` | Cron daemon — triggers Claude Code sessions at market times (09:15, 12:30, 15:45, 17:00 Fri) | `python scripts/scheduler.py [--dry-run]` |

## Data

| Script | Purpose | Usage |
|--------|---------|-------|
| `acquire_historical_data.py` | CLI wrapper around `quantpod.data.AcquisitionPipeline` | `python scripts/acquire_historical_data.py [--symbols SPY QQQ]` |

Bootstrap: `quantpod-bootstrap [--symbols SPY QQQ]` (registered entry point, see `src/quantpod/flows/bootstrap.py`).

## Hooks (in `src/quantpod/hooks/`)

PostToolUse hooks registered as CLI entry points:
- `quantpod-log-decision` — appends trades to trade_journal.md (after `execute_trade`)
- `quantpod-validate-brief` — warns on low signal quality (after `run_analysis`)

## Docker

| Script | Purpose |
|--------|---------|
| `docker-entrypoint.sh` | Container entry point — API server or shell |
