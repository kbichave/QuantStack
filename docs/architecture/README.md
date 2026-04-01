# QuantStack Architecture

---

## Single entry point

Everything starts from `./start.sh`. There is no other way to start the system. The script handles prerequisites, migrations, universe bootstrap, preflight checks, seeds community intelligence (background), and launches five tmux windows. See [../guides/quickstart.md](../guides/quickstart.md).

---

## Repository structure

```
QuantStack/
├── start.sh                          # Single entry point
├── report.sh                         # Performance summary
├── prompts/
│   ├── trading_loop.md               # Trading loop prompt — Claude reads each iteration
│   ├── research_loop.md              # Research loop prompt
│   └── reference/                    # python_toolkit.md, trading_rules.md, etc.
├── src/quantstack/
│   ├── coordination/                 # Supervisor, auto-promoter, preflight
│   │   ├── supervisor.py             # Heartbeat monitor + bug-fix watcher thread
│   │   ├── supervisor_main.py        # Entry point for supervisor tmux window
│   │   └── auto_promoter.py         # forward_testing → live promotion
│   ├── data/
│   │   ├── fetcher.py                # Alpha Vantage client (daily quota guard, priority tiers)
│   │   └── factory.py               # Provider routing + Alpaca OHLCV fallback
│   ├── execution/
│   │   ├── risk_gate.py              # IMMUTABLE — hard-coded pre-trade checks
│   │   ├── kill_switch.py            # Emergency halt (DB sentinel)
│   │   └── broker_routers.py         # Alpaca, PaperBroker
│   ├── mcp/tools/                    # Python toolkit — imported directly in loop prompts
│   │   ├── coordination.py           # record_heartbeat, get/set_loop_context, record_tool_error
│   │   ├── signal.py                 # run_multi_signal_brief
│   │   ├── execution.py              # execute_trade, get_portfolio_state
│   │   └── ...
│   ├── signal_engine/                # 15 concurrent collectors, no LLM, 2–6s
│   ├── core/                         # Indicators, backtesting, ML, options pricing
│   ├── api/                          # FastAPI REST (optional, not required for loops)
│   └── db.py                         # PostgreSQL connection + all migrations
├── scripts/
│   ├── scheduler.py                  # APScheduler cron jobs
│   └── autoresclaw_runner.py         # ARC dispatcher + auto-patch pipeline
├── .claude/
│   ├── agents/                       # Desk agent definitions (trade-debater, risk, etc.)
│   ├── agents/
│   │   ├── community-intel.md        # Weekly quant community discovery
│   │   ├── market-intel.md           # Real-time trading intelligence
│   │   └── ...                       # trade-debater, risk, fund-manager, etc.
│   └── memory/                       # Persistent memory (gitignored)
└── docs/
```

---

## System diagram

```
                        ./start.sh
                            │
          ┌─────────────────┼─────────────────────┐
          ▼                 ▼                     ▼
  ┌──────────────┐  ┌──────────────┐   ┌────────────────────┐
  │   trading    │  │  research    │   │  supervisor        │
  │  (tmux win) │  │  (tmux win) │   │  (tmux win)        │
  │              │  │              │   │                    │
  │ fresh claude │  │ fresh claude │   │ heartbeat monitor  │
  │ every 5 min  │  │ every 2 min  │   │ bug-fix watcher    │
  │              │  │              │   │                    │
  │  spawns:     │  │  spawns:     │   │  ┌──────────────┐  │
  │  position-   │  │  quant-      │   │  │  scheduler   │  │
  │  monitor     │  │  researcher  │   │  │  (tmux win)  │  │
  │  trade-      │  │  ml-scientist│   │  │  cron jobs   │  │
  │  debater     │  │  strategy-rd │   │  └──────────────┘  │
  │  risk        │  │  (BLITZ mode)│   └────────────────────┘
  │  fund-mgr    │  │  community-  │
  └──────┬───────┘  │  intel (10th │
         │          │  iter, AH)   │
         │          └──────┬───────┘
         │                 │
         └────────┬────────┘
                  ▼
        ┌──────────────────────────────────┐
        │          PostgreSQL               │
        │                                  │
        │  positions       loop_heartbeats  │
        │  fills           loop_iteration_  │
        │  strategies        context        │
        │  audit_log       bugs             │
        │  research_queue  system_state     │
        │  universe        ml_experiments   │
        └──────────────────────────────────┘
```

---

## Stateless loop design

Neither loop accumulates in-session state. Each Claude invocation:

1. Reads current context from `loop_iteration_context` (PostgreSQL)
2. Does its work (signals, debate, trades, or research)
3. Writes updated context back to `loop_iteration_context`
4. Exits — the `while :; do ... sleep N; done` wrapper in tmux starts a fresh invocation

**Why no `--continue`:** Claude sessions accumulate context. By day 3–4 of a continuous run, the context window fills and loop steps get silently skipped. Stateless invocations eliminate this completely.

**Context keys per loop:**

| Loop | Key | Purpose |
|------|-----|---------|
| `trading_loop` | `market_intel` | Cached market intelligence (25-min TTL) |
| `trading_loop` | `stale_symbols` | Symbols with stale OHLCV (set each iter) |
| `trading_loop` | `closes_since_review` | Counter triggering weekly trade-reflector |
| `research_loop` | `last_domain` | Last research domain processed |
| `research_loop` | `domain_history` | Rolling 50-entry domain history |
| `research_loop` | `last_execution_audit_at` | Date of last execution-researcher spawn |

---

## Self-healing pipeline

```
loop prompt calls a tool
        │
  exception raised?
        │ yes
        ▼
record_tool_error(tool_name, error, stack_trace, loop_name)
        │
        ▼
  bugs table (UPSERT — dedup by tool_name + loop_name + error_fingerprint)
        │
  consecutive_errors >= 3?
        │ yes
        ▼
research_queue INSERT (task_type='bug_fix', priority=9)
bugs.arc_task_id ← new task_id
        │
        ▼
supervisor._bug_fix_watcher (polls every 60s)
  SELECT bugs JOIN research_queue ORDER BY priority DESC
        │
        ▼
autoresclaw_runner.py --task-id <id>
  ARC: locate code → edit src/ directly → write fix_summary.md
        │
        ▼
_apply_bug_fix():
  1. Read fix_summary.md — low confidence or human-review? → revert
  2. git diff --name-only — protected file touched? → revert
  3. py_compile each changed .py — syntax error? → revert
  4. git add + git commit
  5. _update_bug_status(task_id, "fixed", commit_hash)
  6. _restart_loops_after_fix(changed_files)
```

Reverted fixes reset the bug to `open` so it can be retried or reviewed manually. All outcomes are written to `.claude/memory/session_handoffs.md`.

---

## Execution path

```
Claude decides to trade
        ↓
execute_trade(symbol, side, qty, strategy_id, ...)
        ↓
RiskGate.check()   ← IMMUTABLE — hard-coded Python, no bypass
  ├─ Position size limits
  ├─ Daily loss halt check
  ├─ Liquidity floor
  ├─ Options DTE / premium cap
  └─ forward_testing size scalar (FORWARD_TESTING_SIZE_SCALAR env var)
        ↓ passes
SmartOrderRouter → Alpaca paper API (or PaperBroker)
        ↓
fills table + audit_log
```

---

## Signal engine

16 concurrent Python collectors produce a `SignalBrief`. No LLM calls. Wall-clock 2–6 seconds. Fault-tolerant: individual collector failures return an error flag without blocking the brief.

| Category | Collectors |
|----------|-----------|
| Price structure | Trend, momentum, volatility |
| Volume / microstructure | Volume/OFI, order flow |
| Risk | VaR, drawdown |
| Events | Earnings calendar, macro events |
| Fundamentals | Piotroski F-Score, revenue, insider |
| Sentiment | News NLP, put/call ratio |
| Flow | Options flow, dark pool |
| Cross-asset | Macro, sector rotation, statarb |
| ML | Model predictions (XGBoost/LightGBM ensemble) |
| Regime | ADX + ATR + HMM hidden-state |
| Social | Reddit + Stocktwits community sentiment (no auth needed) |

---

## Agent architecture

Claude's native `Agent` tool handles all parallelism. No external orchestrator. Example patterns from the trading loop:

```
Reviewing 3 open positions:
  Agent(position-monitor, AAPL) ──┐
  Agent(position-monitor, TSLA) ──┤── parallel, same message
  Agent(position-monitor, SPY)  ──┘

Evaluating entry candidates:
  Agent(trade-debater, NVDA)    ──┐
  Agent(trade-debater, MSFT)    ──┤── parallel
  Agent(risk, batch)            ──┘
        ↓ all return
  Agent(fund-manager, batch)    ── sequential (needs debater results)
```

Desk agent definitions live in `.claude/agents/*.md`.

Key agents:
- **trade-debater** — bull/bear/risk debate before entries and exits
- **position-monitor** — HOLD/TRIM/CLOSE/TIGHTEN for open positions
- **market-intel** — real-time web search for news, analyst changes, M&A deals, social buzz
- **fund-manager** — portfolio-level correlation/concentration review before batch entries
- **community-intel** — weekly quant community scanner (Reddit/GitHub/arXiv/X/newsletters → `research_queue`)

---

## PostgreSQL schema (key tables)

```sql
-- All state for active and closed positions
positions (position_id, symbol, qty, avg_entry_price, status, ...)
fills (fill_id, symbol, side, qty, fill_price, realized_pnl, ...)

-- Strategy lifecycle
strategies (strategy_id, name, status, regime_affinity, params, ...)
-- status: draft → forward_testing → live → retired

-- Autonomous research pipeline
research_queue (task_id, task_type, priority, context_json, status, ...)
bugs (bug_id, tool_name, error_fingerprint, consecutive_errors, arc_task_id, status, ...)

-- Loop coordination
loop_heartbeats (loop_name, iteration, started_at, finished_at, status, ...)
loop_iteration_context (loop_name, context_key, context_json, updated_at)

-- Global state
system_state (key, value, updated_at)
-- keys: credit_regime, kill_switch, av_daily_calls_{date}, ...
```

---

## Further reading

- [quantcore.md](./quantcore.md) — Core library modules (indicators, backtesting, ML)
- [mcp_servers.md](./mcp_servers.md) — Tool catalog
- [../guides/quickstart.md](../guides/quickstart.md) — Get running in 10 minutes
- [../guides/deployment.md](../guides/deployment.md) — Environment variables, data paths, cron jobs
- [../guides/execution_setup.md](../guides/execution_setup.md) — Broker config, risk limits
