# QuantStack Architecture

---

## Single entry point

Everything starts from `./start.sh`. There is no other way to start the system. The script handles prerequisites, migrations, universe bootstrap, preflight checks, seeds community intelligence (background), and launches five tmux windows. See [../guides/quickstart.md](../guides/quickstart.md).

---

## Repository structure

```
QuantStack/
├── start.sh                          # Single entry point (Docker Compose)
├── stop.sh / status.sh / report.sh   # Lifecycle & monitoring
├── docker-compose.yml                # All services: postgres, langfuse, ollama, 3 graphs
├── src/quantstack/
│   ├── graphs/                       # LangGraph StateGraphs
│   │   ├── research/                 # Strategy discovery, ML training, hypothesis validation
│   │   │   └── config/agents.yaml    # 8 research agents
│   │   ├── trading/                  # Position monitoring, entry scanning, execution
│   │   │   └── config/agents.yaml    # 10 trading agents + risk gate
│   │   └── supervisor/               # Health monitoring, self-healing, lifecycle mgmt
│   │       └── config/agents.yaml
│   ├── runners/                      # Docker entrypoints for each graph service
│   ├── coordination/                 # Supervisor, auto-promoter, preflight
│   ├── data/
│   │   ├── fetcher.py                # Alpha Vantage client (daily quota guard, priority tiers)
│   │   └── factory.py               # Provider routing + Alpaca OHLCV fallback
│   ├── execution/
│   │   ├── risk_gate.py              # IMMUTABLE — hard-coded pre-trade checks
│   │   ├── kill_switch.py            # Emergency halt (DB sentinel)
│   │   └── broker_routers.py         # Alpaca, PaperBroker
│   ├── tools/
│   │   ├── langchain/                # LLM-facing @tool decorated (agent nodes)
│   │   ├── functions/                # Node-callable deterministic tools
│   │   └── mcp_bridge/              # MCPBridge for MCP server communication
│   ├── mcp/tools/                    # Python toolkit (signal, execution, coordination)
│   ├── signal_engine/                # 16 concurrent collectors, no LLM, 2–6s
│   ├── core/                         # Indicators, backtesting, ML, options pricing
│   ├── health/                       # Health checks for Docker services
│   ├── rag/                          # RAG pipeline (pgvector)
│   ├── llm/                          # LLM config, provider routing, model tiers
│   ├── api/                          # FastAPI REST (optional)
│   └── db.py                         # PostgreSQL connection + all migrations
├── scripts/
│   ├── scheduler.py                  # APScheduler cron jobs
│   └── autoresclaw_runner.py         # ARC dispatcher + auto-patch pipeline
├── .claude/
│   └── memory/                       # Persistent memory (gitignored)
└── docs/
```

---

## System diagram

```
                    docker-compose up (via start.sh)
                            │
          ┌─────────────────┼─────────────────────┐
          ▼                 ▼                     ▼
  ┌──────────────┐  ┌──────────────┐   ┌────────────────────┐
  │   trading    │  │  research    │   │  supervisor        │
  │  (Docker)    │  │  (Docker)    │   │  (Docker)          │
  │              │  │              │   │                    │
  │ LangGraph    │  │ LangGraph    │   │ LangGraph          │
  │ StateGraph   │  │ StateGraph   │   │ StateGraph         │
  │              │  │              │   │                    │
  │ 10 agents:   │  │ 8 agents:    │   │ health monitor     │
  │ daily_planner│  │ quant_       │   │ self-healing       │
  │ position_    │  │  researcher  │   │ strategy lifecycle │
  │  monitor     │  │ ml_scientist │   │                    │
  │ trade_       │  │ strategy_rd  │   │  ┌──────────────┐  │
  │  debater     │  │ hypothesis_  │   │  │  scheduler   │  │
  │ risk_gate    │  │  generator   │   │  │  cron jobs   │  │
  │ fund_manager │  │ community_   │   │  │  strategy_   │  │
  │ ...          │  │  intel       │   │  │  pipeline    │  │
  └──────┬───────┘  │ ...          │   │  └──────┬───────┘  │
         │          └──────┬───────┘   └─────────┼──────────┘
         │                 │                     │
         └────────┬────────┴─────────────────────┘
                  ▼
  ┌──────────────────────────────────────────────┐
  │   PostgreSQL + pgvector    │    LangFuse     │
  │                            │  (observability)│
  │  positions  loop_heartbeats│                 │
  │  fills      loop_iteration_│  traces every   │
  │  strategies   context      │  node, LLM call,│
  │  audit_log  bugs           │  tool invocation│
  │  research_queue            │                 │
  │  system_state              │                 │
  │  universe   ml_experiments │                 │
  └──────────────────────────────────────────────┘
```

---

## Graph execution model

Each graph runs as a Docker service with its own LangGraph StateGraph. State is persisted in PostgreSQL between iterations — graphs do not accumulate in-process state across runs.

Each iteration:
1. Graph runner loads current state from `loop_iteration_context` (PostgreSQL)
2. LangGraph traverses agent nodes (parallel branches where possible)
3. Updated state written back to `loop_iteration_context`
4. Runner sleeps (adaptive: bootstrap vs steady-state) then re-invokes

**Context keys per graph:**

| Graph | Key | Purpose |
|-------|-----|---------|
| `trading` | `market_intel` | Cached market intelligence (25-min TTL) |
| `trading` | `stale_symbols` | Symbols with stale OHLCV (set each iter) |
| `trading` | `closes_since_review` | Counter triggering weekly trade-reflector |
| `research` | `last_domain` | Last research domain processed |
| `research` | `domain_history` | Rolling 50-entry domain history |
| `research` | `last_execution_audit_at` | Date of last execution-researcher spawn |

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

## Strategy promotion pipeline

Strategies progress through a state machine where the DB is the handshake medium between independent processes. No direct inter-process communication.

```
Research loop                  Scheduler                     Research loop
(creates strategies)           (strategy_pipeline_10m)       (Step 2e)

registers as ──────►  draft ──── run_backtest ────►  backtested ──── strategy-rd ────►  forward_testing
  status='draft'       │        (pure Python,         │            agent (LLM           │
                       │         every 10 min)        │             reasoning)          │
                       │                              │                                │
                       │                              └───► retired (REJECT verdict)   │
                       │                                                               │
                       │         AutoPromoter (21+ days paper trading)                  │
                       │         min 15 trades, Sharpe > 0.5, DD < 8%                  │
                       │                                        ┌──────────────────────┘
                       │                                        ▼
                       │                                       live
                       │                                        │
                       │         Monthly lifecycle: degradation  │
                       │         check (30d P&L < 0 → retire)   │
                       │                                        ▼
                       └───────────────────────────────────►  retired
```

**Phase 1 — draft to backtested** runs in the scheduler (`strategy_pipeline_10m`). Pure Python, no LLM. Calls `run_backtest_impl()` which populates `backtest_summary` and transitions status. Heartbeat-guarded to prevent overlap.

**Phase 2 — backtested to forward_testing** runs in the research loop (`Step 2e`). Each iteration checks for up to 2 backtested strategies and spawns a `strategy-rd` agent to reason about whether to PROMOTE, REJECT, or INVESTIGATE. This is intentionally LLM-based: mechanical thresholds miss context about regime fit, overfitting risk, and strategy quality that the strategy-rd agent evaluates.

**Phase 3 — forward_testing to live** is handled by `AutoPromoter` (called during the weekly lifecycle job). Requires 21+ days of paper trading, 15+ trades, Sharpe > 0.5, and drawdown < 8%.

See `docs/ops-runbook.md` for diagnostic queries and common failure modes.

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

LangGraph StateGraphs orchestrate agent nodes. Parallel branches execute concurrently where no data dependency exists. Agent configs live in `graphs/*/config/agents.yaml` (hot-reload supported).

**Trading graph agents (10):** daily_planner, position_monitor, entry_scanner, trade_debater, risk_gate (mandatory), fund_manager, execution, market_intel, trade_reflector, options_analyst.

**Research graph agents (8):** quant_researcher, ml_scientist, strategy_rd, hypothesis_generator, alpha_discovery, execution_researcher, community_intel, domain_selector.

**Supervisor graph:** health_monitor, self_healer, strategy_lifecycle.

Tools are provided to agents via two paths:
- **LLM-facing:** `tools/langchain/*.py` — `@tool` decorated, used by agent nodes that need LLM reasoning
- **Deterministic:** `tools/functions/*.py` — called directly by graph nodes without LLM

---

## PostgreSQL schema (key tables)

```sql
-- All state for active and closed positions
positions (position_id, symbol, qty, avg_entry_price, status, ...)
fills (fill_id, symbol, side, qty, fill_price, realized_pnl, ...)

-- Strategy lifecycle
strategies (strategy_id, name, status, regime_affinity, params, ...)
-- status: draft → backtested → forward_testing → live → retired

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
- [../ops-runbook.md](../ops-runbook.md) — Diagnostic queries, failure modes, recovery procedures
- [../guides/quickstart.md](../guides/quickstart.md) — Get running in 10 minutes
- [../guides/deployment.md](../guides/deployment.md) — Environment variables, data paths, cron jobs
- [../guides/execution_setup.md](../guides/execution_setup.md) — Broker config, risk limits
