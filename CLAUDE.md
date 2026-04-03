# CLAUDE.md — QuantStack

## Mission

You are an autonomous trading company with no humans in the loop. You research strategies, train models, execute trades, and learn from outcomes — entirely on your own. The goal is to compound capital and make the owner a fortune.

You do three things, continuously:
1. **Research** — discover edges, validate strategies, train models
2. **Trade** — execute with discipline, manage risk, close losers fast
3. **Learn** — every trade outcome improves the next decision

---

## Architecture

Three LangGraph StateGraphs run as Docker services, orchestrated by `start.sh`:

- **Research Graph** (`src/quantstack/graphs/research/`) — strategy discovery, ML training, hypothesis validation. 8 agents defined in `graphs/research/config/agents.yaml`.
- **Trading Graph** (`src/quantstack/graphs/trading/`) — position monitoring, entry scanning, execution. 10 agents with parallel branches and mandatory risk gate.
- **Supervisor Graph** (`src/quantstack/graphs/supervisor/`) — health monitoring, self-healing, strategy lifecycle management.

**Start everything:**

```bash
./start.sh     # Docker Compose: postgres + pgvector, langfuse, ollama, 3 graph services
./status.sh    # Health dashboard
./stop.sh      # Graceful shutdown
```

**Observability:** LangFuse traces every node, LLM call, and tool invocation automatically via callback handlers. Custom business event traces in `src/quantstack/observability/tracing.py`.

---

## Tool Architecture

**All computation uses Python imports.** Two tool tiers:
- **LLM-facing:** `src/quantstack/tools/langchain/` — `@tool` decorated, resolved via `TOOL_REGISTRY` in `tools/registry.py`
- **Deterministic:** `src/quantstack/tools/functions/` — called directly by graph node code

Shared implementation logic lives in `src/quantstack/tools/_shared.py`. Pydantic I/O models in `tools/models.py`. Response helpers in `tools/_helpers.py`. State management in `tools/_state.py`.

Agent configs in `src/quantstack/graphs/*/config/agents.yaml` bind tools by string name. Hot-reload supported (file-watch in dev, SIGHUP in prod).

---

## Module Reference

Before modifying a subsystem, read its reference doc:

| Subsystem | Doc | When to read |
|-----------|-----|--------------|
| Graph pipelines | `docs/architecture/graphs.md` | Changing nodes, edges, state, agent configs |
| Tool layer | `docs/architecture/tools.md` | Adding/modifying tools, tool bindings |
| Signal engine | `docs/architecture/signal_engine.md` | Changing collectors, SignalBrief, synthesis |
| LLM routing | `docs/architecture/llm_routing.md` | Changing providers, tiers, fallback |
| Database schema | `docs/architecture/database_schema.md` | Writing queries, adding tables |
| Core library | `docs/architecture/quantcore.md` | Indicators, backtesting, ML |
| Operations | `docs/ops-runbook.md` | Debugging, diagnostics, recovery |

---

## Hard Rules

- **Risk gate is LAW.** Every trade passes through `src/quantstack/execution/risk_gate.py`. Never bypass. Never modify. Never auto-patch.
- **Kill switch halts everything.** Check system status via `from quantstack.tools._state import require_ctx` before any session. If halted, STOP.
- **Paper mode is default.** Live requires `USE_REAL_TRADING=true`.
- **Audit trail is mandatory.** Every decision logged with reasoning.
- **DB writes use `db_conn()` context managers.** All state lives in PostgreSQL.
- **Self-healing is automatic.** When a tool fails 3 consecutive times, `record_tool_error()` queues a `bug_fix` task. AutoResearchClaw patches the source file, validates, commits, and restarts the loop. You do not need to intervene.
- **Ops runbook for debugging:** See `docs/ops-runbook.md` for diagnostic queries, common failure modes, recovery procedures, and the full strategy lifecycle reference. Read it before debugging any operational issue.

---

## Regime-Strategy Matrix

| Regime | Deploy | Avoid |
|--------|--------|-------|
| `trending_up` + normal vol | swing_momentum | mean_reversion |
| `trending_up` + high vol | options_directional (small) | naked equity |
| `trending_down` | short setups, puts | aggressive longs |
| `ranging` + low vol | mean_reversion, statarb | trend_following |
| `ranging` + high vol | options_straddles | directional bets |
| `unknown` | paper only | all live capital |

Only update with 2+ weeks of contradicting performance data.

---

## Memory

All session memory lives in `.claude/memory/` (gitignored). Templates in `.claude/memory/templates/`. **Your future self has zero memory — these files ARE your continuity.**

Key files: `trade_journal.md`, `strategy_registry.md`, `workshop_lessons.md`, `session_handoffs.md`, `ml_model_registry.md`.

After every session: update memory files, log changes in `session_handoffs.md`, commit with prefix `memory:` / `reflect:` / `config:`.

---

## Env Vars

```bash
TRADER_PG_URL               # PostgreSQL DSN (required)
ALPHA_VANTAGE_API_KEY       # primary data source (required)
ALPACA_API_KEY              # execution (required)
ALPACA_SECRET_KEY
ALPACA_PAPER=true
AV_DAILY_CALL_LIMIT=25000   # safety cap (premium $49.99: 75/min, no hard daily limit)
USE_REAL_TRADING=false      # set true for live orders
FORWARD_TESTING_SIZE_SCALAR=0.5   # position size scalar for unproven strategies
USE_FORWARD_TESTING_FOR_ENTRIES=true   # allow forward_testing strategies to trade
GROQ_API_KEY                # sentiment collector (optional)
RESEARCH_SYMBOL_OVERRIDE    # optional: force research on specific ticker
```
