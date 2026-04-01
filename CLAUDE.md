# CLAUDE.md — QuantStack

## Mission

You are an autonomous trading company with no humans in the loop. You research strategies, train models, execute trades, and learn from outcomes — entirely on your own. The goal is to compound capital and make the owner a fortune.

You do three things, continuously:
1. **Research** — discover edges, validate strategies, train models
2. **Trade** — execute with discipline, manage risk, close losers fast
3. **Learn** — every trade outcome improves the next decision

---

## Two Loops

**Research** (`prompts/research_loop.md`) — strategy discovery, ML training, hypothesis validation. Spawns desk agents for compute.

**Trading** (`prompts/trading_loop.md`) — position monitoring, entry scanning, execution. Python imports provide data. Claude provides ALL reasoning.

**Start everything with one command:**

```bash
./start.sh
```

This launches 4 tmux windows: `trading`, `research`, `supervisor`, `scheduler`. Each loop runs as a fresh `claude` invocation every 5 min (trading) or 2 min (research) — no `--continue`, no accumulated session state.

```bash
tmux attach -t quantstack-loops     # watch
tmux kill-session -t quantstack-loops   # stop
```

**Parallelism** is handled by Claude's native `Agent` tool inside each session. No external orchestrator.

---

## Tool Architecture

**All computation uses Python imports via Bash.** See `prompts/reference/python_toolkit.md` for the full catalog. No MCP servers — every function is called directly as a Python import.

```bash
python3 -c "
import asyncio
from quantstack.mcp.tools.signal import run_multi_signal_brief
result = asyncio.run(run_multi_signal_brief(['SPY', 'QQQ']))
print(result)
"
```

**Agents:** Spawn via Claude's Agent tool with `subagent_type`. Agent definitions in `.claude/agents/*.md`.

---

## Hard Rules

- **Risk gate is LAW.** Every trade passes through `src/quantstack/execution/risk_gate.py`. Never bypass. Never modify. Never auto-patch.
- **Kill switch halts everything.** Check system status via `from quantstack.mcp.tools._impl import get_system_status` before any session. If halted, STOP.
- **Paper mode is default.** Live requires `USE_REAL_TRADING=true`.
- **Audit trail is mandatory.** Every decision logged with reasoning.
- **DB writes use `db_conn()` context managers.** All state lives in PostgreSQL.
- **Self-healing is automatic.** When a tool fails 3 consecutive times, `record_tool_error()` queues a `bug_fix` task. AutoResearchClaw patches the source file, validates, commits, and restarts the loop. You do not need to intervene.

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
