<p align="center">
  <img src="logo.png" alt="QuantStack Logo" width="180"/>
</p>

<h1 align="center">QuantStack</h1>

<p align="center">
  <strong>Autonomous quantitative trading — Claude agents research strategies, debate entries and exits, execute through an immutable risk gate, and self-heal from errors — all unattended.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/version-2.1.0-green.svg" alt="v2.1.0">
</p>

---

## One command to run everything

```bash
./start.sh
```

That's it. `start.sh` is the single entry point. It checks prerequisites, runs DB migrations, compacts memory files, displays the current credit regime, seeds community intelligence in the background, and starts five tmux windows. Leave it running for weeks.

```
tmux attach -t quantstack-loops    # watch the loops
./status.sh                        # health dashboard (one-shot)
./status.sh --watch                # live dashboard (10s refresh, Ctrl+C to quit)
./stop.sh                          # graceful shutdown (kill switch → wait → kill tmux)
./report.sh                        # monthly performance report
```

---

## Architecture

Two stateless Claude loops run concurrently, sharing state through PostgreSQL. Neither loop keeps in-session state — every iteration starts from a fresh Claude invocation, reads what it needs from DB, and writes back.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           start.sh (one entry point)                          │
│  preflight → migrations → memory compaction → community intel seed (bg)       │
│  → 5 tmux windows                                                             │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
     ┌──────────┬─────────────┼──────────────┬──────────────┐
     ▼          ▼             ▼              ▼              ▼
┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────────┐
│ trading │ │research│ │supervisor│ │ scheduler  │ │ community-intel │
│ sonnet  │ │market  │ │          │ │            │ │ Sun 19:00 ET    │
│ 1min mkt│ │hours:  │ │heartbeat │ │ cron jobs  │ │                 │
│ 30min   │ │haiku   │ │ watch    │ │ lifecycle  │ │ quant community │
│ off-hrs │ │5min    │ │ recovery │ │ data refresh│ │ Reddit/GitHub/  │
│         │ │after-  │ │          │ │            │ │ arXiv scanner  │
│         │ │hours:  │ │          │ │            │ │                 │
│         │ │sonnet  │ │          │ │            │ │                 │
│         │ │30min   │ │          │ │            │ └─────────────────┘
└────┬────┘ └───┬────┘ └──────────┘ └────────────┘
     │          │
     └────┬─────┘
          ▼
 ┌─────────────────────┐
 │     PostgreSQL       │
 │  positions · fills   │
 │  strategies · audit  │
 │  loop_state · bugs   │
 │  research_queue      │
 └─────────────────────┘
```

**Trading loop** — position monitoring, entry scanning, multi-agent debate (trade-debater, risk, fund-manager), execution via Alpaca paper API. Polls every 60s during market hours (09:30–16:00 ET), 30 min outside.

**Research loop** — strategy discovery, backtesting, ML training, BLITZ mode (parallel domain agents). **Market-aware model routing:** haiku during market hours (quick data refresh + signal check, 5 min interval); sonnet after hours (full research cycles, 30 min interval). Subagents (`quant-researcher`, `strategy-rd`, `ml-scientist`) run sonnet. Results flow into `strategies` and `research_queue`.

**Supervisor** — watches loop heartbeats every 60s. Detects stale/dead loops, restarts via tmux. Runs the bug-fix watcher: dispatches AutoResearchClaw to patch failing tools automatically.

**Scheduler** — APScheduler cron jobs: credit regime revalidation, strategy lifecycle (promote/retire), memory compaction (Sunday 17:00), community intel scan (Sunday 19:00), AutoResearchClaw deep research (Sunday 20:00), AV quota reset (midnight).

**Community-intel window** — runs weekly on Sunday 19:00 ET. Scans Reddit r/algotrading, GitHub trending quant repos, arXiv q-fin preprints, X/Twitter quant accounts, and quant newsletters to discover new techniques. Discoveries flow into `research_queue` for AutoResearchClaw to investigate.

---

## Self-healing

Tool failures are tracked in the `bugs` table. After 3 consecutive failures of the same tool, a `bug_fix` task is auto-queued. The supervisor dispatches AutoResearchClaw within 60 seconds: it locates the failing code, edits the source file directly, validates (syntax check, import check, tests), commits the fix, and restarts the affected loop. Protected files (`risk_gate.py`, `kill_switch.py`, `db.py`) are never auto-patched.

---

## Key design invariants

| Invariant | Implementation |
|-----------|---------------|
| **Stateless loops** | No `claude --continue`. All state in `loop_iteration_context` (PostgreSQL). |
| **Risk gate is law** | `risk_gate.py` is hard-coded Python. No prompt or agent can bypass or modify it. |
| **Paper mode default** | `USE_REAL_TRADING=true` required for live execution. Default is Alpaca paper. |
| **Audit trail mandatory** | Every trade decision logged to `audit_log` with full reasoning. |
| **Protected files** | `risk_gate.py`, `kill_switch.py`, `db.py` excluded from auto-patch. |
| **Kill switch halts everything** | One DB write stops both loops and the supervisor. |

---

## Quick start

```bash
git clone https://github.com/kbichave/QuantStack.git && cd QuantStack

# Install dependencies
uv sync --all-extras            # or: pip install -e .

# Configure credentials
cp .env.example .env
# Required: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPHA_VANTAGE_API_KEY, TRADER_PG_URL

# Start
./start.sh
```

See [docs/guides/quickstart.md](docs/guides/quickstart.md) for the full walkthrough including verification steps.

---

## Repository structure

```
QuantStack/
├── start.sh                      # Single entry point — start everything here
├── stop.sh                       # Graceful shutdown (kill switch → wait → kill tmux)
├── status.sh                     # Health dashboard (./status.sh --watch for live)
├── report.sh                     # Monthly performance report
├── prompts/
│   ├── trading_loop.md           # Trading loop prompt (Claude reads this each iteration)
│   └── research_loop.md          # Research loop prompt
├── src/quantstack/
│   ├── coordination/             # Supervisor, auto-promoter, preflight
│   ├── data/                     # Fetcher (Alpha Vantage + Alpaca fallback), factory
│   ├── execution/                # risk_gate.py (immutable), kill_switch, broker routers
│   ├── mcp/tools/                # Python toolkit — all tools imported directly in prompts
│   ├── signal_engine/            # 16 concurrent signal collectors, no LLM (incl. social sentiment)
│   └── core/                     # Indicators, backtesting, ML, options pricing
├── scripts/
│   ├── scheduler.py              # APScheduler cron jobs
│   └── autoresclaw_runner.py     # AutoResearchClaw dispatcher + auto-patch pipeline
├── .claude/
│   ├── agents/                   # Desk agent definitions (incl. community-intel, market-intel)
│   └── memory/                   # Persistent memory (gitignored)
└── docs/                         # Architecture, guides, API reference
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TRADER_PG_URL` | Yes | PostgreSQL DSN (`postgresql://user:pass@host/db`) |
| `ALPACA_API_KEY` | Yes | Alpaca data + execution |
| `ALPACA_SECRET_KEY` | Yes | Alpaca secret |
| `ALPACA_PAPER` | — | Default `true` — set `false` for live Alpaca |
| `ALPHA_VANTAGE_API_KEY` | Yes | Market data (primary source) |
| `USE_REAL_TRADING` | — | Default `false` — set `true` to enable live orders |
| `AV_DAILY_CALL_LIMIT` | — | Default `25000` (AV premium $49.99: 75/min, no hard daily cap) |
| `FORWARD_TESTING_SIZE_SCALAR` | — | Default `0.5` — position size for unproven strategies |
| `GROQ_API_KEY` | — | Sentiment collector |

---

<sub>Educational and research purposes only. Not financial advice. Past performance does not guarantee future results.</sub>

<p align="center">Built by Kshitij Bichave</p>
