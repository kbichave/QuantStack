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

That's it. `start.sh` is the single entry point. It runs Docker Compose (PostgreSQL, LangFuse, Ollama, 3 graph services), runs DB migrations, and starts the trading, research, and supervisor pipelines. Leave it running for weeks.

```
./status.sh                        # health dashboard (one-shot)
./status.sh --watch                # live dashboard (10s refresh, Ctrl+C to quit)
./stop.sh                          # graceful shutdown
./report.sh                        # monthly performance report
docker compose logs -f trading     # follow a specific service
```

---

## Architecture

Three LangGraph StateGraphs run as Docker services, sharing state through PostgreSQL. No in-process state accumulates across cycles — each iteration reads from DB, does its work, and writes back.

<p align="center">
  <img src="docs/images/architecture.png" alt="QuantStack Architecture" width="900"/>
</p>

**Trading Graph** (12 nodes) — safety check, daily planning, parallel position review + entry scanning, mandatory risk gate (SafetyGate — any rejection kills the batch), portfolio review, options analysis, execution, reflection. 5-minute cycles during market hours.

**Research Graph** (8 nodes) — context loading, domain selection, hypothesis generation, signal validation (conditional gate), backtesting, ML experiments, strategy registration, knowledge update. 2-minute cycles market hours, 5-minute after hours.

**Supervisor Graph** (5 nodes) — health checks, issue diagnosis, recovery execution, strategy lifecycle management (promote/retire), scheduled tasks. Runs continuously.

**Infrastructure** — PostgreSQL + pgvector (60+ tables), LangFuse (observability), Ollama (local embeddings). All orchestrated via `docker-compose.yml`.

**Signal Engine** — 18 concurrent Python collectors produce a SignalBrief in 2-6 seconds. No LLM calls. Fault-tolerant: individual collector failures don't block the brief.

**Tool Layer** — LLM-facing tools (`tools/langchain/`, 19 `@tool` functions) bound to agents via YAML config + TOOL_REGISTRY. Deterministic functions (`tools/functions/`) called directly by graph nodes.

---

## Self-healing

Tool failures are tracked in the `bugs` table. After 3 consecutive failures of the same tool, a `bug_fix` task is auto-queued. The supervisor dispatches AutoResearchClaw within 60 seconds: it locates the failing code, edits the source file directly, validates (syntax check, import check, tests), commits the fix, and restarts the affected loop. Protected files (`risk_gate.py`, `kill_switch.py`, `db.py`) are never auto-patched.

---

## Key design invariants

| Invariant | Implementation |
|-----------|---------------|
| **Stateless graphs** | No in-process state between cycles. All state in `loop_iteration_context` (PostgreSQL). |
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
├── start.sh / stop.sh / status.sh  # Lifecycle scripts (Docker Compose)
├── docker-compose.yml              # All services: postgres, langfuse, ollama, 3 graphs
├── src/quantstack/
│   ├── graphs/                     # LangGraph StateGraphs
│   │   ├── trading/                # 12-node pipeline + config/agents.yaml
│   │   ├── research/               # 8-node pipeline + config/agents.yaml
│   │   └── supervisor/             # 5-node pipeline + config/agents.yaml
│   ├── runners/                    # Docker entrypoints for each graph service
│   ├── tools/
│   │   ├── langchain/              # 19 LLM-facing @tool functions
│   │   ├── functions/              # Deterministic node-callable functions
│   │   └── registry.py             # TOOL_REGISTRY (YAML-driven tool binding)
│   ├── signal_engine/              # 18 concurrent collectors, no LLM, 2-6s
│   ├── execution/                  # risk_gate.py (immutable), kill_switch, brokers
│   ├── llm/                        # Provider config, tier routing, fallback chain
│   ├── health/                     # Heartbeat, retry, shutdown, watchdog
│   ├── rag/                        # pgvector knowledge retrieval
│   ├── core/                       # Indicators, backtesting, ML, options pricing
│   └── db.py                       # PostgreSQL (60+ tables, idempotent migrations)
├── scripts/
│   ├── scheduler.py                # APScheduler cron jobs
│   └── autoresclaw_runner.py       # AutoResearchClaw dispatcher
├── .claude/memory/                 # Persistent session memory (gitignored)
└── docs/                           # Architecture, guides, API reference
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
