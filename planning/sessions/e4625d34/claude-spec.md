# Complete Specification: CrewAI Migration for QuantStack

## 1. Mission

Convert QuantStack from Claude Code CLI orchestration to a fully autonomous CrewAI-based trading system that runs unattended for weeks via Docker Compose, with self-hosted Langfuse observability, local Ollama + ChromaDB RAG, and full LLM-reasoned autonomy (no hardcoded thresholds). The system researches strategies, paper-trades via Alpaca, and self-heals from all failure modes.

## 2. Current Architecture (What Exists)

### Orchestration
- Two stateless loops (Trading + Research) run as fresh `claude` CLI invocations every 2-5 minutes in tmux windows
- 14 agents defined as YAML-like markdown in `.claude/agents/` — spawned via Claude's `Agent` tool
- System prompts in `prompts/` define loop logic (trading_loop.md, research_loop.md, domain researchers)
- Supervisor loop monitors heartbeats; scheduler runs cron jobs

### Tools (30+ async Python functions)
- Located in `src/quantstack/mcp/tools/` — called via `python3 -c "import asyncio; ..."` from Bash
- Domains: signal generation, strategy management, backtesting, ML training, risk, execution, portfolio, intelligence, coordination, research
- All tools are async coroutines returning dict results

### State
- PostgreSQL (`quantstack`) is the single source of truth — strategies, positions, fills, audit_log, heartbeat, ML experiments, research queue
- `.claude/memory/` markdown files for cross-session context (strategy_registry, workshop_lessons, trade_journal, etc.)
- Connection pattern: `pg_conn()` context manager, pool of 2-20

### Execution
- Alpaca (paper default, live behind env var), Paper broker (simulation), eTrade (secondary)
- Risk gate (`risk_gate.py`) — currently IMMUTABLE hardcoded limits
- Kill switch (`kill_switch.py`) — one DB write halts everything
- Order lifecycle: PENDING → FILLED/PARTIAL/CANCELLED/REJECTED

### Signal Engine
- 15+ signal collectors running concurrently (no LLM — pure quant)
- Returns `DailyBrief` with technical, fundamental, momentum, microstructure, sentiment, regime, macro, options signals

### Testing
- pytest + pytest-asyncio + hypothesis
- Tests in `tests/unit/`, coverage excludes external APIs, WebSocket, GPU, LLM agents

## 3. Target Architecture (What To Build)

### 3.1 CrewAI Orchestration — Big-Bang Migration

Build a complete replacement, not an incremental migration. Three crews:

**TradingCrew** (replaces trading_loop.md):
- Runs continuously via loop with fresh kickoffs
- Agents: daily-planner, position-monitor, trade-debater, risk, fund-manager, options-analyst, earnings-analyst, market-intel, trade-reflector
- Hierarchical process with fund-manager as the decision authority

**ResearchCrew** (replaces research_loop.md):
- Runs continuously, parallel to TradingCrew
- Agents: quant-researcher, ml-scientist, strategy-rd, community-intel
- Domain-scoped research (investment, swing, options) via CrewAI task routing
- BLITZ mode: parallel domain research via async_execution tasks

**SupervisorCrew** (new):
- Monitors health of Trading + Research crews
- Self-healing: detects stuck agents, crashed processes, stale data
- Manages LLM provider failover
- Controls system-wide state (kill switch equivalent)

### 3.2 LLM-Reasoned Autonomy (NO Hardcoded Thresholds)

This is the most significant architectural change. The current system uses ~20+ hardcoded numeric thresholds:
- Risk gate: 10% per position, -2% daily loss halt, 500K min volume, etc.
- Backtest validation: IC > 0.02, Sharpe > 0.5, PBO < 0.40, DSR > 0, etc.
- Strategy promotion: 30 days forward testing, specific performance gates

**All of these become LLM-reasoned decisions.** The risk agent doesn't check `position_size < 10%` — it reasons about appropriate position sizing given current market conditions, portfolio state, conviction level, volatility regime, and historical outcomes.

Implementation approach:
- Each decision point gets a **reasoning prompt** instead of a threshold check
- The LLM receives full context (portfolio state, market conditions, historical performance, RAG-retrieved knowledge)
- Decisions are logged with full reasoning chains to Langfuse for audit
- The RAG knowledge base provides empirical evidence for the LLM to reason against
- Over time, the system builds its own evidence base from outcomes

### 3.3 Multi-LLM Provider Support

**Primary**: AWS Bedrock (IAM auth, no per-call keys)
**Fallback chain**: Bedrock → Anthropic Direct → OpenAI → Ollama (local)

Model tiers per agent:

| Tier | Agents | Bedrock | Anthropic | OpenAI | Gemini | Ollama |
|------|--------|---------|-----------|--------|--------|--------|
| Heavy reasoning | fund-manager, quant-researcher, trade-debater, risk, ml-scientist, strategy-rd, options-analyst | claude-sonnet-4 | claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| Medium | earnings-analyst, position-monitor, daily-planner, market-intel, trade-reflector | claude-sonnet-4 | claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| Light (batch) | community-intel, execution-researcher, supervisor | claude-haiku-4-5 | claude-haiku-4-5 | gpt-4o-mini | gemini-2.0-flash | llama3.2 |
| Embeddings | Memory, RAG | — | — | text-embedding-3-small | — | mxbai-embed-large |

Provider selection is config-driven via `LLM_PROVIDER` env var. Each agent gets the appropriate model from a tier map.

### 3.4 Tool Compatibility

Wrap all 30+ existing async Python tools as CrewAI tools:
- Use `@tool` decorator for simple wrappers
- Use `BaseTool` subclass for tools needing typed Pydantic input schemas
- Maintain same function signatures and behavior
- Tools remain async — CrewAI supports async tool execution

Risk gate transforms from a hardcoded checker to a **tool that provides context** (current portfolio state, exposure, daily P&L) to the risk reasoning agent.

### 3.5 Observability — Self-Hosted Langfuse

- Run Langfuse as Docker Compose service (Langfuse + its own Postgres)
- Integrate via OpenInference `CrewAIInstrumentor`
- Trace: all LLM calls, agent decisions, tool invocations, reasoning chains
- Cost tracking per provider, per agent, per crew
- CrewAI event bus for fine-grained operational events

### 3.6 RAG — ChromaDB + Ollama Embeddings

**Knowledge base contents:**
1. Trade outcomes + reflexion episodes (from PostgreSQL)
2. Strategy registry + workshop lessons (from `.claude/memory/`)
3. Research papers + market reports (from community-intel discoveries)

**Architecture:**
- ChromaDB PersistentClient (Docker volume for durability)
- Ollama `mxbai-embed-large` for local embeddings
- Two integration paths:
  - CrewAI Knowledge system for automatic per-agent knowledge
  - Custom `@tool("Search Knowledge Base")` for explicit RAG queries with metadata filtering (ticker, domain, date range)

**Startup ingestion:** One-time migration of `.claude/memory/` files into vector store.

### 3.7 Memory — CrewAI Unified Memory + PostgreSQL

- Replace `.claude/memory/*.md` files with CrewAI's unified `Memory` class
- Hierarchical scopes: `/company/risk`, `/agent/researcher`, `/project/alpha`
- Backed by Ollama embeddings (`mxbai-embed-large`)
- PostgreSQL remains the operational SSOT (positions, strategies, fills)
- Memory handles soft knowledge (lessons, patterns, market observations)

### 3.8 Docker Compose Stack

Services:
1. **trading-crew** — Python process running TradingCrew loop
2. **research-crew** — Python process running ResearchCrew loop
3. **supervisor-crew** — Python process monitoring health
4. **postgres** — System database (existing quantstack DB)
5. **langfuse** — Self-hosted observability (+ its own Postgres)
6. **ollama** — Local LLM for embeddings + light inference
7. **chromadb** — Vector store for RAG

All services with:
- Health checks (file-based heartbeat)
- Auto-restart (restart: unless-stopped)
- Volume mounts for persistence
- Graceful shutdown via SIGTERM handlers

### 3.9 Start/Stop/Status Scripts

- `start.sh` → `docker compose up -d` + health check verification
- `stop.sh` → graceful shutdown (persist state → `docker compose down`)
- `status.sh` → crew health, active positions, Langfuse dashboard URL, container status

### 3.10 Self-Healing (Everything Auto-Recovers)

| Failure | Recovery |
|---------|----------|
| LLM provider down | Auto-failover chain (Bedrock → Anthropic → OpenAI → Ollama) |
| Stuck agent (>10 min) | Watchdog timer → force restart crew cycle |
| Crashed crew process | Docker auto-restart + state recovery from DB |
| Database connection lost | Exponential backoff reconnect |
| Ollama down | Skip embeddings, use cached, restart container |
| ChromaDB down | Degrade gracefully (skip RAG, use memory only) |
| API rate limit | Exponential backoff with jitter |
| Stale market data | Auto-refresh via data adapters |
| Memory corruption | Rebuild from PostgreSQL + ChromaDB |

### 3.11 Capital Allocation

- **Total**: $25,000 paper account
- **Equity**: $20,000
- **Options**: $5,000
- Alpaca paper mode (default, `ALPACA_PAPER=true`)
- LLM reasons about position sizing within these capital bounds

## 4. Constraints

- Paper trading only (Alpaca paper mode)
- Must work on macOS (Docker Desktop)
- PostgreSQL remains system of record
- Alpha Vantage rate limits (75/min premium) respected
- All existing tool logic preserved (just wrapped for CrewAI)
- No hardcoded thresholds — all decision logic is LLM-reasoned

## 5. Non-Goals (This Phase)

- Live trading with real money
- Multi-machine / cloud deployment
- Custom model fine-tuning
- Web UI / mobile app
- Push notifications / alerting (self-healing handles everything)

## 6. Success Criteria

1. System runs unattended for 7+ days without manual intervention
2. Research crew discovers and validates strategies autonomously
3. Trading crew enters and exits positions based on LLM reasoning
4. All decisions traceable in Langfuse with full reasoning chains
5. Self-healing recovers from all enumerated failure modes
6. RAG knowledge base grows with each trade outcome and research discovery
7. Docker Compose `start.sh` brings up entire stack in one command
