# Implementation Plan: QuantStack CrewAI Migration

## Overview

This plan describes the complete replacement of QuantStack's Claude Code CLI orchestration with a CrewAI-based autonomous trading system. The system runs in Docker Compose with self-hosted Langfuse observability, local Ollama + ChromaDB RAG, multi-provider LLM support (Bedrock primary), and LLM-reasoned autonomy within programmatic safety boundaries.

**Estimated LLM cost:** ~$50-100/month at 5-minute trading cycles with Bedrock Sonnet.
**Minimum machine specs:** 16GB RAM (32GB recommended). macOS with Docker Desktop.

**Key architectural shifts:**
1. Claude Code CLI loops → CrewAI Crews with continuous kickoff loops
2. Markdown agent definitions → CrewAI Agent YAML configs
3. Prompt-driven orchestration → CrewAI Tasks with DAG dependencies
4. Bash `python3 -c` tool calls → CrewAI `@tool` decorated functions
5. Hardcoded risk thresholds → LLM-reasoned risk decisions within programmatic safety boundaries
6. Markdown memory files → CrewAI Memory (Ollama embeddings) + ChromaDB RAG
7. tmux process management → Docker Compose with health checks and auto-restart

**Capital:** $25K paper ($20K equity, $5K options). Alpaca paper mode only.

---

## Section 1: Project Scaffolding and Docker Compose Stack

### 1.1 New Directory Structure

The existing `src/quantstack/` code is preserved. New CrewAI orchestration sits alongside it:

```
src/quantstack/
  crews/                          # NEW: CrewAI crew definitions
    trading/
      crew.py                     # TradingCrew class
      config/
        agents.yaml               # Agent definitions (role, goal, backstory)
        tasks.yaml                # Task definitions (description, expected_output)
    research/
      crew.py                     # ResearchCrew class
      config/
        agents.yaml
        tasks.yaml
    supervisor/
      crew.py                     # SupervisorCrew class
      config/
        agents.yaml
        tasks.yaml
  crewai_tools/                   # NEW: CrewAI tool wrappers
    signal_tools.py               # Wraps signal.py functions
    strategy_tools.py             # Wraps strategy.py functions
    backtest_tools.py             # Wraps backtesting.py functions
    ml_tools.py                   # Wraps ml.py functions
    risk_tools.py                 # Wraps qc_risk.py functions
    execution_tools.py            # Wraps execution.py functions
    portfolio_tools.py            # Wraps portfolio.py functions
    intelligence_tools.py         # Wraps intelligence functions
    coordination_tools.py         # Wraps coordination.py functions
    research_tools.py             # Wraps qc_research.py functions
    rag_tools.py                  # NEW: RAG query tools
    web_tools.py                  # NEW: Web search for market-intel
  llm/                            # NEW: LLM provider management
    provider.py                   # Provider selection + fallback chain
    config.py                     # Model tier map per provider
  rag/                            # NEW: RAG pipeline
    ingest.py                     # Document ingestion into ChromaDB
    query.py                      # Retrieval functions
    embeddings.py                 # Ollama embedding wrapper
  runners/                        # NEW: Continuous loop runners
    trading_runner.py             # TradingCrew continuous loop
    research_runner.py            # ResearchCrew continuous loop
    supervisor_runner.py          # SupervisorCrew continuous loop
  health/                         # NEW: Health check utilities
    heartbeat.py                  # File-based heartbeat
    watchdog.py                   # Stuck agent detection
    shutdown.py                   # Graceful SIGTERM handler

docker-compose.yml                # NEW: Full stack definition
Dockerfile                        # UPDATE: CrewAI-compatible image
start.sh                          # UPDATE: docker compose up
stop.sh                           # UPDATE: graceful shutdown
status.sh                         # UPDATE: container health + crew status
```

### 1.2 Docker Compose Services

Seven services in `docker-compose.yml`:

1. **postgres** — Existing quantstack database. Uses a named volume. Health check: `pg_isready`. Exposes port 5432.

2. **ollama** — Official `ollama/ollama` image. Named volume for model storage. Health check: `curl localhost:11434/api/tags`. On startup, a one-shot init container pulls `mxbai-embed-large` and `llama3.2` models if not already present.

3. **chromadb** — Official `chromadb/chroma` image. Named volume for persistent data. Health check: HTTP GET on `/api/v1/heartbeat`. Exposes port 8000.

4. **langfuse** — Official `langfuse/langfuse` image. Requires its own Postgres instance (separate from quantstack). Environment: `DATABASE_URL`, `NEXTAUTH_SECRET`, `SALT`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`. Health check: HTTP GET on `/api/public/health`. Exposes port 3000.

5. **langfuse-db** — Postgres instance dedicated to Langfuse. Separate from the quantstack database.

6. **trading-crew** — Python process (`python -m quantstack.runners.trading_runner`). Depends on: postgres, ollama, chromadb, langfuse (healthy). Health check: file-based heartbeat at `/tmp/trading-heartbeat`. Restart: unless-stopped.

7. **research-crew** — Python process (`python -m quantstack.runners.research_runner`). Same dependencies and health pattern as trading-crew.

8. **supervisor-crew** — Python process (`python -m quantstack.runners.supervisor_runner`). Monitors health of trading-crew and research-crew. Lighter weight — uses haiku-tier LLM.

All crew services share environment variables via `.env` file and have graceful shutdown handlers (SIGTERM → persist state → exit).

### 1.3 Dockerfile

Single multi-stage Dockerfile:
- Base: Python 3.11+ slim
- Install: `uv` for dependency management
- Copy: `pyproject.toml`, `uv.lock`, then `src/`
- Install: project in editable mode (`uv pip install -e ".[crewai]"`)
- Entrypoint: varies per service (set in docker-compose.yml `command`)

### 1.4 Dependencies (pyproject.toml additions)

New optional dependency group `[crewai]`:
- `crewai[anthropic,bedrock,openai,google,litellm,tools]` — core framework + all providers
- `langfuse` — observability SDK
- `openinference-instrumentation-crewai` — Langfuse integration
- `chromadb` — vector store
- `ollama` — local LLM client

---

## Section 2: LLM Provider Management

### 2.1 Provider Abstraction

A central module (`src/quantstack/llm/provider.py`) manages LLM selection. It does NOT wrap CrewAI's LLM class — it provides configuration that CrewAI agents consume.

The module reads `LLM_PROVIDER` env var (default: `bedrock`) and exposes a function that returns the correct model string for a given reasoning tier:

```python
def get_model(tier: str) -> str:
    """Return model string for CrewAI Agent's llm parameter.

    tier: 'heavy', 'medium', 'light', 'embedding'
    Falls back through provider chain on failure.
    """
```

### 2.2 Model Tier Map

A configuration module (`src/quantstack/llm/config.py`) defines the mapping:

```python
@dataclass
class ProviderConfig:
    heavy: str      # fund-manager, quant-researcher, trade-debater, risk, ml-scientist, strategy-rd, options-analyst
    medium: str     # earnings-analyst, position-monitor, daily-planner, market-intel, trade-reflector
    light: str      # community-intel, execution-researcher, supervisor
    embedding: str  # memory, RAG
```

One `ProviderConfig` per supported provider:

| Provider | Heavy | Medium | Light | Embedding |
|----------|-------|--------|-------|-----------|
| bedrock | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | `bedrock/anthropic.claude-haiku-4-5-20251001-v1:0` | `ollama/mxbai-embed-large` |
| anthropic | `anthropic/claude-sonnet-4` | `anthropic/claude-sonnet-4` | `anthropic/claude-haiku-4-5` | `ollama/mxbai-embed-large` |
| openai | `openai/gpt-4o` | `openai/gpt-4o-mini` | `openai/gpt-4o-mini` | `openai/text-embedding-3-small` |
| gemini | `gemini/gemini-2.5-pro` | `gemini/gemini-2.0-flash` | `gemini/gemini-2.0-flash` | `ollama/mxbai-embed-large` |
| ollama | `ollama/llama3:70b` | `ollama/llama3.2` | `ollama/llama3.2` | `ollama/mxbai-embed-large` |

### 2.3 Fallback Chain

The fallback chain is: Bedrock → Anthropic → OpenAI → Ollama.

Implementation: a wrapper that catches LLM provider errors and retries with the next provider in the chain. This wraps CrewAI's `LLM` class initialization — if the primary model fails on first call, it swaps to the fallback model string.

The wrapper lives in `provider.py` and is used by all crew definitions when constructing agents.

### 2.4 Environment Variables

```
LLM_PROVIDER=bedrock              # Primary provider
LLM_FALLBACK_ENABLED=true         # Enable fallback chain
ANTHROPIC_API_KEY=sk-ant-...      # For Anthropic direct
AWS_ACCESS_KEY_ID=...             # For Bedrock
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
OPENAI_API_KEY=sk-...             # For OpenAI
GEMINI_API_KEY=...                # For Gemini
OLLAMA_BASE_URL=http://ollama:11434  # Docker service name
```

---

## Section 3: CrewAI Tool Wrappers

### 3.1 Wrapping Strategy

Every existing async function in `src/quantstack/mcp/tools/` gets a thin CrewAI wrapper in `src/quantstack/crewai_tools/`. The wrappers:

1. Use the `@tool` decorator (preferred for simplicity)
2. Call the underlying async function via the event loop (using `nest_asyncio` to handle nested loops safely — CrewAI's internals may already be inside an event loop)
3. Return string results (CrewAI tools must return strings)
4. Serialize dict results to JSON strings

**Dependency:** Add `nest_asyncio` to project dependencies. Apply `nest_asyncio.apply()` at runner startup before any crew operations.

Pattern for each wrapper:

```python
@tool("Get Signal Brief")
def get_signal_brief_tool(symbol: str) -> str:
    """Generate a comprehensive signal analysis for a stock symbol.

    Returns technical, fundamental, momentum, and regime signals.
    """
```

The body calls the existing async function and returns `json.dumps(result)`.

### 3.2 Tool Modules (one per domain)

**signal_tools.py**: `get_signal_brief_tool`, `run_multi_signal_brief_tool`

**strategy_tools.py**: `register_strategy_tool`, `get_strategy_tool`, `list_strategies_tool`, `update_strategy_status_tool`

**backtest_tools.py**: `run_backtest_tool`, `run_walkforward_tool`, `run_combinatorial_cv_tool`

**ml_tools.py**: `train_ml_model_tool`, `predict_ml_signal_tool`, `analyze_model_shap_tool`, `check_concept_drift_tool`

**risk_tools.py**: `get_portfolio_context_tool` — replaces the old risk_gate.py threshold check. Instead of returning APPROVED/REJECTED, it returns full portfolio context (current exposure, daily P&L, position sizes, volatility, regime) so the risk agent can reason about it.

**execution_tools.py**: `execute_trade_tool`, `close_position_tool`, `get_fills_tool`

**portfolio_tools.py**: `get_portfolio_state_tool`, `get_regime_tool`, `get_daily_equity_tool`

**intelligence_tools.py**: `get_capitulation_score_tool`, `get_institutional_accumulation_tool`, `get_credit_market_signals_tool`, `get_cross_domain_intel_tool`

**coordination_tools.py**: `record_heartbeat_tool`, `publish_event_tool`, `poll_events_tool`, `get_system_status_tool`

**research_tools.py**: `compute_information_coefficient_tool`, `compute_alpha_decay_tool`, `compute_deflated_sharpe_ratio_tool`

**data_tools.py**: `fetch_market_data_tool`, `compute_all_features_tool`, `compute_technical_indicators_tool` — wraps `qc_data.py` and `qc_indicators.py`

**fundamentals_tools.py**: `get_company_facts_tool`, `get_financial_statements_tool` — wraps `qc_fundamentals.py`

**options_tools.py**: `get_options_chain_tool`, `compute_greeks_tool`, `submit_options_order_tool` — wraps `qc_options.py` and `options_execution.py`

**nlp_tools.py**: `analyze_sentiment_tool`, `extract_entities_tool` — wraps `nlp.py`

**attribution_tools.py**: `run_attribution_analysis_tool` — wraps `attribution.py`

**feedback_tools.py**: `record_tool_error_tool`, `get_open_bugs_tool` — wraps `feedback.py` (self-healing trigger)

**learning_tools.py**: `run_reflexion_tool`, `calibrate_model_tool` — wraps `learning.py`

**meta_tools.py**: `generate_daily_digest_tool`, `auto_promote_eligible_tool` — wraps `meta.py`

**intraday_tools.py**: `get_intraday_signals_tool` — wraps `intraday.py`

**analysis_tools.py**: `run_analysis_tool` — wraps `analysis.py` (fallback signal generator)

**rag_tools.py**: `search_knowledge_base_tool`, `remember_knowledge_tool` — new tools for RAG query and storage.

**web_tools.py**: `web_search_tool`, `web_fetch_tool` — for market-intel agent to search news.

### 3.4 Complete Tool Module Mapping

Every module in `src/quantstack/mcp/tools/` maps to a CrewAI wrapper:

| Source Module | CrewAI Wrapper | Tools Wrapped |
|---------------|---------------|---------------|
| `signal.py` | `signal_tools.py` | 2 |
| `strategy.py` + `_impl.py` | `strategy_tools.py` | 4 |
| `backtesting.py` | `backtest_tools.py` | 5 |
| `ml.py` | `ml_tools.py` | 5 |
| `qc_risk.py` | `risk_tools.py` | 3 |
| `execution.py` | `execution_tools.py` | 4 |
| `options_execution.py` | `options_tools.py` | 1 |
| `portfolio.py` | `portfolio_tools.py` | 4 |
| `capitulation.py` + `institutional_accumulation.py` + `macro_signals.py` + `cross_domain.py` | `intelligence_tools.py` | 4 |
| `coordination.py` | `coordination_tools.py` | 6 |
| `qc_research.py` | `research_tools.py` | 4 |
| `qc_data.py` + `qc_indicators.py` | `data_tools.py` | 3 |
| `qc_fundamentals.py` | `fundamentals_tools.py` | 2 |
| `qc_options.py` | `options_tools.py` | 2 |
| `nlp.py` | `nlp_tools.py` | 2 |
| `attribution.py` | `attribution_tools.py` | 1 |
| `feedback.py` | `feedback_tools.py` | 2 |
| `learning.py` | `learning_tools.py` | 2 |
| `meta.py` | `meta_tools.py` | 2 |
| `intraday.py` | `intraday_tools.py` | 1 |
| `analysis.py` | `analysis_tools.py` | 1 |
| `finrl_tools.py` | `finrl_tools.py` | 2 (if FinRL is active) |
| N/A (new) | `rag_tools.py` | 2 |
| N/A (new) | `web_tools.py` | 2 |

**Total: ~60 tool wrappers across 22 modules.**

### 3.3 Risk Gate Transformation

The current `risk_gate.py` has hardcoded limits:
```python
class RiskGate:
    def check(self, symbol, side, quantity, ...) -> Verdict:
        if position_pct > 0.10: return REJECTED
        if daily_loss_pct > 0.02: return REJECTED
        ...
```

This transforms into:
1. `get_portfolio_context_tool` — returns all the data the risk gate currently uses (portfolio equity, current exposure per symbol, daily P&L, volatility, ADV) as a JSON string
2. The **risk agent** receives this context and reasons about whether the proposed trade is appropriate
3. The risk agent's reasoning is logged to Langfuse with full justification
4. The kill switch mechanism is preserved as a coordination tool (`get_system_status_tool`) — the supervisor agent can activate it

The existing `risk_gate.py` file is NOT deleted. It's preserved as a reference for what the risk agent should consider, and as a potential emergency fallback.

---

## Section 4: Agent Definitions (YAML Configs)

### 4.1 Configuration Structure

Each crew has `config/agents.yaml` and `config/tasks.yaml`. CrewAI reads these with `{variable}` interpolation at runtime.

### 4.2 TradingCrew Agents

Convert each `.claude/agents/*.md` to a CrewAI agent definition. The markdown backstory/instructions become the `backstory` field. The tools list maps to CrewAI tool objects.

**Agents to define in `crews/trading/config/agents.yaml`:**

| Agent ID | Role | Goal | Tier | Tools |
|----------|------|------|------|-------|
| `daily_planner` | Daily Trading Strategist | Create actionable daily trading plan with watchlist rankings and exit analysis | medium | portfolio, signal, strategy, intelligence, rag |
| `position_monitor` | Position Risk Analyst | Assess open positions and recommend HOLD/TRIM/CLOSE with reasoning | medium | portfolio, signal, intelligence, rag |
| `trade_debater` | Trade Thesis Analyst | Conduct rigorous bull/bear/risk debate on entry candidates | heavy | signal, strategy, intelligence, backtest, rag |
| `risk_analyst` | Portfolio Risk Reasoner | Reason about position sizing and portfolio risk given current market conditions | heavy | portfolio, risk_context, intelligence, rag |
| `fund_manager` | Portfolio Approval Authority | Review batch of proposed entries for correlation, concentration, regime coherence | heavy | portfolio, strategy, intelligence, rag |
| `options_analyst` | Options Structure Specialist | Select optimal options structure (spread/condor/straddle) with Greeks validation | heavy | portfolio, signal, backtest, rag |
| `earnings_analyst` | Earnings Event Specialist | Analyze earnings catalysts, IV premium, analyst estimates | medium | signal, intelligence, web, rag |
| `market_intel` | Market Intelligence Analyst | Surface real-time news and sentiment for trading decisions | medium | web, intelligence, rag |
| `trade_reflector` | Post-Trade Analyst | Classify trade outcomes, extract lessons, update knowledge base | medium | portfolio, execution, rag, remember_knowledge |
| `executor` | Trade Executor | Execute approved trades through broker with audit logging | medium | execution, coordination |

### 4.3 ResearchCrew Agents

**Agents in `crews/research/config/agents.yaml`:**

| Agent ID | Role | Goal | Tier | Tools |
|----------|------|------|------|-------|
| `quant_researcher` | Senior Quantitative Researcher | Maintain research programs, generate testable hypotheses, direct alpha discovery | heavy | signal, strategy, backtest, research, ml, intelligence, rag |
| `ml_scientist` | Machine Learning Scientist | Design training experiments, select features, tune hyperparameters, detect drift | heavy | ml, signal, research, rag |
| `strategy_rd` | Strategy Validation Specialist | Evaluate hypotheses via backtest, walk-forward, overfitting detection, alpha decay | heavy | backtest, research, strategy, rag |
| `community_intel` | Quant Community Scout | Scan Reddit, GitHub, arXiv, Twitter for new techniques and alpha factors | light | web, rag, remember_knowledge |

### 4.4 SupervisorCrew Agents

**Agents in `crews/supervisor/config/agents.yaml`:**

| Agent ID | Role | Goal | Tier | Tools |
|----------|------|------|------|-------|
| `health_monitor` | System Health Monitor | Detect unhealthy crews, stale heartbeats, and trigger recovery actions | light | coordination, portfolio |
| `self_healer` | Self-Healing Engineer | Diagnose and recover from system failures autonomously | light | coordination, execution (read-only) |
| `strategy_promoter` | Strategy Lifecycle Manager | Reason about strategy promotion/retirement based on evidence, not thresholds | medium | strategy, backtest, research, portfolio, rag |

### 4.5 Backstory Migration

Each agent's backstory is derived from the corresponding `.claude/agents/*.md` file. The key transformation:

**Before** (Claude Code markdown):
```markdown
---
name: fund-manager
description: "Portfolio-level approval agent"
model: sonnet
---
# Fund Manager
You are a portfolio-level approval agent...
## Hard Rules
- Never approve more than 2 entries per iteration
- Check correlation between candidates
...
```

**After** (CrewAI YAML):
```yaml
fund_manager:
  role: "Portfolio Approval Authority"
  goal: "Review proposed entries holistically for correlation, concentration, capital allocation, strategy diversity, and regime coherence. Approve, reject, or modify each candidate."
  backstory: |
    You are a senior portfolio manager responsible for the final gate before any trade executes.
    You receive a batch of entry candidates that have passed individual analysis. Your job is to
    evaluate them as a portfolio — checking for hidden correlations, excessive sector concentration,
    and alignment with the current market regime.

    You reason about risk limits dynamically based on market conditions, portfolio state, and
    historical outcomes — there are no hardcoded position size limits. Instead, you consider
    volatility, conviction level, correlation with existing positions, and available capital.

    You have access to the knowledge base of past trade outcomes and lessons learned.
    Use this evidence base to inform your decisions.
  llm: "{heavy_model}"
  max_iter: 15
  max_execution_time: 300
  memory: true
  verbose: true
  allow_delegation: false
```

The `{heavy_model}` variable is injected at crew instantiation time from the LLM provider module.

---

## Section 5: Task Definitions and Crew Workflows

### 5.1 TradingCrew Workflow

The TradingCrew runs as a **sequential process** (mirroring the existing trading_loop.md step-by-step flow). Each task depends on the previous.

**Task sequence:**

1. **safety_check** — Agent: executor. Check system status, kill switch, data freshness. If halted, skip entire cycle.

2. **daily_plan** — Agent: daily_planner. Generate daily trading plan with watchlist, exit candidates. Runs once per day (check DB flag). Context: portfolio state, regime, memory. Output: ranked watchlist + exit recommendations.

3. **position_review** — Agent: position_monitor. Review each open position. Context: daily_plan output, portfolio state, signals. Output: per-position HOLD/TRIM/CLOSE recommendation with reasoning. Uses `async_execution=True` to review multiple positions in parallel.

4. **execute_exits** — Agent: executor. Execute any CLOSE/TRIM recommendations from position_review. Context: position_review output.

5. **entry_scan** — Agent: trade_debater. For each symbol on the daily plan watchlist, conduct bull/bear/risk debate. Context: daily_plan, signals, regime, RAG knowledge. Output: per-candidate ENTER/SKIP verdict with reasoning. Uses `async_execution=True` for parallel debates.

6. **risk_sizing** — Agent: risk_analyst. For all ENTER verdicts, reason about position sizing. Context: entry_scan results, portfolio state, full market context. Output: per-candidate position size with reasoning.

7. **portfolio_review** — Agent: fund_manager. Review all sized candidates as a batch. Context: risk_sizing output, portfolio state, correlation data. Output: per-candidate APPROVED/REJECTED/MODIFIED.

8. **options_analysis** — Agent: options_analyst. For approved options entries, select structure. Context: portfolio_review output, IV surface, Greeks. Conditional: only runs if approved entries include options.

9. **execute_entries** — Agent: executor. Execute approved entries. Context: portfolio_review + options_analysis outputs.

10. **reflection** — Agent: trade_reflector. For any positions closed this cycle, analyze outcomes, extract lessons, write to RAG knowledge base. Context: exit results from execute_exits.

11. **persist_state** — Agent: executor. Record heartbeat, write audit trail, update coordination tables.

### 5.2 ResearchCrew Workflow

The ResearchCrew runs as a **hierarchical process** with quant_researcher as the manager. This mirrors the current BLITZ mode where the research orchestrator delegates to domain specialists.

**Manager**: quant_researcher (decides which domain to research, which symbols to target)

**Tasks (delegatable):**

1. **load_context** — Load heartbeat, DB state, memory, cross-domain intel. Always first.

2. **domain_selection** — Score research domains (investment/swing/options) by portfolio gaps, recent P&L, strategy diversity. The manager decides.

3. **hypothesis_generation** — Generate testable hypotheses for selected domain + symbol. Uses RAG to check what's been tried before (workshop_lessons, negative results).

4. **signal_validation** — Run IC, alpha decay tests on hypothesized signals. Agent: strategy_rd.

5. **backtest_validation** — Run in-sample backtest, walk-forward, combinatorial CV. Agent: strategy_rd. The agent reasons about whether results are sufficient (no hardcoded Sharpe threshold).

6. **ml_experiment** — Train ML models if hypothesis involves learnable features. Agent: ml_scientist.

7. **strategy_registration** — If validation passes the agent's reasoning, register strategy as draft. Agent: quant_researcher.

8. **knowledge_update** — Write discoveries (positive or negative) to RAG knowledge base. Agent: quant_researcher.

### 5.3 SupervisorCrew Workflow

Runs on a slower cadence (every 5 minutes). Sequential process.

1. **health_check** — Check heartbeats of trading-crew and research-crew containers. Check Langfuse, Ollama, ChromaDB reachability.

2. **diagnose_issues** — If any service is unhealthy, reason about root cause and recovery action.

3. **execute_recovery** — Take recovery action: restart container via Docker API, flush stale data, switch LLM provider, activate kill switch if unrecoverable.

4. **strategy_lifecycle** — Query strategies in `forward_testing` status. For each, reason about whether to promote, extend, or retire based on performance evidence, market conditions, and RAG knowledge. The agent reasons from evidence, within programmatic safety boundaries.

5. **scheduled_tasks** — Replaces the current `scripts/scheduler.py` cron jobs:
   - **Weekly (Sunday 19:00 ET):** Trigger community-intel scan in ResearchCrew
   - **Monthly:** Trigger execution-researcher audit if 20+ fills exist
   - **Every 30 min during market hours:** Check data freshness, trigger refresh if stale
   - **Daily pre-market (9:00 AM ET):** Run preflight checks, verify data sync, check credit regime
   - **Daily post-market (4:30 PM ET):** Generate daily digest, compact memory

Implementation: The supervisor runner tracks scheduled task timestamps in PostgreSQL (`loop_iteration_context` table with key `scheduler_*`). Each cycle, it checks if any scheduled task is due and creates a coordination event for the appropriate crew.

---

## Section 6: RAG Pipeline (ChromaDB + Ollama)

### 6.1 Architecture

ChromaDB runs as a Docker service in client-server mode (`HttpClient` on port 8000). Ollama provides embeddings via `mxbai-embed-large` model.

Three collections in ChromaDB:

1. **`trade_outcomes`** — Historical trades with outcomes, lessons, reflexion episodes. Metadata: ticker, strategy_id, domain, date, pnl, outcome (win/loss/scratch).

2. **`strategy_knowledge`** — Strategy registry entries, workshop lessons (negative results), ML experiment results. Metadata: strategy_name, domain, status, date.

3. **`market_research`** — Community-intel discoveries, research papers, arXiv findings. Metadata: source, topic, date, relevance_score.

### 6.2 Ingestion Pipeline

**Startup ingestion** (`src/quantstack/rag/ingest.py`):
- Read all `.claude/memory/*.md` files
- Parse into chunks (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
- Tag with metadata (source file, section headers as topics)
- Embed via Ollama `mxbai-embed-large`
- Upsert into appropriate ChromaDB collection
- This runs once on first startup; subsequent startups skip if collections are non-empty

**Continuous ingestion**:
- Trade reflector agent calls `remember_knowledge_tool` after each reflection
- Strategy registration writes to `strategy_knowledge` collection
- Community-intel writes discoveries to `market_research` collection

### 6.3 Retrieval

**`search_knowledge_base_tool`** — CrewAI tool available to all agents:

Parameters: `query` (str), `collection` (str, optional), `ticker` (str, optional), `n_results` (int, default 5)

Returns: top-N relevant documents with metadata, formatted as readable text.

Agents use this tool proactively — their backstories instruct them to check the knowledge base before making decisions.

### 6.4 CrewAI Memory Integration

CrewAI's unified `Memory` class runs alongside ChromaDB:
- Memory handles short-term agent context (what happened this cycle, what the agent just learned)
- ChromaDB handles long-term knowledge (historical trades, research, lessons)
- Both use Ollama `mxbai-embed-large` for embeddings

Memory config on each crew:
```python
memory = Memory(
    embedder={"provider": "ollama", "config": {"model_name": "mxbai-embed-large"}},
    llm=get_model("light"),  # for memory analysis/consolidation
)
```

---

## Section 7: Self-Healing System

### 7.1 Failure Taxonomy and Recovery

Every failure mode has an autonomous recovery path. No human notification — the system fixes itself.

**LLM provider failure:**
- Detection: LLM call raises provider-specific error
- Recovery: Fallback chain (Bedrock → Anthropic → OpenAI → Ollama)
- Implementation: Custom LLM wrapper with retry + fallback logic
- Logging: Langfuse traces include provider switches

**Stuck agent (>10 min on single task):**
- Detection: Watchdog timer per crew cycle
- Recovery: Force-terminate current cycle, restart with fresh crew instance
- Implementation: `threading.Timer` in runner loop; on timeout, set flag to skip current result and start new cycle

**Crashed crew process:**
- Detection: Docker health check (heartbeat file not updated)
- Recovery: Docker `restart: unless-stopped` automatically restarts container
- State recovery: Fresh cycle reads state from PostgreSQL (stateless loop pattern preserved)

**Database connection lost:**
- Detection: Connection pool raises `OperationalError`
- Recovery: Exponential backoff reconnect (2s, 4s, 8s, 16s, max 60s)
- Implementation: Wrap `pg_conn()` with retry decorator

**Ollama down:**
- Detection: Embedding call fails
- Recovery: Skip RAG operations, fall back to CrewAI memory cache. Supervisor restarts Ollama container.
- Graceful degradation: Agents continue without RAG, using only DB state and in-cycle memory

**ChromaDB down:**
- Detection: HTTP client connection error
- Recovery: Same as Ollama — skip RAG, supervisor restarts container
- No data loss: ChromaDB uses persistent volume

**API rate limits (Alpha Vantage, Alpaca):**
- Detection: HTTP 429 response
- Recovery: Exponential backoff with jitter
- Implementation: Already exists in data adapters; preserved in CrewAI tool wrappers

**Stale market data:**
- Detection: Data freshness check (timestamp of last OHLCV update)
- Recovery: Trigger data refresh via data adapters
- Implementation: Coordination tool checks freshness; supervisor triggers refresh if stale

### 7.2 Health Check Architecture

Each crew runner writes a heartbeat file every cycle:

```
/tmp/{crew-name}-heartbeat  # Contains: unix timestamp of last successful cycle
```

Docker health check reads this file:
```
test: ["CMD", "python", "-c", "from quantstack.health.heartbeat import check; check('trading')"]
```

The `check()` function returns exit code 0 if heartbeat is less than N seconds old, 1 otherwise. N varies per crew (trading: 120s, research: 600s, supervisor: 360s).

### 7.3 Graceful Shutdown

Each runner registers SIGTERM and SIGINT handlers:
1. Set `should_stop = True` flag
2. Wait for current cycle to complete (max 60s)
3. Flush Langfuse traces
4. Persist any in-flight state to PostgreSQL
5. Exit cleanly

Docker Compose sends SIGTERM on `docker compose down`, then SIGKILL after `stop_grace_period` (90s).

---

## Section 8: Observability (Self-Hosted Langfuse)

### 8.1 Setup

Langfuse runs as a Docker Compose service with its own Postgres instance. Configuration via environment variables:

- `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` — generated at first startup
- `DATABASE_URL` — points to langfuse-db service
- `NEXTAUTH_SECRET` — random secret for auth

### 8.2 CrewAI Integration

In each runner's `__main__` block, before any crew operations:

```python
from openinference.instrumentation.crewai import CrewAIInstrumentor
CrewAIInstrumentor().instrument(skip_dep_check=True)
```

This automatically traces all LLM calls, tool invocations, and agent operations.

### 8.3 Enhanced Tracing

Beyond automatic instrumentation, add explicit traces for:
- **Provider failover events** — when fallback chain activates
- **Self-healing events** — when watchdog triggers, container restarts
- **Strategy lifecycle events** — promotion, retirement, with full reasoning
- **Capital allocation decisions** — position sizing reasoning

Use Langfuse's `@observe()` decorator on runner functions and CrewAI's event bus for fine-grained events.

### 8.4 Cost Tracking

Langfuse automatically tracks token usage and cost per LLM call. Dashboard shows:
- Cost per agent per day
- Cost per crew per cycle
- Total daily cost across all providers
- Provider usage distribution

### 8.5 Flush on Shutdown

Critical: each runner must call `langfuse.flush()` in its shutdown handler. Without this, the last cycle's traces are lost (Langfuse batches async).

---

## Section 9: Continuous Runner Architecture

### 9.1 Runner Pattern

Each crew has a runner module (`src/quantstack/runners/{name}_runner.py`) that implements the continuous loop:

```python
def main():
    """Entry point for Docker container."""
```

The runner:
1. Initializes Langfuse instrumentation
2. Initializes CrewAI Memory with Ollama embeddings
3. Registers signal handlers for graceful shutdown
4. Starts watchdog timer
5. Enters main loop:
   - Create fresh crew instance (avoid memory leaks from reuse)
   - Build inputs dict from current DB state
   - Call `crew.kickoff(inputs=inputs)`
   - Reset watchdog
   - Write heartbeat
   - Save checkpoint to DB
   - Sleep until next cycle (configurable interval)
6. On shutdown signal: flush Langfuse, persist state, exit

### 9.2 Cycle Intervals

| Runner | Market Hours | After Hours | Weekend |
|--------|-------------|-------------|---------|
| Trading | 5 minutes | 30 minutes | Paused |
| Research | 10 minutes | 30 minutes | 2 hours |
| Supervisor | 5 minutes | 5 minutes | 5 minutes |

**Why 5 minutes, not 60 seconds:** Each TradingCrew cycle runs 11 sequential tasks, each involving at least one LLM call (5-30 seconds each). A realistic cycle takes 3-8 minutes. The 5-minute interval ensures the previous cycle completes before the next starts. If a cycle exceeds the interval, the runner skips the sleep and starts immediately.

Market hours detection: check if current time is within NYSE trading hours (9:30 AM - 4:00 PM ET, Mon-Fri, excluding holidays).

### 9.3 Fresh Crew Instances

Critical lesson from research: **never reuse crew instances across cycles**. LLM client objects, tool caches, and context windows accumulate memory. Each cycle creates a new crew via a factory function.

```python
def create_trading_crew() -> Crew:
    """Factory that builds fresh TradingCrew each cycle."""
```

---

## Section 10: Start/Stop/Status Scripts

### 10.1 start.sh

Replaces the current tmux-based launcher. Full startup sequence:

1. Check prerequisites: Docker, Docker Compose, `.env` file exists with required vars
2. Start infrastructure services: `docker compose up -d postgres langfuse-db ollama chromadb langfuse`
3. Wait for infrastructure health checks (postgres ready, ollama responding, chromadb heartbeat, langfuse healthy)
4. Ensure Ollama models are pulled: `docker compose exec ollama ollama pull mxbai-embed-large && ollama pull llama3.2`
5. Run DB migrations (idempotent): `docker compose run --rm trading-crew python -m quantstack.db --migrate`
6. Bootstrap universe if empty (first run): `docker compose run --rm trading-crew python -m quantstack.data.bootstrap`
7. Run preflight checks: verify Alpaca connection, Alpha Vantage key, data freshness
8. Check data freshness → trigger background sync if stale
9. Trigger one-time RAG ingestion if ChromaDB collections are empty
10. Display credit regime and system status
11. Start crew services: `docker compose up -d trading-crew research-crew supervisor-crew`
12. Wait for crew health checks to pass
13. Print status summary + Langfuse URL (http://localhost:3000)

### 10.2 stop.sh

1. `docker compose down` — sends SIGTERM to all containers
2. Crew runners handle graceful shutdown (persist state, flush Langfuse)
3. Containers stop within 90s grace period

### 10.3 status.sh

Displays:
- Container health status (docker compose ps)
- Last heartbeat timestamp per crew
- Active positions count and total P&L
- Current regime
- Langfuse dashboard URL
- Ollama model status
- ChromaDB collection sizes

Optional `--watch` flag for continuous refresh.

---

## Section 11: Memory Migration

### 11.1 One-Time Ingestion

On first startup, when ChromaDB collections are empty:

1. Read all files from `.claude/memory/`:
   - `strategy_registry.md` → `strategy_knowledge` collection
   - `workshop_lessons.md` → `strategy_knowledge` collection (tagged as negative results)
   - `ml_experiment_log.md` → `strategy_knowledge` collection
   - `trade_journal.md` → `trade_outcomes` collection
   - `session_handoff_*.md` → `market_research` collection
   - `tickers/*.md` → `market_research` collection (per-ticker knowledge)

2. Parse markdown into chunks with metadata extraction
3. Embed via Ollama and upsert to ChromaDB

### 11.2 Ongoing Memory

After migration, agents use:
- **CrewAI Memory** for short-term, in-session context (auto-managed)
- **ChromaDB RAG** for long-term knowledge (via `search_knowledge_base_tool` and `remember_knowledge_tool`)
- **PostgreSQL** for structured operational state (positions, strategies, fills)

The `.claude/memory/` directory is no longer written to by the new system.

---

## Section 12: LLM-Reasoned Risk (No Hardcoded Thresholds)

### 12.1 Philosophy

The current system has ~20 hardcoded numeric thresholds (Sharpe > 0.5, PBO < 0.40, position < 10%, daily loss < 2%, etc.). These are all replaced by LLM reasoning.

The key insight: instead of checking `if sharpe > 0.5: approve`, the system asks the LLM "Given this backtest produced a Sharpe of 0.3 over 200 trades in a ranging regime, and similar strategies in our knowledge base have historically produced Sharpe 0.2-0.8, is this strategy worth forward testing? Consider the current market regime, the strategy's edge mechanism, and what we've learned from past failures."

### 12.2 Risk Agent Context

The risk agent receives a comprehensive context bundle for every decision:

- Current portfolio state (equity, cash, exposure by symbol, sector, asset class)
- Current market regime (trending/ranging, volatility level)
- Proposed trade details (symbol, side, size, conviction, strategy)
- Historical outcomes for similar trades (from RAG)
- Lessons from past failures (from RAG)
- Current volatility environment (VIX, sector vol, symbol vol)
- Correlation with existing positions
- Daily P&L so far
- Available capital ($20K equity, $5K options)

The agent reasons about appropriate sizing and risk given all this context. Its reasoning is logged verbatim to Langfuse.

### 12.3 Strategy Promotion Reasoning

The strategy_promoter agent in SupervisorCrew reasons about promotion:

Context provided:
- Strategy definition (entry/exit rules, regime affinity, economic mechanism)
- Forward testing performance (daily P&L, win rate, drawdown, number of trades)
- Duration in forward testing
- Market conditions during testing period
- Similar strategies' historical performance (from RAG)
- Current portfolio needs (domain gaps, strategy diversity)

The agent reasons: "This strategy has been forward testing for 22 days with 15 trades, a 60% win rate, and a Sharpe of 0.8. The testing period included both a pullback and a rally, providing varied market conditions. Similar momentum strategies in our knowledge base show that 15+ trades with win rate > 55% in varied conditions is a reliable signal. I recommend promotion to live paper trading."

No magic numbers. Just evidence-based reasoning.

### 12.4 Programmatic Safety Boundary (Defense-in-Depth)

The LLM reasons about risk decisions, but a programmatic safety boundary prevents catastrophic outcomes from hallucination, prompt injection, or stochastic variance. This is NOT a contradiction of LLM-first philosophy — it's defense-in-depth.

**The existing `risk_gate.py` is preserved as an outer envelope.** After the LLM risk agent produces its recommendation, the programmatic gate validates it before execution. The gate's limits are wider than what a reasonable LLM would recommend, so in normal operation the LLM's reasoning is the binding constraint.

**Hard outer limits (non-negotiable, cannot be overridden by any agent):**
- Max position size: 15% of equity per symbol
- Daily loss halt: -3% triggers automatic halt (deterministic, persists across restarts via DB sentinel)
- Min liquidity: 200,000 ADV (no illiquid names regardless of LLM conviction)
- Max gross exposure: 200% of equity
- Max options premium at risk: 10% of equity
- Kill switch: one DB write halts everything (supervisor can trigger)

**LLM risk decisions use:**
- Temperature 0 (maximum consistency)
- Structured JSON output (parseable, auditable, not free-text)
- Full context bundle (portfolio, market, RAG knowledge)

**How it works in practice:**
1. Risk agent receives full context and produces JSON: `{"symbol": "AAPL", "recommended_size_pct": 8, "reasoning": "..."}`
2. Programmatic gate validates: `8% < 15%` → PASS
3. Trade executes at 8% (the LLM's recommendation, not the gate's maximum)
4. If LLM hallucinated `recommended_size_pct: 40`, the gate rejects it → logged to Langfuse as safety boundary trigger

The kill switch mechanism is preserved: the supervisor agent can write `kill_switch='active'` to the DB. Additionally, the daily loss halt is a deterministic check (not LLM-reasoned) that survives process restarts.

### 12.5 Capital Boundaries

- Total paper account: $25K ($20K equity, $5K options)
- Alpaca paper mode enforced by `ALPACA_PAPER=true`
- Broker API rejects orders exceeding buying power (physical limit)
- Programmatic gate adds inner limits within the capital boundary

---

## Section 13: Testing Strategy

### 13.1 What to Test

**Unit tests (new):**
- LLM provider fallback chain logic
- Tool wrapper correctness (each wrapper calls underlying function correctly)
- RAG ingestion and retrieval
- Health check / heartbeat logic
- Graceful shutdown handler
- Market hours detection
- Runner cycle logic (mocked crew)

**Integration tests (new):**
- Docker Compose stack starts and all services reach healthy state
- Crew can instantiate with correct agent/task configs
- Tool wrappers can call real underlying functions (with test DB)
- RAG end-to-end: ingest → query → retrieve relevant results
- Langfuse traces appear after crew kickoff

**Existing tests:** Preserved. All tests in `tests/unit/` continue to pass since underlying tool implementations are unchanged.

### 13.2 Test Configuration

New test fixtures:
- Mock LLM provider (returns canned responses, no API calls)
- Test ChromaDB (in-memory client for speed)
- Test PostgreSQL (existing fixture, extended for new tables if any)

### 13.3 Testing LLM-Reasoned Decisions

LLM reasoning can't be unit-tested with assert statements. Instead:
- **Langfuse evaluation pipelines** assess reasoning quality over time
- **Backtesting with recorded decisions** — replay historical data through the system, compare LLM decisions to known-good outcomes
- **Verification phase** (see 13.5) validates decisions in read-only mode before enabling execution

### 13.4 Additional Test Cases (from review feedback)

**E2E Smoke Test:** Run one full TradingCrew cycle with a mock LLM provider (returns canned responses) and test database. Verify: all 11 tasks complete, heartbeat written, audit log entries created, no exceptions.

**Provider Fallback Test:** Mock primary provider to raise error, verify fallback to secondary provider completes successfully.

**Graceful Shutdown Test:** Send SIGTERM to runner process, verify: state persisted to DB, Langfuse flushed, clean exit within 60 seconds.

**Watchdog Test:** Set watchdog timeout to 5 seconds, have a task sleep for 10 seconds. Verify watchdog triggers and runner starts a new cycle.

**RAG Degradation Test:** Stop ChromaDB container. Verify crews continue operating (with degraded mode), no crashes.

**Soak Test Specification:** Run the full Docker Compose stack for 24 hours on a test database with mock market data. Monitor: memory usage (should not grow unbounded), connection pool size (should stay within bounds), heartbeat consistency (no gaps), ChromaDB index size, Langfuse trace count.

### 13.5 Verification Phase (Pre-Production)

Before enabling real paper trading, the system runs in **verification mode** for 48 hours:

1. All crews run normally (LLM reasoning, RAG, memory, observability)
2. Execution tools are no-ops — they log what would happen but don't submit orders to Alpaca
3. All other tools work normally (signals, backtests, ML, data)
4. After 48 hours, review Langfuse traces for:
   - Are risk decisions reasonable? (no 50% single-position recommendations)
   - Are strategy evaluations coherent? (reasoning matches evidence)
   - Are self-healing events working? (provoke a failure and verify recovery)
   - Is cost within expectations?
5. If verification passes, enable execution tools by setting `EXECUTION_ENABLED=true`

This replaces the need for a full parallel-run period while still validating the system before capital is at risk.

---

## Section 14: Docker Resource Limits and Cost Estimation

### 14.1 Docker Resource Limits

Each service gets explicit memory limits in `docker-compose.yml`:

| Service | Memory Limit | Notes |
|---------|-------------|-------|
| postgres | 512MB | Sufficient for quantstack workload |
| langfuse-db | 256MB | Lightweight Langfuse metadata |
| langfuse | 512MB | Next.js app + trace processing |
| ollama | 4GB | mxbai-embed-large (2GB) + llama3.2 (2GB) |
| chromadb | 1GB | Vector index + HTTP server |
| trading-crew | 1GB | Python process + CrewAI overhead |
| research-crew | 1GB | Python process + CrewAI overhead |
| supervisor-crew | 512MB | Lightweight monitoring |
| **Total** | **~9.5GB** | Fits in 16GB with room for OS + dev tools |

### 14.2 Cost Estimation

**At 5-minute trading cycles during market hours (6.5 hours = 78 cycles/day):**

| Crew | Cycles/Day | LLM Calls/Cycle | Total Calls/Day |
|------|-----------|----------------|----------------|
| Trading | 78 | ~11 | ~858 |
| Research | 39 | ~8 | ~312 |
| Supervisor | 78 | ~3 | ~234 |
| **Total** | — | — | **~1,404** |

**Estimated daily cost by provider (assuming ~2K input + ~500 output tokens per call):**

| Provider | Heavy Model Cost | Daily Estimate |
|----------|-----------------|---------------|
| Bedrock (Sonnet) | ~$3/1M in, $15/1M out | ~$5-8/day |
| Anthropic Direct | Same pricing | ~$5-8/day |
| OpenAI (GPT-4o) | ~$2.5/1M in, $10/1M out | ~$4-6/day |
| Gemini (2.5 Pro) | ~$1.25/1M in, $5/1M out | ~$2-4/day |

**Monthly estimate: $60-240/month** depending on provider and usage patterns.

### 14.3 Langfuse Retention

Configure Langfuse trace retention:
- Detailed traces: 30 days
- Aggregated metrics: indefinite
- Implement via periodic cleanup job in supervisor (or Langfuse's built-in retention settings if available)

### 14.4 Log Management

Docker Compose log driver configuration:
```yaml
logging:
  driver: json-file
  options:
    max-size: "50m"
    max-file: "5"
```

Applied to all services. Total max log size: ~1.75GB across 7 services.
