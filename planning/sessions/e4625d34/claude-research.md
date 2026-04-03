# Research Findings: CrewAI Migration for QuantStack

## Part 1: Codebase Architecture Analysis

### Executive Summary

QuantStack is a fully autonomous quantitative trading system built on Claude Code with no human-in-the-loop execution. It orchestrates **two stateless Claude loops** (Trading and Research) that run concurrently and share all state through PostgreSQL. The system includes hard-coded risk gates, self-healing capabilities, and a sophisticated multi-agent research framework.

### 1. Project Structure

```
/Users/kshitijbichave/Personal/Trader/
├── start.sh / stop.sh / status.sh / report.sh   # Lifecycle scripts
├── prompts/                    # Claude loop prompts (read each iteration)
│   ├── trading_loop.md         # Main trading agent orchestrator
│   ├── research_loop.md        # Research orchestrator (3 domains)
│   ├── research_shared.md      # Shared research rules & workflow
│   ├── research_equity_*.md    # Domain-specific research prompts
│   ├── research_options.md
│   └── context_loading.md      # Mandatory state loading
├── .claude/agents/             # 14 agent definitions (YAML-like markdown)
├── .claude/memory/             # Persistent cross-session memory (gitignored)
├── src/quantstack/
│   ├── mcp/tools/              # ~30+ tool functions (async Python)
│   ├── mcp/servers/            # Domain-scoped MCP servers
│   ├── signal_engine/          # 15+ signal collectors (no LLM)
│   ├── execution/              # risk_gate.py, kill_switch.py, brokers
│   ├── core/                   # Backtesting, risk, options, features
│   ├── ml/                     # ML pipeline (training, drift, ensemble)
│   ├── finrl/                  # Deep RL trading
│   ├── alpha_discovery/        # Alpha hypothesis generation
│   ├── coordination/           # Supervisor, preflight, loop lifecycle
│   ├── knowledge/              # Knowledge store (trades, policies, learning)
│   ├── data/                   # Multi-provider data adapters
│   └── db.py                   # PostgreSQL pool & schema
├── scripts/                    # scheduler.py, autoresclaw, dashboard
├── tests/unit/                 # pytest + pytest-asyncio + hypothesis
└── pyproject.toml              # Dependencies, coverage config
```

### 2. Agent Definitions (14 agents in `.claude/agents/`)

#### Orchestration-Level Agents
| Agent | Model | Spawned by | Role |
|-------|-------|-----------|------|
| quant-researcher | sonnet | research_loop | Multi-week research programs, hypothesis generation |
| ml-scientist | sonnet | quant-researcher | ML training, features, drift, calibration |
| strategy-rd | sonnet | quant-researcher | Strategy validation, backtest, overfitting |
| fund-manager | sonnet | trading_loop | Portfolio-level approval (correlation, concentration) |
| daily-planner | sonnet | trading_loop | Daily trading plan + exit review |
| community-intel | haiku | scheduler (weekly) | Reddit/GitHub/arXiv/Twitter scan |
| trade-reflector | sonnet | trading_loop | Post-trade reflection & lessons |
| execution-researcher | sonnet | scheduler (monthly) | Fill quality audit |

#### Trading-Specific Agents
| Agent | Model | Condition | Role |
|-------|-------|-----------|------|
| position-monitor | sonnet | Every iteration per position | HOLD/TRIM/CLOSE |
| trade-debater | sonnet | Every entry candidate | Bull/bear/risk debate |
| risk | sonnet | After debater verdicts | Position sizing, VaR, Kelly |
| market-intel | sonnet | Earnings ± 5 days | Real-time news/sentiment |
| earnings-analyst | sonnet | Earnings within 14 days | EPS, guidance, IV |
| options-analyst | sonnet | After fund-manager approval | Structure selection (Greeks, strikes) |

**Key design**: Each agent is stateless per session, reads from DB/memory, writes back.

### 3. Tool Catalog (30+ tools in `src/quantstack/mcp/tools/`)

All tools are **async Python coroutines** called via `python3 -c "import asyncio; ..."`.

#### Signal Generation
- `get_signal_brief(symbol, regime)` → DailyBrief with 15+ collector outputs
- `run_multi_signal_brief(symbols)` → Batch signal scan

#### Strategy Management
- `register_strategy(name, parameters, entry_rules, exit_rules, ...)` → strategy_id
- `get_strategy(strategy_id)` → full strategy details
- `list_strategies()` → all strategies by status
- `update_strategy_status()` → promote/retire

#### Backtesting
- `run_backtest(strategy_id, symbol, start_date, end_date, ...)` → sharpe, max_dd, PF, win_rate
- `run_walkforward(strategy_id, symbol, window_size, ...)` → OOS sharpe, overfit ratio
- `run_combinatorial_cv(strategy_id, symbol)` → PBO (probability of backtest overfitting)

#### Machine Learning
- `train_ml_model(symbol, model_type, feature_tiers, ...)` → model_id, OOS AUC, SHAP
- `predict_ml_signal(model_id, symbol, date)` → probability, confidence, SHAP
- `check_concept_drift(model_id)` → PSI per feature, retrain recommendation

#### Risk Management
- `compute_var(returns, confidence)` → VaR dollar/pct
- `stress_test_portfolio(positions, scenario)` → stressed P&L
- `compute_position_size(symbol, conviction, account_size)` → Kelly-based size

#### Execution
- `execute_trade(symbol, side, quantity, ...)` → order_id, fill_price
- `close_position(symbol, reason)` → realized P&L
- `get_fills(symbol, days_back)` → fill history

#### Portfolio
- `get_portfolio_state()` → equity, cash, exposure, positions
- `get_regime(symbol)` → regime name, confidence, bias
- `get_strategy_pnl(strategy_id)` → daily P&L

#### Intelligence
- `get_capitulation_score(symbol)` → panic/fear/neutral with evidence
- `get_institutional_accumulation(symbol)` → accumulation score
- `get_credit_market_signals()` → credit regime, spreads
- `get_cross_domain_intel()` → wins by domain, correlations

#### Coordination
- `record_heartbeat(loop_name, iteration, status)` → health tracking
- `publish_event(event_type, payload)` / `poll_events(consumer_id)` → event bus
- `record_tool_error(tool_name, error)` → self-healing trigger
- `get_system_status()` → kill switch, risk halt

#### Research
- `compute_information_coefficient(signal, returns, horizon)` → IC
- `compute_alpha_decay(signal, returns, max_horizon)` → half-life
- `compute_deflated_sharpe_ratio(returns, n_trials)` → DSR

### 4. Database Schema (PostgreSQL)

#### Operational Tables
| Table | Purpose |
|-------|---------|
| `strategies` | Strategy registry (status: draft/forward_testing/live/retired/failed) |
| `positions` | Open positions with P&L |
| `fills` | Executed orders |
| `audit_log` | Decision trail (every decision logged with reasoning) |
| `loop_iteration_context` | Persistent loop state (key-value per loop) |
| `system_state` | Global state (kill_switch, data_freshness) |
| `heartbeat` | Loop health tracking |
| `bug_report` | Tool errors (self-healing trigger) |

#### Analytics Tables
| Table | Purpose |
|-------|---------|
| `ml_experiments` | ML training log |
| `strategy_daily_pnl` | Per-strategy P&L |
| `research_programs` | Active research programs |
| `research_queue` | Queued hypotheses |
| `research_wip` | Active research work-in-progress |
| `ohlcv` | Market data cache |
| `universe` | Tradeable symbol universe |
| `reflexion_episodes` | Post-trade reflections |
| `policy_store` | Learned trading policies |

Connection pattern: `pg_conn()` context manager, pool of 2-20 connections.

### 5. Execution Layer

**Risk Gate** (`risk_gate.py`): IMMUTABLE. Limits: 10% per symbol, -2% daily loss halt, 500K min volume, 8% max options premium, 7 DTE minimum.

**Kill Switch** (`kill_switch.py`): IMMUTABLE. One DB write halts everything.

**Brokers**: Alpaca (primary, paper+live), Paper (simulation), eTrade (secondary).

**Order lifecycle**: PENDING → FILLED/PARTIAL/CANCELLED/REJECTED.

### 6. Loop Architecture

**Trading loop** (every 60s market hours, 30 min off-hours):
Safety gate → Daily plan → Context → Position monitor → Entry scan → Trade debate → Risk sizing → Fund manager → Execute → Reflect → Persist

**Research loop** (every 5 min market hours [haiku], 30 min after-hours [sonnet]):
Context loading → Domain selection → Domain research (investment/swing/options) → Validation pipeline → Persist

**BLITZ Mode**: 3 domain agents × 2-3 specialists = 6-9 agents per symbol, up to 27 concurrent.

### 7. Testing

- **Framework**: pytest + pytest-asyncio + hypothesis
- **Location**: `tests/unit/` with conftest.py fixtures
- **Coverage exclusions**: External APIs, WebSocket, GPU, full-system flows, LLM agents
- **Run**: `pytest tests/` or `pytest --cov src/`

### 8. Configuration

Required env vars: `TRADER_PG_URL`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPHA_VANTAGE_API_KEY`, `LLM_PROVIDER`

---

## Part 2: Web Research — CrewAI Architecture Patterns 2025

### Agent/Task/Crew Design

**Agents** use three core attributes: `role`, `goal`, `backstory`. YAML configuration recommended with `{topic}` variable interpolation.

```python
agent = Agent(
    role="Senior Data Scientist",
    goal="Analyze datasets to provide actionable insights",
    backstory="Over 10 years in data science...",
    llm="anthropic/claude-sonnet-4",
    tools=[my_tool],
    memory=True,
    max_iter=20,
    max_execution_time=300,
    respect_context_window=True
)
```

**Tasks** support: `context` dependencies (DAG chaining), `async_execution`, `output_pydantic` for typed results, `guardrails` for validation with retry, callbacks.

**Crew process types**:
- **Sequential**: Linear task execution, each output feeds next
- **Hierarchical**: Manager agent delegates dynamically, validates results

### Tool Wrapping (3 approaches)

1. **`@tool` decorator** — simplest, for wrapping existing functions
2. **`BaseTool` subclass** — for typed input schemas via Pydantic
3. **Async tools** — both decorator and class support `async def`

```python
from crewai.tools import tool

@tool("Calculate Risk Score")
def calculate_risk(ticker: str, position_size: float) -> str:
    """Calculate risk score for a given position."""
    return f"Risk score: {score}"
```

### Memory System (Unified in 2025)

Single `Memory` class with:
- **Hierarchical scopes**: `/project/alpha`, `/agent/researcher`, `/company/knowledge`
- **Composite scoring**: semantic × similarity + recency × decay + importance × weight
- **Auto-analysis**: LLM infers scope, categories, importance
- **Consolidation**: Auto-dedup at 0.85 similarity threshold

```python
memory = Memory(
    embedder={"provider": "ollama", "config": {"model_name": "mxbai-embed-large"}},
    llm="ollama/llama3.2",
    recency_weight=0.5, semantic_weight=0.3, importance_weight=0.2
)
```

### CrewAI Flows (Multi-Crew Orchestration)

Flows use `@start`, `@listen`, `@router` decorators for complex pipelines:
- `@persist` — automatic state recovery across restarts (SQLite backend)
- `and_()`/`or_()` — parallel join semantics
- Structured state via Pydantic BaseModel
- Built-in `self.remember()` and `self.recall()`
- Visualization: `flow.plot("diagram.html")`

### LLM Provider Configuration

Model string format: `provider/model-id`

```python
LLM(model="anthropic/claude-sonnet-4", max_tokens=4096)
LLM(model="openai/gpt-4o", temperature=0.7)
LLM(model="ollama/llama3:70b", base_url="http://localhost:11434")
```

### Gotchas
1. `max_tokens` is mandatory for Anthropic
2. Don't over-delegate — circular delegation loops
3. YAML preferred over code for agent/task definitions
4. Hierarchical process needs capable manager LLM
5. Memory requires explicit embedding provider config
6. `respect_context_window=True` important for long tasks

---

## Part 3: Web Research — Langfuse + CrewAI Observability

### Integration Setup

Uses **OpenInference instrumentation**:

```python
from openinference.instrumentation.crewai import CrewAIInstrumentor
CrewAIInstrumentor().instrument(skip_dep_check=True)
```

### What Gets Traced
- LLM calls (prompts, completions, model params)
- Agent operations (execution flow, collaboration)
- Task processing (inputs, outputs, performance)
- Tool invocations
- Complete execution flow from kickoff to result

### Cost Tracking
- Ingestion-level (preferred): accepts usage from LLM responses
- Inference-level (fallback): auto-calculates via built-in tokenizers
- Supports per-model pricing tiers

### CrewAI Event System
Native event bus for fine-grained observability: crew lifecycle, agent execution, task management, tool operations, LLM calls, memory operations, flow events.

### Key Gotchas
1. Must call `langfuse.flush()` in short-lived processes
2. Self-hosted Langfuse recommended for trading (avoid sending strategy data to cloud)
3. `LANGFUSE_DEBUG=True` essential for diagnosing missing traces

---

## Part 4: Web Research — Ollama + ChromaDB RAG Patterns

### Ollama Embedding Models
| Model | Parameters | Best For |
|-------|-----------|----------|
| `mxbai-embed-large` | 334M | Highest quality, general purpose |
| `nomic-embed-text` | 137M | Good balance quality/speed |
| `all-minilm` | 23M | Fastest, resource-constrained |

Recommendation for financial domain: `mxbai-embed-large` (local) or Voyage AI (cloud).

### ChromaDB Configuration
- `PersistentClient(path=...)` for production (never in-memory)
- Custom `OllamaEmbeddingFunction` for local embeddings
- Supports hybrid search, metadata filtering at query time

### RAG as CrewAI Tool — Two Options

**Option A: Built-in Knowledge system**
```python
crew = Crew(
    agents=[...], tasks=[...],
    knowledge_sources=[PDFKnowledgeSource(file_paths=["research/"])],
    embedder={"provider": "ollama", "config": {"model": "mxbai-embed-large"}}
)
```

**Option B: Custom RAG tool** (more control)
```python
@tool("Search Trading Knowledge Base")
def search_knowledge_base(query: str, ticker: str = None) -> str:
    """Search trading knowledge base."""
    results = collection.query(query_texts=[query], n_results=5, where={"ticker": ticker})
    return "\n---\n".join(results["documents"][0])
```

### Gotchas
1. Embedding model must be consistent (can't query with different model than ingestion)
2. Ollama must be running (`ollama serve`)
3. ChromaDB absolute paths — relative paths break when CWD changes
4. CrewAI Knowledge auto-stores per agent role name — renaming orphans data

---

## Part 5: Web Research — Long-Running Autonomous Agent Systems

### Process Management Options

**Supervisord** (recommended for dev/simplicity): Auto-restart, priority ordering, web UI.
**systemd** (Linux production): WatchdogSec, Restart=on-failure.
**Docker Compose** (containerized): Health checks, depends_on, restart policies.

### Self-Healing Patterns
1. **Exponential backoff** with jitter for API failures
2. **CrewAI Flows `@persist`** for crash recovery (SQLite-backed state)
3. **Database checkpointing** for non-Flow patterns
4. **Graceful shutdown** via SIGTERM/SIGINT handlers with state persistence

### Continuous Operation Pattern

CrewAI has no native "run forever" mode. Recommended: **loop with fresh kickoffs**:

```python
def run_continuous_crew(crew_factory, cycle_interval=300):
    shutdown = GracefulShutdown()
    watchdog = AgentWatchdog(timeout_seconds=600)
    while not shutdown.should_stop:
        crew = crew_factory()  # fresh each cycle
        watchdog.start_cycle()
        result = resilient_crew_kickoff(crew, inputs=get_current_inputs())
        watchdog.end_cycle()
        write_heartbeat()
        save_checkpoint("trading", result.json_dict)
```

### Health Checks
- File-based heartbeat (simplest, Docker-compatible)
- Database-backed health for multi-process monitoring
- Watchdog timer for stuck/hung agent detection

### Key Gotchas
1. Create fresh crew instances each cycle (memory leaks from reuse)
2. `langfuse.flush()` in shutdown handler (or lose last cycle's telemetry)
3. CrewAI AMP (managed deployment) is enterprise/cloud-only — not suitable for local trading

---

## Part 6: Multi-LLM Provider Strategy

### User Requirement
Support Gemini, OpenAI, Claude API, and AWS Bedrock as LLM providers with appropriate default models per agent.

### CrewAI LLM Provider Support
CrewAI uses LiteLLM under the hood, supporting 100+ providers with unified `provider/model-id` format:

| Provider | Format | Install |
|----------|--------|---------|
| Anthropic (direct) | `anthropic/claude-sonnet-4` | `crewai[anthropic]` |
| AWS Bedrock | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | `crewai[bedrock]` |
| OpenAI | `openai/gpt-4o` | `crewai[openai]` (default) |
| Google Gemini | `gemini/gemini-2.5-pro` | `crewai[google]` |
| Ollama (local) | `ollama/llama3.2` | `crewai[litellm]` |

### Recommended Default Models per Agent Role

| Agent | Reasoning Tier | Anthropic | Bedrock | OpenAI | Gemini | Ollama (fallback) |
|-------|---------------|-----------|---------|--------|--------|-------------------|
| **fund-manager** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **quant-researcher** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **trade-debater** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **risk** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **ml-scientist** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **strategy-rd** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **options-analyst** | Heavy reasoning | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o | gemini-2.5-pro | — |
| **earnings-analyst** | Medium | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| **position-monitor** | Medium | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| **daily-planner** | Medium | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| **market-intel** | Medium | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| **trade-reflector** | Medium | claude-sonnet-4 | anthropic.claude-sonnet-4 | gpt-4o-mini | gemini-2.0-flash | — |
| **community-intel** | Light (batch) | claude-haiku-4-5 | anthropic.claude-haiku-4-5 | gpt-4o-mini | gemini-2.0-flash | llama3.2 |
| **execution-researcher** | Light (batch) | claude-haiku-4-5 | anthropic.claude-haiku-4-5 | gpt-4o-mini | gemini-2.0-flash | llama3.2 |
| **Memory/Embeddings** | Embeddings only | — | — | text-embedding-3-small | — | mxbai-embed-large |
| **RAG summarization** | Light | — | — | — | — | llama3.2 |

### Provider Selection Logic
```python
# Config-driven provider selection
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")  # anthropic|bedrock|openai|gemini
LLM_TIER_MAP = {
    "heavy": {"anthropic": "anthropic/claude-sonnet-4", "bedrock": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0", ...},
    "medium": {...},
    "light": {...},
}
```

### Env Vars Required per Provider
```bash
# Anthropic (direct API)
ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GEMINI_API_KEY=...

# Ollama (local, no auth)
OLLAMA_BASE_URL=http://localhost:11434
```
