# QuantPod Architecture

QuantPod is a CrewAI-based multi-agent trading system. The crew produces a `DailyBrief`; Claude Code (acting as portfolio manager) receives it via MCP and makes the final trade decision.

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                  CLAUDE CODE (You)                   │
│  Portfolio Manager · Strategy Researcher · Architect │
└──────────────────────┬──────────────────────────────┘
                       │ MCP: run_analysis → DailyBrief
┌──────────────────────▼──────────────────────────────┐
│            QuantPod MCP Server                       │
│  packages/quant_pod/mcp/server.py                    │
└──────────────────────┬──────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌───────────────┐      ┌───────────────────────┐
│ TradingCrew   │      │  QuantCore MCP         │
│ (stop_at_     │      │  (40+ quant tools)     │
│  assistant)   │      │  packages/quantcore/   │
└───────┬───────┘      │  mcp/server.py         │
        │              └───────────────────────┘
        │ DailyBrief
        ▼
┌───────────────────────────────────────────┐
│              Execution Layer               │
│  RiskGate → SmartOrderRouter → Broker     │
│  KillSwitch · SignalCache · TickExecutor  │
│  packages/quant_pod/execution/            │
└───────────────────────────────────────────┘
```

---

## TradingCrew — Agent Hierarchy

The crew runs with `stop_at_assistant=True`. Layer 3 (Trading Assistant) is the final crew output; Claude Code is Layer 4.

### Layer 1 — Individual Contributors (13 agents, async)

| IC | Pod | Role |
|----|-----|------|
| `data_ingestion_ic` | data | Fetch OHLCV data |
| `market_snapshot_ic` | market_monitor | Current price/volume snapshot |
| `regime_detector_ic` | market_monitor | ADX/ATR regime classification (no LLM) |
| `trend_momentum_ic` | technicals | RSI, MACD, ADX, SMA |
| `volatility_ic` | technicals | ATR, Bollinger Bands, VaR |
| `structure_levels_ic` | technicals | Support/resistance levels |
| `statarb_ic` | quant | ADF test, information coefficient |
| `options_vol_ic` | quant | IV, Greeks, skew |
| `risk_limits_ic` | risk | VaR, stress tests, limit checks |
| `calendar_events_ic` | risk | Earnings, FOMC, economic releases |
| `news_sentiment_ic` | alpha_signals | Alpha Vantage news + sentiment |
| `options_flow_ic` | alpha_signals | Unusual option activity (UOA detection) |
| `fundamentals_ic` | alpha_signals | P/E, debt ratios, earnings quality |

### Layer 2 — Pod Managers (6 agents)

`data_pod_manager`, `market_monitor_pod_manager`, `technicals_pod_manager`, `quant_pod_manager`, `risk_pod_manager`, `alpha_signals_pod_manager`

### Layer 3 — Trading Assistant

Synthesizes all pod outputs into a `DailyBrief` (Pydantic model). Crew stops here.

### Layer 4 — Claude Code

Receives the `DailyBrief` via `run_analysis` MCP tool. Makes the trade decision.

---

## Regime Detection

`regime_detector_ic` is deterministic — uses indicators, not LLMs.

- **ADX (14-period)**: > 25 = trending, < 20 = ranging, 20–25 = transition
- **ATR percentile**: < 25th = low vol, 75–90th = high, > 90th = extreme

Output: `{trend_regime, volatility_regime, adx, atr, atr_percentile, confidence}`

### Regime-Adaptive Crew Config

`packages/quant_pod/crews/regime_config.py` maps regimes to crew configuration:

| Regime | Active ICs | Behaviour |
|--------|------------|-----------|
| `trending` + `normal` vol | Full suite (13 ICs) | Standard analysis |
| `ranging` + `low` vol | Subset (7 ICs, skip trend/momentum) | Faster, simpler |
| Any + `extreme` vol | Full suite + elevated risk thresholds | Tighter limits |
| `unknown` | Full suite | Paper mode only |

---

## LLM Configuration (`packages/quant_pod/llm_config.py`)

Agent model selection is handled by `llm_config.py`. CrewAI uses LiteLLM under the hood; model strings follow `provider/model_id` format.

### Provider tiers

| Tier | Env var | Examples |
|------|---------|---------|
| Tier 1 (recommended) | `LLM_PROVIDER` | `bedrock`, `anthropic`, `openai`, `vertex_ai`, `gemini` |
| Tier 2 | `LLM_PROVIDER` | `azure`, `groq`, `together_ai`, `fireworks_ai`, `mistral` |
| Tier 3 | `LLM_PROVIDER` | `ollama`, `custom_openai` |

`LLM_FALLBACK_CHAIN` (comma-separated) specifies backup providers if the primary is unavailable, e.g. `LLM_FALLBACK_CHAIN=anthropic,openai`. Credentials are detected and cached per provider; missing credentials trigger a graceful skip to the next in chain.

### Default model tiers

| Agent tier | Bedrock default | OpenAI default |
|-----------|----------------|----------------|
| IC (`*_ic`) | `claude-haiku-4-5` | `gpt-4o` |
| Pod manager | `claude-sonnet-4-6` | `gpt-4o` |
| Trading Assistant | `claude-sonnet-4-6` | `gpt-4o` |

Override per-tier:
```bash
LLM_MODEL_IC=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
LLM_MODEL_POD=bedrock/us.anthropic.claude-sonnet-4-6
LLM_MODEL_ASSISTANT=openai/gpt-4o   # mix providers freely
```

---

## Execution Layer

`packages/quant_pod/execution/` — 12 modules, all receiving a single `PgConnection` via dependency injection.

### Execution Path

```
TradingDayFlow.run()
    │
    ├─ 1. KillSwitch.check()
    │      sentinel file at ~/.quant_pod/KILL_SWITCH_ACTIVE
    │
    ├─ 2. TradingCrew.kickoff() → DailyBrief
    │
    ├─ 3. RiskGate.check()
    │      enforces: position size, daily loss, liquidity, restricted list
    │
    ├─ 4. Signal → SignalCache (TTL: 15 min default)
    │      PostgreSQL-backed for crash recovery
    │
    ├─ 5. TickExecutor reads cache → SmartOrderRouter
    │
    └─ 6. SmartOrderRouter → PaperBroker | Alpaca | IBKR | eTrade
               selection: best spread + latency + commission
```

### Module Reference

| Module | Responsibility |
|--------|---------------|
| `portfolio_state.py` | Open positions and cash; source of truth |
| `risk_gate.py` | Hard position/loss/liquidity limits — code-enforced |
| `kill_switch.py` | Emergency halt via sentinel file; survives restarts |
| `risk_state.py` | In-memory hot-path mirror for TickExecutor |
| `signal_cache.py` | TTL-gated signal store (minute analyst → tick executor) |
| `tick_executor.py` | Real-time order submission loop |
| `smart_order_router.py` | Best-execution routing across brokers |
| `paper_broker.py` | Internal fill simulation with slippage/commission |
| `broker_factory.py` | Selects PaperBroker or live broker based on env |
| `order_lifecycle.py` | Order state machine (submitted → filled/rejected) |
| `microstructure_pipeline.py` | Bid-ask estimation, micro-features |

---

## Dependency Injection (TradingContext)

`packages/quant_pod/context.py` is the single wiring point. All services share one `PgConnection` instance.

```python
from quant_pod.context import create_trading_context

# Production — PostgreSQL (TRADER_PG_URL env var)
ctx = create_trading_context()

# Tests — in-memory DuckDB via PgConnection._from_memory(), zero server required
ctx = create_trading_context(db_path=":memory:")

# ctx fields:
#   db           — PgConnection (PostgreSQL pool connection, or DuckDB in-memory for tests)
#   portfolio    — PortfolioState
#   risk_gate    — RiskGate
#   risk_state   — RiskState (hot-path mirror)
#   signal_cache — SignalCache
#   kill_switch  — KillSwitch
#   broker       — PaperBroker (or EtradeBroker when USE_REAL_TRADING=true)
#   audit        — DecisionLog
#   blackboard   — Blackboard (PostgreSQL-backed agent memory)
#   session_id   — UUID threaded through all logs
```

---

## State Management (PostgreSQL + DuckDB)

**Operational state** lives in PostgreSQL (`TRADER_PG_URL`, default `postgresql://localhost/quantpod`). All four MCP server instances share the same pool — true MVCC, no file-lock contention. All services share one `PgConnection` for ACID cross-service transactions.

**Analytics state** (ML experiments, backtests, research programs, reflexion, prompt optimization) stays in DuckDB at `data/trader.duckdb` — append-heavy workloads where DuckDB's columnar engine excels and concurrency is not required.

| Table | Owner | Description |
|-------|-------|-------------|
| `positions` | `PortfolioState` | Open positions and cash balance |
| `cash_balance` | `PortfolioState` | Current cash |
| `closed_trades` | `PortfolioState` | Realized P&L with session tracking |
| `fills` | `PaperBroker` | Order execution with slippage/commission |
| `decision_events` | `DecisionLog` | Every agent decision with context snapshots |
| `agent_memory` | `Blackboard` | Structured agent memory (indexed by symbol/session) |
| `signal_state` | `SignalCache` | TTL-gated signals for crash recovery |
| `strategies` | MCP server | Strategy registry (draft → forward_testing → live) |
| `regime_strategy_matrix` | MCP server | Regime → allocation weights |
| `agent_skills` | `SkillTracker` | IC prediction accuracy |
| `calibration_records` | `Calibration` | Stated confidence vs actual P&L |
| `system_state` | `KillSwitch` | Kill switch and halt flags |

**Why PostgreSQL for operational state?** MVCC allows concurrent reads and writes from all four MCP server instances simultaneously — no file-lock serialization, no degraded mode, no retry logic. ACID transactions still guarantee consistency across services.

---

## Blackboard (Agent Memory)

The `Blackboard` class (`packages/quant_pod/memory/blackboard.py`) is a drop-in replacement for the old markdown file. The public API is unchanged; the backing store is `agent_memory` in DuckDB.

```python
from quant_pod.memory.blackboard import Blackboard

board = Blackboard(conn=conn, session_id=session_id)

# Write structured entry
board.write(
    agent="regime_detector_ic",
    symbol="SPY",
    category="regime",
    content={"trend": "trending_up", "volatility": "normal", "adx": 31.2}
)

# Read latest entries for a symbol
entries = board.read(symbol="SPY", limit=10)

# Read by agent
entries = board.read(agent="regime_detector_ic")
```

---

## Audit Trail

`packages/quant_pod/audit/decision_log.py` logs every agent decision and execution event:

```python
audit.log(
    event_type="ic_analysis",
    agent_name="trend_momentum_ic",
    symbol="SPY",
    action="LONG_SIGNAL",
    confidence=0.78,
    output_summary="RSI 58, MACD bullish crossover, ADX 32",
    market_data_snapshot=snapshot_dict,
    portfolio_snapshot=portfolio_dict,
)
```

Query via MCP: `get_audit_trail(symbol="SPY", limit=20)` or `get_recent_decisions()`.

---

## New Agents

### `portfolio_optimizer_agent.py`

Mean-variance optimizer. Converts per-symbol signals (from all pod managers) into MV-optimal capital weights. Uses the current portfolio state to compute efficient frontier weights subject to the risk gate's position limits.

### `microstructure_signal_agent.py`

Bid-ask spread estimation, order book depth features, and intraday momentum microstructure signals. Used by `alpha_signals_pod_manager` alongside fundamentals and news.

---

## Flows

### `TradingDayFlow` (updated)

New constructor parameters:
- `signal_cache: Optional[SignalCache]` — async signal handoff to TickExecutor
- `signal_ttl_seconds: int = 900` — TTL for signals in SignalCache

New behaviour per run:
1. Kill switch check
2. Inject portfolio state into agent context (agents know current positions)
3. Run TradingCrew → DailyBrief
4. Derive regime-adaptive crew config
5. RiskGate check
6. Publish to SignalCache or execute directly
7. TCA engine records arrival prices + implementation shortfall
8. RL online adapter provides post-trade feedback (shadow mode)

### `IntraDayMonitorFlow` (new)

Intraday monitoring loop. Start with: `quantpod-monitor`

Watches open positions every N minutes, checks risk limits, triggers stop-loss exits or partial reductions when thresholds are breached.

### `StrategyValidationFlow` (new)

Validates a registered strategy against out-of-sample data before promotion from `forward_testing` to `live`.

---

## Directory Structure

```
packages/quant_pod/
├── agents/
│   ├── regime_detector.py          ADX/ATR regime (real indicators)
│   ├── portfolio_optimizer_agent.py  MV-optimal weights
│   └── microstructure_signal_agent.py  Bid-ask, depth features
├── api/                            FastAPI REST server (quantpod-api)
├── audit/
│   ├── decision_log.py             Structured event logging
│   └── models.py                   Pydantic event schemas
├── context.py                      TradingContext DI container
├── crews/
│   ├── config/tasks.yaml           Task definitions
│   ├── decoder_crew.py             Phase 4: reverse-engineer strategies
│   ├── regime_config.py            Regime → IC selection + thresholds
│   ├── registry.py
│   ├── schemas.py                  DailyBrief + all crew schemas
│   └── trading_crew.py             Main 13-IC crew
├── db.py                           PostgreSQL + DuckDB schema, migrations, PgConnection pool
├── execution/                      12-module execution layer
├── flows/
│   ├── intraday_monitor_flow.py    Intraday monitoring
│   ├── strategy_validation_flow.py  OOS validation
│   └── trading_day_flow.py         Main daily flow
├── guardrails/                     Risk rule enforcement
├── knowledge/store.py              Portfolio snapshot time-series
├── learning/
│   ├── calibration.py              Stated confidence vs P&L tracking
│   └── skill_tracker.py            IC prediction accuracy
├── memory/blackboard.py            PostgreSQL-backed agent memory
├── mcp/server.py                   30+ MCP tools (Phases 1–6)
├── monitoring/
│   ├── alpha_monitor.py            Signal quality degradation
│   ├── degradation_detector.py     IS vs OOS divergence
│   └── metrics.py                  Sharpe, drawdown, consistency
└── tools/
    ├── discord/                    Discord MCP bridge
    ├── etrade/                     eTrade auth/client/models
    └── options_flow_tools.py       UOA detection tools
```
