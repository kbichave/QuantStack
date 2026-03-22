# QuantStack Architecture

This document provides an overview of the QuantStack system and how its components interact.

## Repository Structure

```
QuantStack/
├── src/quantstack/           # Unified package
│   ├── core/                 # Research library (200+ indicators, backtesting, ML, options, RL)
│   ├── signal_engine/        # 7 concurrent Python collectors (no LLM)
│   ├── autonomous/           # Unattended trading loops
│   ├── coordination/         # Inter-loop coordination (event bus, locks, promoter, supervisor)
│   ├── alpha_discovery/      # Strategy generation (grid search + Grammar GP)
│   ├── execution/            # Risk gate, order lifecycle, broker routers
│   ├── ml/                   # ML pipeline (LightGBM, XGBoost, CatBoost, TFT)
│   ├── data/                 # Data fetching, storage, streaming
│   ├── learning/             # IC/ICIR tracking, drift detection
│   ├── monitoring/           # AlphaMonitor, DegradationDetector
│   ├── mcp/                  # Unified MCP server (120+ tools)
│   ├── api/                  # FastAPI REST server
│   ├── optimization/         # ReflexionMemory, CreditAssigner, TextGrad
│   └── ...                   # flows, guardrails, risk, crews, features, intraday, knowledge
├── adapters/                 # Broker MCP servers (alpaca_mcp, ibkr_mcp, etrade_mcp)
├── .claude/                  # Skills, agents, memory
├── prompts/                  # Ralph loop prompts
├── scripts/                  # Scheduler, loop launchers
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Package Overview

### quantstack.core — Research Library

Foundation for all quantitative analysis:

- **200+ Technical Indicators**: trend, momentum, volatility, volume, market structure
- **Backtesting Engine**: event-driven with transaction cost modeling, multi-timeframe
- **ML Integration**: LightGBM, XGBoost, CatBoost, SHAP explainability
- **RL Agents**: PPO/DQN for execution, sizing, spread trading (experimental)
- **Options Pricing**: Black-Scholes, Greeks, IV surface
- **Market Microstructure**: order book simulation, impact models, OFI, VPIN
- **Execution Models**: SmartOrderRouter, TCA engine, kill switch, risk gate

### quantstack.signal_engine — Signal Generation

Seven concurrent Python collectors produce a `SignalBrief` — structured output with market bias, conviction, risk environment, and regime detail. No LLM calls, 2–6 seconds. Fault-tolerant: individual collector failures don't block the brief.

### quantstack.execution — Trade Execution

- **RiskGate**: hard-coded pre-trade checks (position size, daily loss, liquidity, options DTE)
- **KillSwitch**: file-sentinel emergency halt, survives restarts
- **SmartOrderRouter**: auto-routes to best available broker
- **OrderLifecycle**: state machine for order management
- **PaperBroker**: zero-config fallback with slippage simulation

### quantstack.coordination — Autonomous Operations

- **UniverseRegistry**: SP500 + NASDAQ-100 + 50 ETFs (~700 symbols)
- **EventBus**: DuckDB pub/sub for inter-loop communication
- **AutoPromoter**: evidence-based forward_testing → live promotion
- **LoopSupervisor**: heartbeat monitoring, crash recovery
- **PortfolioOrchestrator**: correlation, sector cap, position gating

### quantstack.mcp — Unified MCP Server

Single `quantstack-mcp` server exposes 120+ tools across all subsystems. Replaces the previous separate `quantcore-mcp` and `quantpod-mcp` servers.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     CLAUDE CODE (Portfolio Brain)                  │
│        Skills: /trade  /invest  /options  /workshop  /review      │
│        Memory: .claude/memory/ (strategy registry, trade journal) │
└─────────────────────────────┬────────────────────────────────────┘
                              │ MCP calls
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     quantstack-mcp (120+ tools)                   │
│  signals · backtesting · ML · options · execution · coordination  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│ SignalEngine      │ │ Execution Layer │ │ DuckDB State       │
│ 7 collectors      │ │ RiskGate        │ │ positions/fills/   │
│ No LLM, 2–6s     │ │ KillSwitch      │ │ audit/strategies/  │
│ → SignalBrief     │ │ SmartOrderRouter│ │ universe/events    │
└──────────────────┘ │ → Broker        │ └────────────────────┘
                     └─────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Data Sources                               │
│  Alpha Vantage · FD.ai · Alpaca · Polygon (DATA_PROVIDER_PRIORITY)│
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Data Ingestion**: market data fetched via `DATA_PROVIDER_PRIORITY` (FD.ai → Alpaca → Alpha Vantage)
2. **Feature Engineering**: `quantstack.core` computes 200+ indicators + multi-timeframe features
3. **Signal Generation**: `SignalEngine` (7 collectors) produces `SignalBrief`
4. **Decision**: Claude Code reads `SignalBrief` via `get_signal_brief` MCP, makes trade decision
5. **Risk Check**: `RiskGate` enforces position size, daily loss, and liquidity limits
6. **Execution**: `SmartOrderRouter` routes to best available broker (or `PaperBroker`)
7. **Audit**: every decision and fill logged to DuckDB audit trail
8. **Learning**: IC/ICIR tracking, calibration, and optimization modules update from outcomes

---

## Key Design Principles

1. **Hard-coded risk controls**: `RiskGate` and `KillSwitch` are code-enforced, not prompt-enforced. No agent can bypass them.
2. **ACID state**: single DuckDB connection prevents partial-failure state on crash.
3. **Dependency injection**: `TradingContext` wires all services; `:memory:` gives fully isolated test environments.
4. **Unified MCP**: single server exposes the entire tool surface — no split between research and execution.
5. **Paper mode default**: `USE_REAL_TRADING=false` by default; live trading requires explicit opt-in.
6. **No LLM in execution path**: SignalEngine and RiskGate are pure Python. LLMs assist in research and reasoning, not in the hot path.

---

## Further Reading

- [quant_pod.md](./quant_pod.md) — Execution layer and autonomous loop details
- [quantcore.md](./quantcore.md) — Core library modules
- [mcp_servers.md](./mcp_servers.md) — MCP server tool listings
- [../guides/execution_setup.md](../guides/execution_setup.md) — Broker config, risk limits, kill switch
- [../guides/quickstart.md](../guides/quickstart.md) — Get started in 5 minutes
