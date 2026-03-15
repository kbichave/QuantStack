# QuantStack Architecture

This document provides an overview of the QuantStack monorepo and how its packages interact.

## Repository Structure

```
QuantStack/
├── packages/
│   ├── quantcore/          Core quantitative trading library (200+ indicators, backtesting, ML, RL)
│   ├── quant_arena/        Historical simulation engine
│   ├── quant_pod/          Multi-agent trading system (CrewAI)
│   ├── alpaca_mcp/         Alpaca broker MCP server
│   ├── ibkr_mcp/           Interactive Brokers MCP server
│   └── etrade_mcp/         eTrade MCP server
├── scripts/                Utility scripts (bootstrap RL, log decisions, Discord alerts)
├── examples/               Example applications
├── tests/                  Test suite (unit + integration)
├── docs/                   Documentation
│   ├── architecture/       System design (this folder)
│   └── guides/             Setup and operational guides
├── .mcp.json               MCP server config for Claude Code
├── Dockerfile              Container image
└── docker-compose.yml      Multi-service deployment
```

## Package Overview

### quantcore — Core Library

Foundation for all quantitative analysis:

- **200+ Technical Indicators**: trend, momentum, volatility, volume, market structure
- **Backtesting Engine**: event-driven with transaction cost modeling
- **ML Integration**: LightGBM, XGBoost, CatBoost, SHAP
- **RL Agents**: PPO/DQN for execution, sizing, spread trading (experimental)
- **Options Pricing**: Black-Scholes, Greeks, IV surface
- **Market Microstructure**: order book simulation, impact models
- **Execution**: SmartOrderRouter, TCA engine, kill switch, risk gate, unified broker models

See [quantcore.md](./quantcore.md) for module details.

### quant_pod — Multi-Agent System

CrewAI-based trading system with 13 agents across 5 pods. Produces a `DailyBrief` that Claude Code uses to make trade decisions.

- **13 ICs**: data ingestion, regime detection, technicals, quant signals, risk, news/options flow/fundamentals
- **Execution Layer**: risk gate, kill switch, signal cache, tick executor, smart order router
- **DuckDB State**: ACID-safe consolidated database for positions, fills, audit, agent memory
- **Dependency Injection**: `TradingContext` wires all services; `:memory:` for test isolation
- **Audit & Monitoring**: structured decision logging, signal degradation detection
- **MCP Server**: 30+ tools across 6 phases (analysis → strategy → execution → decode → meta → learning)

See [quant_pod.md](./quant_pod.md) for agent hierarchy and execution layer details.

### quant_arena — Simulation Engine

Historical simulation harness for backtesting multi-agent systems with execution realism.

- **SimBroker**: realistic fills with slippage, partial fills, and market impact
- **Historical Clock**: time-synchronized OHLCV replay
- **Risk Metrics**: Sharpe, drawdown, win rate, Calmar

See [quant_arena.md](./quant_arena.md) for simulation details.

### alpaca_mcp / ibkr_mcp / etrade_mcp — Broker MCP Servers

Each broker is a separate MCP server. QuantPod's `SmartOrderRouter` auto-discovers and routes to the best available venue.

| Server | Prerequisites |
|--------|--------------|
| `alpaca-mcp` | Alpaca API key + secret (free paper account available) |
| `ibkr-mcp` | IB Gateway or TWS running locally |
| `etrade-mcp` | eTrade developer keys + OAuth flow |

See [mcp_servers.md](./mcp_servers.md) for tool listings and configuration.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLAUDE CODE (You)                             │
│           Portfolio Manager · Strategy Researcher · Architect        │
└──────────────────────────────┬──────────────────────────────────────┘
                                │ MCP calls
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  quantpod-mcp    │  │  quantcore-mcp   │  │  Broker MCP servers  │
│  (30+ tools)     │  │  (40+ tools)     │  │  alpaca / ibkr /     │
│  execution +     │  │  indicators +    │  │  etrade              │
│  strategy +      │  │  backtesting +   │  └──────────┬───────────┘
│  learning loop   │  │  options + RL    │             │
└────────┬─────────┘  └──────────────────┘             │
         │                                              │
         ▼                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       QuantPod Core                               │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ TradingCrew  │  │ Execution Layer  │  │  DuckDB State        │ │
│  │ 13 ICs       │  │ RiskGate         │  │  positions/fills/    │ │
│  │ 5 Pods       │  │ KillSwitch       │  │  audit/memory/       │ │
│  │ DailyBrief   │  │ SignalCache       │  │  strategies/matrix   │ │
│  └──────────────┘  │ TickExecutor     │  └──────────────────────┘ │
│                    │ SmartOrderRouter │                             │
│                    └─────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                       QuantCore Library                           │
│  Features · Backtesting · ML · RL · Options · Microstructure     │
│  Execution Models · TCA · Portfolio Stats · Research Tools       │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Data Sources                              │
│  Alpaca · Polygon · Alpha Vantage (priority: DATA_PROVIDER_PRIORITY)│
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Data Ingestion**: market data fetched via `DATA_PROVIDER_PRIORITY` (Alpaca → Polygon → Alpha Vantage)
2. **Feature Engineering**: QuantCore computes indicators + multi-timeframe features
3. **Signal Generation**: TradingCrew (13 ICs) produces `DailyBrief`
4. **Decision**: Claude Code reads `DailyBrief` via `run_analysis` MCP, makes trade decision
5. **Risk Check**: `RiskGate` enforces position size, daily loss, and liquidity limits
6. **Execution**: `SmartOrderRouter` routes to best available broker (or `PaperBroker`)
7. **Audit**: every decision and fill logged to `decision_events` and `fills` tables
8. **Learning**: RL feedback + calibration records update agent skill scores

---

## Key Design Principles

1. **Hard-coded risk controls**: `RiskGate` and `KillSwitch` are code-enforced, not prompt-enforced. No agent can bypass them.
2. **ACID state**: single DuckDB connection prevents partial-failure state on crash.
3. **Dependency injection**: `TradingContext` wires all services; `:memory:` gives fully isolated test environments.
4. **MCP integration**: tools exposed via Model Context Protocol for Claude Code access.
5. **Paper mode default**: `USE_REAL_TRADING=false` by default; live trading requires explicit opt-in.

---

## Further Reading

- [quantcore.md](./quantcore.md) — Core library modules
- [quant_pod.md](./quant_pod.md) — Multi-agent system and execution layer
- [quant_arena.md](./quant_arena.md) — Simulation engine
- [mcp_servers.md](./mcp_servers.md) — MCP server tool listings
- [../guides/execution_setup.md](../guides/execution_setup.md) — Broker config, risk limits, kill switch
- [../guides/deployment.md](../guides/deployment.md) — Docker, CI/CD, data paths
- [../guides/quickstart.md](../guides/quickstart.md) — Get started in 5 minutes
