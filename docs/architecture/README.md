# QuantCore Architecture

This document provides an overview of the QuantCore monorepo architecture, describing the major packages and how they interact.

## Repository Structure

```
Trader/
├── packages/                    # All Python packages
│   ├── quantcore/              # Core quantitative trading library
│   ├── quant_arena/            # Historical backtesting simulation
│   ├── etrade_mcp/             # E-Trade MCP server
│   └── quant_pod/              # Multi-agent trading system (CrewAI)
├── configs/                    # Configuration files
├── scripts/                    # Utility scripts
├── examples/                   # Example applications
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## Package Overview

### quantcore (Core Library)

The foundational quantitative trading library providing:

- **200+ Technical Indicators**: Trend, momentum, volatility, volume, and market structure
- **Backtesting Engine**: Event-driven with realistic cost modeling
- **ML Integration**: LightGBM, XGBoost, CatBoost with SHAP explainability
- **RL Agents**: PPO/DQN for execution, sizing, and spread trading
- **Options Pricing**: Black-Scholes, Greeks, IV surface modeling
- **Market Microstructure**: Order book simulation, impact models

See [quantcore.md](./quantcore.md) for detailed module documentation.

### quant_pod (Multi-Agent System)

A CrewAI-based multi-agent trading system for automated market monitoring and execution:

- **Agent Crews**: Specialized agents for market analysis, risk, and execution
- **Flows**: Orchestrated trading workflows
- **Knowledge Store**: Persistent trading insights and policies
- **Memory**: Blackboard pattern for agent communication

See [quant_pod.md](./quant_pod.md) for agent architecture details.

### quant_arena (Simulation Engine)

Historical simulation harness for backtesting multi-agent trading systems:

- **Simulated Broker**: Realistic execution modeling with slippage
- **Historical Clock**: Time-synchronized replay of market data
- **Universe Management**: Multi-asset simulation support

See [quant_arena.md](./quant_arena.md) for simulation details.

### etrade_mcp (E-Trade Integration)

MCP (Model Context Protocol) server for E-Trade brokerage integration:

- **OAuth Authentication**: Secure E-Trade API access
- **Account Operations**: Balances, positions, orders
- **Trade Execution**: Market and limit orders

See [mcp_servers.md](./mcp_servers.md) for MCP server documentation.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Applications                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ QuantArena  │  │  Quant Pod   │  │   Custom Strategies  │  │
│  │     UI       │  │   Agents     │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       QuantCore Library                          │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌───────────┐  │
│  │Features │ │Backtesting│ │ Models │ │  Risk  │ │ Research  │  │
│  │  200+   │ │  Engine   │ │ ML/RL  │ │Controls│ │ Analytics │  │
│  └─────────┘ └──────────┘ └────────┘ └────────┘ └───────────┘  │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌───────────┐  │
│  │Options  │ │Microstruc│ │  Math  │ │Execution│ │Validation │  │
│  │ Pricing │ │  -ture   │ │Stochast│ │ Algos  │ │ Leakage   │  │
│  └─────────┘ └──────────┘ └────────┘ └────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Servers                               │
│  ┌──────────────────┐           ┌──────────────────┐            │
│  │  quantcore-mcp   │           │    etrade-mcp    │            │
│  │ (Indicators, BT) │           │  (Brokerage API) │            │
│  └──────────────────┘           └──────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│   Data Sources      │           │   E-Trade API       │
│ (DuckDB, APIs)      │           │   (Live Trading)    │
└─────────────────────┘           └─────────────────────┘
```

## Data Flow

1. **Data Ingestion**: Market data fetched via APIs (Alpha Vantage, etc.) and stored in DuckDB
2. **Feature Engineering**: QuantCore computes 200+ indicators on OHLCV data
3. **Signal Generation**: ML models or rule-based strategies generate trading signals
4. **Risk Checks**: Position sizing and risk controls applied
5. **Execution**: Orders sent via MCP servers or simulated in QuantArena

## Key Design Principles

1. **Separation of Concerns**: Each package has a single responsibility
2. **MCP Integration**: Tools exposed via Model Context Protocol for AI agent access
3. **Event-Driven Backtesting**: Realistic simulation with transaction costs
4. **Type Safety**: Comprehensive type hints throughout
5. **Testability**: High test coverage with unit, integration, and property tests

## Getting Started

```bash
# Install in development mode
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Start MCP servers
quantcore-mcp  # QuantCore tools
etrade-mcp     # E-Trade integration
```

## Further Reading

- [quantcore.md](./quantcore.md) - Core library modules
- [quant_pod.md](./quant_pod.md) - Multi-agent system
- [quant_arena.md](./quant_arena.md) - Simulation engine
- [mcp_servers.md](./mcp_servers.md) - MCP integrations
