# QuantStack Packages

This directory contains all Python packages that make up the QuantStack unified trading platform.

## Package Overview

| Package | Description | Status |
|---------|-------------|--------|
| [quantcore](./quantcore/) | Core quantitative trading library | Stable |
| [quant_arena](./quant_arena/) | Historical backtesting simulation | Stable |
| [quant_pod](./quant_pod/) | Multi-agent trading system (CrewAI) | Beta |
| [etrade_mcp](./etrade_mcp/) | E-Trade MCP server integration | Beta |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      QuantStack                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  QuantPod    │  │ QuantArena   │  │ E-Trade MCP  │       │
│  │  (Agents)    │  │ (Simulation) │  │ (Execution)  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └────────────┬────┴──────────────────┘               │
│                      │                                       │
│              ┌───────┴───────┐                              │
│              │   QuantCore   │                              │
│              │  (Research)   │                              │
│              └───────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Package Details

### QuantCore
The foundational library providing:
- **200+ Technical Indicators**: Trend, momentum, volatility, volume analysis
- **Backtesting Engine**: Event-driven with realistic cost modeling
- **ML Integration**: LightGBM, XGBoost, CatBoost, ensemble methods
- **Options Pricing**: Black-Scholes, Greeks, IV surface
- **Market Microstructure**: Order book simulation, impact models
- **Research Tools**: Statistical tests, alpha decay, walk-forward validation

### QuantArena
Historical simulation engine for backtesting:
- **Simulation Clock**: Market hours aware time progression
- **Simulated Broker**: Slippage, commission, and fill modeling
- **Data Loading**: DuckDB-backed historical data
- **Multi-Strategy Support**: Test multiple strategies simultaneously

### QuantPod
CrewAI-based multi-agent trading system:
- **Hierarchical Agents**: ICs → Pod Managers → Assistant → SuperTrader
- **Knowledge Store**: Persistent policy and learning storage
- **Memory System**: Blackboard pattern for agent communication
- **MCP Bridge**: Tool integration for external data/execution

### E-Trade MCP
Model Context Protocol server for E-Trade:
- **OAuth Authentication**: Secure credential management
- **Account Access**: Positions, balances, order history
- **Trade Execution**: Market/limit orders with paper trading mode
- **Real-time Quotes**: Live market data

## Installation

All packages are installed together via the root `pyproject.toml`:

```bash
# Using uv (recommended)
uv sync --all-extras

# Using pip
pip install -e ".[all]"
```

## Development

Each package can be developed independently, but they share:
- Common dependencies in root `pyproject.toml`
- Shared test infrastructure in `/tests`
- Unified documentation in `/docs`

```bash
# Run tests for a specific package
uv run pytest tests/unit/test_technical_indicators.py

# Run package-specific tests
uv run pytest tests/quant_pod/
```

## Dependencies

```
quantcore ← (no internal deps)
quant_arena ← quantcore
quant_pod ← quantcore
etrade_mcp ← (no internal deps)
```

## Documentation

- [QuantCore Architecture](../docs/architecture/quantcore.md)
- [QuantArena Architecture](../docs/architecture/quant_arena.md)
- [QuantPod Architecture](../docs/architecture/quant_pod.md)
- [MCP Servers](../docs/architecture/mcp_servers.md)
