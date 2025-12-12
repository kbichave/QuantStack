# MCP Servers Architecture

This document describes the Model Context Protocol (MCP) servers that expose QuantCore capabilities and brokerage integrations to AI agents.

## Overview

MCP (Model Context Protocol) is a standard for exposing tools and resources to AI systems. QuantCore provides two MCP servers:

1. **quantcore-mcp**: Exposes quantitative analysis tools
2. **etrade-mcp**: Provides E-Trade brokerage integration

## quantcore-mcp

The QuantCore MCP server exposes technical indicators, backtesting, and analysis capabilities.

### Location

```
packages/quantcore/mcp/
├── __init__.py
└── server.py
```

### Starting the Server

```bash
# Via CLI entry point
quantcore-mcp

# Or directly
python -m quantcore.mcp.server
```

### Available Tools

#### Technical Indicators

| Tool | Description | Parameters |
|------|-------------|------------|
| `compute_indicator` | Compute single indicator | symbol, indicator, period, timeframe |
| `compute_all_indicators` | Compute all 200+ indicators | symbol, timeframe |
| `get_market_structure` | Support/resistance levels | symbol, timeframe |
| `detect_swing_points` | Swing high/low detection | symbol, lookback |

**Example Usage:**
```json
{
  "tool": "compute_indicator",
  "params": {
    "symbol": "SPY",
    "indicator": "RSI",
    "period": 14,
    "timeframe": "1h"
  }
}
```

#### Backtesting

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_backtest` | Execute strategy backtest | strategy, symbol, start_date, end_date |
| `get_backtest_metrics` | Retrieve performance metrics | backtest_id |
| `compare_strategies` | Compare multiple strategies | strategies[], symbol, date_range |

**Example Usage:**
```json
{
  "tool": "run_backtest",
  "params": {
    "strategy": "mean_reversion",
    "symbol": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "params": {
      "zscore_threshold": 2.0
    }
  }
}
```

#### Regime Detection

| Tool | Description | Parameters |
|------|-------------|------------|
| `detect_regime` | Classify market regime | symbol, method |
| `get_regime_history` | Historical regime changes | symbol, start_date, end_date |

#### Risk Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculate_var` | Value at Risk | portfolio, confidence, horizon |
| `stress_test` | Stress test portfolio | portfolio, scenarios |
| `position_size` | Optimal position sizing | signal, risk_pct, method |

#### Options

| Tool | Description | Parameters |
|------|-------------|------------|
| `price_option` | Black-Scholes pricing | S, K, T, r, sigma, option_type |
| `compute_greeks` | Option Greeks | S, K, T, r, sigma |
| `implied_vol` | IV from market price | S, K, T, r, market_price, option_type |

### Resources

The MCP server exposes these resources:

| Resource URI | Description |
|--------------|-------------|
| `quantcore://indicators/list` | List of available indicators |
| `quantcore://strategies/list` | Available strategy templates |
| `quantcore://data/{symbol}` | Historical data for symbol |

### Server Configuration

```python
# packages/quantcore/mcp/server.py
from fastmcp import FastMCP

mcp = FastMCP("quantcore")

@mcp.tool()
def compute_indicator(
    symbol: str,
    indicator: str,
    period: int = 14,
    timeframe: str = "daily"
) -> dict:
    """Compute a technical indicator for a symbol."""
    # Implementation
    ...

@mcp.resource("quantcore://indicators/list")
def list_indicators() -> list[str]:
    """List all available indicators."""
    return INDICATOR_REGISTRY.keys()
```

## etrade-mcp

The E-Trade MCP server provides brokerage integration for live trading.

### Location

```
packages/etrade_mcp/
├── __init__.py
├── auth.py         # OAuth authentication
├── client.py       # E-Trade API client
├── models.py       # Pydantic models
└── server.py       # MCP server
```

### Starting the Server

```bash
# Via CLI entry point
etrade-mcp

# With custom port
etrade-mcp --port 8081
```

### Authentication

E-Trade uses OAuth 1.0a. The MCP server handles authentication:

```python
# First-time setup (interactive)
etrade-mcp --setup

# This will:
# 1. Open browser for E-Trade authorization
# 2. Store tokens in ~/.etrade/credentials.json
# 3. Auto-refresh tokens when needed
```

### Available Tools

#### Account Operations

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_accounts` | List all accounts | - |
| `get_balance` | Account balance | account_id |
| `get_positions` | Current positions | account_id |
| `get_orders` | Order history | account_id, status |

**Example Usage:**
```json
{
  "tool": "get_positions",
  "params": {
    "account_id": "12345678"
  }
}
```

#### Trading

| Tool | Description | Parameters |
|------|-------------|------------|
| `place_order` | Submit order | account_id, symbol, side, quantity, order_type, limit_price |
| `cancel_order` | Cancel order | account_id, order_id |
| `modify_order` | Modify existing order | account_id, order_id, new_params |

**Example Usage:**
```json
{
  "tool": "place_order",
  "params": {
    "account_id": "12345678",
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "limit",
    "limit_price": 175.50
  }
}
```

#### Market Data

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_quote` | Real-time quote | symbol |
| `get_option_chain` | Options chain | symbol, expiration |

### Order Types Supported

- Market
- Limit
- Stop
- Stop Limit
- Trailing Stop

### Models

```python
# packages/etrade_mcp/models.py
from pydantic import BaseModel

class Order(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: Literal["day", "gtc"] = "day"

class Position(BaseModel):
    symbol: str
    quantity: int
    cost_basis: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

class AccountBalance(BaseModel):
    account_id: str
    cash_balance: float
    buying_power: float
    portfolio_value: float
    margin_used: float
```

### Server Implementation

```python
# packages/etrade_mcp/server.py
from fastmcp import FastMCP
from etrade_mcp.client import ETradeClient
from etrade_mcp.models import Order

mcp = FastMCP("etrade")
client = ETradeClient()

@mcp.tool()
def place_order(
    account_id: str,
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None
) -> dict:
    """Place a trade order."""
    order = Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price
    )
    return client.place_order(account_id, order)

@mcp.tool()
def get_positions(account_id: str) -> list[dict]:
    """Get current positions for an account."""
    positions = client.get_positions(account_id)
    return [p.model_dump() for p in positions]
```

## Using MCP Servers with AI Agents

### From QuantPod

```python
from quant_pod.tools import MCPBridge

# Connect to QuantCore MCP
qc_bridge = MCPBridge(server="quantcore-mcp")

# Get indicators
indicators = await qc_bridge.call(
    "compute_all_indicators",
    {"symbol": "SPY", "timeframe": "4h"}
)

# Connect to E-Trade MCP
et_bridge = MCPBridge(server="etrade-mcp")

# Check positions
positions = await et_bridge.call(
    "get_positions",
    {"account_id": "12345678"}
)
```

### From Claude/Cursor

The MCP servers can be used directly with Claude or Cursor AI:

```json
// .cursor/mcp.json or claude_desktop_config.json
{
  "mcpServers": {
    "quantcore": {
      "command": "quantcore-mcp",
      "args": []
    },
    "etrade": {
      "command": "etrade-mcp",
      "args": ["--port", "8081"]
    }
  }
}
```

## Security Considerations

### E-Trade MCP

- OAuth tokens stored encrypted at rest
- Tokens auto-expire and refresh
- Rate limiting enforced
- Paper trading mode available for testing

```bash
# Run in paper trading mode
etrade-mcp --paper-trading
```

### Network Security

```bash
# Bind to localhost only (default)
quantcore-mcp --host 127.0.0.1

# With authentication token
quantcore-mcp --auth-token $MCP_AUTH_TOKEN
```

## Extending MCP Servers

### Adding New Tools

```python
# In packages/quantcore/mcp/server.py

@mcp.tool()
def my_custom_tool(param1: str, param2: int) -> dict:
    """Description of what this tool does."""
    # Implementation
    result = process(param1, param2)
    return {"status": "success", "data": result}
```

### Adding New Resources

```python
@mcp.resource("quantcore://custom/{resource_id}")
def get_custom_resource(resource_id: str) -> dict:
    """Retrieve custom resource."""
    return load_resource(resource_id)
```

## Troubleshooting

### Common Issues

**quantcore-mcp won't start:**
```bash
# Check dependencies
pip install -e ".[mcp]"

# Verbose mode
quantcore-mcp --verbose
```

**E-Trade authentication fails:**
```bash
# Re-authenticate
etrade-mcp --setup --force

# Check token expiry
etrade-mcp --check-auth
```

**Connection refused:**
```bash
# Check if server is running
curl http://localhost:8080/health

# Check port availability
lsof -i :8080
```
