# E-Trade MCP

MCP (Model Context Protocol) server for E-Trade brokerage integration, enabling AI agents to access account data and execute trades.

## Installation

E-Trade MCP is part of the main repository:

```bash
uv sync --all-extras
```

## Components

| Component | Description |
|-----------|-------------|
| `auth.py` | OAuth 1.0a authentication handling |
| `client.py` | E-Trade API client wrapper |
| `models.py` | Pydantic models (Order, Position, Balance) |
| `server.py` | MCP server implementation |

## Setup

### 1. E-Trade Developer Account

Register at [E-Trade Developer](https://developer.etrade.com/) to get API credentials.

### 2. Authentication

```bash
# First-time setup (opens browser for OAuth)
etrade-mcp --setup
```

Tokens are stored in `~/.etrade/credentials.json`.

## Starting the Server

```bash
etrade-mcp

# With custom port
etrade-mcp --port 8081

# Paper trading mode
etrade-mcp --paper-trading
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_accounts` | List all accounts |
| `get_balance` | Account balance and buying power |
| `get_positions` | Current positions |
| `get_orders` | Order history |
| `place_order` | Submit market/limit orders |
| `cancel_order` | Cancel pending order |
| `get_quote` | Real-time quote |

## Usage Example

```python
from etrade_mcp.client import ETradeClient

client = ETradeClient()

# Get positions
positions = client.get_positions(account_id="12345678")

# Place order
order = client.place_order(
    account_id="12345678",
    symbol="AAPL",
    side="buy",
    quantity=100,
    order_type="limit",
    limit_price=175.50
)
```

## Documentation

See [MCP Servers Documentation](../../docs/architecture/mcp_servers.md) for detailed API reference.
