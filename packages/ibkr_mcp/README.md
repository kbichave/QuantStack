# ibkr_mcp

Interactive Brokers MCP server — market data and order execution via [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) or [TWS](https://www.interactivebrokers.com/en/trading/tws.php).

## Prerequisites

- Interactive Brokers account (paper or live)
- IB Gateway or TWS running and logged in on your machine
- `ib_insync` Python library (optional dependency)

## Installation

```bash
# ib_insync is an optional dependency
uv sync --extra ibkr
# or
pip install -e ".[ibkr]"
```

## Configuration

```bash
# All optional — defaults shown
IBKR_HOST=127.0.0.1   # IB Gateway host
IBKR_PORT=4001         # 4001 = IB Gateway (recommended), 7497 = TWS
IBKR_CLIENT_ID=1       # Must be unique per simultaneous connection (0–999)
```

## Usage

### 1. Start IB Gateway or TWS

- IB Gateway: lighter, recommended for automated trading
  - Default port: `4001` (live), `4002` (paper)
- TWS: full desktop app
  - Default port: `7497` (live), `7496` (paper)

In the IB Gateway / TWS settings, enable **API → Enable ActiveX and Socket Clients** and set **Socket port** to match your `IBKR_PORT`.

### 2. Start the MCP server

```bash
ibkr-mcp

# Or directly
python -m ibkr_mcp.server
```

The server starts in degraded mode if IB Gateway is not reachable. All tools return `{"success": false, "error": "..."}` until connected. Use `connect_gateway` to reconnect without restarting the server.

## Available Tools

### Connection

| Tool | Description |
|------|-------------|
| `get_connection_status` | Check IB Gateway connection status and account ID |
| `connect_gateway` | Explicitly connect to IB Gateway (use if gateway started after server) |

### Account

| Tool | Description |
|------|-------------|
| `get_accounts` | List all managed accounts |
| `get_balance` | Net liquidation, cash, buying power, margin (USD) |
| `get_positions` | Open positions with quantity, avg cost, market value, unrealised P&L |

### Market Data

| Tool | Description |
|------|-------------|
| `get_quote` | Real-time snapshot bid/ask/last for up to 20 symbols |
| `get_historical_bars` | OHLCV history from `reqHistoricalData` (duration: `"30 D"`, `"1 W"`, `"3 M"`) |
| `get_option_chains` | Available expirations and strikes for a symbol |

### Orders

| Tool | Description |
|------|-------------|
| `place_order` | Submit market or limit equity order |
| `cancel_order` | Cancel an open order by IB order ID |
| `get_orders` | Open and completed orders, filterable by status |

## Port Reference

| Application | Paper port | Live port |
|-------------|:----------:|:---------:|
| IB Gateway | `4002` | `4001` |
| TWS | `7496` | `7497` |

## Firewall / Network Notes

IB Gateway listens only on `127.0.0.1` by default. If running in Docker, you must either:
- Run IB Gateway in the same container, or
- Set **Trusted IP addresses** in the IB Gateway API settings to include your container's IP

## MCP Config

```json
{
  "mcpServers": {
    "ibkr": {
      "command": "ibkr-mcp",
      "env": {
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4001",
        "IBKR_CLIENT_ID": "1"
      }
    }
  }
}
```

## Integration with QuantPod

QuantPod's `SmartOrderRouter` detects IBKR credentials automatically. When `IBKR_HOST` is set and IB Gateway is reachable, orders can be routed through IBKR. Alpaca takes precedence if both are configured — adjust `DATA_PROVIDER_PRIORITY` to override.
