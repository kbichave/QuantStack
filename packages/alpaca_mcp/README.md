# alpaca_mcp

Alpaca MCP server — market data and order execution via the [Alpaca](https://alpaca.markets) brokerage API.

## Prerequisites

- Alpaca account (paper account is free at [alpaca.markets](https://alpaca.markets) — no credit card required)
- API key and secret from the Alpaca dashboard

## Installation

```bash
# Alpaca SDK is an optional dependency
uv sync --extra alpaca
# or
pip install -e ".[alpaca]"
```

## Configuration

```bash
# Required
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key

# Paper trading (default: true — safe to omit for paper mode)
ALPACA_PAPER=true   # true = paper endpoint, false = live endpoint
```

## Usage

```bash
# Start the MCP server
alpaca-mcp

# Or directly
python -m alpaca_mcp.server
```

The server starts in degraded mode (all tools return errors) if credentials are missing or invalid. Set the env vars and restart.

## Paper vs Live Mode

| `ALPACA_PAPER` | Endpoint | Money at risk |
|:--------------:|----------|:-------------:|
| `true` (default) | `paper-api.alpaca.markets` | No |
| `false` | `api.alpaca.markets` | **Yes** |

Always validate with paper mode before switching to live.

## Available Tools

### Account

| Tool | Description |
|------|-------------|
| `get_auth_status` | Check API key connectivity and paper/live mode |
| `get_account` | Account summary (status, account type) |
| `get_balance` | Cash, buying power, and portfolio value |
| `get_positions` | Open positions with quantity, entry price, and unrealised P&L |

### Market Data

| Tool | Description |
|------|-------------|
| `get_quote` | Real-time best-bid/offer quotes for up to 50 symbols |
| `get_bars` | Historical OHLCV bars (timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`) |
| `get_option_chains` | Options chain snapshot — requires Alpaca Options Data subscription |

### Orders

| Tool | Description |
|------|-------------|
| `preview_order` | Estimate cost and commission without submitting |
| `place_order` | Submit market, limit, stop, or stop_limit equity order |
| `cancel_order` | Cancel an open order by Alpaca order UUID |
| `get_orders` | Order history, filterable by status (`open`, `filled`, `cancelled`) |

## MCP Config

Add to `.mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "alpaca-mcp",
      "env": {
        "ALPACA_API_KEY": "your_key",
        "ALPACA_SECRET_KEY": "your_secret",
        "ALPACA_PAPER": "true"
      }
    }
  }
}
```

## Integration with QuantPod

QuantPod's `SmartOrderRouter` detects Alpaca credentials automatically. When `ALPACA_API_KEY` is set, orders are routed to Alpaca instead of the internal `PaperBroker`. No code changes required.

```bash
# QuantPod will route through Alpaca when this is set
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
```
