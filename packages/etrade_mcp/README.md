# etrade_mcp

eTrade MCP server — account management, market data, and order execution via the [eTrade API](https://developer.etrade.com).

> **Note**: The OAuth auth layer, API client, and Pydantic models live in `packages/quant_pod/tools/etrade/`. This package is the thin MCP transport wrapper around them.

## Prerequisites

- eTrade brokerage account
- Developer API keys from [developer.etrade.com](https://developer.etrade.com/getting-started)
- Start with sandbox mode (`ETRADE_SANDBOX=true`) — no real money, fake data

## Configuration

```bash
# Required
ETRADE_CONSUMER_KEY=your_consumer_key
ETRADE_CONSUMER_SECRET=your_consumer_secret

# Sandbox mode (default: true — always start here)
# true  → apisb.etrade.com (test environment, fake data)
# false → api.etrade.com   (real market data; paper or live account)
ETRADE_SANDBOX=true
```

## Usage

```bash
# Start in sandbox mode (default)
etrade-mcp

# Switch to production (real data, paper or live account — use with care)
etrade-mcp --production
```

## Authentication (OAuth 1.0a)

eTrade uses a three-step OAuth flow. Tokens expire at midnight Eastern every day.

### Step 1 — Get authorization URL

```
Call: etrade_authorize (no arguments)
Returns: {"auth_url": "https://us.etrade.com/e/t/etws/authorize?..."}
```

### Step 2 — Authorize in browser

Visit the `auth_url`, log in with your eTrade credentials, and approve the application. The page shows a **verifier code**.

### Step 3 — Complete authorization

```
Call: etrade_authorize(verifier_code="XXXX")
Returns: {"success": true, "message": "Authorisation successful."}
```

After this, all trading tools are available for the rest of the day.

### Token refresh

Tokens expire at midnight Eastern. Call `etrade_refresh_token` to extend the session without re-authorizing.

## Available Tools

### Auth

| Tool | Description |
|------|-------------|
| `etrade_authorize` | Start or complete OAuth flow (step 1: get URL; step 3: submit verifier) |
| `etrade_refresh_token` | Renew access token (call before midnight to avoid session expiry) |
| `get_auth_status` | Check authentication state and sandbox mode |

### Account

| Tool | Description |
|------|-------------|
| `get_accounts` | List all accounts — returns `accountIdKey` needed for other tools |
| `get_account_balance` | Cash, margin, and buying power summary |
| `get_positions` | Open positions with P&L (optionally filter by symbol) |

### Market Data

| Tool | Description |
|------|-------------|
| `get_quote` | Real-time quotes for up to 25 symbols (comma-separated string) |
| `get_option_expiry_dates` | Available option expiration dates for a symbol |
| `get_option_chains` | Full option chain with Greeks (calls, puts, or both) |

### Orders

| Tool | Description |
|------|-------------|
| `preview_order` | Estimate cost — **always call before `place_order`** |
| `place_order` | Submit equity or single-leg option order |
| `place_spread_order` | Submit multi-leg option spread (vertical, iron condor, calendar, etc.) |
| `cancel_order` | Cancel an open order |
| `get_orders` | Order history, filterable by status (`OPEN`, `EXECUTED`, `CANCELLED`) |

## Sandbox vs Production

| Setting | API host | Data | Money |
|---------|----------|------|-------|
| `ETRADE_SANDBOX=true` | `apisb.etrade.com` | Fake | No |
| `ETRADE_SANDBOX=false` | `api.etrade.com` | Real | Depends on account |

Always test your full workflow in sandbox before switching to `false`.

## MCP Config

```json
{
  "mcpServers": {
    "etrade": {
      "command": "etrade-mcp",
      "env": {
        "ETRADE_CONSUMER_KEY": "your_key",
        "ETRADE_CONSUMER_SECRET": "your_secret",
        "ETRADE_SANDBOX": "true"
      }
    }
  }
}
```
