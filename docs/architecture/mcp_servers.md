# MCP Servers Architecture

QuantStack exposes trading functionality through five MCP servers. Each server is a separate process; Claude Code (or any MCP client) connects to all of them via `.mcp.json`.

## Server Overview

| Server | CLI command | Package | Purpose |
|--------|-------------|---------|---------|
| `quantcore-mcp` | `quantcore-mcp` | `packages/quantcore/mcp/` | 40+ quant tools: indicators, backtesting, options, RL |
| `quantstack-mcp` | `quantstack-mcp` | `packages/quantstack/mcp/` | Portfolio management, strategy registry, execution, learning loop |
| `alpaca-mcp` | `alpaca-mcp` | `packages/alpaca_mcp/` | Alpaca data + order execution |
| `ibkr-mcp` | `ibkr-mcp` | `packages/ibkr_mcp/` | Interactive Brokers data + order execution |
| `etrade-mcp` | `etrade-mcp` | `packages/etrade_mcp/` | eTrade data + order execution |

All broker MCP servers are optional — QuantStack falls back to its internal `PaperBroker` when none are available.

---

## quantcore-mcp

Exposes the full QuantCore library as MCP tools. Used by QuantStack agents internally and directly by Claude Code for research.

### Start

```bash
quantcore-mcp
python -m quantcore.mcp.server
```

### Tool Categories

| Category | Key tools |
|----------|-----------|
| Data | `fetch_market_data`, `load_market_data`, `list_stored_symbols` |
| Indicators | `compute_technical_indicators`, `compute_all_features`, `list_available_indicators` |
| Backtesting | `run_backtest`, `get_backtest_metrics`, `run_walkforward`, `run_purged_cv` |
| Research | `run_adf_test`, `compute_alpha_decay`, `compute_information_coefficient` |
| Risk | `compute_var`, `check_risk_limits`, `stress_test_portfolio`, `compute_position_size` |
| Options | `price_option`, `price_american_option`, `compute_greeks`, `compute_implied_vol`, `compute_option_chain` |
| Microstructure | `get_order_book_snapshot`, `analyze_liquidity`, `analyze_volume_profile` |
| RL | `get_rl_recommendation` |
| Regime | `get_market_regime_snapshot` |
| ML | `detect_leakage`, `check_lookahead_bias`, `validate_signal`, `diagnose_signal` |

---

## quantstack-mcp

The primary interface for Claude Code. Manages the full trade lifecycle: analysis → strategy → backtest → execution → learning.

### Start

```bash
quantstack-mcp
python -m quantstack.mcp.server
```

### Tool Phases

**Phase 1 — Analysis**
- `run_analysis` — run TradingCrew, return DailyBrief
- `get_portfolio_state` — positions, cash, P&L
- `get_regime` — ADX/ATR regime classification for a symbol
- `get_recent_decisions` — audit trail query
- `get_system_status` — kill switch, risk halt, broker mode

**Phase 2 — Strategy & Backtesting**
- `register_strategy`, `list_strategies`, `get_strategy`, `update_strategy`
- `run_backtest`, `run_walkforward`

**Phase 3 — Execution**
- `execute_trade`, `close_position`, `cancel_order`, `get_fills`
- `get_risk_metrics`, `get_audit_trail`

**Phase 4 — Decoder**
- `decode_strategy` — reverse-engineer strategy from signals
- `decode_from_trades` — decode from system's own trade history

**Phase 5 — Meta Orchestration**
- `run_multi_analysis` — multiple symbols sequentially
- `get_regime_strategies`, `set_regime_allocation`
- `resolve_portfolio_conflicts`

**Phase 6 — Learning Loop**
- `get_rl_status`, `get_rl_recommendation`
- `promote_strategy`, `retire_strategy`
- `get_strategy_performance`, `validate_strategy`
- `update_regime_matrix_from_performance`

---

## alpaca-mcp

Market data and order execution via Alpaca. Paper account is free.

### Start

```bash
ALPACA_API_KEY=... ALPACA_SECRET_KEY=... alpaca-mcp
```

### Tools

| Tool | Description |
|------|-------------|
| `get_auth_status` | Check connectivity and paper/live mode |
| `get_account` | Account summary |
| `get_balance` | Cash, buying power, portfolio value |
| `get_positions` | Open positions with P&L |
| `get_quote` | Real-time quotes (up to 50 symbols) |
| `get_bars` | Historical OHLCV (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w) |
| `preview_order` | Estimate cost without submitting |
| `place_order` | Submit market, limit, stop, stop_limit order |
| `cancel_order` | Cancel by Alpaca order UUID |
| `get_orders` | Order history by status |
| `get_option_chains` | Options chain (requires Alpaca Options subscription) |

### Auth

```bash
ALPACA_API_KEY=your_key        # required
ALPACA_SECRET_KEY=your_secret  # required
ALPACA_PAPER=true              # false = live endpoint
```

See [alpaca_mcp README](../../packages/alpaca_mcp/README.md) for full setup.

---

## ibkr-mcp

Market data and order execution via Interactive Brokers. Requires IB Gateway or TWS running locally.

### Start

```bash
# Start IB Gateway first, then:
IBKR_PORT=4001 ibkr-mcp
```

The server starts in degraded mode if IB Gateway is unreachable. Use `connect_gateway` tool to reconnect after the gateway starts.

### Tools

| Tool | Description |
|------|-------------|
| `get_connection_status` | Check IB Gateway connection |
| `connect_gateway` | Explicitly connect (or reconnect) |
| `get_accounts` | List managed accounts |
| `get_balance` | Net liquidation, cash, buying power, margin |
| `get_positions` | Open positions with P&L |
| `get_quote` | Real-time snapshot (up to 20 symbols) |
| `get_historical_bars` | OHLCV history via `reqHistoricalData` |
| `get_option_chains` | Expirations and strikes |
| `place_order` | Market or limit equity order |
| `cancel_order` | Cancel by IB order ID |
| `get_orders` | Open and completed orders |

### Auth

```bash
IBKR_HOST=127.0.0.1   # default
IBKR_PORT=4001         # 4001=IB Gateway, 7497=TWS
IBKR_CLIENT_ID=1       # unique per simultaneous connection
```

See [ibkr_mcp README](../../packages/ibkr_mcp/README.md) for port reference and firewall notes.

---

## etrade-mcp

Account management and order execution via eTrade. Uses OAuth 1.0a. Always start with sandbox mode.

### Start

```bash
ETRADE_CONSUMER_KEY=... ETRADE_CONSUMER_SECRET=... etrade-mcp
# Add --production flag to switch to live API
```

After starting, complete OAuth via the `etrade_authorize` MCP tool (three-step flow).

### Tools

| Tool | Description |
|------|-------------|
| `etrade_authorize` | Start or complete OAuth flow |
| `etrade_refresh_token` | Renew token (expires at midnight ET) |
| `get_auth_status` | Auth state and sandbox mode |
| `get_accounts` | List accounts → accountIdKey |
| `get_account_balance` | Cash, margin, buying power |
| `get_positions` | Open positions |
| `get_quote` | Real-time quotes (up to 25 symbols) |
| `get_option_expiry_dates` | Available expirations |
| `get_option_chains` | Full chain with Greeks |
| `preview_order` | Cost estimate — call before place_order |
| `place_order` | Equity or single-leg option order |
| `place_spread_order` | Multi-leg option spread |
| `cancel_order` | Cancel open order |
| `get_orders` | Order history by status |

### Auth

```bash
ETRADE_CONSUMER_KEY=your_key     # required
ETRADE_CONSUMER_SECRET=your_secret  # required
ETRADE_SANDBOX=true              # false = production API
```

See [etrade_mcp README](../../packages/etrade_mcp/README.md) for OAuth flow details.

---

## SmartOrderRouter

`packages/quantcore/execution/smart_order_router.py` selects the execution venue for each order. Used by `TradingDayFlow` when `USE_REAL_TRADING=true`.

**Routing priority:**
1. Alpaca — if `ALPACA_API_KEY` is set and server is reachable
2. IBKR — if `IBKR_HOST` is set and gateway is connected
3. eTrade — if `ETRADE_CONSUMER_KEY` is set and authenticated
4. PaperBroker — always available fallback

The router evaluates spread, latency, and commission across available venues and routes to the best option.

---

## .mcp.json Configuration

The repo ships with `.mcp.json`. Claude Code reads this file to discover available MCP servers.

```json
{
  "mcpServers": {
    "quantcore": {
      "command": "quantcore-mcp"
    },
    "quantstack": {
      "command": "quantstack-mcp"
    },
    "alpaca": {
      "command": "alpaca-mcp",
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER": "true"
      }
    },
    "ibkr": {
      "command": "ibkr-mcp",
      "env": {
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4001"
      }
    },
    "etrade": {
      "command": "etrade-mcp",
      "env": {
        "ETRADE_CONSUMER_KEY": "${ETRADE_CONSUMER_KEY}",
        "ETRADE_CONSUMER_SECRET": "${ETRADE_CONSUMER_SECRET}",
        "ETRADE_SANDBOX": "true"
      }
    }
  }
}
```

---

## Security

- All broker servers default to paper/sandbox mode. Live trading requires explicit opt-in via env vars (`ALPACA_PAPER=false`, `ETRADE_SANDBOX=false`, `USE_REAL_TRADING=true`).
- MCP servers bind to `127.0.0.1` only. Do not expose them to the public internet.
- eTrade OAuth tokens expire at midnight Eastern. Refresh with `etrade_refresh_token` before then.
- The `RiskGate` enforces hard limits on every order regardless of which broker is used.
