# Execution Setup Guide

This guide covers broker configuration, risk limits, the kill switch, and SmartOrderRouter routing.

---

## Paper vs Live Trading

QuantPod defaults to **paper mode**. No broker credentials are required to run analysis or paper-trade.

```bash
# Default — uses internal PaperBroker, no external broker needed
USE_REAL_TRADING=false

# Live — routes orders through your configured broker(s)
USE_REAL_TRADING=true
```

When `USE_REAL_TRADING=false`, the `PaperBroker` simulates fills with configurable slippage and commission against your internal DuckDB portfolio state. All risk limits still apply.

---

## Broker Configuration

### Alpaca (recommended for US equities)

1. Create a free paper account at [alpaca.markets](https://alpaca.markets)
2. Copy your API key and secret from the dashboard
3. Set environment variables:

```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_PAPER=true      # false for live trading
```

4. Start the MCP server: `alpaca-mcp`

See [packages/alpaca_mcp/README.md](../../packages/alpaca_mcp/README.md) for full details.

### Interactive Brokers

1. Install IB Gateway from [interactivebrokers.com](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Enable **API → Enable ActiveX and Socket Clients** in IB Gateway settings
3. Set environment variables:

```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=4001         # 4001=IB Gateway, 7497=TWS
IBKR_CLIENT_ID=1
```

4. Start IB Gateway, then: `ibkr-mcp`

See [packages/ibkr_mcp/README.md](../../packages/ibkr_mcp/README.md) for full details.

### eTrade

1. Get developer keys at [developer.etrade.com](https://developer.etrade.com/getting-started)
2. Set environment variables:

```bash
ETRADE_CONSUMER_KEY=your_consumer_key
ETRADE_CONSUMER_SECRET=your_consumer_secret
ETRADE_SANDBOX=true    # always start here
```

3. Start the server: `etrade-mcp`
4. Complete OAuth: call `etrade_authorize` via the MCP tool to get the auth URL, then submit the verifier code

See [packages/etrade_mcp/README.md](../../packages/etrade_mcp/README.md) for the full OAuth flow.

---

## Risk Limits

Hard limits are enforced in `packages/quant_pod/execution/risk_gate.py`. They cannot be bypassed by agents or prompts. All defaults can be overridden via environment variables.

| Rule | Default | Env var | Behavior when breached |
|------|:-------:|---------|------------------------|
| Max position % of equity | 10% | `RISK_MAX_POSITION_PCT` | Order scaled down or rejected |
| Max position notional | $20,000 | `RISK_MAX_POSITION_NOTIONAL` | Order scaled down or rejected |
| Max gross exposure | 150% | — | Order rejected |
| Max net exposure | 100% | — | Order rejected |
| Daily loss limit | 2% | `RISK_DAILY_LOSS_LIMIT_PCT` | Trading **halted for the day** |
| Min daily volume (ADV) | 500,000 shares | `RISK_MIN_DAILY_VOLUME` | Order rejected |
| Max ADV participation | 1% | — | Order quantity capped |
| Restricted symbols | (none) | `RISK_RESTRICTED_SYMBOLS` | Order rejected |

### Configuring via environment

```bash
# .env
RISK_MAX_POSITION_PCT=0.05        # 5% max per symbol
RISK_MAX_POSITION_NOTIONAL=10000  # $10k hard cap
RISK_DAILY_LOSS_LIMIT_PCT=0.01    # 1% daily loss halt
RISK_MIN_DAILY_VOLUME=1000000     # 1M ADV minimum
RISK_RESTRICTED_SYMBOLS=GME,AMC   # comma-separated never-trade list
```

---

## Kill Switch

The kill switch is a hard emergency stop. When active, no orders are submitted and the trading flow exits immediately on every tick.

### Check status

```
MCP tool: get_system_status
Returns: {kill_switch_active, risk_halted, broker_mode, session_id}
```

### Activate / deactivate

The kill switch uses a sentinel file at the path set by `KILL_SWITCH_SENTINEL` (default: `~/.quant_pod/KILL_SWITCH_ACTIVE`):

```bash
# Activate — halts all trading immediately
touch ~/.quant_pod/KILL_SWITCH_ACTIVE

# Deactivate
rm ~/.quant_pod/KILL_SWITCH_ACTIVE
```

The sentinel persists across process restarts. Always call `get_system_status` at the start of every trading session.

---

## Daily Loss Halt

When the portfolio's daily P&L breaches `RISK_DAILY_LOSS_LIMIT_PCT`, trading is halted for the rest of the calendar day. This state persists via a sentinel file and survives process restarts.

```bash
# Manual reset (only if you've investigated and understand why the halt triggered)
rm ~/.quant_pod/DAILY_HALT_ACTIVE
```

The halt resets automatically at the start of the next trading day (based on market calendar).

---

## SmartOrderRouter

`packages/quantcore/execution/smart_order_router.py` selects the execution venue for each order. Routing priority:

1. **Alpaca** — if `ALPACA_API_KEY` is set and `alpaca-mcp` is reachable
2. **IBKR** — if `IBKR_HOST` is set and IB Gateway is connected
3. **eTrade** — if `ETRADE_CONSUMER_KEY` is set and authenticated
4. **PaperBroker** — internal fallback (always available)

The router evaluates spread, latency, and commission from each available venue and routes to the best option. If `USE_REAL_TRADING=false`, the PaperBroker is always used regardless of which broker credentials are configured.

---

## Data Storage

All state is stored in a single DuckDB file:

```bash
# Default path (configurable via TRADER_DB_PATH)
~/.quant_pod/trader.duckdb
```

Tables in the database:

| Table | Contents |
|-------|----------|
| `positions` | Open positions and cash balance |
| `closed_trades` | Realized P&L with session tracking |
| `fills` | Order execution history with slippage |
| `decision_events` | Audit trail — every agent decision logged |
| `agent_memory` | Structured blackboard (replaces markdown) |
| `signal_state` | TTL-gated signals (minute analyst → tick executor) |
| `strategies` | Strategy registry |
| `regime_strategy_matrix` | Regime → strategy allocation weights |

```bash
# Inspect manually
duckdb ~/.quant_pod/trader.duckdb
> SELECT * FROM positions;
> SELECT * FROM fills ORDER BY filled_at DESC LIMIT 10;
```
