# Deployment Guide

Covers Docker, CI/CD, data paths, and utility scripts.

---

## Docker

### Quick start

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env — set at minimum OPENAI_API_KEY and one data provider key

# Start all services
docker compose up

# Detached
docker compose up -d

# Stop
docker compose down
```

### Services

The `docker-compose.yml` defines:

| Service | Description |
|---------|-------------|
| `quantpod` | Main QuantPod process (trading flow + MCP server) |
| `quantcore-mcp` | QuantCore MCP server (indicators, backtesting) |
| `quantpod-api` | FastAPI REST interface for trading operations (port 8420) |
| `alpaca-mcp` | Alpaca broker MCP (optional — starts only if `ALPACA_API_KEY` is set) |
| `ibkr-mcp` | Interactive Brokers MCP (optional — starts only if `IBKR_HOST` is set and IB Gateway is running) |

Environment variables are injected from your `.env` file. No secrets are baked into the image.

### Building the image manually

```bash
docker build -t quantstack:latest .
```

---

## CI/CD

GitHub Actions pipeline is defined in `.github/workflows/ci.yml`.

### What it runs on every push / PR

1. **Lint** — `ruff check packages/`
2. **Type check** — `mypy packages/` (non-blocking warnings only)
3. **Unit tests** — `pytest tests/unit/ -v`
4. **Integration tests** — `pytest tests/ -v --ignore=tests/unit` (skips live broker tests)

### Running locally

```bash
# Full test suite
uv run pytest tests/ -v

# Unit tests only (fast, no network)
uv run pytest tests/unit/ -v

# Specific module
uv run pytest tests/quant_pod/ -v -k "test_blackboard"

# With coverage
uv run coverage run -m pytest tests/ && uv run coverage report
```

---

## Data Paths

All runtime data lives under `~/.quant_pod/` by default. Every path is configurable via environment variables.

```
~/.quant_pod/
├── KILL_SWITCH_ACTIVE     # Sentinel — present = kill switch on
└── DAILY_HALT_ACTIVE      # Sentinel — present = daily loss halt active
```

All state (positions, fills, audit, memory) lives in PostgreSQL: `TRADER_PG_URL`.

### Env overrides

```bash
TRADER_PG_URL=postgresql://localhost/quantpod
KILL_SWITCH_SENTINEL=~/.quant_pod/KILL_SWITCH_ACTIVE
```

Use psql to inspect the database:

```bash
psql $TRADER_PG_URL
=> \dt
=> SELECT * FROM positions;
=> SELECT symbol, realized_pnl, closed_at FROM closed_trades ORDER BY closed_at DESC LIMIT 20;
```

---

## Utility Scripts

Located in `scripts/`.

### `bootstrap_rl_training.py`

Initialise the RL agent with historical data before the first trading session.

```bash
uv run python scripts/bootstrap_rl_training.py --symbol SPY --days 252
```

Fetches OHLCV history, runs feature extraction, and writes the initial RL model checkpoint to `~/.quant_pod/rl/`.

### `log_decision.py`

Append a manual decision event to the audit trail (for human-in-the-loop overrides).

```bash
uv run python scripts/log_decision.py \
  --symbol AAPL \
  --action BUY \
  --reasoning "Breaking out of 6-month consolidation" \
  --confidence 0.75
```

### `notify_discord.py`

Post a message to the configured Discord webhook.

```bash
# Requires DISCORD_WEBHOOK_URL in .env
uv run python scripts/notify_discord.py --message "Daily P&L: +$1,240"
```

Used by `AlphaMonitor` to post degradation alerts automatically after each session.

### `validate_brief_quality.py`

Run quality checks on a `DailyBrief` JSON file (checks for missing fields, low-confidence signals, empty pod notes).

```bash
uv run python scripts/validate_brief_quality.py --brief /tmp/brief.json
```

---

## MCP Server Config (`.mcp.json`)

The repo ships with `.mcp.json` for Claude Code integration. All MCP servers are registered there. Edit to add/remove servers or change env vars:

```json
{
  "mcpServers": {
    "quantcore": { "command": "quantcore-mcp" },
    "quantpod":  { "command": "quantpod-mcp" },
    "alpaca":    { "command": "alpaca-mcp", "env": { "ALPACA_PAPER": "true" } },
    "ibkr":      { "command": "ibkr-mcp" }
  }
}
```

---

## Environment Variables Reference

See `.env.example` for the full annotated list. Summary of the most important variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for CrewAI agents |
| `ALPACA_API_KEY` | — | Alpaca data + trading |
| `ALPACA_PAPER` | `true` | Paper mode toggle |
| `DATA_PROVIDER_PRIORITY` | `alpaca,polygon,alpha_vantage` | Data source order |
| `USE_REAL_TRADING` | `false` | Master live trading switch |
| `TRADER_PG_URL` | `postgresql://localhost/quantpod` | PostgreSQL connection string |
| `RISK_DAILY_LOSS_LIMIT_PCT` | `0.02` | Daily loss halt threshold |
| `DISCORD_WEBHOOK_URL` | — | Alert notifications |
