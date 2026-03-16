# quant_pod

Multi-agent trading system built on CrewAI. Orchestrates 13 specialized AI agents across 5 pods to produce a `DailyBrief`, which Claude Code (acting as portfolio manager) uses to make trade decisions.

## Installation

```bash
uv sync --all-extras
```

## Agent Hierarchy

```
Claude Code (Portfolio Manager — you)
    │
    │  run_analysis MCP tool → DailyBrief returned
    ▼
Trading Assistant  (stop_at_assistant=True — crew stops here)
    │
    ├── Data Pod Manager
    │   └── data_ingestion_ic          Fetch OHLCV data
    │
    ├── Market Monitor Pod Manager
    │   ├── market_snapshot_ic         Current price/volume snapshot
    │   └── regime_detector_ic         ADX/ATR market regime (real indicator-based)
    │
    ├── Technicals Pod Manager
    │   ├── trend_momentum_ic          RSI, MACD, ADX, SMA metrics
    │   ├── volatility_ic              ATR, Bollinger Bands, VaR
    │   └── structure_levels_ic        Support/resistance levels
    │
    ├── Quant Pod Manager
    │   ├── statarb_ic                 ADF test, information coefficient
    │   └── options_vol_ic             IV, Greeks, skew
    │
    ├── Risk Pod Manager
    │   ├── risk_limits_ic             VaR, stress tests, limit checks
    │   └── calendar_events_ic         Earnings, FOMC, event calendar
    │
    └── Alpha Signals Pod Manager      (new)
        ├── news_sentiment_ic          Alpha Vantage news + sentiment
        ├── options_flow_ic            Unusual option activity (UOA detection)
        └── fundamentals_ic            P/E, debt ratios, earnings quality
```

The crew runs with `stop_at_assistant=True`. Claude Code receives the `DailyBrief` and makes the final decision — buy, sell, hold, or register a new strategy.

## Execution Layer

All execution is mediated through hard-coded risk controls. No agent or prompt can bypass them.

```
TradingDayFlow
    │
    ├── KillSwitch.check()    → halt if active
    ├── RiskGate.check()      → enforce position / loss / liquidity limits
    ├── SignalCache           → TTL-gated signal handoff (minute analyst → tick executor)
    ├── TickExecutor          → real-time order submission
    └── SmartOrderRouter      → routes to Alpaca / IBKR / eTrade / PaperBroker
```

Key modules in `packages/quant_pod/execution/`:

| Module | Purpose |
|--------|---------|
| `portfolio_state.py` | Open positions, cash, P&L (DuckDB-backed) |
| `risk_gate.py` | Hard position/loss/liquidity limits |
| `kill_switch.py` | Emergency halt (sentinel file, survives restarts) |
| `paper_broker.py` | Internal fill simulation with slippage |
| `signal_cache.py` | TTL-gated signal store for async handoff |
| `tick_executor.py` | Real-time tick-by-tick order execution |
| `smart_order_router.py` | Best-execution venue selection |

## Dependency Injection

All services share a single DuckDB connection wired together in `TradingContext`:

```python
from quant_pod.context import create_trading_context

# Production — persistent DB at ~/.quant_pod/trader.duckdb
ctx = create_trading_context()

# Tests — fully isolated, no file-system side-effects
ctx = create_trading_context(db_path=":memory:")

# ctx exposes: portfolio, risk_gate, risk_state, signal_cache,
#              kill_switch, broker, audit, blackboard, session_id
```

## State Management (DuckDB)

All runtime state lives in a single ACID-safe DuckDB file. No scattered pickle files or CSV logs.

| Table | Contents |
|-------|---------|
| `positions` | Open positions and cash |
| `closed_trades` | Realized P&L |
| `fills` | Order execution history with slippage |
| `decision_events` | Audit trail — every agent decision |
| `agent_memory` | Structured blackboard (replaces markdown file) |
| `signal_state` | TTL-gated signals for tick executor |
| `strategies` | Strategy registry |
| `regime_strategy_matrix` | Regime → allocation weights |

## Regime Detection

`regime_detector_ic` uses real ADX and ATR indicators — no LLMs, no stubs:

- **Trend regime**: ADX > 25 = trending, ADX < 20 = ranging
- **Volatility regime**: ATR percentile — < 25th = low, 75–90 = high, > 90 = extreme

The regime drives `regime_config.py` which selects which ICs are active and what thresholds they use. Fewer ICs run in ranging/low-vol conditions; full suite in trending.

## Audit & Monitoring

New sub-packages track signal quality and system health:

| Package | Purpose |
|---------|---------|
| `audit/` | `DecisionLog` — structured event logging with portfolio snapshots |
| `monitoring/` | `AlphaMonitor` — signal quality; `DegradationDetector` — IS/OOS divergence |
| `guardrails/` | Risk rule enforcement layer |

## Quick Start

```python
from quant_pod.context import create_trading_context
from quant_pod.flows.trading_day_flow import TradingDayFlow

ctx = create_trading_context()

flow = TradingDayFlow(
    portfolio=ctx.portfolio,
    risk_gate=ctx.risk_gate,
    audit=ctx.audit,
    signal_cache=ctx.signal_cache,
)

# Returns a DailyBrief — Claude Code makes the trade decision from here
brief = await flow.run(symbol="SPY")
```

## MCP Server

QuantPod exposes 30+ tools via its own MCP server:

```bash
quantpod-mcp
```

Key tool groups: `run_analysis`, `get_portfolio_state`, `execute_trade`, `register_strategy`, `run_backtest`, `run_walkforward`, `decode_strategy`, `get_regime_strategies`, `get_rl_recommendation`.

See [CLAUDE.md](../../CLAUDE.md) section 3 for the full tool inventory.

## Configuration

- Agent prompts: `packages/quant_pod/prompts/`
- Task definitions: `packages/quant_pod/crews/config/tasks.yaml`
- Risk limits: env vars (`RISK_*`) or `RiskLimits.from_env()`
- Regime-strategy matrix: `regime_strategy_matrix` DuckDB table (managed via MCP tools)

## Documentation

- [Architecture](../../docs/architecture/quant_pod.md) — detailed agent and execution layer design
- [Execution Setup](../../docs/guides/execution_setup.md) — broker config, risk limits, kill switch
- [Deployment](../../docs/guides/deployment.md) — Docker, CI/CD, data paths
