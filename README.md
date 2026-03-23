<p align="center">
  <img src="logo.png" alt="QuantStack Logo" width="180"/>
</p>

<h1 align="center">QuantStack</h1>

<p align="center">
  <strong>Autonomous quantitative trading — Claude agents research strategies, reason about entries and exits via multi-agent debate, execute through an immutable risk gate, and every loss trains the next iteration.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="v1.0.0">
</p>

---

## Architecture

Three decoupled loops run concurrently. Research thinks. Trading executes. Learning closes the feedback loop.

```
Research Loop (Claude Opus agents)        Trading Loop (Claude, LLM-driven)
  hypothesis → backtest → ML train          SignalEngine (15 collectors, 2-6s)
  walk-forward validation                     → Claude reasons (bull/bear/risk debate)
  overfitting gate (deflated Sharpe, PBO)     → instrument selection (equity/options)
  strategy promotion                          → risk_gate.py (immutable) → Broker
       │                                           │
       └──────── Learning Loop ◄───────────────────┘
                   ReflexionMemory (loss root-cause classification)
                   CreditAssigner (attribute to signal/regime/sizing)
                   TextGradOptimizer (critique → prompt updates)
```

**Why this matters:** MCP tools provide data (signals, portfolio, options chains, news). Claude provides ALL reasoning — entry, exit, instrument selection, sizing, hold/trim decisions. The immutable risk gate is the last line of defense. Losses feed back into research as structured attributions.

## What's Novel

- **Two-loop separation** — Research runs Claude Opus sub-agents (quant-researcher, ml-scientist, strategy-rd, risk, execution-researcher). Trading runs a Claude-driven loop with position-monitor and trade-debater agents for TradingAgents-style multi-agent debate. They share state through DuckDB.

- **15-collector SignalEngine** — Trend, momentum, volatility, regime (ADX+ATR+HMM), volume/OFI, risk (VaR, drawdown), events, fundamentals, sentiment, flow, sector rotation, cross-asset, statarb, ML predictions, macro, quality, options flow. Runs concurrently, fault-tolerant per collector, wall-clock 2–6s.

- **Immutable risk gate** — Hard-coded in Python, not configurable by any agent or prompt. Position limits, daily loss halt (persists via sentinel file across restarts), liquidity floors, options premium caps. Kill switch halts everything in one op.

- **Self-improving from losses** — Every losing trade > 1% is classified by root cause, attributed to a pipeline step, and fed back into the research loop. HypothesisJudge gates new research against known failures and data snooping. Dormant modules (OPROLoop, TrajectoryEvolution) activate at 500+ trades.

- **Overfitting-aware backtesting** — Walk-forward with purged embargo gaps, deflated Sharpe ratios, probability of backtest overfitting (PBO), IS/OOS ratio checks. Monte Carlo simulation for sparse-signal strategies.

## Quick Start

```bash
git clone https://github.com/kbichave/QuantStack.git && cd QuantStack
uv sync --all-extras
cp .env.example .env   # set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPHA_VANTAGE_API_KEY

# Preflight
source .env && python -m quantstack.coordination.preflight SPY

# Run
FORCE_LOOPS=1 ./scripts/start_research_loop.sh   # research loop still gated
./scripts/start_trading_loop.sh                   # trading loop runs autonomously
```

Supports Alpaca, IBKR, eTrade, and a zero-config PaperBroker. MCP tools expose the full stack to Claude Code (`quantstack-mcp`). REST API at `quantstack-api`.

---

<sub>Educational and research purposes only. Not financial advice. Past performance does not guarantee future results.</sub>

<p align="center">Built by Kshitij Bichave</p>
