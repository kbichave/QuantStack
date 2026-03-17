<p align="center">
  <img src="logo.png" alt="QuantStack Logo" width="180"/>
</p>

<h1 align="center">QuantStack</h1>

<p align="center">
  <strong>An AI-native quantitative trading system — research, execution, and learning in one stack.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/version-0.5.0-green.svg" alt="v0.5.0">
  <img src="https://img.shields.io/badge/MCP%20tools-97-purple.svg" alt="97 MCP Tools">
</p>

<p align="center">
  <a href="docs/architecture/README.md">Architecture</a> •
  <a href="docs/guides/quickstart.md">Quick Start</a> •
  <a href="https://github.com/kbichave/QuantStack/issues">Issues</a>
</p>

---

QuantStack replaces the traditional "strategy coded in isolation" model with a system where **Claude Code acts as the portfolio manager** — reasoning over market data, backtesting hypotheses, executing trades, and learning from outcomes across sessions.

Three trading styles, one system:

| Style | Skill | Cadence | Decision inputs |
|-------|-------|---------|-----------------|
| **Equity swing** | `/trade` | Daily | SignalEngine brief, regime, technicals |
| **Long-term investing** | `/invest` | Weekly | DCF, quality scorecard, insider flow |
| **Options** | `/options` | Per-event | IV rank, event calendar, Greeks |

The system runs **unattended** via `AutonomousRunner`, discovers new strategies overnight via `AlphaDiscoveryEngine`, and tracks signal quality with IC/ICIR metrics that alert on Discord when an agent degrades.

---

## How It Works

```
Claude Code (Portfolio Brain)
│  Skills: /trade  /invest  /options  /workshop  /review  /reflect  /meta
│  Memory: .claude/memory/ (strategy registry, trade journal, lessons)
│
├── get_signal_brief(symbol)   ← 2–6 seconds, no LLM calls
│   └── SignalEngine (7 concurrent Python collectors)
│       ├── Technical indicators (trend, momentum, volatility, structure)
│       ├── Regime classification (ADX + ATR + HMM)
│       ├── Volume & microstructure (OBV, VWAP, OFI)
│       ├── Risk (VaR, drawdown, liquidity headroom)
│       ├── Events (earnings, FOMC)
│       ├── Fundamentals (P/E, FCF, quality)
│       └── Sentiment (news scoring)
│
├── run_backtest_mtf / run_walkforward    ← Strategy validation
│   └── 2-stage: IS screen → OOS walk-forward → register if passing
│
├── execute_trade()
│   └── RiskGate → SmartOrderRouter → Alpaca / IBKR / eTrade / PaperBroker
│
└── AutonomousRunner (unattended)
    └── Overnight: AlphaDiscoveryEngine discovers draft strategies
```

**Previous approach (TradingCrew):** 13 LLM agents → 3–5 minutes per symbol.
**Current approach (SignalEngine):** 7 pure-Python collectors → 2–6 seconds. TradingCrew is still available as a fallback for cases that benefit from LLM synthesis.

---

## Key Capabilities

### Signal Engine
Seven collectors run concurrently and produce a `SignalBrief` — a structured output with market bias, conviction score, risk environment, and regime detail. Fault-tolerant: individual collector failures don't block the brief.

### Strategy Workshop
Design and validate strategies with a full research toolkit:
- **Multi-timeframe backtesting** (`run_backtest_mtf`) — setup on H4/D1, trigger on M15/H1
- **Walk-forward validation** — IS/OOS folds with purged embargo gaps
- **Sparse-signal handling** — auto-adjusts OOS window to guarantee minimum trades per fold
- **Overfitting detection** — DSR, PBO, IS/OOS Sharpe ratio comparisons
- **ML integration** — LightGBM/XGBoost/CatBoost with SHAP explainability

### Options Trading
Full options workflow in `/options`:
- Live chain fetching via Alpaca → Polygon fallback → synthetic fallback
- IV surface: `iv_rank`, `atm_iv_30d`, `skew_25d`, term structure
- Structure analysis: iron condors, credit spreads, debit spreads, straddles
- Built-in decision matrix: IV rank × regime × event → structure
- Options-specific risk gate: premium at risk ≤ 2% equity, DTE 7–60 days

### Long-Term Investing
Fundamental investing workflow in `/invest`:
- DCF shortcut with margin-of-safety gate (≥ 20% required)
- Quality scorecard: Quality + Value + Momentum + Insider Signal (0–10)
- Conviction tiers: High → 5% equity, Moderate → 2.5%, Low → 1.25%
- Weekly review cadence — not daily signal chasing

### Autonomous Runner
Runs unattended without a Claude Code session:
- Checks kill switch → loads active strategies → runs SignalEngine per symbol → routes to execution
- Every decision (including skips) logged to DuckDB audit trail
- `paper_mode=True` hard default; live requires explicit env var

### Alpha Discovery
Overnight strategy generation (60-minute budget):
- Detects regime, selects parameter templates, iterates bounded grid (≤ 200 combinations)
- Two-stage filter: IS screen (fast) → OOS walk-forward
- Registers candidates as `status='draft'` — never auto-promotes, always requires human review

### Learning Loop
Signal quality degrades. QuantStack measures it:
- **IC (Information Coefficient)**: correlation between signal and forward returns
- **ICIR**: IC / IC_std — consistency metric; > 0.5 good, > 1.0 institutional-grade
- **AlphaMonitor**: rolling 30-day IC per agent; Discord alert when IC < 0
- **DegradationDetector**: live Sharpe vs IS Sharpe; flags when IS/OOS ratio > 2×
- **StrategyValidationFlow**: weekly walk-forward re-validation for all registered strategies

---

## Repository Structure

```
QuantStack/
├── packages/
│   ├── quantcore/          # Research library (200+ indicators, ML, options, RL)
│   ├── quant_pod/          # Execution system (signal engine, strategies, agents)
│   │   ├── signal_engine/  # 7 concurrent Python collectors
│   │   ├── autonomous/     # Unattended trading loop
│   │   ├── alpha_discovery/# Overnight strategy generation
│   │   ├── execution/      # Risk gate, order lifecycle, broker routers
│   │   ├── learning/       # IC/ICIR tracking, calibration, expectancy
│   │   ├── monitoring/     # AlphaMonitor, DegradationDetector, Prometheus
│   │   ├── flows/          # TradingDayFlow, IntradayMonitorFlow, ValidationFlow
│   │   ├── guardrails/     # Signal plausibility, TradeTrap defense
│   │   ├── audit/          # Immutable decision log (DuckDB)
│   │   ├── crews/          # TradingCrew (LLM fallback, 13 ICs + 6 pods)
│   │   └── api/            # FastAPI REST server (28 endpoints)
│   ├── alpaca_mcp/         # Alpaca broker MCP (11 tools)
│   ├── ibkr_mcp/           # Interactive Brokers MCP (11 tools)
│   └── etrade_mcp/         # eTrade MCP (OAuth 1.0a)
├── .claude/
│   ├── skills/             # Session type definitions (trade, invest, options, etc.)
│   └── memory/             # Persistent brain (git-ignored, local only)
├── scripts/                # Scheduler, health checks, pipeline runners
├── tests/
├── docs/
├── pyproject.toml          # Unified workspace (uv)
├── CLAUDE.md               # AI system operating manual
└── docker-compose.yml
```

---

## MCP Tool Surface (97 tools)

QuantStack exposes its entire research and execution stack as MCP tools — callable directly from Claude Code without writing scripts.

### QuantCore (54 tools)

| Category | Tools |
|----------|-------|
| **Market data** | `fetch_market_data`, `load_market_data`, `list_stored_symbols`, `get_symbol_snapshot`, `get_market_regime_snapshot` |
| **Fundamentals** | `get_financial_statements`, `get_financial_metrics`, `get_earnings_data`, `get_insider_trades`, `get_institutional_ownership`, `get_analyst_estimates`, `get_company_news`, `screen_stocks` |
| **Technical analysis** | `compute_technical_indicators`, `compute_all_features`, `compute_feature_matrix`, `compute_quantagent_features`, `list_available_indicators` |
| **Backtesting** | `run_backtest`, `get_backtest_metrics`, `run_walkforward`, `run_purged_cv`, `run_monte_carlo`, `run_adf_test` |
| **Signal research** | `validate_signal`, `diagnose_signal`, `detect_leakage`, `check_lookahead_bias`, `compute_alpha_decay`, `compute_information_coefficient` |
| **Options** | `price_option`, `price_american_option`, `compute_greeks`, `compute_implied_vol`, `compute_option_chain`, `analyze_option_structure`, `compute_multi_leg_price`, `score_trade_structure`, `simulate_trade_outcome`, `get_options_chain`*, `get_iv_surface`* |
| **Risk & portfolio** | `compute_position_size`, `compute_portfolio_stats`, `compute_var`, `compute_max_drawdown`, `check_risk_limits`, `stress_test_portfolio` |
| **Market microstructure** | `analyze_liquidity`, `analyze_volume_profile` |
| **Calendar & events** | `get_trading_calendar`, `get_event_calendar` |
| **Trade generation** | `generate_trade_template`, `validate_trade`, `run_screener` |

*\* New in v0.5.0 — live broker chain with Alpaca → Polygon → synthetic fallback*

### QuantPod (43 tools)

| Category | Tools |
|----------|-------|
| **Analysis** | `get_signal_brief`, `run_multi_signal_brief`, `run_analysis` (TradingCrew fallback), `get_regime` |
| **Portfolio** | `get_portfolio_state`, `get_recent_decisions`, `get_system_status`, `get_risk_metrics` |
| **Strategy** | `register_strategy`, `list_strategies`, `get_strategy`, `update_strategy` |
| **Backtesting** | `run_backtest`, `run_backtest_mtf`, `run_backtest_options`, `run_walkforward`, `run_walkforward_mtf`, `walk_forward_sparse_signal` |
| **Execution** | `execute_trade`, `close_position`, `cancel_order`, `get_fills`, `get_fill_quality`, `get_position_monitor` |
| **Audit** | `get_audit_trail` |
| **Learning** | `get_strategy_performance`, `validate_strategy`, `promote_strategy`, `retire_strategy`, `get_rl_status`, `get_rl_recommendation`, `update_regime_matrix_from_performance` |
| **Orchestration** | `run_multi_analysis`, `resolve_portfolio_conflicts`, `get_regime_strategies`, `set_regime_allocation` |
| **Decode** | `decode_strategy`, `decode_from_trades` |
| **IC-level** | `list_ics`, `run_ic`, `run_pod`, `run_crew_subset`, `get_last_ic_output` |

---

## Installation

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kbichave/QuantStack.git
cd QuantStack

# Install everything
uv sync --all-extras

# Or install specific broker support only
uv sync --extra alpaca    # Alpaca SDK
uv sync --extra ibkr      # Interactive Brokers (ib_insync)
uv sync --extra polygon   # Polygon.io data
```

Copy `.env.example` to `.env` and fill in your keys. At minimum:

```bash
# Data + execution
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true        # safe default

# LLM (for AlphaDiscovery and TradingCrew fallback; not needed for SignalEngine)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

---

## LLM Configuration

The **Signal Engine** (primary analysis path) requires **no LLM**. LLMs are used for:
- TradingCrew (optional fallback for deep synthesis)
- AlphaDiscoveryEngine (hypothesis generation)
- AutonomousRunner's GroqPM (non-routine decisions)
- `/workshop` deep reasoning

### Option A — Local Ollama (free)

```bash
brew install ollama
ollama serve
ollama pull qwen3.5:9b   # ~6.6 GB
```

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_IC=ollama/qwen3.5:9b
LLM_MODEL_POD=ollama/qwen3.5:9b
LLM_MODEL_ASSISTANT=ollama/qwen3.5:9b
```

### Option B — Cloud

```bash
LLM_PROVIDER=groq
LLM_MODEL_IC=groq/llama-3.3-70b-versatile   # free tier, fast
```

| Provider | Key | Est. cost per TradingCrew run |
|----------|-----|-------------------------------|
| Local Ollama | — | $0.00 |
| Groq (free tier) | `GROQ_API_KEY` | $0.00 |
| Bedrock Haiku ICs + Sonnet pods | AWS credentials | ~$0.02 |
| OpenAI GPT-4o (all agents) | `OPENAI_API_KEY` | ~$0.12 |

Mix tiers freely — e.g., free local ICs + stronger cloud model for pod synthesis:

```bash
LLM_MODEL_IC=ollama/qwen3.5:9b
LLM_MODEL_POD=bedrock/us.anthropic.claude-sonnet-4-6
LLM_MODEL_ASSISTANT=bedrock/us.anthropic.claude-sonnet-4-6
```

---

## Broker Support

`SmartOrderRouter` selects the execution venue automatically based on which credentials are present.

| Broker | Mode | MCP server | Notes |
|--------|------|------------|-------|
| **Alpaca** | paper + live | `alpaca-mcp` | US equities; options chain requires Options Data subscription |
| **Interactive Brokers** | paper + live | `ibkr-mcp` | Equities + options; requires IB Gateway running |
| **eTrade** | paper + live | `etrade-mcp` | OAuth 1.0a, multi-leg options |
| **PaperBroker** | paper | built-in | Zero-config fallback with slippage simulation |

Priority: IBKR → Alpaca → PaperBroker. Override with `DATA_PROVIDER_PRIORITY`.

---

## Risk Controls

Hard rules enforced in `packages/quant_pod/execution/risk_gate.py`. Not bypassable.

| Rule | Default | Env override |
|------|---------|-------------|
| Max position % of equity | 10% | `RISK_MAX_POSITION_PCT` |
| Max position notional | $20,000 | `RISK_MAX_POSITION_NOTIONAL` |
| Daily loss halt | 2% | `RISK_DAILY_LOSS_LIMIT_PCT` |
| Min daily volume | 500,000 | `RISK_MIN_DAILY_VOLUME` |
| Options: max premium at risk | 2% equity | `RISK_MAX_PREMIUM_AT_RISK_PCT` |
| Options: total premium book | 8% equity | `RISK_MAX_TOTAL_PREMIUM_PCT` |
| Options: DTE at entry | 7–60 days | `RISK_MIN_DTE_ENTRY` / `RISK_MAX_DTE_ENTRY` |

Daily halt state persists via sentinel file — survives process restarts.

---

## Persistent State (DuckDB, 12 tables)

All decisions, trades, signals, and audit events are persisted locally.

| Table | What's stored |
|-------|--------------|
| `strategies` | Registry: rules, parameters, status, backtest summary, instrument type, time horizon |
| `trades` | Every trade: symbol, direction, entry/exit, P&L, regime, strategy |
| `audit_trail` | Every decision: agent, output, confidence, risk verdict |
| `positions` | Open positions: quantity, entry price, unrealized P&L |
| `regime_history` | Regime transitions: date, regime, confidence, ADX, ATR |
| `agent_skills` | IC/ICIR per agent over time |
| `outcomes` | Closed trade outcomes for learning loop |
| `options_chains` | Live options snapshots (populated by `get_options_chain`) |
| `calibration` | Confidence calibration data |
| `waves_regime` | Wave analysis + regime switching states |

---

## Session Skills

Claude Code uses skill files to run structured sessions. Each skill is a step-by-step workflow backed by MCP tools.

| Skill | Purpose |
|-------|---------|
| `/trade` | Daily equity swing: SignalEngine → reason → decide → execute |
| `/invest` | Long-term: fundamental scorecard → DCF → margin of safety → size |
| `/options` | Short-term options: event calendar → IV rank → structure → risk check |
| `/workshop` | Strategy R&D: hypothesize → backtest → walk-forward → register |
| `/review` | Position review: P&L, DTE checks, strategy lifecycle (promote/retire) |
| `/reflect` | Weekly: outcome analysis, IC review, memory update, signal fixes |
| `/meta` | Multi-symbol orchestration: regime matrix, conflict resolution |
| `/decode` | Reverse-engineer strategies from trade history |
| `/compact-memory` | Distill memory files when they exceed 200 lines |

---

## Automated Scheduling

```bash
python scripts/scheduler.py          # start scheduled sessions
python scripts/scheduler.py --run-now morning_routine
```

| Time (ET) | Days | Session |
|-----------|------|---------|
| 09:15 | Mon–Fri | /review → /meta → /trade |
| 12:30 | Mon–Fri | /review (mid-day) |
| 15:45 | Mon–Fri | /review (pre-close) |
| 17:00 | Friday | /reflect (weekly) |

---

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| `signal_engine` | Stable | Primary analysis path (no LLM) |
| `features` (200+ indicators) | Stable | |
| `backtesting` | Stable | MTF, walk-forward, sparse-signal |
| `models` (ML) | Stable | LightGBM, XGBoost, CatBoost + SHAP |
| `options` | Stable | Live chain fetching + Greeks + structure scoring |
| `execution` | Stable | Risk gate, SmartOrderRouter, TCA, kill switch |
| `autonomous` | Stable | Unattended loop, GroqPM |
| `alpha_discovery` | Stable | Overnight discovery, 60-min budget |
| `learning` | Stable | IC/ICIR tracking, calibration, AlphaMonitor |
| `monitoring` | Stable | Degradation detection, Discord alerts |
| `portfolio` (optimization) | Stable | Mean-variance, Ledoit-Wolf shrinkage |
| `microstructure` | Stable | OFI, VPIN, Kyle's lambda |
| `rl` (reinforcement learning) | Experimental | PPO/DQN, shadow mode |
| `crews` (TradingCrew) | Maintenance | LLM fallback; primary path is SignalEngine |

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check packages/

# Start QuantPod MCP server
quantpod-mcp

# Start QuantCore MCP server
quantcore-mcp

# Start REST API
quantpod-api   # http://localhost:8000

# Trigger intraday monitor manually
python -c "from quant_pod.flows.intraday_monitor_flow import IntradayMonitorFlow; print(IntradayMonitorFlow().run())"
```

### CLI Entry Points

| Command | Description |
|---------|-------------|
| `quantcore-mcp` | QuantCore MCP server (54 tools) |
| `quantpod-mcp` | QuantPod MCP server (43 tools) |
| `alpaca-mcp` | Alpaca broker MCP server |
| `ibkr-mcp` | Interactive Brokers MCP server |
| `quantpod-api` | FastAPI REST server (28 endpoints) |
| `quantpod-monitor` | Intraday monitoring loop |

---

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Do not use it for real trading without understanding the risks, implementing appropriate safeguards, and complying with applicable regulations. Past performance does not guarantee future results.

---

<p align="center">Built by Kshitij Bichave</p>
