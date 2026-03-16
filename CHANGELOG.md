# Changelog

All notable changes to QuantCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-15

### Added

#### New MCP Packages
- `alpaca_mcp` — FastMCP server for Alpaca brokerage (11 tools: account queries, market data, order management, option chains; paper + live mode via `ALPACA_PAPER`)
- `ibkr_mcp` — FastMCP server for Interactive Brokers via IB Gateway (11 tools; degraded-startup design — server starts even if gateway is offline, tools return errors until reconnected)

#### QuantCore: Production Execution Layer (`packages/quantcore/execution/`)
- `BrokerInterface` — abstract base with typed exceptions (`BrokerConnectionError`, `BrokerOrderError`, `BrokerAuthError`)
- `UnifiedModels` — broker-agnostic data model layer (`UnifiedBalance`, `UnifiedOrder`, `UnifiedOrderResult`, `UnifiedPosition`, `UnifiedQuote`)
- `PreTradeRiskGate` — 6 sequential pre-trade checks: kill switch, order size, position size, open positions cap, orders-per-minute rate limiter, daily drawdown halt
- `KillSwitch` — file-sentinel emergency halt (`/tmp/KILL_TRADING` by default); survives process restarts; shell-activatable with `touch`
- `FillTracker` — thread-safe in-memory position and P&L tracker with daily ledger and mark-to-market; `reset_daily_pnl()` called at market open
- `SmartOrderRouter` — best-execution routing: equities → Alpaca primary / IBKR fallback; futures/FX → IBKR only; paper mode forces Alpaca paper; per-broker health-check cache (5-min TTL)
- `TCAEngine` — Perold (1988) implementation shortfall with Almgren-Chriss square-root market impact; pre-trade cost forecast and algo recommendation (IMMEDIATE / TWAP / VWAP / POV / LIMIT); post-trade IS measurement vs arrival price, VWAP, and TWAP
- `AsyncExecutionLoop` — async event loop for non-blocking order processing

#### QuantCore: Microstructure Features (`packages/quantcore/microstructure/`)
- `MicrostructureFeatureEngine` — real-time tick-data signals: Order Flow Imbalance (OFI), Kyle's Lambda (price impact via OLS), bid-ask spread stats, VPIN (Volume-Synchronized Probability of Informed Trading), trade intensity, Roll (1984) effective spread estimator; pluggable `on_trade`/`on_quote` callbacks for streaming ingestion
- `OrderBookReconstructor` — reconstructs limit order book state from `TradeTick` + `QuoteTick` streams

#### QuantCore: Portfolio Optimization (`packages/quantcore/portfolio/`)
- `MeanVarianceOptimizer` — Markowitz mean-variance with Ledoit-Wolf covariance shrinkage; objectives: `MAX_SHARPE`, `MIN_VARIANCE`, `RISK_PARITY`, `MAX_DIVERSIFICATION`; scipy SLSQP solver; per-position weight bounds; sector-level constraints; turnover cost penalty; automatic fallback to equal-weight on infeasible constraints

#### QuantCore: RL Production System (`packages/quantcore/rl/`)
- `RLProductionConfig` — versioned, env-driven config (`QUANTRL_` prefix); per-tier feature flags; shadow mode on by default; promotion thresholds (63-day shadow minimum for execution/sizing, 126-day for meta)
- `KnowledgeStoreRLBridge` — maps backtest results and trade history into RL training datasets
- State vector modules for execution (8 features), sizing (10 features), and meta (36 features) agent tiers
- `OnlineAdapter` — live trade outcome → real-time weight update with catastrophic forgetting guard and convergence check
- `PromotionGate` — shadow → live validation: walk-forward folds (≥60% positive), Monte Carlo significance (p < 0.05), max drawdown (≤12%), Sharpe threshold (≥0.5)
- `ShadowMode` — RL recommendations tagged `[SHADOW]`; observations collected for future promotion validation without affecting execution
- `RLTools` — MCP tool implementations backing Phase 6 endpoints (`get_rl_recommendation`, `promote_strategy`, etc.)

#### QuantCore: Other
- `backtesting/stats.py` — extended backtest performance statistics (Calmar, Sortino, tail ratio)
- `research/overfitting.py` — overfitting detection and deflated Sharpe ratio utilities

#### QuantPod: Execution Layer (`packages/quant_pod/execution/`)
- 12-module execution layer: `PaperBroker`, `EtradeBroker`, `OrderLifecycle` (state machine: submitted → open → filled/cancelled), `RiskGate` (QuantPod wrapper), `RiskState` (hot-path in-memory mirror), `BrokerFactory`, `MicrostructurePipeline`, `TickExecutor`, `SignalCache` (DuckDB-backed with configurable TTL)

#### QuantPod: New Subsystems
- `api/` — FastAPI REST interface for trading operations (`quantpod-api` CLI entry point, port 8420)
- `audit/` — immutable `DecisionLog` event store logging every agent decision, execution event, and risk check
- `guardrails/` — runtime sanity checks: signal plausibility validation, market halt detection
- `monitoring/` — `AlphaMonitor` (signal quality degradation alerts), `DegradationDetector` (IS vs OOS Sharpe divergence), `Metrics` (Sharpe, drawdown, consistency tracking)
- `db.py` — DuckDB schema and migrations; 12 tables shared via a single connection injected through `TradingContext`
- `context.py` — `TradingContext` dependency injection container; all services share one DuckDB connection for ACID cross-service transactions
- `llm_config.py` — multi-provider LLM routing (Bedrock, Anthropic, OpenAI, Vertex AI, Gemini, plus tier-2/3 providers); per-tier model overrides (`LLM_MODEL_IC`, `LLM_MODEL_POD`, etc.); automatic credential detection with fallback chain

#### QuantPod: New Agents and Flows
- `portfolio_optimizer_agent.py` — converts pod manager signals into mean-variance optimal capital weights subject to risk gate position limits
- `microstructure_signal_agent.py` — generates trading signals from OFI, VPIN, and Kyle's Lambda tick-data features
- `IntraDayMonitorFlow` — real-time position monitoring loop that triggers stop-loss exits and partial reductions when thresholds are breached (`quantpod-monitor` CLI)
- `StrategyValidationFlow` — out-of-sample validation gate before strategy promotion from `forward_testing` to `live`
- `DecoderCrew` — Phase 4 crew for reverse-engineering strategy rules from trade signal history

#### QuantPod: Alpha Signals IC Group
- `news_sentiment_ic` — Alpha Vantage news ingestion + sentiment scoring
- `options_flow_ic` — unusual options activity (UOA) detection
- `fundamentals_ic` — P/E ratio, debt metrics, earnings quality assessment
- `alpha_signals_pod_manager` — synthesizes all three alpha signal ICs
- `OptionsFlowTool`, `PutCallRatioTool` — singleton tools registered in `TOOL_REGISTRY`

#### Multi-LLM Provider Support (`packages/quant_pod/llm_config.py`)
- **12 supported providers** across three tiers:
  - Tier 1 (recommended): `bedrock`, `anthropic`, `openai`, `vertex_ai`, `gemini`
  - Tier 2: `azure`, `groq`, `together_ai`, `fireworks_ai`, `mistral`
  - Tier 3: `ollama`, `custom_openai`
- `LLM_PROVIDER` env var selects the primary provider (default: `bedrock`)
- `LLM_FALLBACK_CHAIN` — comma-separated fallback providers tried in order if primary credentials are unavailable
- Per-tier model overrides: `LLM_MODEL_IC`, `LLM_MODEL_POD`, `LLM_MODEL_ASSISTANT`
- Automatic per-provider credential detection (cached); agents degrade gracefully when a provider is unconfigured

#### QuantCore: Settings Refactor (`packages/quantcore/config/settings.py`)
- Nested provider settings with isolated env prefixes (prevents field name collisions):
  - `AlpacaSettings` (`ALPACA_` prefix): `api_key`, `secret_key`, `paper`
  - `PolygonSettings` (`POLYGON_` prefix): `api_key`
  - `IBKRSettings` (`IBKR_` prefix): `host`, `port`, `client_id`, `timeout`
- `data_provider_priority` field (default: `alpaca,polygon,alpha_vantage`); registry skips providers with missing credentials automatically
- `data_end_date` defaults to today (was hardcoded `2024-12-31`); validator warns early when `ALPHA_VANTAGE_API_KEY` is unset

#### QuantCore: Extended Timeframes (`packages/quantcore/config/timeframes.py`)
- Intraday timeframes added to `Timeframe` enum and `TIMEFRAME_HIERARCHY`: `M30` (30-min), `M15`, `M5`, `M1` (order flow), `S5` (5-second HFT bars)
- Per-timeframe `TimeframeParams` tuned for intraday scales (EMA 9/21, RSI/ATR 9 at M5 and below)

#### QuantCore: Walk-forward & Overfitting (`packages/quantcore/research/`)
- `CPCVEvaluator` — Combinatorial Purged CV evaluator; feeds OOS returns into DSR (Deflated Sharpe Ratio) and PBO (Probability of Backtest Overfitting) for formal overfitting detection
- `OverfittingReport` with verdict: `GENUINE | SUSPECT | OVERFIT`

#### QuantCore: RL Environment Production Guards
- `AlphaSelectionEnvironment`: `production_mode=True` raises `ValueError` instead of generating synthetic returns; `from_knowledge_store()` factory populates real IC signal history
- `SizingEnvironment`: same `production_mode` guard and `from_knowledge_store()` factory
- `RLOrchestrator`: accepts `knowledge_store` kwarg; environments seeded with real trade history when provided; graceful warning (not error) when no store is supplied

#### QuantArena: Simulation Enhancements
- `SimulationResult` extended with institutional reporting fields:
  - Sharpe ratio 95% CI (Lo 2002 method), Calmar ratio, sample size adequacy flag
  - Benchmark comparison: `alpha`, `beta`, `information_ratio` vs `benchmark_symbol` (default: SPY)
  - `overfitting_verdict` and summary from `CPCVEvaluator`
  - `tca_report` — aggregate TCA from `TCAEngine` recorded throughout simulation
  - `walk_forward_summary` — per-fold IS/OOS results when `walk_forward_mode=True`
- `HistoricalConfig` additions:
  - `benchmark_symbol` for alpha/beta/IR reporting
  - `max_portfolio_correlation` — correlation-concentration filter rejecting BUY orders that push avg pairwise portfolio correlation above threshold (default: 1.0 = disabled)
  - `walk_forward_mode`, `walk_forward_n_folds`, `walk_forward_test_days`
- `SimBroker` enhancements:
  - `max_daily_loss_pct` now enforced in broker (was in config but not wired)
  - `update_prices()` accepts `volumes`/`volatilities` enabling Almgren-Chriss impact-based slippage (replaces flat bps model when volume data is available)
  - Immutable `_order_audit` trail separate from `_orders` (never filtered or modified post-hoc)
  - Intraday daily loss tracking: `_day_open_equity` reset on each new trading day

#### QuantPod: State Management Migration
- **`Blackboard`** migrated from markdown file to DuckDB (`agent_memory` table): O(log n) SQL reads replace O(n) full-file scans; JSON content prevents prompt injection; concurrent writes are ACID-safe; public API unchanged — no callers needed to change
- **`SkillTracker`** migrated to DuckDB; adds IC (Information Coefficient) and ICIR (IC Information Ratio) metrics alongside win-rate — separates signal quality from position sizing and execution noise

#### Environment Configuration (`.env.example`)
- Full overhaul: all 12 LLM providers documented with example values
- Data provider section with `DATA_PROVIDER_PRIORITY`, Alpaca, Polygon, Alpha Vantage, IBKR blocks
- AWS Bedrock credential chain documented; `AWS_PROFILE` example for SSO
- All previously undocumented env vars added (broker credentials, risk limits, Discord, RL config)

#### QuantArena
- `historical/risk_metrics.py` — comprehensive backtesting risk metrics (Calmar ratio, Sortino ratio, drawdown distribution)

#### Infrastructure
- `Dockerfile` — Python 3.11-slim production image built with `uv`; health check on port 8420; volume mount for DuckDB persistence
- `docker-compose.yml` — local development stack with volume persistence and source hot-reload
- `.github/workflows/ci.yml` — CI/CD pipeline: lint (`ruff`) → type check (`mypy`) → unit tests → integration tests

#### Scripts
- `scripts/scheduler.py` — APScheduler-based daily session triggers: 09:15 morning routine (review → meta → trade), 12:30 mid-day check, 15:45 pre-close check, 17:00 Friday weekly reflect
- `scripts/bootstrap_rl_training.py` — RL model initialization from historical OHLCV data
- `scripts/log_decision.py` — manual decision logging to the audit trail (for human-in-the-loop overrides)
- `scripts/notify_discord.py` — Discord webhook notification helper (used by `AlphaMonitor` for degradation alerts)
- `scripts/validate_brief_quality.py` — `DailyBrief` quality checks (missing fields, low-confidence signals, empty pod notes)
- `scripts/docker-entrypoint.sh` — container entrypoint with command dispatch (`api`, etc.)

### Changed
- Restructured to `packages/` monorepo layout with `quantcore`, `quant_arena`, `quant_pod`, `etrade_mcp`, `alpaca_mcp`, `ibkr_mcp`
- Renamed package from `trader` to `quantcore`
- Consolidated `math_models`, `microstructure`, `research`, `signals` into main `quantcore` package
- eTrade OAuth layer moved from `packages/etrade_mcp/` into `packages/quant_pod/tools/etrade/`; `etrade_mcp` is now a thin FastMCP wrapper delegating to the quant_pod tools
- `TradingDayFlow` updated: regime-adaptive crew config via `regime_config.py`, signal cache handoff to `TickExecutor`, TCA arrival-price recording, RL online adapter shadow feedback loop

## [0.1.0] - 2024-12-04

### Added

#### Core Framework
- Multi-timeframe hierarchical trading system
- Event-driven backtesting engine with realistic costs
- Configurable position sizing and risk management
- DuckDB-based data storage and retrieval

#### Feature Engineering
- 200+ technical indicators across categories:
  - Trend: EMA, SMA, MACD, ADX, Aroon
  - Momentum: RSI, Stochastic, Williams %R, CCI
  - Volatility: ATR, Bollinger Bands, Keltner Channels
  - Volume: OBV, VWAP, Volume Profile
  - Market Structure: Support/Resistance, Swing Points
  - Gann Analysis: Swing points, retracements
  - Wave Analysis: Elliott Wave detection
- Multi-timeframe feature factory
- Feature scaling and normalization utilities

#### Machine Learning
- LightGBM, XGBoost, CatBoost model training
- Ensemble model with weighted averaging
- SHAP-based feature importance
- Hyperparameter tuning with Optuna

#### Reinforcement Learning (Optional)
- PPO and DQN agents for trading
- Custom Gymnasium environments
- Multi-objective reward shaping
- Experience replay and prioritized sampling

#### Research Tools
- Statistical tests (ADF, Granger causality)
- Alpha decay analysis
- Walk-forward validation
- Harvey-Liu multiple testing correction
- Leakage detection utilities

#### Market Microstructure
- Limit order book simulation
- Price impact models (Almgren-Chriss, Bouchaud)
- Execution algorithms (TWAP, VWAP, IS)
- Market making simulation (Avellaneda-Stoikov)

#### Mathematical Models
- Geometric Brownian Motion
- Stochastic volatility (Heston, SABR)
- Kalman and particle filters
- Portfolio optimization utilities

#### Strategies
- Mean reversion (z-score based)
- Momentum (multi-timeframe)
- Composite strategy framework
- Options strategies (covered calls, spreads)

#### Data Sources
- Alpha Vantage integration
- FRED economic data
- Earnings calendar
- News sentiment (extensible)

### Infrastructure
- Pydantic-based configuration
- Loguru structured logging
- Comprehensive test suite
- Type hints throughout

---

## Version History

- `0.1.0` - Initial release

[Unreleased]: https://github.com/kbichave/QuantStack/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kbichave/QuantStack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kbichave/QuantStack/releases/tag/v0.1.0

