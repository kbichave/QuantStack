# Audit Findings

*Running accumulator. Agents READ this before starting their work.*
*Each wave builds on previous waves' discoveries.*
*Last updated: 2026-04-07T16:15:00Z*

## Wave 0: Quick Scan
<!-- source: quick-scan, confidence: high, wave: 0 -->

- **Language**: Python 3.11+ (target 3.11–3.12)
- **Framework**: LangGraph 0.4.11+ (StateGraph-based agentic orchestration)
- **LLM Layer**: LiteLLM 1.82.2+ (multi-provider routing with fallback chain)
- **Codebase size**: ~719 .py files, ~193k LOC
- **Problem domain**: Autonomous quantitative trading — AI-native alpha discovery, execution, and learning
- **Key technologies**: PostgreSQL 16 + pgvector (60+ tables), LangFuse, Prometheus, Loki/Grafana, Alpaca/IBKR brokers, Alpha Vantage/Polygon data, AWS Bedrock + Anthropic + OpenAI LLMs, LightGBM/XGBoost/CatBoost ML, FinRL (Torch + SB3), Ollama embeddings

### Architecture Overview
- 3 LangGraph StateGraphs as Docker services
- 22 agents across trading (11), research (7), supervisor (4) graphs
- 27 signal collectors → SignalBrief aggregation
- Immutable risk gate + kill switch + paper mode default
- 55 active tools (manifest) + ~91 remaining stubs in planned state

### CTO Audit Status: IMPLEMENTED
- 169 findings documented 2026-04-06, implementation confirmed by user
- Key fixes applied: ACTIVE/PLANNED tool split, stop-loss enforcement, CI/CD re-enabled
- Remaining: most findings at code-level completed, but systemic gaps persist (see below)

## Wave 1: Deep Research — CTO Audit Cross-Reference
<!-- source: main-claude, confidence: high, wave: 1-synthesis -->

### What's Fixed (Post CTO Audit Implementation)
1. Tool registry split into ACTIVE_TOOLS (55) vs PLANNED_TOOLS — agents no longer call stubs
2. Stop-loss enforcement at OMS level (trade_service.py validates stop_price)
3. CI/CD re-enabled (.github/workflows/ci.yml)
4. BaseCheckpointSaver injectable (not hardcoded MemorySaver)
5. Groq hybrid provider added for cost optimization
6. Bracket order support in execution monitor
7. TradingWindow enum for time gating
8. EWF signal pipeline integrated
9. Bedrock LLM routing with fallback chain
10. Candlestick pattern recognition added

### What's STILL Missing (The "Harvard IB Fund" Gap)

#### Tier 1: EXISTENTIAL (System can't be trusted without these)
1. **Signal IC tracking** — Still stubbed. No signal has been validated against forward returns.
2. **91 tool stubs** — Not bound to agents anymore, but functionality still missing (ML, FinRL, TCA, walk-forward, Monte Carlo)
3. **5 ghost learning modules** — OutcomeTracker, SkillTracker, ICAttribution, ExpectancyEngine, StrategyBreaker all implemented but zero callers
4. **Feedback loops broken** — IC→weight, cost→sizing, loss→research, live-vs-backtest demotion — none closed

#### Tier 2: TABLE STAKES (Any real fund has these)
5. **Execution algorithms** — TWAP/VWAP selected but execute as single fills
6. **Options Greeks in risk gate** — DTE + premium only, no delta/gamma/vega limits
7. **TCA feedback loop** — Pre-trade estimates don't calibrate from realized costs
8. **Intraday circuit breaker** — Only daily loss limit, no unrealized P&L triggers
9. **Liquidity model** — Only ADV check, no spread/depth/time-of-day modeling
10. **Model versioning** — Latest-overwrites, no A/B, no rollback
11. **Hyperparameter optimization** — Hardcoded defaults, no Optuna/Bayesian search

#### Tier 3: DIFFERENTIATORS (What separates a top fund)
12. **Reinforcement learning pipeline** — FinRL stubs exist, 11 tools planned, zero functional
13. **Options market-making** — Currently directional only, no vol arb/dispersion/market-making
14. **Alternative data** — No satellite, credit card, web traffic, patent data
15. **Multi-asset expansion** — Equity + options only, no futures/forex/crypto/fixed income
16. **Meta-learning** — No prompt optimization, no strategy-of-strategies
17. **Causal inference** — No causal alpha discovery (just correlation-based)
18. **Conformal prediction** — No prediction intervals on ML outputs

#### Tier 4: NICE-TO-HAVE (Future competitive edge)
19. **Graph neural networks** — For market structure/sector relationship modeling
20. **Deep hedging** — Neural network-based optimal hedging (Buehler et al.)
21. **Transformer forecasting** — PatchTST, TimesFM for time series
22. **Market microstructure ML** — Order flow toxicity, Kyle lambda, VPIN
23. **NLP evolution** — Beyond FinBERT to domain-specific fine-tuned models

## Stakeholder Interview Summary
<!-- source: interview, wave: 1 -->

- **Vision**: Autonomous trading company, no humans, Harvard-IB-grade
- **Priority**: Full parallel, dependency-graph ordered
- **Scale**: <$100K personal capital
- **Expansion**: RL trading + options MM + alt data + multi-asset (all 4)
- **CTO audit**: Already implemented — baseline is post-fix
