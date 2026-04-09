# Target Architecture: QuantStack Autonomous Fund

**Date:** 2026-04-07
**Horizon:** After all 16 phases (P00–P15)

---

## Architecture Principles

1. **Every decision closes a loop.** No action without feedback. No feedback without behavioral change.
2. **Statistical validation gates everything.** No signal enters synthesis without IC validation. No strategy goes live without walk-forward + PBO.
3. **Risk gate is immutable law.** Strengthened with Greeks, intraday circuits, liquidity model — never weakened.
4. **Multi-asset by design.** AssetClass abstraction enables instruments beyond equity without architectural changes.
5. **Self-improving cognition.** Agent prompts, signal weights, strategy allocations all adapt from measured outcomes.

---

## System Topology (Post P15)

```
┌──────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR GRAPH                              │
│  Health Monitor → Strategy Lifecycle → Self-Healing → Meta-Learn │
│  - Agent quality scorecards           - OPRO prompt optimization │
│  - Strategy promotion/demotion gates  - Research prioritization  │
│  - Kill switch + circuit breakers     - Weekly perf report       │
└────────┬────────────────────────────────┬────────────────────────┘
         │                                │
    ┌────▼────────────────┐    ┌──────────▼──────────────┐
    │   TRADING GRAPH     │    │    RESEARCH GRAPH        │
    │                     │    │                          │
    │  Signal Engine      │    │  Hypothesis Generator    │
    │  (27+ collectors)   │    │  (causal + correlation)  │
    │       │             │    │       │                  │
    │  IC Validator       │    │  Walk-Forward Validator  │
    │  (P01 - filter)     │    │  (PBO, Monte Carlo)      │
    │       │             │    │       │                  │
    │  Adaptive Synthesis │    │  ML Training Pipeline    │
    │  (IC-weighted, P05) │    │  (Optuna, registry, A/B) │
    │       │             │    │       │                  │
    │  Risk Gate          │    │  RL Training             │
    │  (Greeks, circuits) │    │  (FinRL environments)    │
    │       │             │    │       │                  │
    │  Execution Engine   │    │  Causal Discovery        │
    │  (TWAP/VWAP real)   │    │  (DoWhy, EconML)         │
    │       │             │    │       │                  │
    │  TCA Feedback ──────┼────┼───→ Cost Model Update    │
    │       │             │    │                          │
    │  Outcome Tracker ───┼────┼───→ Strategy Breaker     │
    │       │             │    │       │                  │
    │  Learning Modules   │    │  Conformal Predictor     │
    │  (all 5 wired)      │    │  (calibrated intervals)  │
    └─────────────────────┘    └──────────────────────────┘
```

---

## Data Architecture (Post P07)

```
┌─────────────────────────────────────────────┐
│              DATA LAYER                      │
│                                              │
│  Multi-Provider Failover:                    │
│    Alpha Vantage → Polygon → Yahoo → Cache   │
│                                              │
│  Point-in-Time Store:                        │
│    Every feature timestamped at observation   │
│    Look-ahead bias impossible by construction │
│                                              │
│  Alt Data Feeds (P11):                       │
│    Quiver (congressional) → collector        │
│    USPTO (patents) → collector               │
│    SimilarWeb (traffic) → collector          │
│                                              │
│  Database (PostgreSQL + pgvector):            │
│    60+ tables, partitioned by date            │
│    db/ module: schema.py, queries.py,         │
│    migrations.py (incremental extraction)     │
│    New tables via SQLAlchemy, legacy raw SQL  │
│                                              │
│  Staleness Alerting:                         │
│    Per-source freshness SLA                   │
│    Auto-switch on provider degradation        │
└─────────────────────────────────────────────┘
```

---

## Execution Architecture (Post P02)

```
Order Decision (fund_manager agent)
       │
       ▼
┌─────────────────────────────────────┐
│  PRE-TRADE ANALYSIS                 │
│  - Liquidity model (ADV + spread +  │
│    depth + time-of-day)             │
│  - Cost estimate (calibrated from   │
│    TCA EWMA, not hardcoded)         │
│  - Conformal interval → size adj    │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  RISK GATE (IMMUTABLE)              │
│  - Daily loss limit                 │
│  - Position concentration           │
│  - Gross exposure cap               │
│  - Stop-loss mandatory              │
│  - Greeks limits (delta/gamma/vega) │
│  - Intraday circuit breaker         │
│  - Liquidity floor                  │
│  - TradingWindow validation         │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  EXECUTION ENGINE                   │
│  - TWAP: N child orders / T mins    │
│  - VWAP: volume-weighted schedule   │
│  - Participation: max 5% ADV/slice  │
│  - Smart routing (if multi-venue)   │
│  - Bracket orders (stop + target)   │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  POST-TRADE ANALYSIS                │
│  - TCA: realized vs estimated       │
│  - EWMA cost model update           │
│  - Outcome tracking → learning      │
│  - Best execution audit trail       │
└─────────────────────────────────────┘
```

---

## Multi-Asset Framework (Post P12)

```python
class AssetClass(Enum):
    EQUITY = "equity"
    EQUITY_OPTION = "equity_option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    FIXED_INCOME_ETF = "fi_etf"

# Each asset class registers:
# - Collectors (signal sources)
# - Risk parameters (margin, leverage limits)
# - Execution venue + algo
# - Data provider
# - Trading hours
```

**Venue mapping:**
| Asset Class | Venue | Data Provider | Hours |
|------------|-------|--------------|-------|
| Equity | Alpaca | Alpha Vantage + Polygon | 9:30-16:00 ET |
| Options | Alpaca/IBKR | Alpha Vantage + CBOE | 9:30-16:00 ET |
| Futures | IBKR (CME) | CME DataMine | Near 24/5 |
| Forex | OANDA | OANDA API | 24/5 |
| Crypto | Binance/Coinbase | Exchange WebSocket | 24/7 |
| FI ETFs | Alpaca | Alpha Vantage | 9:30-16:00 ET |

---

## Options Market-Making Architecture (Post P08)

```
Vol Surface Builder
  │
  ├─→ Vol Arbitrage Engine (realized vs implied)
  │     └─→ Delta-neutral straddle/strangle
  │
  ├─→ Dispersion Engine (index vs components)
  │     └─→ Sell index vol, buy component vol
  │
  ├─→ Gamma Scalping Engine
  │     └─→ Long options + trade underlying
  │
  └─→ Hedging Engine
        ├─→ Dynamic delta hedge (continuous)
        ├─→ Deep hedging (learned policy, P14)
        └─→ Pin risk management (near expiry)
```

---

## Learning & Feedback Loops (Post P00 + P05 + P10)

```
 ┌──────────────────────────────────────────────────────────┐
 │                    5 CLOSED LOOPS                         │
 │                                                          │
 │  Loop 1: IC → Signal Weights                            │
 │    IC drops below threshold → weight decreases → fewer   │
 │    trades from that signal → weight recovers or signal   │
 │    gets retired                                          │
 │                                                          │
 │  Loop 2: Realized Cost → Cost Model                     │
 │    Actual slippage logged → EWMA updates estimate →      │
 │    backtest uses realistic costs → better sizing          │
 │                                                          │
 │  Loop 3: Trade Loss → Research Priority                  │
 │    Losing trade → OutcomeTracker logs → research graph   │
 │    prioritizes investigation → hypothesis generated →     │
 │    strategy updated or retired                            │
 │                                                          │
 │  Loop 4: Live vs Backtest → Strategy Demotion            │
 │    Live Sharpe < 0.5 × Backtest Sharpe for 20+ trades → │
 │    StrategyBreaker demotes to forward_testing →           │
 │    research graph investigates divergence                 │
 │                                                          │
 │  Loop 5: Agent Quality → Prompt Improvement              │
 │    SkillTracker scores agents → OPRO generates prompt    │
 │    variants → A/B test on paper trades → promote winner  │
 └──────────────────────────────────────────────────────────┘
```

---

## 24/7 Operations Model (Post P15)

| Time Window | Trading Graph | Research Graph | Supervisor |
|-------------|--------------|----------------|------------|
| **Market Hours** (9:30-16:00 ET) | Full cycle: scan → enter → monitor → exit | Lightweight: quick hypothesis checks | Health + risk monitoring |
| **Extended Hours** (16:00-20:00, 04:00-09:30) | Position monitoring only | Earnings processing, overnight signals | EOD reconciliation, data sync |
| **Overnight/Weekend** | Dormant (or crypto if P12) | FULL COMPUTE: ML training, RL training, hypothesis generation, walk-forward | Community intel, strategy lifecycle, backups |

---

## Observability (Enhanced)

| Layer | Current | Target |
|-------|---------|--------|
| Tracing | LangFuse (every node/LLM call) | + IC attribution traces, cost model traces |
| Metrics | Prometheus | + Greeks dashboard, TCA metrics, agent quality scores |
| Logs | Loki/Grafana | + structured trade decision logs with reasoning chains |
| Alerting | Kill switch only | + Discord/email for: drawdown, agent failures, data staleness, Greeks breach, strategy demotion |
| Reports | None | Weekly automated: Sharpe, alpha decomposition, research velocity, strategy lifecycle |

---

## Technology Stack Additions

| Capability | Package | Version | Why |
|-----------|---------|---------|-----|
| Hyperparameter optimization | Optuna | 4.2+ | Bayesian search, LightGBM/XGBoost native |
| Portfolio optimization | Riskfolio-Lib | 6.4+ | HRP, risk parity, CVaR — 30+ methods |
| Options pricing | QuantLib-Python | 1.35+ | Vol surface fitting, exotic pricing, Greeks |
| Causal inference | DoWhy + CausalML + EconML | Latest | Causal graphs, treatment effects, double ML |
| Conformal prediction | MAPIE | 0.9+ | Calibrated prediction intervals |
| Transformer forecasting | NeuralForecast | 1.8+ | PatchTST, iTransformer, TFT |
| GNN | torch_geometric | Latest | Market structure graph modeling |
| RL framework | FinRL (existing) | Latest | Portfolio optimization, execution, strategy selection |
