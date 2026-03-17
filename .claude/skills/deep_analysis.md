---
name: deep_analysis
description: QuantCore tool usage guide — when and how to enrich decisions with raw data, options intelligence, and risk checks beyond the DailyBrief.
user_invocable: false
---

# Deep Analysis — QuantCore Tool Integration Guide

This skill is a reference for **when to call which QuantCore MCP tools** during
trading sessions.  It is consumed by `/trade` and `/meta`; do not invoke it directly.

---

## Pre-Trade Intelligence (run before every /trade and /meta session)

### Economic Calendar Check
**Tool:** `mcp__quantcore__get_event_calendar(symbol, days_ahead=2)`
**When:** Before any session.
**Decision rule:**
- FOMC or CPI within 2 hours → reduce all position sizes 50% or skip
- Earnings today/tomorrow → flag in trade plan, options only (defined risk)
- No major events → proceed normally

### Market Regime Snapshot
**Tool:** `mcp__quantcore__get_market_regime_snapshot(end_date)`
**When:** Opening of any session; supplement `get_regime` with broad market context.
**What to look for:** VIX regime, breadth (% above 200-day MA), market phase.

---

## Signal Enrichment (run when DailyBrief confidence > 0.65)

### Raw Price Action
**Tool:** `mcp__quantcore__load_market_data(symbol, timeframe, start_date, end_date)`
**When:** DailyBrief flags critical levels — verify the levels are real in the data.
**What to look for:** Did price actually bounce at support? Is the resistance clean?

### Volume Profile
**Tool:** `mcp__quantcore__analyze_volume_profile(symbol, timeframe, lookback_days=20)`
**When:** Before entering at a support or resistance level.
**Decision rule:**
- High volume node (HVN) at entry level → strong support/resistance, higher conviction
- Low volume node (LVN) at entry level → price may slice through quickly, tighten stop

### Multi-Timeframe Alignment
**Tool:** `mcp__quantcore__compute_technical_indicators(symbol, timeframe, indicators)`
**When:** When 1-day signal looks good but context is unclear.
**Timeframes to check:** weekly (trend direction), daily (entry timing), 4h (entry precision)
**Rule:** Enter only if weekly and daily agree on direction.

### Relative Strength vs Sector
**Tool:** `mcp__quantcore__run_screener(symbols, trend_filter, ...)`
**When:** For sector-specific trades.
**Decision rule:** Entry symbol should be outperforming its sector peers.

---

## Options Intelligence (run when volatility_ic flags elevated IV or options signal)

### IV Rank and Skew

**For strategy design and backtesting (synthetic chain):**
**Tool:** `mcp__quantcore__compute_option_chain(symbol, expiry_date)`
**When:** `/workshop` sessions — evaluating whether a signal has options convexity, running `run_backtest_options`.
**Important:** This chain is **synthetic** (hardcoded 25% IV, fake bid/ask spreads). Do NOT use for live /options execution.
**What to extract:** Theoretical IV percentile, skew shape, options Sharpe vs equity Sharpe comparison.

**For live /options execution (live broker chain):**
**Tool:** `get_options_chain(symbol, expiry_min_days=7, expiry_max_days=45)`
**When:** `/options` sessions — selecting strikes and structures for real trades.
**What to extract:** Real bid/ask, live IV per contract, open interest, volume.

**IV surface (live):**
**Tool:** `get_iv_surface(symbol)`
**When:** Before any `/options` entry decision.
**What to extract:** `iv_rank` (0–100), `iv_percentile`, `skew_25d`, `atm_iv_30d`, `atm_iv_60d`.
Use this — not `compute_option_chain` — for the IV rank decision matrix in `/options`.

### Put/Call Ratio
**Tools:** `mcp__quantcore__analyze_option_structure` (or options_flow_ic output)
**Decision rule:**
- P/C ratio > 1.2 → bearish sentiment (potential contrarian long setup)
- P/C ratio < 0.7 → complacent / bullish (fade if overextended)

### Trade Template
**Tool:** `mcp__quantcore__generate_trade_template(symbol, direction, structure_type, expiry_days, risk_amount)`
**When:** Executing an options position.
**Follow with:** `mcp__quantcore__validate_trade` and `mcp__quantcore__score_trade_structure`

### Options Convexity Assessment
**When:** Evaluating whether an equity strategy's edge is better captured via options.
**Decision rule:**
- Options Sharpe > 2× equity Sharpe → deploy as options strategy
- IV crush > 60% of trades → avoid options, stick with equity
- Set IV rank max to 0.5 to filter entries when vol is elevated (reduces premium cost)

---

## Risk Enrichment (run before any position > 3% allocation)

### Portfolio Beta Impact
**Tool:** `mcp__quantcore__compute_portfolio_stats(equity_curve)`
**When:** Before any position sized at "half" or "full".

### VaR Check
**Tool:** `mcp__quantcore__compute_var(returns, confidence_levels, method)`
**When:** When gross_exposure is already > 80%.
**Decision rule:** If adding this position pushes 99% VaR beyond 3% of equity, reduce size.

### Stress Test
**Tool:** `mcp__quantcore__stress_test_portfolio(positions, scenarios)`
**When:** Earnings season or macro events on the calendar.
**Scenarios:** Include 2008, COVID, Volmageddon as minimum.
**Decision rule:** If any scenario shows > 10% portfolio loss, reduce gross exposure.

### Liquidity Check
**Tool:** `mcp__quantcore__analyze_liquidity(symbol, timeframe, window)`
**When:** For any symbol outside the S&P 500.
**Decision rule:** If bid-ask spread estimate > 10 bps, factor into expected return.

---

## Post-Trade Monitoring (run during /review)

### Position Status
**Tool:** `mcp__quantpod__get_position_monitor(symbol)` ← NEW E5 tool
**When:** Every /review session for each open position.
**Flags to act on:**
- `near_stop=True` → tighten stop or close
- `near_target=True` → consider partial exit
- `days_held` > strategy max holding period → time stop, consider close

### Fill Quality Audit
**Tool:** `mcp__quantpod__get_fill_quality(order_id)` ← NEW E5 tool
**When:** /reflect sessions — audit last 20 fills.
**Metrics to track:**
- Average slippage (target: < 5 bps)
- fill_vs_vwap_bps (target: < 3 bps absolute)
- Worst 3 fills: what time of day, what size, what symbol?
**Action:** If avg slippage > 5 bps, check:
  1. Are we trading illiquid names? (analyze_liquidity)
  2. Are we sizing too large vs ADV? (check_risk_limits)
  3. Are we trading at market open/close? (time pattern analysis)

---

## ML Signal Enrichment (use in /workshop when rule-based Sharpe < 0.5)

### Causal Feature Filtering (when # features > 30)
**How:** `from quantcore.validation.causal_filter import CausalFilter`
**When:** After variance filtering but before model training — drops spurious correlations
**What it adds:** Granger causality tests identify which features actually predict forward
returns; Bonferroni-corrected p-values prevent false discoveries across 200+ features;
optional transfer entropy captures nonlinear causal relationships
**Decision rule:** Features surviving the causal filter have stronger OOS stability than
correlation-filtered features alone. If < 10 features survive, fall back to top
correlations — the signal may be too noisy for causal detection at current sample size
**Diagnostics:** `cf.get_result()` returns per-feature p-values, stationarity status,
and which features were dropped — feed this into /reflect for feature quality analysis

### Feature Importance Check
**How:** Directly import `SHAPExplainer` from `quantcore.models.explainer`
**When:** After a backtest fails — understand which features actually drove signals
**What to look for:** If top SHAP features are lagged price (not indicators), the
signal may be autocorrelation artefact, not a real edge

### Regime Probability (when `get_regime` confidence < 0.6)
**How:** `from quantcore.hierarchy.regime.hmm_model import HMMRegimeModel`
**When:** `get_regime()` returns confidence < 0.6 or regime has changed 3x in 5 days
**What it adds:** State transition probabilities — "40% chance staying in ranging,
35% trending_up, 25% trending_down" is more useful than a single label

### ML-Backed Direction Signal
**How:** `from quantcore.equity.pipeline import run_ml_strategy`
**When:** Rule-based workshop strategies repeatedly fail Sharpe > 1.0 threshold
**What it adds:** GradientBoosting classifier trained on 200+ features with
CausalFilter pre-selection + TimeSeriesSplit CV — provides calibrated probability
of up/down move

### Changepoint Detection
**How:** `from quantcore.hierarchy.regime.changepoint import BayesianChangepointDetector`
**When:** Entering a new trade after a regime that has been stable for 20+ days
**Decision rule:** If changepoint probability > 0.3, reduce position size 50%
— you may be near a regime transition

---

## Conditional Usage Rules

### When to call these tools vs relying on DailyBrief

| Signal strength | Action |
|-----------------|--------|
| DailyBrief conviction >= 0.8, pod_agreement="unanimous" | Trust the brief, do minimal enrichment (just calendar check) |
| DailyBrief conviction 0.65-0.8 | Run signal enrichment (volume profile + multi-TF) |
| DailyBrief conviction 0.50-0.65 | Run full enrichment before trading |
| DailyBrief conviction < 0.50 | Don't trade — no amount of enrichment justifies weak signal |

### Never enrich past the point of diminishing returns
3 confirming signals > 8 confirming signals.  Calling 10 QuantCore tools per
trade is analysis paralysis. Pick 2-3 relevant enrichment tools and move on.

### Cost-awareness
Each QuantCore tool call has latency cost (~50-500ms).  Batch calls where possible
using `compute_all_features` when you need multiple indicator values.
