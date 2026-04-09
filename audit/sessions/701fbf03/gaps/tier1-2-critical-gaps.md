# Tier 1-2 Gaps: Existential & Table Stakes

**Date:** 2026-04-07
**Baseline:** Post-CTO audit (169 findings implemented)

---

## Tier 1: EXISTENTIAL — System Can't Be Trusted Without These

These gaps mean the system is making decisions on unvalidated foundations. No real fund operates this way.

### G1: Zero Signal Validation (→ P01)

**Current:** 27 signal collectors produce scores. No collector has been tested against forward returns. IC (Information Coefficient) = unknown for every signal.

**Impact:** The system may be trading on noise. Without IC tracking, there's no way to know if any signal has predictive power.

**What "fixed" looks like:**
- IC computed daily for every collector against 1d/5d/20d forward returns
- Confidence intervals via bootstrap (reject signals where CI includes 0)
- Signal decay curves — half-life per collector
- Auto-demote signals with IC < threshold for N consecutive days

**Ref:** CTO audit QS-S1, QS-S2, QS-S3

---

### G2: Ghost Learning Modules — Built but Never Called (→ P00)

**Current:** 5 fully-implemented learning modules exist:
1. `OutcomeTracker` — records trade outcomes (BUILT, 0 callers)
2. `SkillTracker` — tracks agent skill per strategy (BUILT, 0 callers)
3. `ICAttribution` — attributes alpha to signals (BUILT, 0 callers)
4. `ExpectancyEngine` — computes per-strategy expectancy (BUILT, 0 callers)
5. `StrategyBreaker` — detects regime-broken strategies (BUILT, 0 callers)

**Impact:** The system records losses but behavior never changes. This is the #1 gap separating QuantStack from a real fund. A real fund would demote a losing strategy within days.

**What "fixed" looks like:**
- Wire 1: OutcomeTracker → strategy selection (prefer strategies with positive expectancy)
- Wire 2: OutcomeTracker → position sizing (scale with realized edge)
- Wire 3: StrategyBreaker → trade execution (auto-halt broken strategies)
- Wire 4: SkillTracker → agent confidence weighting
- Wire 5: ICAttribution → signal weight adjustment
- Wire 6: TradeEvaluator → agent prompt updates

**Ref:** CTO audit DO-1 through DO-5

---

### G3: 91 Tool Stubs — Functionality Missing (→ P03, P09)

**Current:** ACTIVE/PLANNED split prevents agents from calling stubs (fixed). But the functionality those 91 tools represent is still missing: ML training, RL environments, TCA, walk-forward validation, Monte Carlo simulation.

**Impact:** The system has the skeleton of an ML-driven quant fund but the muscle (actual ML, RL, statistical validation) is stub code.

**What "fixed" looks like:** Implement the 5 highest-priority ML tools (P03), 11 RL tools (P09), and 6 options tools (P06) — the rest can remain planned until needed.

**Ref:** CTO audit TC1

---

### G4: Five Broken Feedback Loops (→ P00, P01, P02, P05)

**Current:** None of these loops are closed:
1. **IC → signal weight:** Signal weights are static/regime-based, not IC-driven
2. **Realized cost → cost model:** Pre-trade cost estimates never calibrate from actual fills
3. **Trade loss → research priority:** Losses don't trigger focused research on what went wrong
4. **Live vs backtest divergence → strategy demotion:** No automatic demotion when live underperforms backtest
5. **Agent quality → prompt improvement:** Agent mistakes don't improve future prompts

**Impact:** The system doesn't learn from experience. Every trading day is day 1.

**What "fixed" looks like:** Each loop has a measurable behavior change (e.g., signal weight decreases when IC drops; cost model updates EWMA; losing strategy gets demoted).

---

## Tier 2: TABLE STAKES — Any Real Fund Has These

These gaps are "expected features" at any institutional fund. Their absence limits the system to hobby-grade trading.

### G5: Phantom TWAP/VWAP Execution (→ P02)

**Current:** The system selects TWAP or VWAP as execution algorithms, but both execute as single market fills. No child order slicing, no time scheduling, no participation rate.

**Impact:** Market impact on any position > $5K is uncontrolled. For options with wide spreads, this is especially costly.

**What "fixed" looks like:**
- TWAP: N child orders over T minutes, random jitter
- VWAP: Volume-weighted scheduling using historical intraday volume profile
- Participation rate: max 5% of ADV per interval
- Fallback to market order if fills stall

---

### G6: No Greeks in Risk Gate (→ P02, P06)

**Current:** Options risk checks: DTE minimum + premium cap. No delta, gamma, vega, or theta limits at portfolio level.

**Impact:** A portfolio could accumulate unlimited gamma risk or vega exposure. Any vol event would produce outsized losses with no circuit breaker.

**What "fixed" looks like:**
- Portfolio-level limits: |net delta| < 100, |gamma| < 50, |vega| < $5K, |theta| > -$500/day
- Per-position Greeks computed at entry and monitored continuously
- Auto-hedge triggers when limits approach (80% threshold)

---

### G7: No TCA Feedback (→ P02)

**Current:** Pre-trade cost estimates (slippage, impact) are static constants. Realized execution costs are logged but never fed back to calibrate estimates.

**Impact:** Position sizing and execution strategy are based on incorrect cost assumptions. If actual slippage is 3x the estimate, every backtest is too optimistic.

**What "fixed" looks like:**
- EWMA of realized vs estimated slippage per symbol
- Cost model auto-calibrates every 20 fills
- Backtest cost inputs updated from realized data (not hardcoded 10 bps)

---

### G8: No Intraday Circuit Breaker (→ P02)

**Current:** Daily loss limit exists (DB-level). No unrealized P&L monitoring within the day.

**Impact:** A flash crash could produce catastrophic losses before the daily loss limit triggers on realized P&L.

**What "fixed" looks like:**
- Mark-to-market unrealized P&L computed every 60 seconds
- Triggers: -2% portfolio → reduce exposure, -3% → close new entries, -5% → liquidate
- Kill switch auto-engages on cascading failures

---

### G9: No Liquidity Model (→ P02)

**Current:** Only ADV (average daily volume) check. No bid-ask spread modeling, order book depth, or time-of-day liquidity variation.

**Impact:** System may enter illiquid names or trade at high-spread times, paying excessive execution costs.

**What "fixed" looks like:**
- Liquidity score combining ADV, spread, depth
- Time-of-day adjustment (avoid first/last 15 min for illiquid names)
- Position size capped at max(2% ADV, $50K) per order

---

### G10: No Model Versioning or A/B Testing (→ P03)

**Current:** ML models overwrite previous versions. No rollback, no A/B comparison, no champion/challenger framework.

**Impact:** Can't tell if a new model is better or worse. No way to safely deploy experimental models.

**What "fixed" looks like:**
- Model registry table: model_id, version, metrics, created_at, status (champion/challenger/retired)
- A/B testing: challenger gets 20% of signal weight, promote after N trades with better Sharpe
- Rollback: one-click revert to previous champion

---

### G11: Hardcoded ML Hyperparameters (→ P03)

**Current:** LightGBM/XGBoost use default or manually-set hyperparameters. No systematic search.

**Impact:** Models are likely significantly underperforming their potential. Hyperparameter optimization typically yields 10-30% metric improvement.

**What "fixed" looks like:**
- Optuna integration with purged CV objective
- Automated sweep on schedule (weekly after-hours)
- Best params stored in model registry with full trial history
