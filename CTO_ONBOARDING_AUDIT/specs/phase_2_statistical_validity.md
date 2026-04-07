# Phase 2: Statistical Validity — Deep Plan Spec

**Timeline:** Week 2-4
**Effort:** 26-32 days (parallelizable to ~14 days with 2 engineers)
**Gate:** Daily IC computed. Backtests trustworthy. Walk-forward mandatory.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 2 determines **whether the system has any right to trade at all**. The system currently trades on unvalidated signals — the quant equivalent of selling drugs without clinical trials. No signal has ever been validated against forward returns. Backtests use survivorship-biased universes, underapplied transaction costs, and no look-ahead bias protection.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`02_STATISTICAL_VALIDITY.md`](../02_STATISTICAL_VALIDITY.md)
**Supporting sections:** [`03_EXECUTION_LAYER.md`](../03_EXECUTION_LAYER.md) (items 2.12-2.14 in roadmap)

---

## Objective

Build the statistical validation infrastructure to answer: "Do our signals predict returns?" with data, not hope. After this phase, every signal has IC tracking, every backtest is trustworthy, and no strategy advances without out-of-sample validation.

---

## Items

### 2.1 Signal IC Computation + Tracking

- **Finding:** QS-S1 | **Severity:** CRITICAL | **Effort:** 3-4 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.1](../02_STATISTICAL_VALIDITY.md)
- **Problem:** `signal_ic` DB table exists. `compute_information_coefficient()` tool returns `{"error": "Tool pending implementation"}`. No signal ever validated against forward returns. 16 concurrent collectors producing unvalidated conviction scores.
- **Fix:**
  1. Implement `ICTracker` module (new `src/quantstack/learning/ic_tracker.py`)
  2. Compute IC daily for all 16 collectors (scheduled job in supervisor)
  3. Store in `signal_ic` table (already exists, empty)
  4. Add gate: `if rolling_63d_IC < 0.02: disable collector from synthesis`
  5. Alert if IC negative for >5 consecutive days
  6. Wire `ic_attribution.record()` into signal engine after synthesis
- **Required metrics:** Daily IC, IC Stability (std over 63-day window), IC Decay Curve (lag 0/1/5/21 days), IC by Regime, t-statistic, IC Attribution per collector
- **Key files:** New `learning/ic_tracker.py`, `src/quantstack/signal_engine/engine.py`, `src/quantstack/db.py` (signal_ic table)
- **Acceptance criteria:**
  - [ ] Daily IC values for all 16 collectors stored in `signal_ic` table
  - [ ] Collectors with IC < 0.02 for 63 days automatically disabled
  - [ ] IC dashboard or query available for human review
  - [ ] `ICAttribution` module (currently ghost — zero callers) wired and producing data

### 2.2 Signal Confidence Intervals

- **Finding:** QS-S2 | **Severity:** CRITICAL | **Effort:** 2 days
- **Depends on:** 2.1
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.2](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Signals output point estimates (`consensus_conviction = 0.75`). No confidence bounds. Can't distinguish high-agreement (12 collectors) from single-noisy-collector. Position sizing treats both identically.
- **Fix:**
  1. Add `uncertainty_estimate` field to `SignalBrief`
  2. Compute from collector agreement distribution (bootstrap)
  3. Propagate to position sizing: `size = base_size * (1 - confidence_width)`
  4. When confidence interval spans zero: skip trade entirely
- **Key files:** `src/quantstack/signal_engine/models.py` (SignalBrief), position sizer
- **Acceptance criteria:**
  - [ ] Every `SignalBrief` includes uncertainty bounds
  - [ ] Position sizing scales inversely with uncertainty width
  - [ ] Signals with wide uncertainty intervals produce smaller positions or no trade

### 2.3 Signal Decay Modeling

- **Finding:** QS-S3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.3](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Signal cache TTL is 60 minutes. 59-minute-old signal has identical weight to 1-minute-old. In fast-moving markets, 59-minute-old RSI is noise.
- **Fix:** Implement exponential decay per collector type. Formula: `effective_conviction = conviction * exp(-age_minutes / half_life)`. Half-lives: Technical=15min, ML=30min, Options=60min, Sentiment=4h, Fundamentals=24h, Macro=7d. Calibrate from IC decay curves once 2.1 is live.
- **Key files:** Signal engine synthesis logic
- **Acceptance criteria:**
  - [ ] Each collector has a configured half-life
  - [ ] Signal age reduces effective conviction via exponential decay
  - [ ] Stale signals (>2x half-life) produce negligible conviction contribution

### 2.4 Look-Ahead Bias Detection

- **Finding:** QS-S4 | **Severity:** CRITICAL | **Effort:** 2-3 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.4](../02_STATISTICAL_VALIDITY.md)
- **Problem:** `check_lookahead_bias()` returns stub. No automated check that features at signal time don't include future data. Earnings from AV may include data announced at 16:05 ET for 16:00 ET signal. If look-ahead bias exists, backtest IC inflated by 0.10+.
- **Fix:**
  1. Add `FeatureTimestamp` metadata: `(value, as_of_timestamp, known_since_timestamp)`
  2. Validate: `known_since_timestamp < signal_time < forward_return_start`
  3. Flag violations as `LOOKAHEAD_BIAS_DETECTED`
  4. Implement point-in-time data semantics for fundamentals
- **Key files:** Feature engineering pipeline, backtest framework
- **Acceptance criteria:**
  - [ ] All features have `(as_of_date, available_date)` metadata
  - [ ] Automated validation rejects backtests where `available_date >= signal_time`
  - [ ] Fundamentals correctly timestamped with availability date, not reporting period

### 2.5 Conviction-Scaled Position Sizing

- **Finding:** QS-S6 | **Severity:** CRITICAL | **Effort:** 1 day
- **Depends on:** 2.2
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.5](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Position sizing uses ATR-based stop distance and equity fraction — not signal strength. 0.95 and 0.10 conviction get same size if same stop distance.
- **Fix:** Add conviction scaling to `ATRPositionSizer`: <0.30 → 0.5x, 0.30-0.60 → 1.0x, 0.60-0.80 → 1.25x, >0.80 → 1.5x (capped by risk gate).
- **Key files:** `src/quantstack/core/risk/position_sizer.py` (ATRPositionSizer)
- **Acceptance criteria:**
  - [ ] Position size scales with signal conviction
  - [ ] Low-conviction signals produce materially smaller positions
  - [ ] Risk gate position caps still enforced

### 2.6 Walk-Forward Validation Gate

- **Finding:** QS-B4 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.6](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Walk-forward framework exists in `core/research/walkforward.py` but tool wrapper `run_walkforward()` is stubbed. Strategies advance from `draft` → `backtested` without out-of-sample testing.
- **Fix:**
  1. Implement `run_walkforward()` tool — wire to existing `walkforward.py`
  2. Make WFV mandatory before strategy advances past `backtested`
  3. Gate: OOS Sharpe >= 50% of IS Sharpe
  4. Log IS/OOS ratio for overfitting monitoring
- **Key files:** `src/quantstack/core/research/walkforward.py`, walk-forward tool wrapper, strategy lifecycle
- **Acceptance criteria:**
  - [ ] No strategy can reach `forward_testing` without passing walk-forward validation
  - [ ] OOS Sharpe ratio >= 50% of IS Sharpe required
  - [ ] IS/OOS ratio logged and queryable per strategy

### 2.7 Survivorship Bias Adjustment

- **Finding:** QS-B2 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.7](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Backtests use current universe constituents. Bankrupt companies excluded, creating 2-5% annual positive bias. `delisted_at` column exists but not used in backtest universe filtering.
- **Fix:**
  1. Implement `universe_as_of(date)` function
  2. Filter: not delisted before date, not IPO'd after date
  3. Use in all backtest calls
  4. Populate `delisted_at` for known delistings
- **Key files:** Universe management, backtest framework
- **Acceptance criteria:**
  - [ ] `universe_as_of(date)` function exists and is called by all backtests
  - [ ] Backtests on 2020-2024 include symbols later delisted
  - [ ] Strategy Sharpe ratios re-validated with survivorship-adjusted universe

### 2.8 Realistic Transaction Costs (30 bps)

- **Finding:** QS-B1 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.8](../02_STATISTICAL_VALIDITY.md)
- **Problem:** Default backtest cost is 10 bps (commissions only). Missing: spread (1-50 bps), impact (2-20 bps), opportunity cost. Strategy with 50 bps gross and 10 bps modeled cost looks like 40 bps net; realistic 30-40 bps cost means actual net alpha is 10-20 bps.
- **Fix:** Raise default to 30 bps for large-cap equities, 60 bps for small-cap, 5% of bid-ask for options. Recalculate all existing strategy Sharpe ratios.
- **Key files:** Backtest cost model, strategy pipeline
- **Acceptance criteria:**
  - [ ] Default transaction cost raised from 10 bps to 30 bps for equities
  - [ ] All existing strategy Sharpe ratios recalculated with realistic costs
  - [ ] Strategies unprofitable at realistic costs flagged for review

### 2.9 Hyperparameter Optimization (Optuna)

- **Finding:** QS-M1 | **Severity:** CRITICAL | **Effort:** 2-3 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.9](../02_STATISTICAL_VALIDITY.md)
- **Problem:** ML model hyperparameters hardcoded: `learning_rate=0.05, max_depth=6, n_estimators=500`. No grid search, random search, or Bayesian optimization. Default params leave 10-20% of performance on table.
- **Fix:**
  1. Add `optuna` integration to `ml/trainer.py`
  2. Run 100-trial Bayesian optimization with purged cross-validation on each retraining
  3. Cache optimal params per symbol/horizon in `model_hyperparams` table
  4. Compare IC: optimized vs. default — log improvement
- **Key files:** `src/quantstack/ml/trainer.py`, new `model_hyperparams` table
- **Acceptance criteria:**
  - [ ] Every model retraining runs hyperparameter optimization
  - [ ] Optimal parameters stored per symbol/horizon
  - [ ] IC improvement from optimization measured and logged

### 2.10 Feature Multicollinearity Audit

- **Finding:** QS-B6 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.10](../02_STATISTICAL_VALIDITY.md)
- **Problem:** 150+ features with no VIF analysis, no correlation matrix. Technical indicators (RSI, MACD, Stochastic, Williams %R) are 0.7+ correlated. Model thinks 150 independent features; effective rank ~30. Causes massive overfitting.
- **Fix:**
  1. Compute pairwise feature correlation matrix weekly
  2. Remove features with VIF > 10
  3. Report effective feature count (eigenvalues > 0.1)
  4. Consider PCA/autoencoders for dimensionality reduction
- **Key files:** Feature engineering pipeline, ML training pipeline
- **Acceptance criteria:**
  - [ ] Weekly feature correlation audit runs automatically
  - [ ] Features with VIF > 10 flagged and removed from model training
  - [ ] Effective feature count reported (expect ~30 from 150+)

### 2.11 Monte Carlo Validation

- **Finding:** QS-B3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Depends on:** 2.6
- **Audit section:** [`02_STATISTICAL_VALIDITY.md` §2.11](../02_STATISTICAL_VALIDITY.md)
- **Problem:** `run_monte_carlo()` returns stub. Without it, backtest Sharpe of 0.8 could have 95% CI of [0.2, 1.4] — essentially noise. Can't quantify overfitting risk.
- **Fix:** Implement bootstrap Monte Carlo: resample daily returns with replacement, compute Sharpe distribution. Reject strategies where 5th percentile Sharpe < 0.3.
- **Key files:** Backtest framework, strategy validation pipeline
- **Acceptance criteria:**
  - [ ] Monte Carlo produces confidence intervals on backtest Sharpe
  - [ ] Strategies with 5th percentile Sharpe < 0.3 automatically rejected
  - [ ] Parameter sensitivity analysis available

### 2.12 Options Greeks in Risk Gate

- **Finding:** QS-E3 | **Severity:** CRITICAL | **Effort:** 3 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.2](../03_EXECUTION_LAYER.md)
- **Problem:** Risk gate options checks limited to DTE bounds (7-60) and premium at risk (2%/position, 8% total). No delta/gamma/vega/theta limits. 50-contract short straddle gets same treatment as delta call spread. Capability exists in `core/risk/options_risk.py` (444 lines) but risk gate never calls it.
- **Fix:**
  1. Add Greeks aggregation call to `portfolio_state.py`
  2. Wire `options_risk.py` checks into `risk_gate.check()`
  3. Block trades pushing Greeks outside limits (delta ±$50K/1%, gamma $10K/1%², vega $5K/vol point)
  4. Add pin risk detection: DTE < 3 AND near strike → force exit/roll
- **Key files:** `src/quantstack/execution/risk_gate.py`, `src/quantstack/core/risk/options_risk.py`, `src/quantstack/execution/portfolio_state.py`
- **Acceptance criteria:**
  - [ ] Pre-trade Greeks check in risk gate for all options orders
  - [ ] Portfolio-level Greeks aggregated and limit-checked
  - [ ] Short straddle with excessive gamma rejected pre-trade

### 2.13 Intraday Circuit Breaker

- **Finding:** QS-E5 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.4](../03_EXECUTION_LAYER.md)
- **Problem:** Daily loss limit (-2%) only triggers after losses realized. No real-time unrealized P&L monitoring. Can lose 5% in 5 minutes before halt.
- **Fix:** Add `IntraDayCircuitBreaker` to execution monitor: -1.5% → halt entries; -2.5% → systematic exit weakest; -5.0% → emergency liquidation; -1% in 5min velocity → halt regardless.
- **Key files:** `src/quantstack/execution/execution_monitor.py`
- **Acceptance criteria:**
  - [ ] Unrealized P&L checked every tick cycle
  - [ ] Graduated response: halt entries → systematic exit → emergency liquidation
  - [ ] Velocity check catches fast drops even when daily level not breached

### 2.14 Fed Event Enforcement in Risk Gate

- **Finding:** DO-4 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.11](../03_EXECUTION_LAYER.md)
- **Problem:** Events collector outputs `has_fomc_24h`, `has_macro_event`. Agent prompts say "reduce sizing 50% within 24h of FOMC." But this is prompt guidance, not code. No hard rule in `risk_gate.py`.
- **Fix:** Add mandatory sizing multipliers: FOMC within 4h → 0.5x (no naked options); CPI/NFP within 2h → 0.75x; VIX > 30 → 0.7x; VIX > 50 → 0.0x (paper only).
- **Key files:** `src/quantstack/execution/risk_gate.py`
- **Acceptance criteria:**
  - [ ] Macro calendar checked in risk gate, not just agent prompts
  - [ ] FOMC proximity automatically reduces position sizes
  - [ ] VIX > 50 halts all live trading

---

## Dependencies

- **Depends on:** Phase 1 (safety hardening must be in place)
- **2.2 depends on 2.1** (need IC computation before confidence intervals)
- **2.5 depends on 2.2** (need confidence intervals for conviction scaling)
- **2.11 depends on 2.6** (Monte Carlo builds on walk-forward framework)
- **Feeds into:** Phase 7 (feedback loops need IC tracking from 2.1)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 2.1: IC computation may show all signals are noise (IC < 0.02) | This is information, not a failure. Better to know than to trade on noise. Prepare pivot plan. |
| 2.7: Survivorship adjustment may invalidate most existing strategies | Re-run full pipeline with corrected universe. This is a feature, not a bug. |
| 2.8: Realistic costs may make most strategies unprofitable | Expected. Strategies that can't survive realistic costs are not strategies. |
| 2.9: Optuna adds significant compute time to training | Run overnight. Budget 8 hours for full optimization. |

---

## Validation Plan

After this phase, answer these questions with data:
1. Which of the 16 collectors have IC > 0.02? (From 2.1)
2. What is the 95% CI on our best strategy's Sharpe? (From 2.11)
3. How many strategies survive walk-forward validation? (From 2.6)
4. How many strategies survive realistic transaction costs? (From 2.8)
5. What is the effective feature count after VIF filtering? (From 2.10)
