# 02 — Statistical Validity: Know If The Signals Work

**Priority:** P1 — Must Fix Before Trusting Any Backtest
**Timeline:** Week 2-4
**Gate:** Daily IC computed for all collectors. Confidence intervals on conviction. Walk-forward mandatory.

---

## Why This Section Is Second

You can't make money if the signals don't predict returns. You can't know if the signals predict returns without statistical validation. The system currently trades on unvalidated signals — the quant equivalent of selling drugs without clinical trials. Everything in Section 01 prevents catastrophic loss. This section determines whether the system has any right to trade at all.

---

## 2.1 Signal IC Computation and Tracking

**Finding ID:** QS-S1
**Severity:** CRITICAL
**Effort:** 3-4 days

### The Problem

The `signal_ic` database table exists. The `compute_information_coefficient()` tool is defined. But it returns `{"error": "Tool pending implementation"}`. No signal has ever been validated against forward returns.

The system has 16 concurrent signal collectors producing conviction scores. None of these scores have been validated as predictive. A random number generator would produce conviction scores too — the question is whether these are better than random.

### What Must Exist

| Metric | Definition | Frequency | Threshold |
|--------|-----------|-----------|-----------|
| Daily IC | Rank correlation of signal value vs. 1/5/21-day forward returns | Daily | IC > 0.02 to stay active |
| IC Stability | Standard deviation of daily IC over rolling 63-day window | Weekly | std(IC) < IC mean |
| IC Decay Curve | IC at lag 0, 1, 5, 21 days — how fast does alpha decay? | Weekly | Half-life documented |
| IC by Regime | Signal performance segmented by trending/ranging/unknown | Weekly | IC > 0.01 in each regime |
| Statistical Significance | t-statistic of IC | Daily | t > 2.0 to remain active |
| IC Attribution | Per-collector contribution to final conviction quality | Daily | Identifies best/worst collectors |

### The Fix

| Step | Action | Location |
|------|--------|----------|
| 1 | Implement `ICTracker` module | New `src/quantstack/learning/ic_tracker.py` |
| 2 | Compute IC daily for all 16 collectors | Scheduled job (add to supervisor) |
| 3 | Store in `signal_ic` table (already exists, empty) | `db.py` |
| 4 | Add gate: `if rolling_63d_IC < 0.02: disable collector from synthesis` | `signal_engine/engine.py` |
| 5 | Alert if IC negative for >5 consecutive days | Supervisor health check |
| 6 | Wire `ic_attribution.record()` into signal engine after synthesis | `signal_engine/engine.py` |

### Acceptance Criteria

- [ ] Daily IC values for all 16 collectors stored in `signal_ic` table
- [ ] Collectors with IC < 0.02 for 63 days automatically disabled
- [ ] IC dashboard or query available for human review
- [ ] `ICAttribution` module (currently ghost — zero callers) wired and producing data

---

## 2.2 Signal Confidence Intervals

**Finding ID:** QS-S2
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Signals output point estimates: `consensus_conviction = 0.75`. No confidence bounds. The system can't distinguish between:
- "0.75 conviction from 12 agreeing collectors" (high confidence)
- "0.75 conviction from 1 noisy collector" (low confidence)

Position sizing treats both identically. Kelly criterion requires probability estimates with uncertainty — a point estimate is not sufficient.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add `uncertainty_estimate` field to `SignalBrief` |
| 2 | Compute from collector agreement distribution (bootstrap) |
| 3 | Propagate to position sizing: `size = base_size * (1 - confidence_width)` |
| 4 | When confidence interval spans zero (e.g., [−0.1, 0.6]): skip trade entirely |

### Acceptance Criteria

- [ ] Every `SignalBrief` includes uncertainty bounds
- [ ] Position sizing scales inversely with uncertainty width
- [ ] Signals with wide uncertainty intervals produce smaller positions or no trade

---

## 2.3 Signal Decay Modeling

**Finding ID:** QS-S3
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Signal cache TTL is 60 minutes. A 59-minute-old signal has identical weight to a 1-minute-old signal. In fast-moving markets, a 59-minute-old RSI signal is essentially noise.

### The Fix

Implement exponential decay per collector type:

| Collector Type | Half-Life | Rationale |
|---------------|-----------|-----------|
| Technical (RSI, MACD, volume) | 15 minutes | Fast-moving, price-dependent |
| ML signal | 30 minutes | Model predictions stale quickly |
| Options flow (GEX, IV) | 60 minutes | Options data updates less frequently |
| Sentiment | 4 hours | News sentiment persists |
| Fundamentals | 24 hours | Quarterly data, slow-changing |
| Macro indicators | 7 days | GDP, CPI — monthly at best |

Formula: `effective_conviction = conviction * exp(-age_minutes / half_life)`

Calibrate half-lives from IC decay curves (once IC tracking is live from 2.1).

### Acceptance Criteria

- [ ] Each collector has a configured half-life
- [ ] Signal age reduces effective conviction via exponential decay
- [ ] Stale signals (>2x half-life) produce negligible conviction contribution

---

## 2.4 Look-Ahead Bias Detection

**Finding ID:** QS-S4
**Severity:** CRITICAL
**Effort:** 2-3 days

### The Problem

`check_lookahead_bias()` returns `{"error": "Tool pending implementation"}`. No automated check that features at signal time don't include future data:

- Earnings from AV: may include data announced at 16:05 ET for a signal computed at 16:00 ET
- Options flow: uses live delta/gamma, but IC may be computed against 5-day forward returns — features overlap the prediction window
- Fundamentals refreshed nightly but signals fire intraday

If look-ahead bias exists, backtest IC is inflated by 0.10+. Strategies that appear profitable are actually unprofitable live.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add `FeatureTimestamp` metadata to every feature: `(value, as_of_timestamp, known_since_timestamp)` |
| 2 | Validate: `known_since_timestamp < signal_time < forward_return_start` |
| 3 | Flag any violation as `LOOKAHEAD_BIAS_DETECTED` |
| 4 | Implement point-in-time data semantics for fundamentals: Q3 data available 2025-01-25 not 2024-09-30 |

### Acceptance Criteria

- [ ] All features have `(as_of_date, available_date)` metadata
- [ ] Automated validation rejects any backtest where `available_date >= signal_time`
- [ ] Fundamentals data correctly timestamped with availability date, not reporting period

---

## 2.5 Conviction-Scaled Position Sizing

**Finding ID:** QS-S6
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

Signals produce conviction [0.05, 0.95]. But position sizing uses ATR-based stop distance and equity fraction — not signal strength. A 0.95 conviction signal and a 0.10 conviction signal get the same position size if they have the same stop distance. This is alpha-destroying.

### The Fix

Add conviction scaling to `ATRPositionSizer`:

```
position_size = base_size * min(conviction / conviction_threshold, 2.0)
```

| Conviction Range | Sizing Multiplier | Rationale |
|-----------------|-------------------|-----------|
| < 0.30 | 0.5x base | Low confidence — small bet |
| 0.30 - 0.60 | 1.0x base | Normal confidence |
| 0.60 - 0.80 | 1.25x base | High confidence — lean in |
| > 0.80 | 1.5x base (capped by risk gate) | Very high confidence |

### Acceptance Criteria

- [ ] Position size scales with signal conviction
- [ ] Low-conviction signals produce materially smaller positions
- [ ] Risk gate position caps still enforced (conviction scaling cannot exceed gate limits)

---

## 2.6 Walk-Forward Validation Gate

**Finding ID:** QS-B4
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Walk-forward framework exists in `core/research/walkforward.py` but the tool wrapper `run_walkforward()` is stubbed. Strategies can advance from `draft` → `backtested` without out-of-sample testing. No OOS Sharpe gate exists.

### The Fix

| Step | Action |
|------|--------|
| 1 | Implement `run_walkforward()` tool — wire to existing `walkforward.py` |
| 2 | Make WFV mandatory before any strategy advances past `backtested` |
| 3 | Gate: OOS Sharpe must be >= 50% of in-sample Sharpe |
| 4 | Log IS/OOS ratio for ongoing monitoring of overfitting tendency |

### Acceptance Criteria

- [ ] No strategy can reach `forward_testing` without passing walk-forward validation
- [ ] OOS Sharpe ratio >= 50% of IS Sharpe required
- [ ] IS/OOS ratio logged and queryable per strategy

---

## 2.7 Survivorship Bias Adjustment

**Finding ID:** QS-B2
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Backtests use current universe constituents. Companies that went bankrupt (e.g., Bed Bath & Beyond 2023) are excluded, creating a positive bias of 2-5% annual returns. `delisted_at` column exists but there's no evidence it's used to filter the backtest universe point-in-time.

### The Fix

| Step | Action |
|------|--------|
| 1 | Implement `universe_as_of(date)` function — returns only symbols active at that date |
| 2 | Filter: not delisted before date, not IPO'd after date |
| 3 | Use in all backtest calls |
| 4 | Populate `delisted_at` for known delistings |

### Acceptance Criteria

- [ ] `universe_as_of(date)` function exists and is called by all backtests
- [ ] Backtests on 2020-2024 period include symbols that were later delisted
- [ ] Strategy Sharpe ratios re-validated with survivorship-adjusted universe

---

## 2.8 Realistic Transaction Costs

**Finding ID:** QS-B1
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

Default backtest transaction cost is 10 bps (commissions only). Missing: bid-ask spread (1-50 bps), market impact (2-20 bps), opportunity cost. A strategy with 50 bps gross alpha and 10 bps modeled cost looks like 40 bps net. Realistic cost of 30-40 bps means actual net alpha is 10-20 bps — possibly noise.

### The Fix

| Asset Type | Default All-In Cost | Components |
|-----------|-------------------|------------|
| Large-cap equity | 30 bps | 5 bps commission + 5 bps spread + 10 bps impact + 10 bps opp cost |
| Small-cap equity | 60 bps | 5 bps commission + 20 bps spread + 25 bps impact + 10 bps opp cost |
| Options | 5% of bid-ask spread | Commission + spread (highly variable) |

Use 30 bps as default all-in cost until TCA feedback loop provides realized estimates.

### Acceptance Criteria

- [ ] Default transaction cost raised from 10 bps to 30 bps for equities
- [ ] All existing strategy Sharpe ratios recalculated with realistic costs
- [ ] Strategies that become unprofitable at realistic costs flagged for review

---

## 2.9 Hyperparameter Optimization

**Finding ID:** QS-M1
**Severity:** CRITICAL
**Effort:** 2-3 days

### The Problem

ML model hyperparameters are hardcoded: `learning_rate=0.05, max_depth=6, n_estimators=500`. No grid search, random search, or Bayesian optimization. Default parameters leave 10-20% of achievable performance on the table.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add `optuna` integration to `ml/trainer.py` |
| 2 | Run 100-trial Bayesian optimization with purged cross-validation on each retraining |
| 3 | Cache optimal parameters per symbol/horizon in `model_hyperparams` table |
| 4 | Compare IC: optimized vs. default parameters — log improvement |

### Acceptance Criteria

- [ ] Every model retraining runs hyperparameter optimization
- [ ] Optimal parameters stored per symbol/horizon
- [ ] IC improvement from optimization measured and logged

---

## 2.10 Feature Multicollinearity Audit

**Finding ID:** QS-B6
**Severity:** HIGH
**Effort:** 2 days

### The Problem

150+ features with no VIF analysis, no correlation matrix, no dimensionality reduction. Technical indicators (RSI, MACD, Stochastic, Williams %R) are 0.7+ correlated — they all measure momentum. Including all of them makes the model think it has 150 independent features when effective rank is ~30. This causes massive overfitting.

### The Fix

| Step | Action |
|------|--------|
| 1 | Compute pairwise feature correlation matrix weekly |
| 2 | Remove features with VIF > 10 |
| 3 | Report effective feature count (eigenvalues > 0.1 of correlation matrix) |
| 4 | Consider PCA or autoencoders for dimensionality reduction before model training |

### Acceptance Criteria

- [ ] Weekly feature correlation audit runs automatically
- [ ] Features with VIF > 10 flagged and removed from model training
- [ ] Effective feature count reported (expect ~30 from 150+)

---

## 2.11 Monte Carlo Validation

**Finding ID:** QS-B3
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

`run_monte_carlo()` returns `{"error": "pending implementation"}`. Without Monte Carlo, a backtest Sharpe of 0.8 could have 95% CI of [0.2, 1.4] — essentially noise. Can't quantify overfitting risk, parameter sensitivity, or probability of ruin.

### The Fix

Implement bootstrap Monte Carlo: resample daily returns with replacement, compute Sharpe distribution. Reject strategies where lower 5th percentile Sharpe < 0.3.

### Acceptance Criteria

- [ ] Monte Carlo simulation produces confidence intervals on backtest Sharpe
- [ ] Strategies with 5th percentile Sharpe < 0.3 automatically rejected
- [ ] Parameter sensitivity analysis available for strategy validation

---

## 2.12 Point-in-Time Data Semantics

**Finding ID:** QS-B5
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Features used in backtests don't have explicit "as-of-date" and "known-since-date" fields. Example: Q3 2024 earnings released 2025-01-25 but labeled as 2024-09-30. If a signal fires on 2025-01-24, it shouldn't use Q3 data — but it does because there's no PIT enforcement. For fundamentals data, this inflates backtest IC by 0.05-0.20.

### The Fix

Add `(value, as_of_date, available_date)` triple to all fundamental features. Filter: `available_date < signal_timestamp`.

### Acceptance Criteria

- [ ] All fundamental features have `(as_of_date, available_date)` metadata
- [ ] Backtests filter features by availability date, not reporting period
- [ ] PIT violation logged and flagged during backtest runs

---

## 2.13 Feature Importance: Consensus Across Methods

**Finding ID:** QS-M4
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Three importance methods mentioned (MDI, MDA, SFI) but only MDI (built-in `feature_importances_`) actually runs. MDI is biased toward high-cardinality features. SHAP explainer in `explainer.py:82-124` is "best-effort" — falls back to MDI on failure.

### The Fix

Implement all three methods. Use consensus: feature is "important" only if ranked top-20 by >= 2/3 methods. Discard features important by only one method (likely noise).

### Acceptance Criteria

- [ ] MDI, MDA, and SFI all implemented and running
- [ ] Feature importance requires consensus (2/3 methods agree)
- [ ] Single-method-only importance flagged as unreliable

---

## 2.14 Cross-Validation Must Account for Regimes

**Finding ID:** QS-M5
**Severity:** HIGH
**Effort:** 1 day

### The Problem

Purged K-fold CV exists (good for leakage prevention). But: fixed `test_size=0.2` regardless of holding period, no stratification on regime label, no expanding-window CV. Model tested on trending data but not on ranging data in same fold — achieves 0.04 IC in CV (mixed regimes) but 0.01 IC in live (current ranging regime).

### The Fix

Add regime-stratified CV: ensure each fold contains proportional representation of all regime types. Add expanding-window CV to simulate production learning.

### Acceptance Criteria

- [ ] Each CV fold contains proportional representation of all regime types
- [ ] Expanding-window CV available as alternative to fixed K-fold
- [ ] Per-regime IC reported for each model

---

## Summary: Week 2-4 Delivery Checklist

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 2.1 | Signal IC computation | 3-4 days | Know if signals predict returns |
| 2.2 | Signal confidence intervals | 2 days | Position sizing reflects certainty |
| 2.3 | Signal decay modeling | 2 days | Stale signals produce less conviction |
| 2.4 | Look-ahead bias detection | 2-3 days | Backtests trustworthy |
| 2.5 | Conviction-scaled sizing | 1 day | Bet big on high confidence, small on low |
| 2.6 | Walk-forward validation gate | 2 days | No strategy promoted without OOS test |
| 2.7 | Survivorship bias adjustment | 2 days | Backtests reflect reality |
| 2.8 | Realistic transaction costs | 1 day | Net alpha estimates accurate |
| 2.9 | Hyperparameter optimization | 2-3 days | 15-30% IC improvement |
| 2.10 | Feature multicollinearity audit | 2 days | Reduce overfitting |
| 2.11 | Monte Carlo validation | 2 days | Confidence intervals on Sharpe |

| 2.12 | Point-in-time data semantics | 2 days | Fundamentals not backdated |
| 2.13 | Feature importance consensus | 2 days | Reliable feature selection |
| 2.14 | Regime-stratified CV | 1 day | Model tested across all regimes |

**Total estimated effort: 25-29 engineering days (overlapping with Section 01).**
**After this section:** you can answer "do our signals predict returns?" with data, not hope.
