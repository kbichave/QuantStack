# Phase 7: Feedback Loops & Learning — Deep Plan Spec

**Timeline:** Week 6-10
**Effort:** 23-26 days
**Gate:** Losses drive research. IC decay adjusts weights. System learns.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 7 addresses **the single biggest gap all three audits identified**: the system has **five fully-implemented learning modules with zero consumers**. When a trade loses 2%, the loss is recorded in 3 different tables. No downstream system reads them. Position sizing the next morning is identical — as if the loss never happened. This is the difference between a trading system and a trading company.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`07_FEEDBACK_LOOPS.md`](../07_FEEDBACK_LOOPS.md)
**Supporting sections:** [`02_STATISTICAL_VALIDITY.md`](../02_STATISTICAL_VALIDITY.md) (IC tracking dependency), [`09_DATA_SIGNALS.md`](../09_DATA_SIGNALS.md) (signal correlation + conflict)

---

## The Ghost Component Registry

Five modules exist, are implemented, have write paths — but read paths are stubbed or never called:

| Component | Status | Problem |
|-----------|--------|---------|
| `OutcomeTracker` | **SINK** | Write called on fill hook. `get_regime_strategies()` returns stub. Data goes in, never comes out. |
| `SkillTracker` | **GHOST** | ~250 lines, fully implemented, zero callers anywhere. |
| `ICAttribution` | **GHOST** | ~200 lines, fully implemented, zero callers anywhere. |
| `ExpectancyEngine` | **ORPHAN** | ~200 lines. Sizing uses `core/kelly_sizing.py` instead. |
| `StrategyBreaker` | **GHOST** | ~200 lines, fully implemented, zero callers anywhere. |
| `TradeEvaluator` | **SINK** | Scores written to `trade_quality_scores`. Nobody reads. |

---

## Objective

Wire the 6 ghost module readpoints, implement failure mode taxonomy, connect IC degradation to weight adjustment, add live vs. backtest Sharpe demotion, and track agent decision quality. After this phase: every loss makes the next decision better.

---

## Items

### 7.1 Wire 6 Ghost Module Readpoints

- **Finding:** DO-1 | **Severity:** CRITICAL | **Effort:** 2-3 days
- **Audit section:** [`07_FEEDBACK_LOOPS.md` §readpoints](../07_FEEDBACK_LOOPS.md)
- **Problem:** 6 specific code changes needed to connect ghost modules to the live system.
- **Fix:**

| # | Missing Wire | From → To | Change |
|---|-------------|-----------|--------|
| 1 | Implement `get_regime_strategies()` | `meta_tools.py` → reads `strategies.regime_affinity` | Replace stub with DB query returning affinity-weighted allocations |
| 2 | `risk_sizing` checks regime_affinity | `trading/nodes.py` → `outcome_tracker.get_affinity()` | Multiply Kelly fraction by affinity before sizing |
| 3 | `execute_entries` checks strategy_breaker | `trading/nodes.py` → `strategy_breaker.get_scale_factor()` | Reduce/block orders for SCALED/TRIPPED strategies |
| 4 | Trade hooks populate SkillTracker | `trade_hooks.py` → `skill_tracker.update_agent_skill()` | Call on every trade close with agent_name + outcome |
| 5 | Signal engine records IC attribution | `signal_engine/engine.py` → `ic_attribution.record()` | After synthesis, record each collector's contribution vs. forward return |
| 6 | `daily_planner` queries quality scores | `trading/nodes.py` → `SELECT * FROM trade_quality_scores` | Surface patterns: "exit_evaluator gives HOLD on positions that then lose >3%" |

- **Key files:** `meta_tools.py`, `trading/nodes.py`, `trade_hooks.py`, `signal_engine/engine.py`
- **Acceptance criteria:**
  - [ ] All 6 readpoints wired and producing data
  - [ ] `get_regime_strategies()` returns real data, not stub
  - [ ] StrategyBreaker scale factor applied to position sizing
  - [ ] SkillTracker updated on every trade close

### 7.2 Failure Mode Taxonomy

- **Finding:** Loop-1 | **Severity:** CRITICAL | **Effort:** 2 days
- **Depends on:** 7.1
- **Audit section:** [`07_FEEDBACK_LOOPS.md` Loop 1](../07_FEEDBACK_LOOPS.md)
- **Problem:** One-directional path exists (`trade_hooks.py:118-144`): loss > 1% → `research_queue` with `bug_fix` task type. But all losses are generic "bug_fix." No distinction between regime mismatch, factor crowding, data staleness. No aggregation. No graduated response.
- **Fix:**
  1. Add failure mode taxonomy enum: `REGIME_MISMATCH`, `FACTOR_CROWDING`, `DATA_STALE`, `TIMING_ERROR`, `THESIS_WRONG`, `BLACK_SWAN`
  2. Classify each loss using regime-at-entry vs. regime-at-exit, factor overlap, data freshness
  3. Aggregate failure modes daily in supervisor (16:30 ET)
  4. Research queue priority = `f(cumulative_loss_30d * recency_weight)`
  5. Top failure mode by cumulative P&L impact → highest priority research task
- **Key files:** `src/quantstack/hooks/trade_hooks.py`, supervisor daily batch, research queue
- **Acceptance criteria:**
  - [ ] Every loss classified with specific failure mode
  - [ ] Daily aggregation identifies top failure mode by cumulative P&L impact
  - [ ] Research queue priority driven by loss severity, not just binary threshold

### 7.3 IC Degradation → Weight Adjustment

- **Finding:** Loop-3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Depends on:** Phase 2 item 2.1 (IC tracking must be live)
- **Audit section:** [`07_FEEDBACK_LOOPS.md` Loop 3](../07_FEEDBACK_LOOPS.md)
- **Problem:** Even when IC tracking exists (Phase 2), no mechanism adjusts synthesis weights. Technical IC drops 0.05 → 0.01 → nobody notices → still gets 25% weight → conviction inflated.
- **Fix:**
  1. Daily IC per collector (from Phase 2)
  2. If rolling 21-day IC < 0.02: halve collector weight in synthesis
  3. Publish `SIGNAL_DEGRADATION` event to EventBus
  4. Research graph picks up investigation task
- **Key files:** Signal engine synthesis weights, EventBus
- **Acceptance criteria:**
  - [ ] IC < 0.02 for 21 days automatically halves collector weight
  - [ ] `SIGNAL_DEGRADATION` event published on degradation
  - [ ] Research graph receives investigation task

### 7.4 Live vs. Backtest Sharpe Demotion

- **Finding:** Loop-4 | **Severity:** HIGH | **Effort:** 1 day
- **Depends on:** Phase 2 item 2.6 (walk-forward validation)
- **Audit section:** [`07_FEEDBACK_LOOPS.md` Loop 4](../07_FEEDBACK_LOOPS.md)
- **Problem:** `strategy_promoter` handles promotion. `strategy_breaker` handles 5% drawdown or 3 consecutive losses. But no slow-bleed detection: strategy live 30 days with Sharpe 0.2 (backtest was 1.5) keeps trading until 5% drawdown.
- **Fix:**
  1. Add live-vs-backtest Sharpe comparison to `strategy_promoter`
  2. Gate: live Sharpe < 50% of backtest for 21+ days → auto-demote to `forward_testing`
  3. Queue research task for strategy degradation investigation
- **Key files:** Strategy lifecycle, strategy promoter
- **Acceptance criteria:**
  - [ ] Live Sharpe < 50% of backtest for 21 days → auto-demote
  - [ ] Demotion triggers research investigation task
  - [ ] Demoted strategies reduce position sizes by 75%

### 7.5 Agent Decision Quality Tracking

- **Finding:** Loop-5 | **Severity:** HIGH | **Effort:** 3 days
- **Audit section:** [`07_FEEDBACK_LOOPS.md` Loop 5](../07_FEEDBACK_LOOPS.md)
- **Problem:** `trade_debater` recommends ENTER → position loses 3% → `trade_reflector` logs "thesis wrong" → debater's next call uses exact same prompt. No learning.
- **Fix:**
  1. Track per-agent recommendation → outcome
  2. Compute per-agent win rate over rolling 30 trades
  3. Alert when win rate drops below 40% (baseline should be >50%)
  4. Manual prompt improvement until meta-optimization (Phase 10)
- **Key files:** Agent executor, trade outcome tracking, alerting
- **Acceptance criteria:**
  - [ ] Per-agent win rate tracked over rolling 30 trades
  - [ ] Alert when any agent win rate < 40%
  - [ ] Decision quality data available for prompt improvement

### 7.6 Loss Aggregation in Supervisor

- **Finding:** DO-1 | **Severity:** HIGH | **Effort:** 1 day
- **Depends on:** 7.2 (failure mode taxonomy)
- **Audit section:** [`07_FEEDBACK_LOOPS.md`](../07_FEEDBACK_LOOPS.md)
- **Problem:** Individual losses recorded but never aggregated for pattern detection.
- **Fix:** Daily 16:30 ET supervisor job: aggregate losses by failure mode, strategy, symbol. Report top 3 failure patterns. Auto-generate targeted research tasks.
- **Key files:** Supervisor graph batch nodes
- **Acceptance criteria:**
  - [ ] Daily loss aggregation runs at 16:30 ET
  - [ ] Top 3 failure patterns identified and reported
  - [ ] Targeted research tasks auto-generated

### 7.7 Signal Correlation Tracking

- **Finding:** QS-S5 | **Severity:** CRITICAL | **Effort:** 2 days
- **Depends on:** Phase 2 item 2.1 (IC tracking)
- **Audit section:** [`09_DATA_SIGNALS.md` §8.6](../09_DATA_SIGNALS.md)
- **Problem:** 22 collectors run independently with static weights. Technical RSI and ML direction often >0.7 correlated. Effective independent count may be 10-12, not 22.
- **Fix:**
  1. Weekly: compute pairwise signal correlation matrix
  2. If `corr(A, B) > 0.7`: halve weight of weaker signal
  3. Report effective signal count = eigenvalues > 0.1
  4. Store correlation matrix for trend analysis
- **Key files:** Signal engine, new correlation analysis job
- **Acceptance criteria:**
  - [ ] Weekly signal correlation matrix computed and stored
  - [ ] Highly correlated signals have reduced weight
  - [ ] Effective independent signal count reported

### 7.8 Conflicting Signal Resolution

- **Finding:** QS-S9 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`09_DATA_SIGNALS.md` §8.7](../09_DATA_SIGNALS.md)
- **Problem:** When technical says "bullish" but ML says "bearish" and sentiment says "neutral," system computes weighted average → trades on weak, conflicting signal.
- **Fix:** Conflict detection: `if max_signal - min_signal > 0.5: flag as conflicting`. Cap conviction at 0.3 or skip trade. Log conflicts.
- **Key files:** Signal engine synthesis
- **Acceptance criteria:**
  - [ ] Signal conflict detected when max-min spread > 0.5
  - [ ] Conflicting signals cap conviction at 0.3 (or skip trade)
  - [ ] Conflict events logged

### 7.9 Conviction Calibration (Multiplicative)

- **Finding:** QS-S8 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 2 item 2.1 (IC tracking for calibration data)
- **Audit section:** [`09_DATA_SIGNALS.md` §8.9](../09_DATA_SIGNALS.md)
- **Problem:** Conviction adjustments are additive/fixed (ADX > 25 = +0.10). Not calibrated. +0.10 on 0.15 base is 67% increase; on 0.85 it's 11%.
- **Fix:** Replace additive with multiplicative: `adjusted = base * adx_factor * stability_factor * conflict_factor`. Calibrate quarterly from realized performance.
- **Key files:** Signal engine conviction adjustment logic
- **Acceptance criteria:**
  - [ ] Multiplicative factors replace additive adjustments
  - [ ] Factors calibrated from historical performance data

### 7.10 Concept Drift Detection (IC + Label + Interaction)

- **Finding:** QS-M2 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 2 item 2.1 (IC tracking)
- **Audit section:** [`02_STATISTICAL_VALIDITY.md`](../02_STATISTICAL_VALIDITY.md)
- **Problem:** PSI drift check exists but only monitors feature distributions. No label drift (target variable distribution shifts) or interaction drift (feature-target relationship changes).
- **Fix:** Add IC-based drift (feature-target correlation shift), label drift (target distribution change), and interaction drift monitoring. Auto-trigger model retraining when detected.
- **Key files:** ML pipeline, drift detection module
- **Acceptance criteria:**
  - [ ] IC-based concept drift detected within 5 trading days
  - [ ] Label drift triggers automatic model retraining
  - [ ] Drift events published to EventBus

### 7.11 Model Versioning + A/B

- **Finding:** QS-M3 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`02_STATISTICAL_VALIDITY.md`](../02_STATISTICAL_VALIDITY.md)
- **Problem:** No model versioning. No A/B testing. Can't compare new model vs. incumbent before promoting.
- **Fix:** Version all models with metadata. Champion/challenger framework: new model runs in shadow mode for 21 days. Promote only if IC > champion.
- **Key files:** ML model registry, model serving layer
- **Acceptance criteria:**
  - [ ] All models versioned with train date, features, hyperparams
  - [ ] Champion/challenger comparison running
  - [ ] Promotion requires IC improvement over incumbent

### 7.12 Regime Transition Detection

- **Finding:** QS-S7 | **Severity:** HIGH | **Effort:** 3 days
- **Audit section:** [`09_DATA_SIGNALS.md` §8.8](../09_DATA_SIGNALS.md)
- **Problem:** HMM regime model: 3 states, no transition detection. Most losses occur during transitions when model is most uncertain.
- **Fix:**
  1. Add transition probability output from HMM
  2. P(transition) > 0.3 → reduce all signal weights 50%, halve position sizes
  3. Add vol-conditioned sub-regimes (trending_up_low_vol vs. trending_up_high_vol)
  4. During 2-3 day transition window → paper-only for new entries
- **Key files:** Regime detection module, risk gate
- **Acceptance criteria:**
  - [ ] Regime transition probability computed and available to risk gate
  - [ ] High transition probability reduces sizing automatically
  - [ ] Vol-conditioned sub-regimes implemented

---

## Dependencies

- **Depends on:** Phase 1 (safety), Phase 2 item 2.1 (IC tracking — critical for 7.3, 7.7, 7.9, 7.10)
- **7.2 depends on 7.1** (need readpoints wired before taxonomy can classify)
- **7.6 depends on 7.2** (need taxonomy before aggregation)
- **7.4 depends on Phase 2 item 2.6** (walk-forward for backtest Sharpe baseline)

---

## Validation Plan

1. **Readpoints (7.1):** Query `regime_affinity` → verify non-stub data. Execute trade with TRIPPED strategy → verify blocked.
2. **Taxonomy (7.2):** Close trade at loss in wrong regime → verify classified as `REGIME_MISMATCH`.
3. **IC degradation (7.3):** Inject declining IC series for one collector → verify weight halved after 21 days.
4. **Sharpe demotion (7.4):** Run strategy with live Sharpe 0.3 vs. backtest 1.5 for 21 days → verify demotion.
5. **Agent quality (7.5):** Run 30 trades → verify win rate computed per agent, alerts fire for <40%.
6. **Signal conflict (7.8):** Inject conflicting signals (technical bullish, ML bearish) → verify conviction capped.
