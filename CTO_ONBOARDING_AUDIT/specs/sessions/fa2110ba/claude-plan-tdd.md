# TDD Plan — Phase 7: Feedback Loops & Learning

**Testing framework:** pytest (existing). Fixtures in `tests/unit/conftest.py`. Class-based tests. Async via `run_async` fixture. OHLCV generators available.

**Conventions:** Test files at `tests/unit/test_*.py` or `tests/unit/<subdir>/test_*.py`. Mock DB via `mock_settings` fixture. No real DB calls in unit tests.

---

## Section 1: Ghost Module API Audit

### OutcomeTracker affinity formula fix
- Test: new formula step size for -2% loss produces ~0.11 step (not 0.019)
- Test: exponential decay with 20-trade halflife — outcome 20 trades ago contributes half as much
- Test: cold-start — fewer than 5 outcomes returns affinity 1.0 with no adjustment
- Test: affinity stays within [0.1, 1.0] bounds after many consecutive wins/losses
- Test: recency weighting — recent loss has more impact than old loss of same magnitude

### SkillTracker ICIR adjustment simplification
- Test: `get_confidence_adjustment()` with ICIR=3.0 caps at 0.3 (not 0.6)
- Test: adjustment stays within [0.5, 1.5] for all edge cases

### StrategyBreaker PostgreSQL migration
- Test: save/load round-trip through DB matches original JSON behavior
- Test: TRIPPED state persists across mock "restart" (new instance reads from DB)
- Test: concurrent reads don't block (simulated with multiple calls)

### ICAttributionTracker PostgreSQL migration
- Test: save/load round-trip through DB matches JSON behavior
- Test: data persists across mock "restart"

---

## Section 2: Wire 6 Ghost Module Readpoints

### Wire 1: get_regime_strategies()
- Test: returns strategies sorted by affinity for given regime
- Test: includes strategy status (ACTIVE/SCALED/TRIPPED) from StrategyBreaker
- Test: returns empty list for regime with no strategies
- Test: filters out retired strategies

### Wire 2: StrategyBreaker in risk_sizing
- Test: ACTIVE strategy (factor 1.0) — alpha_signal unchanged
- Test: SCALED strategy (factor 0.5) — alpha_signal halved
- Test: TRIPPED strategy (factor 0.0) — alpha_signal zeroed
- Test: defensive bounds — get_scale_factor() raising exception defaults to 1.0
- Test: defensive bounds — factor > 1.0 clamped to 1.0

### Wire 3: StrategyBreaker in execute_entries
- Test: TRIPPED strategy order skipped entirely
- Test: ACTIVE strategy order placed normally
- Test: skip logged with breaker reason

### Wire 4: SkillTracker in trade hooks
- Test: profitable trade close calls update_agent_skill with prediction_correct=True
- Test: unprofitable trade close calls with prediction_correct=False
- Test: agent_name extracted correctly from trade context

### Wire 5: ICAttribution in signal engine
- Test: after synthesis, record() called for each collector with signal_value
- Test: forward_return initially None
- Test: on_trade_close backfills forward_return for matching records

### Wire 6: Trade quality in daily_plan
- Test: rolling 30-trade averages computed correctly per dimension
- Test: weakest dimension identified correctly
- Test: with fewer than 5 scored trades, quality context omitted from prompt

---

## Section 3: Failure Mode Taxonomy

### FailureMode enum
- Test: all 7 modes are valid FailureMode values
- Test: enum is str-compatible (serializes to string)

### Rule-based classifier
- Test: regime mismatch detected when entry_regime != exit_regime
- Test: data_stale detected when data freshness > threshold
- Test: black_swan detected when loss > 3 std deviations
- Test: timing_error detected when entry within 2 bars of key level
- Test: returns UNCLASSIFIED when no rule matches

### Async LLM fallback
- Test: UNCLASSIFIED queued for async LLM classification (not blocking)
- Test: LLM timeout does not block trade close hook
- Test: LLM result backfills failure_mode in strategy_outcomes

### Research queue enhancement
- Test: priority computation with cumulative_loss_30d and recency_weight
- Test: higher priority for persistent failure patterns vs isolated losses
- Test: task_type matches failure_mode (not generic 'bug_fix')

---

## Section 4: Loss Aggregation in Supervisor

### run_loss_aggregation()
- Test: groups losses by failure_mode, strategy, symbol over 30 days
- Test: top 3 patterns ranked by absolute P&L impact
- Test: auto-generates research tasks with failure_mode as type
- Test: aggregation stored in loss_aggregation table
- Test: handles empty losses (no trades in 30 days) gracefully
- Test: UNCLASSIFIED losses still appear in aggregation

---

## Section 5: IC Degradation → Weight Adjustment

### Sigmoid IC factor function
- Test: IC=0.05 → factor ≈ 1.0
- Test: IC=0.02 → factor ≈ 0.5
- Test: IC=0.00 → factor ≈ 0.0
- Test: IC=-0.02 → factor ≈ 0.0
- Test: smooth transition — no discrete jumps

### IC_IR penalty
- Test: IC_IR < 0.1 applies 0.7× penalty
- Test: IC_IR >= 0.1 no penalty

### Weight floor check
- Test: all collectors near-zero IC → fall back to equal static weights
- Test: SIGNAL_DEGRADATION alert published on floor trigger

### Config flag
- Test: FEEDBACK_IC_WEIGHT_ADJUSTMENT=false → ic_factor always 1.0
- Test: flag=true → IC factors applied

### Cold-start
- Test: < 21 days of IC data → ic_factor = 1.0

---

## Section 6: Signal Correlation Tracking

### Correlation matrix computation
- Test: pairwise Spearman computed correctly for known inputs
- Test: effective independent signal count via eigenvalues

### Continuous correlation penalty
- Test: corr=0.4 → penalty = 1.0 (no penalty)
- Test: corr=0.6 → penalty = 0.8
- Test: corr=0.8 → penalty = 0.2 (floor)
- Test: weaker signal (lower IC) gets the penalty, not the stronger one

### Config flag
- Test: FEEDBACK_CORRELATION_PENALTY=false → penalty always 1.0

---

## Section 7: Conflicting Signal Resolution

### Conflict detection
- Test: max-min spread > 0.5 → flagged as conflicting
- Test: max-min spread <= 0.5 → not flagged
- Test: SIGNAL_CONFLICT event published with correct payload

### Conviction cap
- Test: conflicting signals → conviction capped at 0.3
- Test: non-conflicting → conviction not capped

---

## Section 8: Conviction Calibration — Multiplicative

### Individual factors
- Test: ADX factor — ADX=15 → 1.0, ADX=50 → 1.15
- Test: Stability factor — stability=0.0 → 0.85, stability=1.0 → 1.05
- Test: Timeframe factor — contradicting → 0.80, agreeing → 1.0
- Test: Regime agreement factor — disagree → 0.85, agree → 1.0
- Test: ML confirmation factor — confirms → 1.10, not → 1.0
- Test: Data quality factor — failure → 0.75, ok → 1.0

### Combined behavior
- Test: all factors worst case → product is ~0.43 (document expected range)
- Test: all factors best case → product is ~1.35
- Test: final result clipped to [0.05, 0.95]
- Test: missing inputs default factor to 1.0

### Config flag
- Test: FEEDBACK_CONVICTION_MULTIPLICATIVE=false → reverts to additive

---

## Section 9: Agent Decision Quality Tracking

### Win rate computation
- Test: 30 trades, 18 wins → win_rate = 0.60
- Test: rolling window — old trades drop off

### Alert threshold
- Test: win_rate < 0.40 → AGENT_DEGRADATION event published
- Test: win_rate >= 0.40 → no alert
- Test: research task queued with agent details

### Cold-start
- Test: < 30 trades → no alert, confidence = 1.0

---

## Section 10: Live vs. Backtest Sharpe Demotion

### Live Sharpe computation
- Test: rolling 21-day Sharpe from known returns
- Test: handles missing return days

### Demotion gate
- Test: live Sharpe < 50% of backtest for 21 days → auto-demote
- Test: live Sharpe < 50% for 20 days → not yet demoted
- Test: STRATEGY_DEMOTED event published
- Test: 0.25× sizing multiplier applied via force_scale()

### Config flag
- Test: FEEDBACK_SHARPE_DEMOTION=false → no demotion check

### Cold-start
- Test: < 21 days of live data → skip check

---

## Section 11: Concept Drift Detection

### Layer 1: IC-based drift
- Test: IC drops > 2 std from baseline → alert
- Test: IC stable → no alert
- Test: detection within 5 trading days of injected shift

### Layer 2: Label drift
- Test: KS test detects shifted return distribution (p < 0.01)
- Test: stable returns → no alert

### Layer 3: Interaction drift
- Test: adversarial classifier AUC > 0.60 → flag for investigation
- Test: stable joint distribution → AUC near 0.50, no flag

### Auto-retrain decision
- Test: gradual IC decline → auto-retrain triggered
- Test: abrupt IC drop → MODEL_DEGRADATION event, no auto-retrain
- Test: cooldown — second retrain blocked within 20 trading days

### Config flag
- Test: FEEDBACK_DRIFT_DETECTION=false → skip all drift checks

---

## Section 12: Model Versioning + Champion/Challenger

### Model registry
- Test: register new model → version auto-increments
- Test: query champion model for strategy → returns correct version
- Test: retire old model → status changes to retired

### Shadow mode
- Test: challenger predictions logged to shadow_predictions table
- Test: champion predictions drive real signals (not challenger)

### Promotion
- Test: IC improvement > 0.005 + Sharpe improvement > 0.15 + no DD regression → promote
- Test: any criterion not met → no promotion
- Test: after 60 days without promotion → retire challenger

### Cold-start
- Test: no champion in registry → falls back to disk-based model loading

---

## Section 13: Regime Transition Detection

### Filtered transition probability
- Test: use predict_proba() output, not static transmat_
- Test: high uncertainty (max prob < 0.5) → high transition_probability
- Test: confident state (max prob > 0.9) → low transition_probability

### Sizing response
- Test: P(transition) < 0.10 → factor = 1.0
- Test: P(transition) = 0.20 → factor = 0.75
- Test: P(transition) = 0.40 → factor = 0.50
- Test: P(transition) = 0.60 → factor = 0.25

### Degraded mode
- Test: HMM fit failure → transition_probability defaults to 0.0
- Test: risk_sizing handles None transition_probability → factor = 1.0

### Minimum tradeable size floor
- Test: compound factors producing < $100 position → trade skipped
- Test: compound factors producing >= $100 → trade placed

### Vol-conditioned sub-regimes
- Test: low vol (< 30th percentile) → correct sub-regime label
- Test: high vol (> 70th percentile) → correct sub-regime label

### Config flag
- Test: FEEDBACK_TRANSITION_SIZING=false → transition_factor always 1.0
