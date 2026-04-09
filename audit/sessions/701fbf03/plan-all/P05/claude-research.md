# P05 Codebase Research: Adaptive Signal Synthesis

## Current Implementation State (Much Already Exists)

### synthesis.py — Already Has P05 Features

1. **Vol-conditioned sub-regime profiles** (§5.2): `_WEIGHT_PROFILES` already defines 8 sub-regime profiles (`trending_up_low_vol`, `trending_up_high_vol`, etc.) in addition to the 4 base regimes. `_get_weights()` checks `sub_regime` first.

2. **Ensemble methods** (§5.4): Three methods already implemented:
   - `_ensemble_weighted_avg` — default
   - `_ensemble_weighted_median` — robust to outlier voters
   - `_ensemble_trimmed_mean` — drops highest/lowest, averages rest
   - Hash-based A/B assignment: `ensemble_fn = _ENSEMBLE_METHODS[hash(symbol) % len(_ENSEMBLE_METHODS)]`

3. **IC-driven weights** (§5.1): Wire 5b integration at lines 530-544 calls `ICAttributionTracker().get_weights_for_regime()` and replaces static weights when sufficient data exists. Flag-gated via `ic_driven_weights_enabled()`.

4. **Transition dampening** (§5.2): Lines 589-599 already halve the score when `transition_probability > 0.3`. Flag-gated via `transition_signal_dampening_enabled()`.

5. **Multiplicative conviction** (§5.3): `_conviction_multiplicative()` method computes 6 multiplicative factors: ADX, stability, timeframe, regime_agreement, ML confirmation, data quality.

6. **IC gate** (P01 §1.1): Zeros out collectors with rolling 63d IC < 0.02.

7. **Correlation penalty** (P01 §1.4): Halves weight of redundant collectors via `CrossSectionalICTracker.compute_pairwise_correlation()`.

### ic_attribution.py — Full Tracker Implementation

- `ICAttributionTracker` class with DB persistence
- `record()` — stores observations with regime tagging
- `get_collector_ic()` — rolling Spearman IC
- `get_weights()` — IC-normalized collector weights (positive IC → weight, else 0)
- `get_weights_for_regime()` — regime-conditioned IC weights with min_days filter
- `get_report()` — full IC attribution report with status, trend, degraded list
- `_compute_trend()` — IC trend comparison (improving/stable/declining)

### ic_weights.py — Sigmoid IC Factors

- `ic_factor()` — continuous sigmoid: `1 / (1 + exp(-50 * (ic - 0.02)))`
- `compute_ic_factors()` — per-collector with ICIR consistency penalty (0.7x if ICIR < 0.1)
- `compute_ic_factors_gated()` — env var gated wrapper
- `check_weight_floor()` — safety floor at 10% total effective weight

### cross_sectional_ic.py — Cross-Sectional IC Tracker

- `CrossSectionalICTracker` — daily cross-sectional IC computation
- `compute_daily_ic()` — rank correlation of signal votes vs forward returns
- `get_rolling_ic()`, `get_ic_stability()`, `get_ic_gate_status()`
- `compute_pairwise_correlation()` — signal redundancy detection
- `_store_ic()` — persists to `signal_ic` table with ICIR metrics

### regime.py — Regime Collector

- HMM-based regime detection with `hmmlearn` library
- Provides `trend_regime`, `hmm_stability`, `regime_confidence`
- **Already exposes `transition_probability`** in output dict (used by synthesis.py transition dampening)
- Sub-regime detection via vol classification

### feedback_flags.py — Feature Flags

Relevant P05 flags already exist:
- `ic_driven_weights_enabled()` — Wire 5b
- `ic_gate_enabled()` — P01 IC gate
- `correlation_penalty_enabled()` — P01 correlation penalty
- `ensemble_ab_test_enabled()` — P05 §5.4 A/B test
- `transition_signal_dampening_enabled()` — P05 §5.2
- `signal_ci_enabled()` — P01 bootstrap CI

### Testing Setup

- Framework: pytest with loguru capture
- Test directory: `tests/unit/` (flat structure, some subdirs like `ml/`, `execution/`)
- Existing signal engine tests: `tests/unit/test_signal_scorer.py`, `tests/unit/test_drift_detection_enhanced.py`
- Pattern: direct function/class testing, monkeypatching for DB, no fixtures for signal engine

## What's Actually Remaining (Gaps)

### Gap 1: Weekly IC Weight Batch Job
No scheduled job exists to periodically recompute IC-driven weights. Currently, `ICAttributionTracker` is instantiated fresh on every synthesis call, loading from DB each time. Need: a periodic job (weekly) that precomputes and caches regime-conditioned weights.

### Gap 2: IC Weight Caching
Each synthesis call creates a new `ICAttributionTracker()`, which loads ALL observations from DB. This is expensive for every signal. Need: singleton/cached instance or precomputed weight table.

### Gap 3: Position Sizing During Transitions
Transition dampening reduces signal score, but the spec also asks for halved position sizes. Position sizing happens downstream in `nodes.py` — need to check if conviction reduction alone achieves this or if explicit sizing reduction is needed.

### Gap 4: Quarterly Conviction Factor Calibration
Multiplicative factors use hardcoded parameters (ADX: 0.15 ramp over [15,50], stability: 0.85-1.05). No automated calibration from realized signal-to-return performance. Need: quarterly job that optimizes factor parameters.

### Gap 5: A/B Test Result Tracking
Ensemble A/B assignment exists (hash-based) but there's no mechanism to compare results across methods. Need: table to track per-method performance metrics and a comparison job.

### Gap 6: Integration Tests
No end-to-end test for the IC-driven weight → synthesis → conviction flow.
