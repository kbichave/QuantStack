# Section 13: Concept Drift Detection

## Overview

The existing `DriftDetector` class (`src/quantstack/learning/drift_detector.py`, 487 lines) only monitors feature distributions via PSI (Population Stability Index, implemented as a scaled KS statistic). It misses the two most dangerous drift types for trading models: **label drift** (the target distribution changes) and **interaction drift** (the feature-target relationship changes while individual distributions look stable).

This section extends `DriftDetector` with three new detection layers, adds auto-retrain decision logic with a 20-day cooldown, and wires new supervisor batch nodes to run these checks on appropriate schedules.

**Config flag:** `FEEDBACK_DRIFT_DETECTION` (default `false`). When `false`, all drift checks are skipped. Data collection (IC computation, signal storage) continues regardless of this flag so that data accumulates for when the flag is enabled.

**Dependencies:** None -- this section is parallelizable in Batch 1. However, it benefits from IC data populated by nightly `run_ic_computation()` (already exists in `src/quantstack/graphs/supervisor/nodes.py`). The `MODEL_DEGRADATION` event type already exists in `EventType` enum (`src/quantstack/coordination/event_bus.py`, line 72).

---

## Existing Code

The current `DriftDetector` class lives at `src/quantstack/learning/drift_detector.py`. Key elements:

- `DriftDetector` -- compares current signal features against per-strategy baselines stored as JSON files at `~/.quantstack/drift_baselines/{strategy_id}.json`
- `DriftReport` dataclass -- holds per-feature PSI scores, overall severity ("NONE"/"WARNING"/"CRITICAL"), and drifted feature list
- `compute_psi()` -- actually a scaled KS statistic (not binned PSI), pure numpy, no scipy dependency
- `StrategyDriftMonitor` -- sits above `DriftDetector`, combines drift with performance signals for lifecycle decisions
- `TRACKED_FEATURES` -- 6 features: rsi_14, atr_pct, adx_14, bb_pct, volume_ratio, regime_confidence
- Baselines stored as JSON on disk (same container-restart risk as StrategyBreaker -- section-01 addresses persistence migration for other modules, but drift baselines are read-only in the hot path so the risk is lower)

The `EventType` enum already includes `MODEL_DEGRADATION = "model_degradation"` and `DEGRADATION_DETECTED = "degradation_detected"`.

---

## Tests

All tests go in `tests/unit/test_drift_detection.py`. Use pytest with class-based organization. Mock DB calls via the existing `mock_settings` fixture. No real DB or scipy calls in unit tests.

### Layer 1: IC-based Concept Drift (daily)

- **Test: IC drop triggers alert.** Given a baseline IC of 0.04 with std 0.005 for a feature, when the rolling IC drops to 0.02 (> 2 std below baseline), the detector should return a drift alert for that feature. Verify the alert includes the feature name, current IC, baseline IC, and the magnitude of deviation.

- **Test: stable IC produces no alert.** Given a baseline IC of 0.04 with std 0.005, when the rolling IC is 0.038, no alert is produced.

- **Test: detection latency.** Inject a step-change in IC at day N. The detector should flag the drift within 5 trading days of the shift (not 20+ days). This validates that the rolling window is responsive enough.

### Layer 2: Label Drift (weekly, via KS test)

- **Test: shifted return distribution detected.** Construct two return distributions: baseline with mean=0.001, std=0.02 (training period) and current with mean=-0.005, std=0.04 (shifted). The KS test should return p < 0.01, triggering a label drift alert.

- **Test: stable returns produce no alert.** Two distributions drawn from the same parameters. KS test p-value should be well above 0.01. No alert.

### Layer 3: Interaction Drift (monthly, adversarial validation)

- **Test: shifted joint distribution flagged.** Create "training" data where feature X positively correlates with returns, and "recent" data where the correlation is inverted. An adversarial classifier (logistic regression) distinguishing the two periods should achieve AUC > 0.60, triggering an investigation flag.

- **Test: stable joint distribution not flagged.** When training and recent data are drawn from the same distribution, the adversarial classifier AUC should be near 0.50 (no better than random). No flag.

### Auto-Retrain Decision Tree

- **Test: gradual IC decline triggers auto-retrain.** When IC has been declining over 60+ days (gradual slope, not step change) and current IC is below 0.01, `should_retrain()` returns `True` with reason "gradual_ic_decline".

- **Test: abrupt IC drop publishes MODEL_DEGRADATION, no auto-retrain.** When IC drops sharply (step change within 5 days), `should_retrain()` returns `False`, and instead a `MODEL_DEGRADATION` event payload is returned for the caller to publish.

- **Test: cooldown blocks second retrain within 20 days.** After a successful retrain decision on day 0, calling `should_retrain()` on day 15 returns `False` even if IC is still degraded. On day 21, it returns `True` again.

### Config Flag

- **Test: FEEDBACK_DRIFT_DETECTION=false skips all checks.** When the env var is `false`, all three drift layers return no-op results (no alerts, no retrain decisions). Verify by mocking the env var and calling the top-level drift check.

---

## Implementation Details

### Layer 1: IC-based Concept Drift

**File:** `src/quantstack/learning/drift_detector.py` -- extend the existing `DriftDetector` class.

Add a method `check_ic_drift()` that:
1. Accepts a dict of `{feature_name: current_rolling_ic}` and a dict of `{feature_name: (baseline_ic_mean, baseline_ic_std)}` from the training period
2. For each feature, computes `z_score = (baseline_ic - current_ic) / baseline_ic_std` (a positive z-score means IC has dropped)
3. If `z_score > 2.0` for any feature, flags that feature as drifted
4. Returns a new dataclass `ICDriftReport` with per-feature z-scores and the list of drifted features

The baseline IC statistics (mean, std) should be stored alongside the existing PSI baselines, either in the same JSON file or a companion file at `~/.quantstack/drift_baselines/{strategy_id}_ic_baseline.json`.

**IC source:** Read from the `signal_ic` table populated by the nightly `run_ic_computation()` supervisor node. For each feature tracked by the ML model, compute Spearman correlation between that feature and realized 5-day forward returns over a rolling window. The existing IC computation infrastructure handles this -- this layer just compares the result against the stored baseline.

**Schedule:** Daily, after `run_ic_computation()` completes in the supervisor nightly batch.

### Layer 2: Label Drift (KS Test)

Add a method `check_label_drift()` that:
1. Accepts `training_returns: np.ndarray` (return distribution from training period) and `recent_returns: np.ndarray` (rolling 63-day return distribution)
2. Computes a two-sample KS test. Use the existing pure-numpy KS implementation in `compute_psi()` as a template -- extract the raw KS statistic before the PSI scaling step, then compute an approximate p-value using the asymptotic formula: `p_approx = 2 * exp(-2 * n_eff * ks_stat^2)` where `n_eff = (n1 * n2) / (n1 + n2)`. This avoids adding scipy as a dependency.
3. If `p < 0.01`, return a `LabelDriftReport` flagging significant distribution shift
4. Include the KS statistic, p-value, and summary statistics (mean/std of both distributions) in the report

**Schedule:** Weekly (include in the Friday `run_signal_correlation()` batch or the daily run since it is lightweight). The plan specifies it is lightweight enough for daily inclusion.

### Layer 3: Interaction Drift (Adversarial Validation)

Add a method `check_interaction_drift()` that:
1. Accepts `training_data: np.ndarray` (feature-target pairs from training) and `recent_data: np.ndarray` (feature-target pairs from last 63 days)
2. Labels training data as class 0, recent data as class 1
3. Trains a logistic regression classifier to distinguish the two periods. Use a simple implementation -- sklearn's `LogisticRegression` if available, or a minimal numpy-based logistic regression. Since this runs monthly, the sklearn dependency is acceptable here.
4. Evaluates with AUC (use train/test split or cross-validation)
5. If AUC > 0.60, the joint distribution has shifted -- return an `InteractionDriftReport` flagging investigation needed

**Schedule:** Monthly, 1st trading day. New supervisor batch node `run_adversarial_validation()`.

### Auto-Retrain Decision Tree

Add a standalone function `evaluate_retrain_decision()` (or a method on `DriftDetector`) that:

1. Accepts: current IC, IC history (last 60+ days), last retrain date, current drift reports
2. Decision logic:
   - **Feature drift + IC still healthy (> 0.01):** Log warning, return `RetrainDecision(should_retrain=False, reason="benign_covariate_shift")`
   - **IC degradation + gradual (declining slope over 60+ days):** Return `RetrainDecision(should_retrain=True, reason="gradual_ic_decline", data_window=252)` -- retrain with recent 252-day data
   - **IC degradation + abrupt (step change):** Return `RetrainDecision(should_retrain=False, reason="abrupt_shift", publish_event=True)` with `MODEL_DEGRADATION` event payload for the caller to publish. Abrupt shifts need human/research investigation, not blind retraining.
   - **Cooldown check:** If last retrain was < 20 trading days ago, return `RetrainDecision(should_retrain=False, reason="cooldown")` regardless of other signals

3. The "gradual vs abrupt" distinction: compute the slope of IC over the last 60 days via linear regression. If R-squared > 0.5 and slope is negative, it is gradual. If IC dropped more than 2 std in a 5-day window, it is abrupt.

New dataclass `RetrainDecision`:
```python
@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    data_window: int | None = None  # days of data for retraining
    publish_event: bool = False
    event_payload: dict | None = None
```

### New Supervisor Batch Nodes

**File:** `src/quantstack/graphs/supervisor/nodes.py`

**`run_drift_detection()`** -- daily, after IC computation:
1. Check `FEEDBACK_DRIFT_DETECTION` env var. If `false`, return early with `{"skipped": True, "reason": "flag_disabled"}`.
2. For each active/forward_testing strategy with >= 63 days of feature data:
   a. Run Layer 1 (IC drift) using latest IC values from `signal_ic` table
   b. Run Layer 2 (label drift) using rolling 63-day returns vs training-period returns
   c. Run existing PSI-based `check_drift()` as Layer 0
3. Evaluate the retrain decision tree for each strategy
4. If retrain is recommended, queue a research task of type `model_retrain` with the strategy context
5. If `MODEL_DEGRADATION` event should be published, publish it via EventBus
6. Return summary: `{"strategies_checked": N, "drift_alerts": [...], "retrain_queued": [...], "errors": [...]}`

**`run_adversarial_validation()`** -- monthly, 1st trading day:
1. Check `FEEDBACK_DRIFT_DETECTION` env var.
2. For each strategy with an ML model:
   a. Load training-period feature-target data (from model metadata or stored baselines)
   b. Load recent 63-day feature-target data
   c. Run Layer 3 (adversarial validation)
3. If any strategy flags AUC > 0.60, publish `MODEL_DEGRADATION` event and queue investigation task
4. Return summary

Wire these into the supervisor's nightly batch sequence in the same pattern as `run_ic_computation()` and `run_signal_scoring()`:
- `run_drift_detection()` runs after `run_ic_computation()` (it needs fresh IC data)
- `run_adversarial_validation()` runs conditionally on the 1st trading day of the month

### Cold-Start Behavior

- If a strategy has < 63 days of feature data, skip all drift checks for that strategy. Return a no-op report with `severity="NONE"` and a note indicating insufficient data.
- If no IC baseline exists for a strategy (newly promoted, never had baselines set), skip IC drift. PSI drift still runs if PSI baselines exist.
- If no training-period return distribution is stored, skip label drift.

### Rollback

Set `FEEDBACK_DRIFT_DETECTION=false`. All drift checks become no-ops. The existing PSI-based drift detection in `DriftDetector.check_drift()` is unaffected by this flag (it predates this section and should continue to work as before). Only the new layers (IC drift, label drift, interaction drift) and the auto-retrain logic are gated by the flag.

---

## Files to Create or Modify

| File | Action |
|------|--------|
| `src/quantstack/learning/drift_detector.py` | Modify -- add `check_ic_drift()`, `check_label_drift()`, `check_interaction_drift()`, `evaluate_retrain_decision()` methods/functions. Add `ICDriftReport`, `LabelDriftReport`, `InteractionDriftReport`, `RetrainDecision` dataclasses. |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify -- add `run_drift_detection()` and `run_adversarial_validation()` async functions. Wire into nightly batch after `run_ic_computation()`. |
| `tests/unit/test_drift_detection.py` | Create -- all tests listed above. |

---

## Key Design Decisions

**Why not scipy for KS test?** The existing codebase deliberately avoids scipy in the drift detector hot path (the `compute_psi()` function implements KS in pure numpy). For label drift, the same approach works -- extract the raw KS statistic and use the asymptotic p-value approximation. This keeps the dependency footprint small. The adversarial validation layer (monthly, cold path) can use sklearn since it already exists in the environment for ML model training.

**Why gradual vs abrupt distinction for retraining?** Blind auto-retraining on any IC drop is dangerous. An abrupt shift (earnings surprise, regime change, black swan) means the recent data is non-stationary -- retraining on it would fit the model to a transient event. Gradual decline means the market structure has slowly evolved and retraining on a longer window can adapt. The 20-day cooldown prevents overfitting to noise from too-frequent retraining.

**Why 63-day windows?** One calendar quarter of trading days. Long enough for statistical significance in correlation and distribution tests, short enough to detect drift before it costs multiple months of performance. Matches the window used in section-08 (signal correlation).
