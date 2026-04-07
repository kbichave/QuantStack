# Section 08: Autonomous Feature Factory (AR-10)

## Overview

The system's feature set is manually defined in `src/quantstack/core/features.py` and `tools/functions/data_functions.py`. There is no systematic discovery of new features or detection of decaying ones beyond the per-collector IC attribution in `learning/ic_attribution.py`. The Feature Factory closes this gap with a three-phase pipeline: LLM-assisted enumeration, IC screening with correlation filtering, and daily drift monitoring with auto-replacement.

This section delivers a new subsystem under `src/quantstack/research/` that runs weekly (enumeration + screening) and daily (monitoring), producing a curated set of 50-100 predictive features from 500+ candidates while automatically retiring features that decay.

## Dependencies

- **section-01-db-migrations**: The `feature_candidates` table must exist before any phase of the pipeline can persist results. The table schema is defined in `db.py`'s `ensure_schema()`.
- **section-04-budget-discipline**: The experiment prioritization formula references `novelty_score` and `estimated_compute_cost`, which interact with how the feature factory candidates feed into the research pipeline.
- **section-05-event-bus-extensions**: The `FEATURE_DECAYED` and `FEATURE_REPLACED` event types must be registered in the `EventType` enum before Phase 3 monitoring can publish events.

## Tests First

All tests go in `tests/unit/test_feature_factory.py`. The test file covers all three phases of the pipeline.

```python
"""tests/unit/test_feature_factory.py"""

# --- Phase 1: Enumeration ---

# Test: programmatic enumeration from 10 base features produces >100 candidates
#   Given 10 base features, the cross-interaction pairs alone yield C(10,2)=45 ratios.
#   Combined with lags (6 per feature = 60) and rolling stats (4 stats * 4 windows * 10 = 160),
#   total exceeds 100 easily. Assert len(candidates) > 100.

# Test: enumeration respects 2000 candidate hard cap
#   Feed in enough base features that cross-interactions would produce >2000.
#   Assert len(candidates) == 2000. Verify truncation uses expected_novelty ranking.

# Test: LLM-assisted enumeration adds novel features not in programmatic set
#   Mock the Haiku LLM call to return 5 feature definitions.
#   Assert those 5 appear in the final candidate list.
#   Assert they are tagged with source="haiku_generated".

# Test: LLM failure falls back to programmatic-only (no crash)
#   Mock the Haiku LLM call to raise an exception.
#   Assert enumeration completes successfully with only programmatic candidates.
#   Assert no exception propagates.

# --- Phase 2: Screening ---

# Test: IC screening filters candidates below IC 0.01
#   Provide candidates with IC values [0.005, 0.01, 0.02, 0.05].
#   Assert candidates with IC < 0.01 are excluded.
#   Assert candidates with IC >= 0.01 are included.

# Test: IC screening filters candidates below stability 0.5
#   Provide candidates with IC stability values [0.3, 0.5, 0.7].
#   Assert candidates with stability < 0.5 are excluded.
#   Assert candidates with stability >= 0.5 are included.

# Test: correlation check drops features with >0.95 Pearson to already-selected
#   Provide two features with 0.96 Pearson correlation. First is selected.
#   Assert second feature is dropped.
#   Provide two features with 0.94 correlation. Assert both are kept.

# Test: screening output is 50-100 features from 500+ input
#   Provide 500 candidates with a realistic IC distribution.
#   Assert 50 <= len(curated) <= 100.

# --- Phase 3: Monitoring ---

# Test: daily monitoring detects PSI > 0.25 as CRITICAL decay
#   Set a feature's PSI to 0.30.
#   Assert the feature is flagged as decayed.

# Test: daily monitoring detects IC < 0.005 for 10 days as decay
#   Provide a feature with IC below 0.005 for 10 consecutive days.
#   Assert the feature is flagged as decayed.
#   Provide a feature with IC below 0.005 for only 9 days. Assert not flagged.

# Test: auto-replacement selects next-best feature from screening pool
#   Mark a curated feature as decayed. Screening pool has 3 unused candidates.
#   Assert the replacement is the candidate with the highest IC that does not
#   have >0.95 correlation with any remaining curated feature.

# Test: FEATURE_DECAYED and FEATURE_REPLACED events published
#   Mock the event bus. Trigger a decay + replacement cycle.
#   Assert publish was called with EventType.FEATURE_DECAYED payload containing feature_id, psi, ic_current.
#   Assert publish was called with EventType.FEATURE_REPLACED payload containing old_feature_id, new_feature_id.
```

## Data Model

A new table `feature_candidates` must exist (created by section-01-db-migrations in `db.py`'s `ensure_schema()`):

```python
@dataclass
class FeatureCandidate:
    feature_id: str           # UUID
    feature_name: str         # Human-readable, e.g. "rsi_14_lag_5_div_bb_pct_21"
    definition: str           # Computable expression, e.g. "rsi_14_lag_5 / bb_pct_21"
    source: str               # "programmatic" | "haiku_generated"
    ic: float                 # Spearman rank correlation with 5-day forward returns
    ic_stability: float       # Inverse of rolling 63-day IC standard deviation
    correlation_group: str    # Cluster ID for correlated features
    status: str               # "candidate" | "curated" | "active" | "decayed" | "replaced"
    screening_date: str       # ISO date of last screening
    decay_date: str | None    # ISO date when decay was detected
    created_at: datetime      # TIMESTAMPTZ
```

Primary key: `feature_id` (UUID). Idempotent writes via `ON CONFLICT (feature_id) DO UPDATE`.

## Implementation

### File: `src/quantstack/research/feature_factory.py` (NEW)

This is the top-level orchestrator for all three phases. It exposes three entry points corresponding to the three phases, plus a convenience `run_full_pipeline()` that chains them.

Key responsibilities:
- `enumerate_features(base_features: list[str]) -> list[FeatureCandidate]` -- calls into `feature_enumerator.py`, enforces the 2000 hard cap.
- `screen_features(candidates: list[FeatureCandidate], universe_symbols: list[str]) -> list[FeatureCandidate]` -- calls into `feature_screener.py`, returns 50-100 curated features.
- `monitor_features(curated_features: list[FeatureCandidate]) -> list[FeatureEvent]` -- runs daily drift check, triggers replacement, publishes events.
- `run_full_pipeline()` -- weekly orchestrator: enumerate, screen, persist to `feature_candidates` table.

The orchestrator reads base features from `src/quantstack/core/features.py` (the existing feature definitions: RSI, MACD, ADX, Bollinger Band %, SMA crossovers, volume metrics). It does not hardcode feature names -- it introspects whatever the core features module exports.

### File: `src/quantstack/research/feature_enumerator.py` (NEW)

Responsible for Phase 1: generating candidate features from base features.

**Programmatic enumeration** generates candidates by applying transformations to base features:

- **Lags**: For each base feature, create lagged versions at 1, 2, 3, 5, 10, 21 days.
- **Rolling statistics**: For each base feature, compute mean, std, skew, zscore over windows of 5, 10, 21, 63 days.
- **Cross-interactions**: For all pairs of base features, compute the ratio (feature_A / feature_B). This is O(N^2) in the number of base features. With 10 base features, this yields 45 ratios; with 20 base features, 190 ratios.
- **Regime-conditional**: For each base feature, multiply by a binary regime indicator (trending=1, ranging=0).

If the total exceeds 2000, truncate by ranking candidates on expected novelty. Expected novelty is approximated by how dissimilar a candidate's transformation type is from already-selected candidates (prefer diversity of transformation types over many lags of the same feature).

**LLM-assisted enumeration** calls Haiku with the base feature list and current market context:

```
Prompt structure:
- System: "You are a quantitative feature engineer."
- User: "Given these base features: {list}. Current regime: {regime}.
  Suggest 20 novel composite features for predicting 5-day forward returns.
  Return as JSON array of {name, definition, rationale}."
```

Parse the response. Tag each with `source="haiku_generated"`. If the LLM call fails (timeout, rate limit, malformed response), log the error and proceed with programmatic-only candidates. This fallback is critical -- the pipeline must never crash due to an LLM failure.

### File: `src/quantstack/research/feature_screener.py` (NEW)

Responsible for Phase 2: filtering candidates down to a curated set.

For each candidate feature:

1. **Compute feature values** across all universe symbols over the trailing 2 years of daily data.
2. **Calculate IC**: Spearman rank correlation between the feature values and 5-day forward returns. This is a standard quantitative finance metric. Use `scipy.stats.spearmanr`.
3. **Calculate IC stability**: Compute IC on rolling 63-day windows. Stability = 1 / std(rolling_ICs). Higher is better (more consistent predictive power).
4. **Filter**: Keep candidates where IC > 0.01 AND stability > 0.5.
5. **Correlation deduplication**: Sort surviving candidates by IC descending. Walk the list; for each candidate, compute Pearson correlation of its feature values against all already-selected candidates. If any correlation > 0.95, drop it. This greedy approach ensures the final set has no near-duplicate features.
6. **Target output**: 50-100 features. If fewer than 50 survive, relax the IC threshold to 0.005 and re-run. If more than 100 survive, take the top 100 by IC.

Persist results to the `feature_candidates` table with status `"curated"`.

### File: `src/quantstack/learning/drift_detector.py` (MODIFY)

Integrate Phase 3 monitoring into the existing drift detector. The drift detector already computes PSI for model features. Extend it to also monitor curated feature candidates.

Add a function (or extend an existing one) that:

1. Loads all features with status `"curated"` or `"active"` from the `feature_candidates` table.
2. For each feature, computes PSI vs. its distribution at screening time.
3. Tracks rolling IC per feature over the last 10 trading days.
4. Flags a feature as decayed if:
   - PSI > 0.25 (CRITICAL distribution shift), OR
   - IC < 0.005 for 10 consecutive trading days (sustained loss of predictive power)
5. On decay detection:
   - Update the feature's status to `"decayed"` and set `decay_date`.
   - Select the replacement: the highest-IC unused candidate from the screening pool that does not have >0.95 correlation with any remaining curated feature.
   - Update the replacement's status to `"curated"`.
   - Publish `FEATURE_DECAYED` event with payload: `{feature_id, psi, ic_current}`.
   - Publish `FEATURE_REPLACED` event with payload: `{old_feature_id, new_feature_id}`.

The event publishing uses the existing `event_bus.publish()` interface. The new event types (`FEATURE_DECAYED`, `FEATURE_REPLACED`) are defined in section-05-event-bus-extensions.

## Scheduling

- **Enumeration + Screening (Phase 1 + 2)**: Runs weekly during the overnight window (e.g., Saturday night as part of the weekend research runner, or a dedicated weekly slot). This is compute-intensive but infrequent.
- **Monitoring (Phase 3)**: Runs daily as part of the existing drift detection cycle in `learning/drift_detector.py`. This is lightweight -- just PSI and IC computations on ~100 features.

## Key Design Decisions

**IC > 0.01 threshold is intentionally low.** An IC of 0.01 is barely predictive on its own, but the feature factory's job is to cast a wide net. Downstream consumers (the research graph, backtest validation, ML models) will further filter. The factory optimizes for recall, not precision.

**0.95 correlation cutoff is standard in factor research.** It prevents multicollinearity in downstream models without being so aggressive that it eliminates useful feature variants. A stricter cutoff (e.g., 0.80) would reduce the feature set too aggressively.

**2000 candidate hard cap prevents runaway compute.** Cross-interactions are O(N^2) in base features. With 20 base features, lags, rolling stats, and cross-interactions can produce 5000+ candidates. Computing IC for all of them on 2 years of daily data across 50+ symbols is expensive on home hardware. The 2000 cap, with novelty-based truncation, keeps wall-clock time manageable.

**LLM fallback to programmatic-only is non-negotiable.** The pipeline must run to completion even if Haiku is unavailable. Programmatic enumeration alone produces hundreds of candidates -- sufficient for the screening phase. LLM-assisted features are a bonus, not a requirement.

**PSI > 0.25 and IC < 0.005 for 10 days are independent decay triggers.** PSI catches distribution shift (the feature's values have changed meaning). IC decay catches loss of predictive power even when the distribution is stable. Both are actionable and warrant replacement.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/research/feature_factory.py` | NEW | Top-level orchestrator for all 3 phases |
| `src/quantstack/research/feature_enumerator.py` | NEW | Phase 1: programmatic + LLM-assisted feature generation |
| `src/quantstack/research/feature_screener.py` | NEW | Phase 2: IC screening + correlation deduplication |
| `src/quantstack/learning/drift_detector.py` | MODIFY | Phase 3: daily feature monitoring + decay replacement |
| `tests/unit/test_feature_factory.py` | NEW | All unit tests for the 3-phase pipeline |
