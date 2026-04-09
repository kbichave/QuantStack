<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-schema-migrations
section-02-ic-weight-precompute  depends_on:section-01-schema-migrations
section-03-transition-zone  depends_on:section-01-schema-migrations
section-04-conviction-calibration  depends_on:section-01-schema-migrations
section-05-ab-test-tracking  depends_on:section-01-schema-migrations
section-06-synthesis-integration  depends_on:section-02-ic-weight-precompute,section-03-transition-zone,section-04-conviction-calibration,section-05-ab-test-tracking
section-07-scheduler-jobs  depends_on:section-02-ic-weight-precompute,section-04-conviction-calibration,section-05-ab-test-tracking
section-08-unit-tests  depends_on:section-02-ic-weight-precompute,section-03-transition-zone,section-04-conviction-calibration,section-05-ab-test-tracking
section-09-integration-test  depends_on:section-06-synthesis-integration,section-07-scheduler-jobs,section-08-unit-tests
END_MANIFEST -->

# P05 Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-schema-migrations | - | 02, 03, 04, 05 | Yes |
| section-02-ic-weight-precompute | 01 | 06, 07, 08 | Yes (with 03, 04, 05) |
| section-03-transition-zone | 01 | 06, 08 | Yes (with 02, 04, 05) |
| section-04-conviction-calibration | 01 | 06, 07, 08 | Yes (with 02, 03, 05) |
| section-05-ab-test-tracking | 01 | 06, 07, 08 | Yes (with 02, 03, 04) |
| section-06-synthesis-integration | 02, 03, 04, 05 | 09 | No |
| section-07-scheduler-jobs | 02, 04, 05 | 09 | Yes (with 06, 08) |
| section-08-unit-tests | 02, 03, 04, 05 | 09 | Yes (with 06, 07) |
| section-09-integration-test | 06, 07, 08 | - | No |

## Execution Order

1. section-01-schema-migrations (no dependencies)
2. section-02-ic-weight-precompute, section-03-transition-zone, section-04-conviction-calibration, section-05-ab-test-tracking (parallel after 01)
3. section-06-synthesis-integration, section-07-scheduler-jobs, section-08-unit-tests (parallel after 02-05)
4. section-09-integration-test (final)

## Section Summaries

### section-01-schema-migrations
Add 4 new tables: precomputed_ic_weights, conviction_calibration, ensemble_ab_results, ensemble_config. Idempotent CREATE TABLE IF NOT EXISTS.

### section-02-ic-weight-precompute
New function compute_and_store_ic_weights() in ic_attribution.py. Queries IC observations per regime, applies gate/penalty/floor, normalizes, upserts into precomputed_ic_weights.

### section-03-transition-zone
Add transition_zone boolean to SymbolBrief schema. Set in synthesis.py when P(transition) > 0.3. Apply 0.5× position scalar in nodes.py when transition_zone=True.

### section-04-conviction-calibration
Persist conviction_factors in signals.metadata. New calibrate_conviction_factors() function. Read calibrated params in _conviction_multiplicative().

### section-05-ab-test-tracking
Record ensemble method + signal_value in ensemble_ab_results table. New evaluate_ensemble_ab() comparison job. ensemble_config for active method selection.

### section-06-synthesis-integration
Replace ICAttributionTracker() instantiation with precomputed weight lookup. Wire all new features into synthesis flow. Wire weight_floor safety check.

### section-07-scheduler-jobs
Add weekly_ic_weight_precompute, quarterly_conviction_calibration, weekly_ensemble_ab_evaluate to scheduler.

### section-08-unit-tests
Unit tests for all new functions: precompute, transition zone, calibration, A/B tracking.

### section-09-integration-test
End-to-end test: seed IC data → precompute → synthesize → verify weights/transition/factors/AB recorded.
