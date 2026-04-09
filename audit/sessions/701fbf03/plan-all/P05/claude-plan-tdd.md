# P05 TDD Plan: Adaptive Signal Synthesis

Testing framework: pytest (existing). Test location: `tests/unit/`. DB mocking: monkeypatch `db_conn`.

## Section 3: IC Weight Precomputation

### Unit Tests (test_ic_weight_precompute.py)

```python
# Test: compute_and_store_ic_weights produces correct weights for regime with sufficient data
# Test: compute_and_store_ic_weights skips regime with <60 days of IC observations
# Test: compute_and_store_ic_weights applies IC gate (drops collectors with IC < 0.02)
# Test: compute_and_store_ic_weights applies ICIR penalty (0.7x for ICIR < 0.1)
# Test: compute_and_store_ic_weights checks weight floor (skips if total < 0.1)
# Test: compute_and_store_ic_weights normalizes weights to sum=1.0
# Test: precomputed weights upserted correctly (no duplicates on re-run)
# Test: synthesis reads precomputed weights instead of instantiating ICAttributionTracker
# Test: synthesis falls back to static when precomputed weights >7 days old
# Test: synthesis falls back to static when no precomputed rows for regime
```

## Section 4: Transition Zone Position Sizing

### Unit Tests (test_transition_zone.py)

```python
# Test: SymbolBrief.transition_zone defaults to False
# Test: synthesis sets transition_zone=True when transition_probability > 0.3
# Test: synthesis sets transition_zone=False when transition_probability <= 0.3
# Test: synthesis sets transition_zone=False when transition_probability is None
# Test: position sizing applies 0.5x scalar when transition_zone=True
# Test: position sizing unchanged when transition_zone=False
# Test: transition_position_sizing_enabled flag controls feature
```

## Section 5: Conviction Factor Calibration

### Unit Tests (test_conviction_calibration.py)

```python
# Test: conviction_factors added to signals.metadata JSONB
# Test: calibrate_conviction_factors produces correct params with synthetic data
# Test: calibrate_conviction_factors returns defaults when <100 trades
# Test: calibrate_conviction_factors returns defaults when R² < 0.01
# Test: _conviction_multiplicative uses calibrated params from conviction_calibration table
# Test: _conviction_multiplicative falls back to hardcoded when no calibration data
```

## Section 6: A/B Test Result Tracking

### Unit Tests (test_ensemble_ab.py)

```python
# Test: ensemble method name recorded in ensemble_ab_results on each synthesis
# Test: evaluate_ensemble_ab computes per-method IC correctly
# Test: evaluate_ensemble_ab promotes method with p < 0.05 improvement
# Test: evaluate_ensemble_ab keeps default when no significant winner
# Test: synthesis reads ensemble_config to select active method
# Test: synthesis falls back to weighted_avg when ensemble_config empty
```

## Section 8: Schema Migrations

```python
# Test: precomputed_ic_weights table created idempotently
# Test: conviction_calibration table created idempotently
# Test: ensemble_ab_results table created idempotently
# Test: ensemble_config table created idempotently
```

## Section 9: Integration Test

```python
# Test: end-to-end IC→weights→synthesis flow uses precomputed weights
# Test: transition_zone=True propagates to position sizing reduction
# Test: conviction factors appear in signals.metadata after synthesis
# Test: ensemble method recorded in ensemble_ab_results after synthesis
```
