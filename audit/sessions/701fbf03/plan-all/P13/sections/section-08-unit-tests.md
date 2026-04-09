# Section 08: Unit Tests

## Objective

Comprehensive test suite for the entire P13 causal alpha discovery system. Tests are organized by module and emphasize edge cases, failure modes, and integration correctness over happy-path coverage.

## Dependencies

- Section 01 (Causal Discovery Engine)
- Section 02 (Treatment Effects)
- Section 03 (Robustness Checks)
- Section 05 (Causal Signal Collector)
- Section 06 (Counterfactual Analysis)

## Files to Create

### `tests/unit/core/causal/__init__.py`
- Package init for causal test module.

### `tests/unit/core/causal/test_discovery.py`
- **`test_known_dag_recovery`**: Generate synthetic data from A -> B -> C (with noise). Run PC algorithm. Verify A->B and B->C edges discovered, and A->C is correctly identified as non-direct.
- **`test_independent_features_no_edges`**: Generate independent random features. Verify empty graph (no spurious edges).
- **`test_domain_validation_all_confirmed`**: Provide a graph that matches all priors. Verify `domain_agreement_score = 1.0`.
- **`test_domain_validation_partial_match`**: Provide a graph missing some expected edges. Verify correct agreement score and missing_edges list.
- **`test_empty_dataframe`**: Pass empty DataFrame. Verify graceful return (empty graph, no crash).
- **`test_single_feature`**: Only one feature column. Verify no crash, trivial graph.
- **`test_constant_column`**: Feature matrix with an all-constant column. Verify it's excluded from graph.
- **`test_graph_serialization_roundtrip`**: Build graph, convert to JSON adjacency list, reconstruct, verify equality.

### `tests/unit/core/causal/test_treatment_effects.py`
- **`test_dml_known_effect`**: Synthetic data with 2% treatment effect + confounders. Verify `LinearDML` ATE is within 95% CI of 0.02.
- **`test_dml_no_effect`**: Treatment has no causal effect (correlated through confounder only). Verify ATE CI includes 0.
- **`test_psm_balance_check`**: After matching, verify all covariates have SMD < 0.1.
- **`test_psm_caliper_exclusion`**: Some treated units have no good match. Verify they're excluded and n_treated reflects actual matched count.
- **`test_cate_heterogeneity`**: Treatment effect = 5% for large-cap, 0% for small-cap. Verify CausalForestDML detects the heterogeneity.
- **`test_insufficient_treated_units`**: Fewer than 30 treated observations. Verify warning and graceful degradation.
- **`test_continuous_treatment`**: Continuous treatment variable (not binary). Verify DML handles it correctly.
- **`test_dml_psm_consistency`**: Well-specified data. Verify DML and PSM produce ATEs within 1 SE of each other.

### `tests/unit/core/causal/test_robustness.py`
- **`test_placebo_rejects_spurious`**: Correlated but non-causal relationship. Verify placebo p-value > 0.05 (permuted effects are as large as real effect).
- **`test_placebo_passes_genuine`**: Genuine causal effect. Verify placebo p-value < 0.05.
- **`test_random_cause_robust`**: Genuine effect survives random confounder addition (< 30% change).
- **`test_random_cause_fragile`**: Fragile effect where adding noise changes ATE by > 30%. Verify test fails.
- **`test_subset_stable`**: Genuine effect is stable across subsets (CV < 0.5).
- **`test_subset_unstable`**: Effect exists only in a specific subgroup. Verify CV > 0.5.
- **`test_run_all_all_pass`**: Genuine effect passes all three. Verify `overall_passed = True`.
- **`test_run_all_one_fails`**: One refutation fails. Verify `overall_passed = False`.

### `tests/unit/core/causal/test_factor_library.py`
- **`test_register_and_retrieve`**: Register a factor, retrieve by ID, verify fields match.
- **`test_lifecycle_transitions`**: discovered -> validated -> active -> retired. Verify each step.
- **`test_activation_gate`**: Factor with regime_stability = 0.5. Verify activation fails. Set to 0.8, verify activation succeeds.
- **`test_get_active_factors_ordering`**: Register factors with different ATEs. Verify returned in descending ATE magnitude order.
- **`test_uniqueness_upsert`**: Register same factor_name twice with different ATEs. Verify single record with updated values.
- **`test_retire_with_reason`**: Retire a factor, verify reason is stored.

### `tests/unit/core/causal/test_counterfactual.py`
- **`test_perfect_synthetic_control`**: Target = 0.5*A + 0.5*B exactly. Verify weights recovered within tolerance.
- **`test_known_treatment_effect`**: Post-treatment divergence of exactly 5%. Verify causal_alpha close to 0.05.
- **`test_poor_fit_flagged`**: Control pool cannot match target. Verify R-squared < 0.8 detected.
- **`test_single_control_stock`**: Only one peer available. Verify weight = 1.0, still works.
- **`test_short_pre_period`**: Pre-period < 20 days. Verify warning logged but computation proceeds.
- **`test_missing_data_handling`**: Some control stocks have gaps. Verify they're excluded gracefully.

### `tests/unit/signal_engine/test_causal_collector.py`
- **`test_empty_library_returns_empty`**: No active factors. Verify `{}` returned.
- **`test_single_active_factor`**: One active insider_buy factor. Symbol has recent insider buy. Verify non-zero causal_signal.
- **`test_no_treatment_present`**: Active factor exists but treatment condition not met for symbol. Verify causal_signal = 0 or not present.
- **`test_multi_factor_aggregation`**: Three active factors, mixed signs. Verify correct aggregation and clipping to [-1, 1].
- **`test_exception_returns_empty`**: Force an exception (e.g., DB down). Verify `{}` returned, warning logged.
- **`test_confidence_computation`**: Verify causal_confidence is weighted average of factor confidences.

### `tests/unit/tools/test_causal_tools.py`
- **`test_discover_tool_valid_input`**: Call with valid features. Verify structured output with adjacency list.
- **`test_estimate_tool_registers_factor`**: Call estimate_treatment_effect with valid data. Verify factor appears in library.
- **`test_counterfactual_tool_invalid_trade`**: Call with non-existent trade_id. Verify graceful error message.
- **`test_list_factors_filter`**: Register factors with various statuses. Verify filter works correctly.
- **`test_tools_in_registry`**: Verify all four tools appear in `TOOL_REGISTRY`.

## Implementation Details

1. **Synthetic Data Generation**: Create a `tests/unit/core/causal/conftest.py` with shared fixtures:
   - `make_causal_data(n=1000, ate=0.02, n_confounders=3)` -> DataFrame with known causal structure.
   - `make_spurious_data(n=1000)` -> DataFrame with correlation but no causation.
   - `make_heterogeneous_data(n=1000, ate_group1=0.05, ate_group2=0.0)` -> DataFrame with heterogeneous effects.

2. **DB Mocking**: Use a test PostgreSQL database or mock `db_conn()` for factor library tests. Prefer real DB tests where possible (the project uses PostgreSQL).

3. **Library Mocking**: For collector tests, mock `CausalFactorLibrary` to return controlled factor lists. Do not require DoWhy/EconML for collector tests.

4. **Test Organization**: Each test file corresponds to one section's module. Fixtures in `conftest.py` are shared. Parametrize where it adds value (e.g., different treatment effect sizes, different sample sizes).

5. **Performance**: Refutation tests are slow by design (many re-estimations). Mark them with `@pytest.mark.slow` so they can be excluded from CI fast path.

## Test Requirements

This IS the test section. All tests listed above must pass.

## Acceptance Criteria

- [ ] All test files created with the tests enumerated above
- [ ] `conftest.py` provides shared synthetic data fixtures
- [ ] Slow tests (refutation) marked with `@pytest.mark.slow`
- [ ] All tests pass: `uv run pytest tests/unit/core/causal/ tests/unit/signal_engine/test_causal_collector.py tests/unit/tools/test_causal_tools.py -v`
- [ ] No test depends on external services (DoWhy/EconML computations use synthetic data, DB uses test instance or mocks)
- [ ] Edge cases covered: empty data, insufficient samples, missing dependencies, DB failures
- [ ] Test coverage for the causal subsystem exceeds 85%
