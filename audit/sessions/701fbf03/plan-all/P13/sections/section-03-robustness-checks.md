# Section 03: Robustness Checks

## Objective

Implement the three mandatory refutation tests that every causal estimate must pass before being considered validated. This is the quality gate that prevents spurious correlations from being treated as causal.

## Dependencies

- Section 02 (Treatment Effects) -- robustness checks wrap treatment effect results.

## Files to Create

### `src/quantstack/core/causal/robustness.py`
- **Class `CausalRefuter`**: runs all refutation tests on a treatment effect estimate.
- **Method `refute_placebo(estimator: TreatmentEffectEstimator, treatment: str, outcome: str, confounders: list[str], data: pd.DataFrame, n_permutations: int = 100) -> RefutationResult`**:
  - Randomize treatment assignment (permutation test).
  - Re-estimate ATE on permuted data.
  - Effect should vanish: p-value of permuted ATE distribution should be > 0.05 (i.e., the real ATE is outside the permuted distribution).
  - Returns: permuted_ate_mean, permuted_ate_std, real_ate, p_value, passed (bool).
- **Method `refute_random_common_cause(estimator: TreatmentEffectEstimator, treatment: str, outcome: str, confounders: list[str], data: pd.DataFrame, n_trials: int = 10) -> RefutationResult`**:
  - Add a random confounder (standard normal) to the model.
  - Re-estimate ATE with the added confounder.
  - Effect should persist: ATE should not change by more than 30%.
  - Returns: original_ate, augmented_ate_mean, pct_change, passed (bool).
- **Method `refute_subset(estimator: TreatmentEffectEstimator, treatment: str, outcome: str, confounders: list[str], data: pd.DataFrame, subset_fraction: float = 0.8, n_subsets: int = 20) -> RefutationResult`**:
  - Estimate ATE on random 80% subsets.
  - Effect should be stable: coefficient of variation across subsets < 0.5.
  - Returns: subset_ates (list), cv, mean_ate, passed (bool).
- **Method `run_all(estimator, treatment, outcome, confounders, data) -> FullRefutationReport`**:
  - Run all three refutations.
  - Returns consolidated report with overall_passed flag.
  - A factor is "validated" only if all three pass; otherwise marked "unvalidated".

### `src/quantstack/core/causal/models.py` (extend)
- **Dataclass `RefutationResult`**: test_name, passed (bool), details (dict), p_value (optional float), effect_change_pct (optional float).
- **Dataclass `FullRefutationReport`**: placebo, random_common_cause, subset (all RefutationResult), overall_passed (bool), summary (str).

## Implementation Details

1. **Placebo Treatment Test**:
   - Shuffle the treatment column `n_permutations` times.
   - For each permutation, estimate ATE using the same DML pipeline.
   - Compute the p-value: fraction of permuted ATEs >= real ATE (one-sided).
   - Pass criterion: p < 0.05 (real effect is unlikely under random assignment).

2. **Random Common Cause Test**:
   - Generate `n_trials` random standard normal columns.
   - Add each as an additional confounder and re-estimate ATE.
   - Compute mean augmented ATE and percentage change from original.
   - Pass criterion: |pct_change| < 30%.

3. **Subset Validation Test**:
   - Sample 80% of rows without replacement, `n_subsets` times.
   - Estimate ATE on each subset.
   - Compute coefficient of variation (std/mean).
   - Pass criterion: CV < 0.5.

4. **Performance Considerations**: Refutation is expensive (100+ re-estimations for placebo). Use `LinearDML` (fast) for refutation even if the original estimate used `CausalForestDML`. Cache intermediate results. Consider `joblib.Parallel` for subset/placebo parallelism.

5. **Failure Semantics**: If any single refutation fails, the factor status becomes `"unvalidated"` in the causal_factors table. It may still be stored for research purposes but will NOT be used by the signal collector (Section 05).

## Test Requirements

- **True causal relationship passes all refutations**: Synthetic data with genuine treatment effect. All three tests should pass.
- **Spurious correlation fails placebo**: Synthetic data where treatment and outcome are correlated only through a hidden confounder (not included). Placebo test should catch this.
- **Unstable effect fails subset test**: Synthetic data where effect exists only in a specific subgroup. Subset test should detect instability.
- **Robust effect survives random common cause**: Verify that adding random noise columns does not break a genuine effect.

## Acceptance Criteria

- [ ] `refute_placebo()` correctly rejects spurious correlations (permutation p-value)
- [ ] `refute_random_common_cause()` verifies robustness to omitted variable bias
- [ ] `refute_subset()` detects unstable effects via coefficient of variation
- [ ] `run_all()` returns consolidated report with correct overall_passed flag
- [ ] Genuine causal effects in synthetic data pass all three refutations
- [ ] Non-causal correlations fail at least one refutation
- [ ] Tests pass: `uv run pytest tests/unit/core/causal/test_robustness.py`
