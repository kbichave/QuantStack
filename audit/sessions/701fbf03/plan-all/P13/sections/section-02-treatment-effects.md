# Section 02: Treatment Effect Estimation

## Objective

Implement Double Machine Learning (DML) and Propensity Score Matching for estimating causal treatment effects of features on forward returns. This module answers: "How much does insider buying *cause* returns to increase, controlling for confounders?"

## Dependencies

- Section 01 (Causal Discovery Engine) -- uses `CausalGraph` to identify confounders and treatment/outcome pairs.

## Files to Create

### `src/quantstack/core/causal/treatment_effects.py`
- **Class `TreatmentEffectEstimator`**: orchestrates treatment effect estimation.
- **Method `estimate_dml(treatment: str, outcome: str, confounders: list[str], data: pd.DataFrame, method: str = "linear") -> TreatmentEffectResult`**:
  - Uses EconML's `LinearDML` (default) or `CausalForestDML` (when `method="forest"`).
  - Treatment: binary (e.g., insider_buy=1/0) or continuous (e.g., earnings_revision_pct).
  - Outcome: forward return (5d, 10d, 30d column).
  - Confounders: sector, market_cap, momentum, volatility, regime.
  - Returns: ATE, ATE confidence interval, p-value, CATE per observation.
- **Method `estimate_psm(treatment: str, outcome: str, covariates: list[str], data: pd.DataFrame, caliper: float = 0.05) -> TreatmentEffectResult`**:
  - Propensity Score Matching fallback when DML assumptions are questionable.
  - Logistic regression for propensity scores.
  - Caliper matching with configurable tolerance (default 0.05).
  - Post-matching covariate balance check: standardized mean differences < 0.1.
- **Method `estimate_cate(treatment: str, outcome: str, confounders: list[str], effect_modifiers: list[str], data: pd.DataFrame) -> CATEResult`**:
  - Conditional Average Treatment Effect per symbol/group.
  - Uses `CausalForestDML` for heterogeneous treatment effect estimation.
  - Returns CATE values indexed by observation + confidence intervals.

### `src/quantstack/core/causal/models.py` (extend from Section 01)
- **Dataclass `TreatmentEffectResult`**: ate, ate_ci_lower, ate_ci_upper, ate_p_value, method (str), treatment, outcome, confounders, n_treated, n_control, covariate_balance (dict).
- **Dataclass `CATEResult`**: cate_values (pd.Series), cate_ci_lower (pd.Series), cate_ci_upper (pd.Series), effect_modifiers, heterogeneity_score.

## Implementation Details

1. **Double Machine Learning (DML)**:
   - Stage 1: Regress outcome on confounders (get residualized outcome).
   - Stage 2: Regress treatment on confounders (get residualized treatment).
   - Stage 3: Regress residualized outcome on residualized treatment.
   - EconML handles cross-fitting automatically (K=5 folds default).
   - Use `LinearDML` for interpretable linear effects, `CausalForestDML` for non-linear heterogeneous effects.

2. **Propensity Score Matching**:
   - Fit logistic regression: P(treatment=1 | covariates).
   - For each treated unit, find nearest control unit within caliper.
   - Discard unmatched units.
   - Verify balance: for each covariate, compute standardized mean difference between matched treated/control. All must be < 0.1.
   - ATE = mean(outcome_treated) - mean(outcome_control) over matched pairs.

3. **Confounders Selection**: When a `CausalGraph` from Section 01 is available, use it to identify the minimal adjustment set (backdoor criterion). When not available, use the default set: `[sector, market_cap_log, momentum_20d, volatility_20d, regime]`.

4. **Error Handling**: If EconML is not installed, raise `ImportError` with installation instructions. If treatment variable has < 30 treated units, return result with `ate=None` and a warning flag.

## Test Requirements

- **Known treatment effect recovery**: Generate synthetic data where treatment has a known 2% effect on outcome. Verify DML recovers ATE within the 95% CI.
- **Binary treatment PSM**: Synthetic data with confounded treatment. Verify PSM produces balanced groups (SMD < 0.1) and ATE estimate is reasonable.
- **CATE heterogeneity**: Synthetic data where treatment effect varies by a modifier (e.g., large-cap vs small-cap). Verify `CausalForestDML` detects heterogeneity.
- **Insufficient data**: Verify graceful handling when fewer than 30 treated units.
- **Method consistency**: For well-specified data, DML and PSM should produce similar ATE estimates (within 1 standard error).

## Acceptance Criteria

- [ ] `estimate_dml()` returns valid `TreatmentEffectResult` with ATE, CI, and p-value
- [ ] `estimate_psm()` produces balanced matched groups and valid ATE
- [ ] `estimate_cate()` detects heterogeneous treatment effects
- [ ] Known synthetic treatment effects are recovered within 95% CI
- [ ] Covariate balance check enforced (SMD < 0.1 threshold)
- [ ] Graceful degradation for small sample sizes
- [ ] Tests pass: `uv run pytest tests/unit/core/causal/test_treatment_effects.py`
