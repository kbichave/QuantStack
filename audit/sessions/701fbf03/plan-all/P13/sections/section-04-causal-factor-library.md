# Section 04: Causal Factor Library

## Objective

Create the persistent storage and lifecycle management for validated causal factors. This is the registry where discovered, validated, and active causal factors live, along with their treatment effect estimates and refutation results.

## Dependencies

- Section 03 (Robustness Checks) -- factors enter the library only after passing refutation.

## Files to Create

### `src/quantstack/core/causal/factor_library.py`
- **Class `CausalFactorLibrary`**: manages the lifecycle of causal factors (discovered -> validated -> active -> retired).
- **Method `register_factor(factor: CausalFactorRecord) -> int`**:
  - Insert a new factor into the `causal_factors` table with status `discovered`.
  - Returns the factor ID.
- **Method `validate_factor(factor_id: int, refutation_report: FullRefutationReport) -> bool`**:
  - If `refutation_report.overall_passed`, update status to `validated`.
  - Store refutation scores (placebo_p, subset_cv) in the record.
  - Return True if validated, False if refutation failed.
- **Method `activate_factor(factor_id: int, regime_stability_score: float) -> bool`**:
  - Promote validated factor to `active` if `regime_stability_score >= 0.7`.
  - Only active factors are used by the signal collector.
- **Method `retire_factor(factor_id: int, reason: str) -> None`**:
  - Mark factor as `retired` with reason. No longer used in signal generation.
- **Method `get_active_factors() -> list[CausalFactorRecord]`**:
  - Return all factors with status `active`, ordered by ATE magnitude descending.
- **Method `get_factor(factor_id: int) -> CausalFactorRecord | None`**:
  - Retrieve single factor by ID.
- **Method `update_regime_stability(factor_id: int, score: float) -> None`**:
  - Update regime stability score after regime transition evaluation.

### `src/quantstack/core/causal/models.py` (extend)
- **Dataclass `CausalFactorRecord`**:
  - `factor_id: int | None`
  - `factor_name: str` (e.g., "insider_buy_60d_return")
  - `treatment_variable: str`
  - `outcome_variable: str`
  - `outcome_horizon_days: int`
  - `ate: float`
  - `ate_ci_lower: float`
  - `ate_ci_upper: float`
  - `ate_p_value: float`
  - `method: str` (dml_linear, dml_forest, psm)
  - `confounders: list[str]`
  - `refutation_placebo_p: float | None`
  - `refutation_subset_cv: float | None`
  - `refutation_random_cause_pct: float | None`
  - `regime_stability_score: float` (0-1, survives how many regime transitions)
  - `status: Literal["discovered", "validated", "active", "retired"]`
  - `created_at: datetime`
  - `updated_at: datetime`
  - `retire_reason: str | None`

## Files to Modify

### `src/quantstack/db.py`
- Add `causal_factors` table DDL within `ensure_tables()`:
  ```sql
  CREATE TABLE IF NOT EXISTS causal_factors (
      id SERIAL PRIMARY KEY,
      factor_name TEXT NOT NULL UNIQUE,
      treatment_variable TEXT NOT NULL,
      outcome_variable TEXT NOT NULL,
      outcome_horizon_days INTEGER NOT NULL,
      ate REAL NOT NULL,
      ate_ci_lower REAL NOT NULL,
      ate_ci_upper REAL NOT NULL,
      ate_p_value REAL NOT NULL,
      method TEXT NOT NULL,
      confounders TEXT[] NOT NULL DEFAULT '{}',
      refutation_placebo_p REAL,
      refutation_subset_cv REAL,
      refutation_random_cause_pct REAL,
      regime_stability_score REAL NOT NULL DEFAULT 0.0,
      status TEXT NOT NULL DEFAULT 'discovered',
      retire_reason TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS idx_causal_factors_status ON causal_factors(status);
  CREATE INDEX IF NOT EXISTS idx_causal_factors_treatment ON causal_factors(treatment_variable);
  ```

## Implementation Details

1. **Status Lifecycle**: `discovered` -> `validated` -> `active` -> `retired`. Only forward transitions are allowed (no re-activating retired factors; create a new one instead).

2. **Regime Stability Score**: Computed externally (by running the factor through historical regime transitions). Score = fraction of regime transitions where the ATE remains significant (p < 0.05). Must be >= 0.7 for activation. Updated periodically by the research graph.

3. **Factor Naming Convention**: `{treatment}_{horizon}d_{outcome}` (e.g., `insider_buy_60d_return`, `earnings_revision_30d_return`).

4. **DB Access Pattern**: All methods use `db_conn()` context manager from `quantstack.db`. Reads use `SELECT`, writes use `INSERT ... ON CONFLICT` or `UPDATE ... WHERE id = %s`.

5. **Uniqueness**: Factor name is unique. Re-running discovery for the same treatment/outcome/horizon replaces the existing factor (UPDATE, not INSERT duplicate).

## Test Requirements

- **CRUD lifecycle**: Register a factor (discovered), validate it, activate it, retire it. Verify status transitions at each step.
- **Activation gate**: Verify that factors with regime_stability_score < 0.7 cannot be activated.
- **get_active_factors**: Register 5 factors with various statuses. Verify only active ones are returned, ordered by ATE magnitude.
- **Uniqueness constraint**: Register same factor_name twice. Verify upsert behavior (second call updates, not duplicates).
- **Invalid transitions**: Verify that retiring a "discovered" factor works but re-activating a retired one raises an error.

## Acceptance Criteria

- [ ] `causal_factors` table created by `ensure_tables()` with correct schema and indices
- [ ] Full lifecycle works: register -> validate -> activate -> retire
- [ ] Activation gate enforces `regime_stability_score >= 0.7`
- [ ] `get_active_factors()` returns only active factors ordered by ATE magnitude
- [ ] Factor name uniqueness is enforced at DB level
- [ ] All DB operations use `db_conn()` context manager
- [ ] Tests pass: `uv run pytest tests/unit/core/causal/test_factor_library.py`
