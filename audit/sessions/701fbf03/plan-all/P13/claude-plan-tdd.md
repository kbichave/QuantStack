# P13 TDD Plan: Causal Alpha Discovery

**Testing framework:** pytest (existing codebase)
**Test locations:** `tests/unit/causal/`, `tests/integration/`
**Fixtures:** DB mocking via `monkeypatch`, `MagicMock` for DoWhy/EconML models, synthetic datasets with known causal structure
**Key libraries under test:** DoWhy, EconML, CausalML

---

## Section 1: Causal Graph Builder (`core/causal/discovery.py`)

```python
# tests/unit/causal/test_discovery.py

import pytest
import pandas as pd
import numpy as np


class TestCausalGraphBuilder:
    """Tests for PC-algorithm-based causal discovery."""

    def test_recovers_known_dag_from_synthetic_data(self):
        """Generate data from A->B->C structure, verify PC algorithm recovers edges."""

    def test_returns_empty_dag_for_independent_features(self):
        """All features are independent random noise -- DAG should have no edges."""

    def test_handles_single_feature_input(self):
        """Edge case: only one feature column + returns -- no possible edges."""

    def test_handles_missing_values_in_feature_matrix(self):
        """NaN values in feature matrix should raise ValueError, not silently produce wrong DAG."""

    def test_handles_constant_feature_column(self):
        """A feature with zero variance should be excluded from discovery, not crash."""

    def test_dag_serialization_roundtrip(self):
        """Serialize DAG to JSONB adjacency list and deserialize -- graph structure identical."""

    def test_dag_stored_to_causal_graphs_table(self, mock_db):
        """After discovery, DAG is persisted to causal_graphs table with correct schema."""

    def test_domain_validation_flags_implausible_edges(self):
        """If discovered DAG has return->earnings_revision edge (wrong direction), flag it."""

    def test_large_feature_matrix_does_not_timeout(self):
        """50 features x 1000 rows completes within 60 seconds (guards against combinatorial blow-up)."""

    def test_deterministic_output_with_fixed_seed(self):
        """Same input + same seed produces identical DAG."""
```

---

## Section 2: Treatment Effect Estimation (`core/causal/treatment_effects.py`)

```python
# tests/unit/causal/test_treatment_effects.py

import pytest
import numpy as np


class TestDoubleMachineLearning:
    """Tests for DML-based ATE and CATE estimation."""

    def test_ate_recovers_known_effect_synthetic(self):
        """Synthetic data with true ATE=0.05: estimated ATE within 95% CI of true value."""

    def test_ate_is_zero_when_no_causal_effect(self):
        """Treatment is random noise, outcome is independent -- ATE CI should contain 0."""

    def test_cate_varies_across_subgroups(self):
        """Synthetic data where treatment effect differs by sector -- CATE captures heterogeneity."""

    def test_binary_treatment_accepted(self):
        """Binary treatment (insider_buy=0/1) runs without error."""

    def test_continuous_treatment_accepted(self):
        """Continuous treatment (earnings_revision_pct) runs without error."""

    def test_output_contains_required_fields(self):
        """Output dict has ate, ate_ci_lower, ate_ci_upper, ate_p_value keys."""

    def test_empty_treatment_group_raises(self):
        """All treatment=0 (no treated units) raises ValueError."""

    def test_single_confounder_works(self):
        """DML runs with just one confounder column."""

    def test_highly_collinear_confounders_handled(self):
        """Near-duplicate confounders do not crash DML (regularization should handle)."""


class TestPropensityScoreMatching:
    """Tests for PSM fallback method."""

    def test_matched_pairs_have_balanced_covariates(self):
        """After matching, standardized mean differences across covariates all < 0.1."""

    def test_caliper_rejects_poor_matches(self):
        """With caliper=0.05, treated units with no close control match are excluded."""

    def test_small_sample_returns_warning(self):
        """Fewer than 30 matched pairs triggers a low-power warning."""

    def test_all_treated_unmatched_raises(self):
        """If caliper excludes all treated units, raise InsufficientMatchesError."""


class TestRefutationChecks:
    """Tests for DoWhy refutation pipeline."""

    def test_placebo_catches_spurious_correlation(self):
        """Treatment is randomized: placebo refutation p-value should be > 0.05 (effect vanishes)."""

    def test_random_common_cause_preserves_real_effect(self):
        """True causal effect persists after adding random confounder (effect change < 30%)."""

    def test_subset_validation_stable_for_real_effect(self):
        """ATE on 80% subset within 30% of full-sample ATE."""

    def test_failed_refutation_marks_factor_unvalidated(self):
        """If placebo p < 0.05, factor status set to 'unvalidated'."""

    def test_all_three_refutations_run(self):
        """Output contains results for placebo, random_common_cause, and subset."""

    def test_refutation_on_tiny_dataset_does_not_crash(self):
        """50-row dataset: refutation either runs or raises clear error, no silent failure."""
```

---

## Section 3: Causal Factor Library (`causal_factors` table + lifecycle)

```python
# tests/unit/causal/test_causal_factor_library.py

import pytest


class TestCausalFactorSchema:
    """Tests for causal_factors table writes and reads."""

    def test_insert_validated_factor(self, mock_db):
        """Insert factor with status='validated' -- all required columns populated."""

    def test_status_transitions_are_valid(self):
        """Only allowed transitions: discovered->validated->active->retired. Others raise."""

    def test_regime_stability_score_range(self):
        """regime_stability_score must be in [0.0, 1.0]."""

    def test_ate_confidence_interval_ordering(self):
        """ate_ci_lower <= ate <= ate_ci_upper enforced."""

    def test_duplicate_factor_name_horizon_raises(self, mock_db):
        """Same factor_name + outcome_horizon_days is a unique constraint violation."""

    def test_retire_factor_preserves_history(self, mock_db):
        """Retiring a factor sets status='retired' but does not delete the row."""
```

---

## Section 4: Causal Signal Collector (`signal_engine/collectors/causal.py`)

```python
# tests/unit/causal/test_causal_collector.py

import pytest
from unittest.mock import MagicMock


class TestCausalSignalCollector:
    """Tests for causal signal collector integration with SignalBrief."""

    def test_active_factor_produces_signal(self, mock_db):
        """Active causal factor with positive CATE emits bullish causal_signal."""

    def test_no_active_factors_returns_neutral(self, mock_db):
        """No active factors for symbol: causal_signal is 0.0 / neutral."""

    def test_signal_weight_formula(self):
        """Weight = ATE magnitude x regime_stability x refutation_confidence. Verify calculation."""

    def test_retired_factor_excluded(self, mock_db):
        """Factor with status='retired' is not included in signal computation."""

    def test_unvalidated_factor_excluded(self, mock_db):
        """Factor with status='unvalidated' (failed refutation) is not included."""

    def test_multiple_factors_aggregated(self, mock_db):
        """Three active factors: signal is weighted sum of individual CATEs."""

    def test_initial_synthesis_weight_is_005(self):
        """Default synthesis weight for causal collector is 0.05."""

    def test_collector_handles_db_failure_gracefully(self, mock_db):
        """DB read failure returns neutral signal with logged warning, not exception."""

    def test_output_matches_symbol_brief_schema(self, mock_db):
        """Output includes causal_signal field compatible with SymbolBrief."""
```

---

## Section 5: Counterfactual Analysis (`core/causal/counterfactual.py`)

```python
# tests/unit/causal/test_counterfactual.py

import pytest
import numpy as np


class TestSyntheticControl:
    """Tests for post-trade synthetic control attribution."""

    def test_synthetic_control_recovers_known_alpha(self):
        """Stock returned 10%, synthetic control returned 6%: causal_alpha = 4%."""

    def test_no_similar_stocks_raises(self):
        """If no non-traded stocks in universe are similar, raise InsufficientControlsError."""

    def test_weights_sum_to_one(self):
        """Synthetic control weights across donor stocks sum to 1.0."""

    def test_weights_are_nonnegative(self):
        """No donor stock gets a negative weight."""

    def test_pre_trade_fit_is_close(self):
        """Synthetic control tracks actual stock within 2% RMSE in pre-trade window."""

    def test_single_donor_stock_works(self):
        """Edge case: only one donor stock available -- weight is 1.0 for that stock."""

    def test_counterfactual_return_stored_on_trade_outcome(self, mock_db):
        """After analysis, trade outcome row updated with counterfactual_return and causal_alpha."""


class TestTradeJournalIntegration:
    """Tests for counterfactual fields in trade outcome logging."""

    def test_causal_alpha_positive_for_outperformance(self, mock_db):
        """Trade beat synthetic control: causal_alpha > 0."""

    def test_causal_alpha_negative_for_underperformance(self, mock_db):
        """Trade underperformed synthetic control: causal_alpha < 0."""

    def test_missing_counterfactual_defaults_to_null(self, mock_db):
        """If counterfactual analysis not available, fields are NULL (not zero)."""
```

---

## Section 6: Research Graph Integration

```python
# tests/unit/causal/test_causal_tools.py

import pytest
from unittest.mock import MagicMock


class TestCausalHypothesisTools:
    """Tests for LLM-facing causal tools in research graph."""

    def test_discover_causal_graph_returns_dag(self):
        """discover_causal_graph tool returns serialized DAG with nodes and edges."""

    def test_estimate_treatment_effect_returns_ate(self):
        """estimate_treatment_effect tool returns ATE + CI + p-value."""

    def test_run_counterfactual_returns_attribution(self):
        """run_counterfactual tool returns counterfactual_return and causal_alpha."""

    def test_invalid_trade_id_returns_error_message(self):
        """run_counterfactual with non-existent trade_id returns user-friendly error."""

    def test_hypothesis_template_fields_populated(self):
        """Generated hypothesis has treatment, outcome, mechanism, confounders, accept_if."""
```

---

## Section 7: Integration Test

```python
# tests/integration/test_causal_pipeline_e2e.py

import pytest


class TestCausalPipelineEndToEnd:
    """End-to-end: discovery -> estimation -> refutation -> signal -> counterfactual."""

    def test_full_pipeline_synthetic_data(self, mock_db):
        """
        Synthetic data with known causal structure:
        1. Discovery recovers correct DAG
        2. DML estimates ATE within CI of true value
        3. Refutations all pass
        4. Factor stored as 'validated'
        5. Collector produces non-zero signal
        6. After simulated trade close, counterfactual attribution computed
        """

    def test_spurious_factor_rejected_by_pipeline(self, mock_db):
        """
        Synthetic data with correlation but no causation:
        1. DML may estimate non-zero ATE
        2. Placebo refutation catches it (p < 0.05)
        3. Factor marked 'unvalidated'
        4. Collector excludes it from signal
        """

    def test_regime_transition_stability(self, mock_db):
        """
        Factor estimated in regime A re-estimated in regime B:
        - If ATE changes > 50%, regime_stability_score reduced
        - If factor survives 2 regime transitions, regime_stability > 0.7
        """
```

---

## Test Execution Order

For TDD, write and verify tests in this order:

1. **Section 1 tests** -- Causal graph discovery (synthetic data, no DB needed for core logic)
2. **Section 2 tests** -- Treatment effects + refutation (pure computation, synthetic data)
3. **Section 3 tests** -- Factor library schema (DB mocks)
4. **Section 4 tests** -- Collector (mocks factors, no real inference)
5. **Section 5 tests** -- Counterfactual (synthetic data + DB mock for storage)
6. **Section 6 tests** -- Research tools (mock underlying engines)
7. **Section 7 tests** -- End-to-end (requires all sections implemented)
