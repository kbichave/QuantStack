# P14 TDD Plan: Advanced ML (Transformers, GNNs, Deep Hedging)

**Testing framework:** pytest (existing codebase)
**Test locations:** `tests/unit/ml/`, `tests/integration/`
**Fixtures:** DB mocking via `monkeypatch`, synthetic time series, mock model checkpoints
**Key libraries under test:** MAPIE, neuralforecast, torch_geometric, HuggingFace transformers

---

## Section 1: Conformal Prediction Wrapper (`ml/conformal.py`)

```python
# tests/unit/ml/test_conformal.py

import pytest
import numpy as np


class TestConformalWrapper:
    """Tests for MAPIE-based conformal prediction on existing models."""

    def test_coverage_guarantee_90pct(self):
        """On held-out synthetic data, 90% CI covers >= 87% of outcomes (3% tolerance)."""

    def test_coverage_guarantee_80pct(self):
        """On held-out synthetic data, 80% CI covers >= 77% of outcomes."""

    def test_coverage_guarantee_95pct(self):
        """On held-out synthetic data, 95% CI covers >= 92% of outcomes."""

    def test_prediction_intervals_are_ordered(self):
        """For every prediction: lower_80 <= lower_90 <= lower_95 <= point <= upper_95 <= upper_90 is wrong --
        actually: lower_95 <= lower_90 <= lower_80 <= point <= upper_80 <= upper_90 <= upper_95."""

    def test_wraps_lightgbm_model(self):
        """Conformal wrapper accepts trained LGBMRegressor and produces intervals."""

    def test_wraps_xgboost_model(self):
        """Conformal wrapper accepts trained XGBRegressor and produces intervals."""

    def test_point_prediction_unchanged(self):
        """Point prediction from conformal wrapper matches underlying model prediction."""

    def test_narrow_ci_for_low_noise_data(self):
        """Near-deterministic synthetic data: average CI width is small (< 0.01)."""

    def test_wide_ci_for_high_noise_data(self):
        """High-variance synthetic data: average CI width is large."""

    def test_empty_calibration_set_raises(self):
        """Fitting conformal wrapper with 0 calibration samples raises ValueError."""

    def test_single_sample_calibration_raises(self):
        """Fitting with 1 calibration sample raises ValueError (need minimum for jackknife+)."""


class TestPositionSizingIntegration:
    """Tests for CI-width-based position sizing."""

    def test_narrow_ci_gives_larger_size(self):
        """CI width at 10th percentile: size_scalar > 0.8."""

    def test_wide_ci_gives_smaller_size(self):
        """CI width at 90th percentile: size_scalar < 0.4."""

    def test_size_scalar_floored_at_02(self):
        """Even maximum CI width produces size_scalar >= 0.2, never 0."""

    def test_size_scalar_capped_at_10(self):
        """Zero CI width (impossible in practice) caps at 1.0."""

    def test_formula_correctness(self):
        """size_scalar = 1.0 - (ci_width / max_ci_width), verify with known values."""

    def test_max_ci_width_zero_raises(self):
        """If max_ci_width is 0 (all identical predictions), raise ZeroDivisionError or handle gracefully."""


class TestModelRegistryExtension:
    """Tests for conformal metadata in model registry."""

    def test_coverage_fields_stored(self, mock_db):
        """Model registry entry includes coverage_80, coverage_90, coverage_95."""

    def test_avg_interval_width_stored(self, mock_db):
        """Model registry entry includes avg_interval_width."""

    def test_calibration_drift_flagged(self, mock_db):
        """Empirical coverage deviating >3% from target triggers calibration_warning flag."""

    def test_calibration_ok_not_flagged(self, mock_db):
        """Empirical coverage within 3% of target: no warning flag."""
```

---

## Section 2: Transformer Time Series Forecaster (`ml/transformers/forecaster.py`)

```python
# tests/unit/ml/test_transformer_forecaster.py

import pytest
import numpy as np
import pandas as pd


class TestPatchTSTForecaster:
    """Tests for PatchTST-based time series prediction."""

    def test_train_on_synthetic_trend(self):
        """Train on synthetic uptrend data: predicted direction is positive."""

    def test_train_on_synthetic_downtrend(self):
        """Train on synthetic downtrend data: predicted direction is negative."""

    def test_input_shape_validation(self):
        """Input must be (n_samples, 20 features) -- wrong shape raises ValueError."""

    def test_horizon_is_5_days(self):
        """Output is 5-day ahead return prediction (single value per sample)."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Save model checkpoint, load from same path, predictions match."""

    def test_cpu_inference_works(self):
        """Inference runs on CPU without CUDA, no errors."""

    def test_missing_values_in_input_raises(self):
        """NaN in input features raises ValueError, not silent corruption."""

    def test_constant_input_returns_near_zero(self):
        """All-constant features: prediction should be near zero (no signal)."""


class TestChronosFallback:
    """Tests for Chronos zero-shot fallback model."""

    def test_zero_shot_returns_prediction(self):
        """Chronos produces a prediction without any training on our data."""

    def test_prediction_is_finite(self):
        """Output is not NaN or Inf."""

    def test_direction_reasonable_for_strong_trend(self):
        """Strong synthetic uptrend: Chronos predicts positive direction."""


class TestTransformerSignalCollector:
    """Tests for signal_engine/collectors/transformer_signal.py."""

    def test_loads_latest_checkpoint(self, mock_model_dir):
        """Collector loads most recent model checkpoint from directory."""

    def test_produces_direction_and_confidence(self, mock_model):
        """Output includes predicted_return and direction_confidence."""

    def test_default_synthesis_weight_010(self):
        """Initial synthesis weight for transformer collector is 0.10."""

    def test_missing_checkpoint_returns_neutral(self, mock_model_dir):
        """No checkpoint file: returns neutral signal, logs warning."""

    def test_stale_checkpoint_returns_neutral(self, mock_model_dir):
        """Checkpoint older than 14 days: returns neutral signal with stale warning."""

    def test_inference_error_returns_neutral(self, mock_model):
        """Model raises during inference: returns neutral signal, logs error."""
```

---

## Section 3: Graph Neural Network (`ml/gnn/`)

```python
# tests/unit/ml/test_gnn.py

import pytest
import numpy as np


class TestMarketGraphConstruction:
    """Tests for ml/gnn/market_graph.py."""

    def test_nodes_match_universe_symbols(self):
        """Graph has one node per symbol in the universe."""

    def test_correlation_edge_threshold(self):
        """Only symbol pairs with rolling 63-day correlation > 0.5 get an edge."""

    def test_same_sector_edge_added(self):
        """Two stocks in same sector get an edge regardless of correlation."""

    def test_no_self_loops(self):
        """No symbol has an edge to itself."""

    def test_edge_features_include_correlation_and_type(self):
        """Each edge has correlation_strength and relationship_type attributes."""

    def test_node_features_shape(self):
        """Node features: returns + volume + technicals + sector one-hot. Verify dimension."""

    def test_empty_universe_raises(self):
        """Zero symbols raises ValueError."""

    def test_single_symbol_produces_isolated_node(self):
        """One symbol: graph has 1 node, 0 edges."""

    def test_graph_serialization_to_snapshot_table(self, mock_db):
        """Graph stored to gnn_graph_snapshots table with correct n_nodes, n_edges."""


class TestGATModel:
    """Tests for ml/gnn/model.py (Graph Attention Network)."""

    def test_forward_pass_produces_per_node_prediction(self):
        """Model output shape: (n_nodes, 1) -- one return prediction per symbol."""

    def test_attention_weights_sum_to_one(self):
        """Per-node attention weights across neighbors sum to 1.0."""

    def test_model_trains_on_synthetic_graph(self):
        """Loss decreases over 10 epochs on synthetic graph with known structure."""

    def test_cpu_inference(self):
        """Inference on CPU completes without error."""

    def test_handles_disconnected_graph(self):
        """Graph with isolated nodes: model still produces predictions for all nodes."""


class TestGNNContagionCollector:
    """Tests for signal_engine/collectors/gnn_contagion.py."""

    def test_neighbor_drop_propagates_bearish_signal(self):
        """Highly-connected neighbor drops >3%: target symbol gets bearish signal."""

    def test_sector_leader_rally_propagates_bullish(self):
        """Sector leader rallies: other sector members get bullish signal."""

    def test_no_significant_neighbor_moves_gives_neutral(self):
        """All neighbors flat: contagion signal is neutral."""

    def test_default_synthesis_weight_005(self):
        """Initial synthesis weight for GNN contagion collector is 0.05."""

    def test_graph_not_available_returns_neutral(self):
        """No graph snapshot in DB: returns neutral signal, logs warning."""
```

---

## Section 4: Deep Hedging (`core/options/deep_hedging.py`)

```python
# tests/unit/ml/test_deep_hedging.py

import pytest
import numpy as np


class TestDeepHedgingNetwork:
    """Tests for LSTM-based deep hedging (requires P08 infrastructure)."""

    def test_output_is_hedge_ratio(self):
        """Network output is a single float (hedge ratio: shares of underlying)."""

    def test_hedge_ratio_bounded(self):
        """Hedge ratio in [-2.0, 2.0] (allows slight over-hedge, no extreme values)."""

    def test_cvar_loss_improves_over_training(self):
        """CVaR loss decreases over 50 epochs on simulated GBM paths."""

    def test_beats_bs_delta_hedge_on_simulated_data(self):
        """CVaR of deep hedge < CVaR of BS delta hedge by > 10% on stochastic vol paths."""

    def test_transaction_costs_included(self):
        """Hedging P&L accounts for 0.1% per rebalance cost."""

    def test_feature_flag_default_false(self):
        """deep_hedging_enabled() returns False by default."""

    def test_feature_flag_respects_env_var(self):
        """Setting DEEP_HEDGING_ENABLED=true makes deep_hedging_enabled() return True."""

    def test_input_validation(self):
        """Missing Greeks or negative time_to_expiry raises ValueError."""

    def test_zero_dte_handled(self):
        """time_to_expiry = 0 does not crash (edge: expiration day)."""
```

---

## Section 5: Financial NLP (`ml/nlp/financial_sentiment.py`)

```python
# tests/unit/ml/test_financial_nlp.py

import pytest


class TestFinBERTSentiment:
    """Tests for FinBERT-based sentiment scoring."""

    def test_positive_text_returns_positive_score(self):
        """'Revenue beat expectations by 20%' produces score > 0.3."""

    def test_negative_text_returns_negative_score(self):
        """'Company missed earnings and lowered guidance' produces score < -0.3."""

    def test_neutral_text_returns_near_zero(self):
        """'The company held its annual meeting' produces score near 0."""

    def test_score_range_bounded(self):
        """All scores in [-1.0, 1.0]."""

    def test_empty_string_returns_neutral(self):
        """Empty input returns 0.0, not error."""

    def test_very_long_text_truncated_gracefully(self):
        """Input exceeding model max tokens is truncated, not crashed."""


class TestSentimentEnsemble:
    """Tests for FinBERT + LLM ensemble in sentiment collector."""

    def test_ensemble_weight_formula(self):
        """Final score = 0.6 * finbert + 0.4 * llm_sentiment. Verify with known values."""

    def test_finbert_unavailable_falls_back_to_llm_only(self):
        """If FinBERT fails, use 100% LLM sentiment with logged warning."""

    def test_llm_unavailable_falls_back_to_finbert_only(self):
        """If LLM sentiment fails, use 100% FinBERT with logged warning."""

    def test_both_unavailable_returns_neutral(self):
        """Both fail: return neutral signal, log error."""

    def test_ic_tracked_per_source(self, mock_db):
        """IC attribution stores separate entries for finbert and llm sentiment sources."""
```

---

## Section 6: Schema Extensions

```python
# tests/unit/ml/test_advanced_ml_schema.py

import pytest


class TestModelRegistryCategoryExtension:
    """Tests for model_category field in model_registry."""

    def test_valid_categories_accepted(self, mock_db):
        """traditional_ml, transformer, gnn, rl, deep_hedge all accepted."""

    def test_unknown_category_raises(self, mock_db):
        """Category 'magic' raises ValueError or constraint violation."""


class TestConformalCoverageTable:
    """Tests for conformal_coverage tracking table."""

    def test_insert_coverage_record(self, mock_db):
        """Insert with model_id, date, target_coverage, empirical_coverage, avg_width."""

    def test_duplicate_model_date_target_raises(self, mock_db):
        """Same (model_id, date, target_coverage) is unique constraint violation."""


class TestGNNGraphSnapshotsTable:
    """Tests for gnn_graph_snapshots tracking table."""

    def test_insert_snapshot(self, mock_db):
        """Insert with date, graph_json, n_nodes, n_edges."""

    def test_graph_json_is_valid_json(self, mock_db):
        """Invalid JSON string in graph_json field raises error."""
```

---

## Section 7: Integration Tests

```python
# tests/integration/test_advanced_ml_e2e.py

import pytest


class TestConformalPipelineEndToEnd:
    """End-to-end: train model -> wrap conformal -> position sizing."""

    def test_train_wrap_and_size(self, mock_db):
        """
        1. Train LightGBM on synthetic data
        2. Wrap with conformal predictor
        3. Generate prediction with CI
        4. Compute position size from CI width
        5. Verify size_scalar in [0.2, 1.0]
        6. Verify coverage metadata stored in registry
        """


class TestTransformerPipelineEndToEnd:
    """End-to-end: train -> checkpoint -> collector -> synthesis."""

    def test_train_and_collect_signal(self, mock_db, tmp_path):
        """
        1. Train PatchTST on synthetic OHLCV
        2. Save checkpoint
        3. Collector loads checkpoint and produces signal
        4. Signal has predicted_return and direction_confidence
        """


class TestGNNPipelineEndToEnd:
    """End-to-end: graph build -> train -> contagion signal."""

    def test_graph_build_train_signal(self, mock_db):
        """
        1. Build graph from synthetic universe (10 symbols)
        2. Train GAT for 5 epochs
        3. Run contagion collector
        4. Inject neighbor drop > 3%, verify bearish signal propagates
        """
```

---

## Test Execution Order

For TDD, write and verify tests in this order:

1. **Section 1 tests** -- Conformal prediction (highest priority, wraps existing models)
2. **Section 5 tests** -- Financial NLP (incremental, standalone)
3. **Section 6 tests** -- Schema extensions (DB mocks, fast)
4. **Section 2 tests** -- Transformer forecaster (requires neuralforecast setup)
5. **Section 3 tests** -- GNN (requires torch_geometric setup)
6. **Section 4 tests** -- Deep hedging (requires P08, lowest priority)
7. **Section 7 tests** -- End-to-end integration (requires all sections)
