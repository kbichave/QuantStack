# Section 10: Comprehensive Unit Tests

## Objective

Write integration-level tests that verify the cross-module interactions between all P14 components. Individual unit tests are specified in sections 01-09; this section covers tests that span multiple modules and validate end-to-end behavior.

## Dependencies

- **section-01-conformal-prediction**
- **section-03-transformer-forecaster**
- **section-05-gnn-market-graph**
- **section-07-deep-hedging**
- **section-08-financial-nlp**

## Files to Create/Modify

### New Files

- **`tests/unit/ml/test_conformal.py`** — Conformal prediction unit tests (from section-01).
- **`tests/unit/ml/test_conviction_sizing.py`** — CI-based sizing tests (from section-02).
- **`tests/unit/ml/test_transformer_forecaster.py`** — Transformer training/inference tests (from section-03).
- **`tests/unit/signal_engine/test_transformer_signal.py`** — Transformer collector tests (from section-04).
- **`tests/unit/ml/test_gnn_market_graph.py`** — Graph construction tests (from section-05).
- **`tests/unit/ml/test_gnn_model.py`** — GAT model tests (from section-05).
- **`tests/unit/signal_engine/test_gnn_contagion.py`** — Contagion collector tests (from section-06).
- **`tests/unit/options/test_deep_hedging.py`** — Deep hedging tests (from section-07).
- **`tests/unit/ml/test_financial_nlp.py`** — FinBERT tests (from section-08).
- **`tests/unit/signal_engine/test_sentiment_ensemble.py`** — Ensemble sentiment tests (from section-08).
- **`tests/unit/test_schema_migrations.py`** — Schema migration tests (from section-09).
- **`tests/integration/test_p14_integration.py`** — Cross-module integration tests (this section).

## Integration Test Details

### `tests/integration/test_p14_integration.py`

#### 1. Conformal + Position Sizing End-to-End

```
def test_conformal_sizing_pipeline():
    """Train model -> conformal calibrate -> predict with CI -> compute size scalar."""
    # 1. Train LightGBM on synthetic data
    # 2. Calibrate ConformalPredictor on held-out set
    # 3. Predict on new data, get intervals
    # 4. Compute ci_size_scalar from interval width
    # 5. Verify: narrow intervals -> scalar near 1.0, wide -> near floor
```

#### 2. Transformer -> Signal Collector Pipeline

```
def test_transformer_signal_pipeline():
    """Train transformer -> save checkpoint -> collector loads and predicts."""
    # 1. Train PatchTST on synthetic trend data
    # 2. Save checkpoint
    # 3. Call collect_transformer_signal with mocked store
    # 4. Verify output dict has all expected keys
    # 5. Verify direction matches trend direction
```

#### 3. GNN -> Contagion Collector Pipeline

```
def test_gnn_contagion_pipeline():
    """Build graph -> train GAT -> inject neighbor drop -> verify contagion signal."""
    # 1. Build market graph from synthetic correlated returns
    # 2. Train GAT model
    # 3. Simulate: set one high-attention neighbor to -5% daily return
    # 4. Run contagion collector
    # 5. Verify bearish contagion detected for connected symbol
```

#### 4. Sentiment Ensemble Pipeline

```
def test_sentiment_ensemble_pipeline():
    """FinBERT + LLM sentiment -> ensemble -> verify combined score."""
    # 1. Mock headlines
    # 2. Run FinBERT scorer
    # 3. Mock LLM sentiment
    # 4. Verify ensemble math: 0.6 * finbert + 0.4 * llm
    # 5. Verify backwards compatibility of output keys
```

#### 5. Model Registry Category Filtering

```
def test_model_registry_categories():
    """Register models of different categories -> filter by category."""
    # 1. Register a traditional_ml model
    # 2. Register a transformer model
    # 3. Register a gnn model
    # 4. Query by model_category='transformer' -> returns only transformer
    # 5. Query all -> returns all three
```

#### 6. Deep Hedging Feature Flag

```
def test_deep_hedging_flag_gate():
    """Deep hedging is gated behind feature flag."""
    # 1. With deep_hedging_enabled=False, engine uses BS delta
    # 2. With deep_hedging_enabled=True, engine uses deep hedge ratio
    # 3. Both paths produce valid hedge ratios
```

### Test Fixtures

Create shared fixtures in `tests/conftest.py` or `tests/fixtures/p14.py`:

- `synthetic_ohlcv(n_days, trend)` — generate synthetic OHLCV data with configurable trend
- `synthetic_correlated_returns(n_symbols, n_days, correlation)` — returns matrix with known correlation structure
- `mock_store()` — DataStore mock that returns synthetic data
- `trained_lgbm()` — pre-trained LightGBM model on synthetic data (for conformal tests)

## Test Requirements

All tests must:
1. Run without GPU (CPU only).
2. Run without network access (all models/data mocked or synthetic).
3. Run without PostgreSQL (use in-memory SQLite or mock `db_conn`).
4. Complete in under 60 seconds total (no expensive training in CI).
5. Use `pytest` fixtures for shared setup.

## Acceptance Criteria

- [ ] All individual section tests from sections 01-09 are implemented
- [ ] Integration tests verify cross-module interactions for all 6 scenarios above
- [ ] Shared test fixtures reduce duplication across test files
- [ ] All tests pass with `uv run pytest tests/unit/ml/ tests/unit/signal_engine/ tests/unit/options/ tests/integration/test_p14_integration.py`
- [ ] No test requires GPU, network access, or PostgreSQL
- [ ] Total test runtime < 60 seconds
