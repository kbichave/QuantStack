# P14 Research: Advanced ML

## Codebase Research

### What Exists
- **ML trainer**: `src/quantstack/ml/trainer.py` — LightGBM/XGBoost training with Optuna hyperparameter optimization
- **Training service**: `src/quantstack/ml/training_service.py` — orchestrates model training pipeline
- **Model registry**: P03 infrastructure for model versioning, A/B promotion
- **Feature importance**: `src/quantstack/ml/feature_importance.py` — SHAP/permutation importance
- **Sentiment collector**: `src/quantstack/signal_engine/collectors/sentiment.py` — LLM-based sentiment
- **FinRL config**: `src/quantstack/finrl/config.py` — torch/SB3 dependencies already in pyproject.toml
- **Ollama**: local LLM infrastructure for inference

### What's Needed (Gaps)
1. **Conformal prediction**: No uncertainty quantification on ML outputs — need MAPIE wrapper
2. **Transformer forecaster**: No time series transformer models — need PatchTST/Chronos via neuralforecast
3. **GNN market structure**: No graph-based models — need torch_geometric for sector contagion signals
4. **Deep hedging**: No neural hedging — need custom LSTM implementation (requires P08 first)
5. **Financial NLP**: Sentiment uses generic LLM — could improve with FinBERT

## Domain Research

### Conformal Prediction for Finance
- MAPIE provides model-agnostic prediction intervals
- Jackknife+ method: robust, works with any sklearn-compatible model
- Key guarantee: coverage validity (90% CI covers ≥90% of outcomes)
- Position sizing application: narrow intervals → higher conviction → larger size

### Transformer Time Series Models
- PatchTST (2023): patch-based tokenization of time series, state-of-art on financial data
- Chronos (Amazon, 2024): pre-trained foundation model, zero-shot capable
- neuralforecast library wraps both with consistent API
- CPU inference is feasible (200ms per prediction), GPU needed only for training

### GNN for Market Structure
- GAT (Graph Attention Networks) learn which graph neighbors matter most
- Applied to stock markets: sector relationships, supply chain, correlation structure
- torch_geometric provides efficient batched processing
- Monthly retraining is sufficient — market structure changes slowly

### FinBERT vs LLM Sentiment
- FinBERT: fine-tuned BERT on financial text, faster inference (~10ms), less accurate on nuanced sentiment
- LLM (current): better understanding of context, slower (~500ms), more expensive
- Ensemble approach: weight both, let IC tracking decide optimal blend
