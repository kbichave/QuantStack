# P14 Spec: Advanced ML

## Deliverables

### D1: Conformal Prediction (Priority 1)
- MAPIE wrapper for existing LightGBM/XGBoost models
- Calibrated prediction intervals: 80%, 90%, 95%
- Position sizing integration: CI width → conviction scalar
- Coverage monitoring in model registry

### D2: Transformer Forecasting (Priority 2)
- PatchTST via neuralforecast for 5-day ahead prediction
- Chronos (Amazon) as zero-shot fallback
- Weekly retrain overnight, CPU training
- New transformer_signal collector, weight = 0.10

### D3: GNN Market Structure (Priority 3)
- Graph construction: correlation + sector + supply chain edges
- GAT model (2 layers, 4 heads) via torch_geometric
- Monthly retrain
- New gnn_contagion collector, weight = 0.05

### D4: Deep Hedging (Priority 4, requires P08)
- LSTM + FC network for hedge ratio prediction
- CVaR loss with transaction costs
- Deploy only if >10% CVaR improvement vs BS delta hedge
- Feature flag gated, A/B test alongside BS hedge

### D5: Financial NLP (Priority 5)
- FinBERT integration via HuggingFace
- Ensemble with existing LLM sentiment (0.6/0.4)
- Separate IC tracking per source

## Dependencies
- P03 (ML Pipeline): model registry, walk-forward validation
- P08 (Options Market-Making): required for deep hedging
- P05 (Signal Synthesis): IC-driven weight for new collectors
