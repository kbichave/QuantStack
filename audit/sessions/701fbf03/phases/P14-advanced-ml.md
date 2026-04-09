# P14: Advanced ML (Transformers, GNNs, Deep Hedging)

**Objective:** Integrate cutting-edge ML techniques: transformer time series forecasting, graph neural networks for market structure, deep hedging for options, and conformal prediction for uncertainty.

**Scope:** New: ml/transformers/, ml/gnn/, core/options/deep_hedging.py

**Depends on:** P03 (ML pipeline)

**Effort estimate:** 2-3 weeks

---

## What Changes

### 14.1 Transformer Time Series Forecasting
- **Models:** PatchTST, TimesFM (Google), Chronos (Amazon), Lag-Llama
- **Use case:** Multi-step ahead price prediction with attention-based context
- **Integration:** New collector `transformer_signal.py` that runs inference
- **Packages:** `neuralforecast` (Nixtla), `chronos-forecasting` (Amazon)

### 14.2 Graph Neural Networks for Market Structure
- **Models:** GAT (Graph Attention Network), GCN
- **Use case:** Model sector/industry relationships, supply chain links, correlation structure
- **Integration:** GNN provides "sector contagion" signal — if a key supplier drops, predict downstream effect
- **Packages:** `torch_geometric`, `dgl`

### 14.3 Deep Hedging (Buehler et al.)
- **Use case:** Neural network learns optimal hedging strategy for complex option portfolios
- **Advantage:** Handles transaction costs, discrete hedging, market frictions that Black-Scholes ignores
- **Integration:** Replace rule-based delta hedging (P08) with learned hedging policy
- **Package:** Custom implementation (research code available on GitHub)

### 14.4 Conformal Prediction for Uncertainty
- **Use case:** Calibrated prediction intervals on ML model outputs
- **Advantage:** "Model predicts AAPL +2% with 90% CI [+0.5%, +3.5%]" — enables proper position sizing
- **Integration:** Wrap existing LightGBM/XGBoost with conformal predictor
- **Package:** `mapie` (Model Agnostic Prediction Interval Estimator)

### 14.5 Domain-Specific Financial NLP
- **Models:** FinGPT, BloombergGPT successors, fine-tuned LLMs on SEC filings
- **Use case:** Better sentiment analysis than generic FinBERT
- **Integration:** Replace/augment existing sentiment collectors
- **Package:** `fingpt` or fine-tune Llama on financial corpus (Ollama local)

## Priority Within Phase
1. **Conformal prediction** (easiest, highest immediate value — calibrated uncertainty)
2. **Transformer forecasting** (moderate effort, proven results)
3. **GNN market structure** (novel signal, moderate effort)
4. **Deep hedging** (complex, requires P08 first)
5. **Financial NLP** (incremental improvement over existing sentiment)

## Acceptance Criteria

1. At least 1 transformer model producing time series forecasts
2. Conformal prediction intervals calibrated (90% CI covers 90% of outcomes)
3. GNN produces sector contagion signals
4. All new models tracked in model_registry (P03)
