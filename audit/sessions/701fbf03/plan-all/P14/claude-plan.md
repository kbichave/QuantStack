# P14 Implementation Plan: Advanced ML (Transformers, GNNs, Deep Hedging)

## 1. Background

P03 (ML pipeline) provides walk-forward validation, model registry, and A/B promotion. P14 adds cutting-edge ML: conformal prediction for calibrated uncertainty, transformer forecasting, GNN market structure signals, deep hedging (requires P08), and financial NLP improvements.

## 2. Anti-Goals

- **Do NOT build custom transformer architectures** — use PatchTST/Chronos from neuralforecast/Amazon
- **Do NOT require GPU for inference** — GPU for training only, inference must run on CPU
- **Do NOT replace existing LightGBM/XGBoost** — new models are additional signals, not replacements
- **Do NOT deploy any model without walk-forward validation** — same P03 pipeline applies
- **Do NOT implement deep hedging before P08** — requires options market-making infrastructure
- **Do NOT fine-tune LLMs locally** — use pre-trained financial models, fine-tuning is out of scope

## 3. Conformal Prediction (Priority 1 — Highest Value)

### 3.1 Conformal Wrapper

New `src/quantstack/ml/conformal.py`:
- Wraps existing LightGBM/XGBoost models from P03
- Method: MAPIE `MapieRegressor` with `method="plus"` (jackknife+)
- Output: point prediction + calibrated prediction intervals (80%, 90%, 95%)
- Key property: guaranteed coverage — 90% CI covers ≥90% of outcomes

### 3.2 Position Sizing Integration

Use prediction interval width for position sizing:
- Narrow CI → high conviction → larger position
- Wide CI → low conviction → smaller position
- Formula: `size_scalar = 1.0 - (ci_width / max_ci_width)`, floored at 0.2
- Integrates with existing conviction system (P05)

### 3.3 Model Registry Extension

Extend model metadata:
- `coverage_80`, `coverage_90`, `coverage_95`: empirical coverage rates
- `avg_interval_width`: average CI width (narrower = more informative)
- Calibration check: flag if empirical coverage deviates >3% from target

## 4. Transformer Time Series Forecasting (Priority 2)

### 4.1 Model Setup

New `src/quantstack/ml/transformers/forecaster.py`:
- Primary model: PatchTST (patch-based, efficient for financial time series)
- Fallback: Chronos (Amazon, pre-trained foundation model — zero-shot)
- Library: `neuralforecast` (Nixtla) for PatchTST, `chronos-forecasting` for Chronos

### 4.2 Training Pipeline

- Input: OHLCV + technical indicators (20 features per symbol)
- Horizon: 5-day ahead return prediction
- Training: walk-forward, retrain weekly (overnight batch)
- Hardware: CPU training with batch size 32 (GPU optional via CUDA flag)
- Checkpoint: save to `models/transformers/` directory

### 4.3 Signal Collector

New `src/quantstack/signal_engine/collectors/transformer_signal.py`:
- Load latest model checkpoint
- Run inference on current features
- Output: predicted 5-day return + direction confidence
- Synthesis weight: 0.10, adjusted by IC

## 5. Graph Neural Networks for Market Structure (Priority 3)

### 5.1 Graph Construction

New `src/quantstack/ml/gnn/market_graph.py`:
- Nodes: symbols in universe
- Edges: correlation > 0.5 (rolling 63-day), same sector, supply chain relationship
- Node features: returns, volume, technical indicators, sector one-hot
- Edge features: correlation strength, relationship type

### 5.2 Model

New `src/quantstack/ml/gnn/model.py`:
- Architecture: GAT (Graph Attention Network) — 2 layers, 4 attention heads
- Library: `torch_geometric`
- Task: node-level regression (predict 5-day return per symbol)
- Training: walk-forward, retrain monthly (overnight)

### 5.3 Sector Contagion Signal

New `src/quantstack/signal_engine/collectors/gnn_contagion.py`:
- Use GNN attention weights to identify contagion risk
- If highly-connected neighbor drops >3%, propagate bearish signal
- If sector leader rallies, propagate bullish signal to sector
- Synthesis weight: 0.05, adjusted by IC

## 6. Deep Hedging (Priority 4 — Requires P08)

### 6.1 Hedging Network

New `src/quantstack/core/options/deep_hedging.py`:
- Architecture: LSTM + fully connected (Buehler et al. 2019 design)
- Input: current portfolio Greeks, underlying price, time to expiry, IV
- Output: hedge ratio (shares of underlying to hold)
- Loss function: CVaR of hedging error (accounts for transaction costs)

### 6.2 Training

- Simulate paths using GBM with stochastic volatility
- Transaction costs: 0.1% per rebalance
- Rebalance frequency: daily
- Benchmark: compare against Black-Scholes delta hedging (P06)
- Deploy only if CVaR improvement > 10% vs BS delta hedge

### 6.3 Integration

Replace P06 HedgingEngine's delta computation when deep hedging is active:
- Feature flag: `deep_hedging_enabled()`, default False
- A/B test: run deep hedge alongside BS hedge, compare P&L

## 7. Financial NLP (Priority 5 — Incremental)

### 7.1 Model Selection

New `src/quantstack/ml/nlp/financial_sentiment.py`:
- Primary: FinBERT (already available via HuggingFace)
- Enhancement: fine-tuned sentiment on SEC 10-K/10-Q filing language
- Use existing Ollama infrastructure for local inference

### 7.2 Collector Upgrade

Modify existing `sentiment` collector:
- Add FinBERT score alongside current LLM-based sentiment
- Ensemble: 0.6 × FinBERT + 0.4 × LLM sentiment
- Track IC separately for each source

## 8. Schema

- Extend `model_registry`: add `model_category` (traditional_ml, transformer, gnn, rl, deep_hedge)
- `conformal_coverage`: (model_id, date, target_coverage, empirical_coverage, avg_width)
- `gnn_graph_snapshots`: (date, graph_json, n_nodes, n_edges)

## 9. Testing

- Conformal: verify coverage guarantees on held-out data (90% CI covers ≥87%)
- Transformer: train on synthetic trend → verify prediction direction
- GNN: synthetic graph with known contagion → verify signal propagation
- Deep hedging: compare CVaR vs BS delta hedge on simulated paths
- NLP: verify FinBERT produces valid sentiment scores [-1, 1]
