# P14 Self-Interview: Advanced ML

## Q1: Why is conformal prediction the highest priority when transformers seem more exciting?
**A:** Conformal prediction provides the highest immediate value with the lowest effort. It wraps existing LightGBM/XGBoost models — no new model architecture, no new training pipeline, no GPU requirements. The output (calibrated prediction intervals) directly improves position sizing, which has immediate P&L impact. Transformers are higher effort and their alpha is unproven in our specific universe.

## Q2: Can PatchTST train on CPU in reasonable time?
**A:** For our use case (50 symbols, 20 features, 252-day windows, 5-day horizon): yes. PatchTST with patch_length=16, d_model=128, n_heads=4 trains in ~2-4 hours on CPU with batch_size=32. This fits in the overnight compute window. For larger universes, GPU becomes necessary. The plan supports optional GPU via CUDA flag without requiring it.

## Q3: How does the GNN handle symbols entering/leaving the universe?
**A:** The graph is reconstructed monthly. Node addition/removal is handled at graph construction time — new symbols get added as nodes, removed symbols get deleted. The GAT model handles variable-size graphs natively (attention mechanism is node-count agnostic). Node features are zero-padded for the first month a symbol is in the universe.

## Q4: What's the risk of deep hedging vs Black-Scholes delta hedging?
**A:** Deep hedging learns to minimize CVaR of hedging error, accounting for transaction costs and discrete rebalancing — theoretically superior. Risk: model uncertainty (neural network may learn spurious patterns in training data). Mitigation: (a) benchmark against BS delta hedge on out-of-sample data, (b) deploy only if CVaR improvement > 10%, (c) A/B test alongside BS hedge, (d) feature flag with kill switch.

## Q5: How do you prevent the transformer from overfitting on financial noise?
**A:** Three mechanisms: (a) walk-forward validation — same P03 pipeline, train on T-252 to T-21, validate on T-21 to T; (b) patch-based tokenization (PatchTST) reduces the effective sequence length, making overfitting harder; (c) Chronos as fallback — pre-trained on millions of time series, zero-shot inference avoids overfitting entirely. IC tracking will quickly reveal if the transformer signal adds value.

## Q6: How many new dependencies does P14 add?
**A:** 4 new packages: `mapie` (conformal), `neuralforecast` (transformers), `torch_geometric` (GNN), `chronos-forecasting` (Amazon foundation model). torch is already a dependency (via FinRL/SB3). This is a significant dependency footprint — but each is well-maintained and widely used. Pin exact versions in pyproject.toml.

## Q7: How does the GNN contagion signal work in practice?
**A:** After training, the GAT model produces attention weights for each edge (how much one node influences another). The contagion signal: if a highly-connected neighbor (high attention weight) drops >3%, propagate a bearish signal to the target node, scaled by attention weight. If a sector leader rallies, propagate bullish. This captures "if NVDA drops, AMD/INTC are likely affected" beyond simple correlation.

## Q8: What's the integration path for FinBERT alongside existing LLM sentiment?
**A:** Non-disruptive upgrade. Add FinBERT as a second sentiment source in the existing sentiment collector. Ensemble: 0.6 × FinBERT + 0.4 × LLM sentiment (FinBERT weighted higher because it's faster and specifically trained on financial text). Track IC separately for each source — if LLM sentiment has higher IC, the weights will naturally adjust via P05.
