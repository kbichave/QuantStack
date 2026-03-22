---
name: ML Research Program
description: Self-evolving research priorities for the autonomous ML loop.
type: project
---

# ML Research Program

## Current State (2026-03-21)

- 8 symbols trained (SPY, QQQ, NVDA, AAPL, TSLA, IWM, XLK, XLF, XLE)
- 23 experiments run across 3 batches
- 8 champions deployed (IWM rejected)
- All 5 validated strategy symbols covered (SPY, QQQ, XLK, XLF, XLE)
- CausalFilter tested — drops 0 features (all 84 pass Granger causality)
- Concept drift: 91-96% of features drifted; bop and dx are regime-invariant
- 0 fundamental features loaded (cache empty)
- 6 critical bugs fixed in training pipeline

## Active Research Questions

### Q1: Does CausalFilter improve OOS performance?
- Status: BLOCKED (API was broken, now fixed)
- Hypothesis: Dropping non-causal features reduces noise and improves OOS AUC by 0.02+
- Next step: Retrain SPY with apply_causal_filter=True (API fixed)
- Success: OOS AUC improves vs 0.7397 baseline

### Q2: Can fundamentals add signal beyond technicals?
- Status: BLOCKED (no fundamental data cached)
- Hypothesis: Adding PE ratio, earnings growth adds 0.02+ AUC
- Next step: Populate fundamental cache via Alpha Vantage, then retrain
- Risk: Fundamentals change slowly, may not help daily prediction

### Q3: Why does IWM fail completely?
- Status: OPEN
- Hypothesis: Small-cap ETF behavior is too diffuse for single-stock technical features
- Next step: Try CatBoost (ordered boosting), try sector rotation features
- Alternative: Accept IWM as unmodellable and focus on SPY/QQQ

### Q4: Feature ablation -- less is more?
- Status: PLANNED for next iteration
- Hypothesis: Removing bottom 50% SHAP features reduces collinearity and improves OOS
- Rationale: 84 features with many correlated MAs (8 variants x 4 periods = 32 MA features alone)
- Next step: Train SPY with top 42 features only

### Q5: Multi-horizon agreement signal
- Status: PLANNED
- Hypothesis: When 1-day and 5-day models agree, conviction should be higher
- Next step: Train 5-day label models, compare signal overlap
- Benefit: Portfolio-level confidence scoring

## Iteration Plan (Next Cycle)

1. **CatBoost on XLE** — worst validated model (0.5748), ordered boosting may handle noise
2. **Add macro tier to XLE** — energy is macro-driven, technicals insufficient
3. **XGBoost challengers on XLK/XLF** — XGB won SPY, may win other ETFs
4. **Focused feature set experiment** — ADX family + MACD family only (~15 features)
5. **Multi-horizon labels on SPY** — train 5-day model alongside current event labels
6. **Cache fundamentals** — prerequisite for fundamental tier experiments

## Proven Patterns

- XGBoost > LightGBM on SPY only. LightGBM dominates 7 of 8 symbols.
- LightGBM > XGBoost on volatile single stocks (NVDA, TSLA) AND sector ETFs (XLK, XLF, XLE)
- **macdext_histogram is the #1 cross-symbol feature** — top-5 SHAP in 4 of 5 validated symbols
- ADX family (adx, adxr, dx) consistently important for trend strength
- natr (normalized ATR) uniquely important for financials (XLF) and energy (XLE)
- bb_width uniquely important for energy (XLE) — vol expansion signal
- bop and dx are the most regime-invariant features (no drift across SPY/QQQ/XLK)
- CausalFilter drops 0 features — all 84 technical indicators pass Granger causality
- XLF has the best IS-OOS calibration (gap 0.014) of any model trained

## Breakthrough Features (for cross-pod sharing)

| Feature | Cross-Symbol Rank | Interpretation | Strategy Alignment |
|---------|-------------------|----------------|-------------------|
| macdext_histogram | #1 (4/5 symbols) | Extended MACD momentum divergence | aligns with regime_momentum |
| adxr | #2 (4/8 symbols) | Smoothed trend strength | aligns with regime_momentum |
| natr | #3 (sector ETFs) | Normalized volatility | aligns with vol_compress (XLE) |
| minus_di | #4 (QQQ, XLK) | Negative directional pressure | new signal for short setups |
| bb_width | #5 (XLE) | Bollinger bandwidth (vol expansion) | aligns with vol_compress |

## Known Failure Modes

- XGBoost CV with sklearn 1.8 was misidentifying classifier (FIXED via manual CV)
- CausalFilter API mismatch: .filter() vs .fit_transform() (FIXED)
- EventLabeler API: .label() vs .label_trades() (FIXED)
- TrainingResult attribute mismatch: .test_auc vs .metrics["auc"] (FIXED)
- QQQ LightGBM over-relies on RSI (SHAP=0.87) -- fixed in Batch 2 retrain
- DuckDB write lock prevents register_model when MCP server holds connection
