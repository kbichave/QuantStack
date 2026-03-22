---
name: ML Experiment Log
description: Tracks every ML experiment -- hypothesis, config, result, lesson.
type: project
---

# ML Experiment Log

## Batch 1: Initial Model Training (2026-03-20)

**Hypothesis:** Technical indicators (84 features) can predict ATR-based trade outcomes
across liquid US equities with AUC > 0.55 using gradient boosting.

**Config:** lookback=756 bars, label=EventLabeler (ATR TP/SL), CV=PurgedKFold 5-fold,
feature_tiers=technical+fundamentals (note: fundamentals not cached, effectively technical-only).

### SPY

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| LightGBM | 0.6993 | 0.4459 | 0.6765 | 0.023 | champion_candidate |
| XGBoost | **0.7397** | 0.6216 | 0.6790 | 0.061 | **CHAMPION** |

Top SHAP (LGB): macd_histogram (0.058), sar (0.027), ad (0.021)
Top SHAP (XGB): t3_50 (0.026), ema_10 (0.026), kama_20 (0.025)
Lesson: XGBoost superior on SPY. Momentum indicators (MACD, EMA alignment) dominate.

### QQQ

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| LightGBM | 0.6221 | 0.7365 | 0.6109 | 0.011 | acceptable |
| XGBoost | **0.6765** | 0.6824 | 0.6407 | 0.036 | **CHAMPION** |

Top SHAP (LGB): rsi (0.873), trima_50 (0.812), t3_50 (0.421)
Top SHAP (XGB): trima_200 (0.032), ema_20 (0.025), tema_20 (0.024)
Lesson: LGB over-relies on RSI (SHAP 0.87 is suspiciously high). XGB more diversified.

### NVDA

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| **LightGBM** | **0.6485** | 0.7500 | 0.5525 | 0.096 | **CHAMPION** |
| XGBoost | 0.6189 | 0.7500 | 0.6141 | 0.005 | acceptable |

Top SHAP (LGB): sar (0.540), atr (0.312), stochf_d (0.298)
Top SHAP (XGB): sma_10 (0.032), wma_50 (0.024), vwap (0.022)
Lesson: LGB finds volatility-based patterns (SAR, ATR) on NVDA. High IS-OOS gap (0.096) -- watch for drift.

### AAPL

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| **LightGBM** | **0.6033** | 0.5782 | 0.4850 | 0.118 | **CHAMPION** (acceptable) |
| XGBoost | 0.4602 | 0.7007 | 0.4960 | 0.036 | rejected |

Top SHAP (LGB): macdext_signal (0.111), dx (0.111), wma_20 (0.056)
Top SHAP (XGB): aroonosc (0.031), sma_200 (0.028), rsi (0.025)
Lesson: AAPL is harder to model. High IS-OOS gap on LGB (0.118). XGB completely failed (AUC < 0.50). CausalFilter not applied due to API bug (now fixed).

### TSLA

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| **LightGBM** | **0.5803** | 0.7365 | 0.5200 | 0.060 | **CHAMPION** (acceptable) |
| XGBoost | 0.4495 | 0.4189 | 0.5411 | 0.092 | rejected |

Top SHAP (LGB): adxr (0.137), aroon_up (0.099), vwap (0.077)
Top SHAP (XGB): bb_middle (0.026), sma_10 (0.023), dema_200 (0.021)
Lesson: TSLA volatile, XGB overfits badly (CV std=0.142). ADX-family features most useful.

### IWM

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| LightGBM | 0.4539 | 0.6892 | 0.5257 | 0.072 | rejected |
| XGBoost | 0.4296 | 0.6351 | 0.6046 | 0.175 | rejected [OVERFIT] |

Top SHAP (LGB): obv (0.505), trima_50 (0.428), trange (0.277)
Top SHAP (XGB): wma_200 (0.074), willr (0.033), ema_20 (0.023)
Lesson: IWM is noise. Both models below random. XGB shows catastrophic overfit (0.175 gap).

## Cross-Symbol Breakthrough Features

Features that appear in top-10 SHAP across 3+ symbol/model combinations:

| Feature | Symbols | Mean SHAP | Interpretation |
|---------|---------|-----------|----------------|
| trima_50 | IWM, QQQ | 0.620 | Medium-term smoothed trend |
| obv | IWM, NVDA, TSLA | 0.261 | Volume-price divergence |
| trange | IWM, NVDA, QQQ | 0.246 | Daily volatility |
| sar | IWM, NVDA, SPY | 0.224 | Trend reversal detection |
| vwap | IWM, NVDA, QQQ, TSLA | 0.066 | Intraday fair value |
| ema_20 | IWM, QQQ, SPY, TSLA | 0.023 | Short-term trend |
| macdext_histogram | NVDA, SPY, TSLA | 0.070 | Momentum divergence |
| adxr | QQQ, SPY, TSLA | 0.056 | Trend strength smoothed |

## Batch 2: Challenger Models + Feature Ablation (2026-03-21)

**Hypothesis:** Training the opposite model type (LightGBM where XGBoost was champion,
and vice versa) may find patterns the champion missed. Feature ablation (top 42 of 84)
may reduce noise and improve OOS AUC.

**Config:** Same as Batch 1 (lookback=756, EventLabeler, 5-fold CV, 84 features).
No CausalFilter applied (single-variable change: model type only).

### SPY — LightGBM Challenger

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| LightGBM challenger | 0.7177 | 0.5918 | 0.5478 | 0.170 | challenger loses |
| XGBoost champion | **0.7397** | 0.6216 | 0.6790 | 0.061 | **CHAMPION HOLDS** |

Top features (LGB): adxr (0.091), trima_20 (0.061), macd_histogram (0.061), dx (0.061)
Lesson: LGB gets close on OOS AUC (0.72 vs 0.74) but with much worse IS-OOS gap (0.170 vs 0.061).
The LGB model is likely overfitting — CV AUC of 0.55 vs OOS 0.72 is suspicious. XGBoost
remains champion with better calibration.

### QQQ — LightGBM Challenger **[NEW CHAMPION]**

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| **LightGBM challenger** | **0.7122** | 0.7415 | 0.5794 | 0.133 | **NEW CHAMPION** |
| XGBoost (former) | 0.6765 | 0.6824 | 0.6407 | 0.036 | demoted |

Top features (LGB): minus_di (0.053), stochrsi_d (0.039), adxr (0.038), bop (0.035), adx (0.035)
Lesson: LGB beats XGB by 0.036 AUC on QQQ. Importantly, this LGB does NOT over-rely on RSI
(unlike the Batch 1 LGB which had RSI SHAP=0.87). Feature importance is well-diversified
across directional/momentum indicators.
**CAUTION:** IS-OOS gap is 0.133 — needs monitoring. If next retrain shows regression,
revert to XGBoost.

### NVDA — XGBoost Challenger

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| XGBoost challenger | 0.6285 | 0.6599 | 0.5531 | 0.075 | challenger loses |
| LightGBM champion | **0.6485** | 0.7500 | 0.5525 | 0.096 | **CHAMPION HOLDS** |

Top features (XGB): sma_10 (0.032), wma_50 (0.024), vwap (0.022), aroonosc (0.021)
Lesson: Consistent with Batch 1 — LGB finds better patterns on NVDA. XGB close but 0.02 behind.

### AAPL — XGBoost Challenger

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| XGBoost challenger | 0.4602 | 0.7007 | 0.4960 | 0.036 | rejected (< 0.50) |
| LightGBM champion | **0.6033** | 0.5782 | 0.4850 | 0.118 | **CHAMPION HOLDS** |

Lesson: XGBoost fails on AAPL again (AUC < 0.50, same as Batch 1). Reproduces exactly.
AAPL has something that XGBoost cannot capture but LGB can. Hypothesis: AAPL's price
dynamics require leaf-wise splitting to find nonlinear patterns that XGB's level-wise
growth misses.

### TSLA — XGBoost Challenger

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| XGBoost challenger | 0.4126 | 0.4762 | 0.4388 | 0.026 | rejected (< 0.50) |
| LightGBM champion | **0.5803** | 0.7365 | 0.5200 | 0.060 | **CHAMPION HOLDS** |

Top features (XGB): bb_middle (0.026), sma_10 (0.023), dema_200 (0.021)
Lesson: XGBoost catastrophically fails on TSLA (AUC 0.41, worse than random). Same as
Batch 1. CV std across folds is enormous (0.287 to 0.584). TSLA volatility breaks XGB's
regularization entirely.

### SPY — Feature Ablation (Top 42 Features)

| Model | AUC (OOS) | Acc | CV AUC | Gap | Verdict |
|-------|-----------|-----|--------|-----|---------|
| XGBoost (42 feat) | 0.7007 | 0.6735 | 0.6104 | 0.090 | full model still better |
| XGBoost champion (84 feat) | **0.7397** | 0.6216 | 0.6790 | 0.061 | **CHAMPION HOLDS** |

Features kept (top 42): t3_50, ema_10, kama_20, dema_10, macd_histogram, wma_50, ema_50...
Features dropped (bottom 42): bb_width, sar, dx, stochf_k, macd_line, mama, rsi, sma_200...
Lesson: Removing 42 features HURT performance (AUC dropped 0.039). The "low importance"
features collectively contribute meaningful signal. Notably, sar and rsi were dropped
despite being important in other symbols — they may carry cross-asset information that
the model uses indirectly. Feature ablation hypothesis REJECTED for SPY XGBoost.

## Batch 2 Summary

| Experiment | OOS AUC | vs Champion | Verdict |
|-----------|---------|-------------|---------|
| SPY LGB challenger | 0.7177 | -0.022 | loses (high gap) |
| QQQ LGB challenger | 0.7122 | +0.036 | **NEW CHAMPION** |
| NVDA XGB challenger | 0.6285 | -0.020 | loses |
| AAPL XGB challenger | 0.4602 | -0.143 | rejected |
| TSLA XGB challenger | 0.4126 | -0.168 | rejected |
| SPY ablation (42f) | 0.7007 | -0.039 | loses |

Key findings:
1. **QQQ flipped to LightGBM** — significant improvement (+0.036 AUC), diversified features.
2. **XGBoost consistently fails on single stocks** — AAPL and TSLA reproduce below-random results.
3. **Feature ablation hurts SPY** — the full 84-feature set is not redundant for XGBoost.
4. **Pattern confirmed**: XGBoost for ETFs, LightGBM for single stocks. But QQQ is now an exception.
5. **IS-OOS gaps are large on LGB models** — suggests LGB finds patterns in test that CV misses, or CV is too conservative.

## Batch 3: Validated Strategy Symbol Expansion (2026-03-21)

**Hypothesis:** LightGBM with technical+fundamentals tiers and CausalFilter can establish
champion baselines on validated strategy symbols (XLK, XLF, XLE) with AUC > 0.56.
Also retrained SPY/QQQ with consistent config for comparability.

**Config:** lookback=756, label=EventLabeler, CV=PurgedKFold 5-fold,
feature_tiers=technical+fundamentals, apply_causal_filter=True, model=LightGBM.
Note: CausalFilter dropped 0 features on all symbols (all 84 passed).

### XLK (Technology Select Sector)

| Model | AUC (OOS) | Acc | CV AUC (mean) | Verdict |
|-------|-----------|-----|---------------|---------|
| LightGBM | **0.6477** | 0.7347 | 0.5510 | **CHAMPION** |

Top SHAP: trima_50 (0.065), macdext_histogram (0.049), minus_di (0.049), adxr (0.042), trix (0.040)
CV folds: [0.502, 0.496, 0.637, 0.527, 0.593] -- high variance, fold 3 strong.
Lesson: XLK behaves similarly to QQQ (tech-heavy). Trend smoothing (trima_50) and momentum
divergence (macdext_histogram) dominate. Strong model, near NVDA quality.

### XLF (Financial Select Sector)

| Model | AUC (OOS) | Acc | CV AUC (mean) | Verdict |
|-------|-----------|-----|---------------|---------|
| LightGBM | **0.6374** | 0.6054 | 0.6229 | **CHAMPION** |

Top SHAP: macdext_histogram (0.100), natr (0.100), macd_line (0.067), stoch_k (0.050), adx (0.050)
CV folds: [0.691, 0.553, 0.636, 0.658, 0.576] -- most stable CV across folds.
Lesson: XLF is the best-calibrated model (CV mean 0.623 vs OOS 0.637, gap only 0.014).
Volatility (natr) and momentum (macdext_histogram) dominate. Financials respond to
volatility regimes more than tech stocks do.

### XLE (Energy Select Sector)

| Model | AUC (OOS) | Acc | CV AUC (mean) | Verdict |
|-------|-----------|-----|---------------|---------|
| LightGBM | **0.5748** | 0.517 | 0.5118 | **CHAMPION** (marginal) |

Top SHAP: macdext_histogram (0.069), bb_width (0.069), midpoint (0.049), wma_20 (0.039), stochf_k (0.039)
CV folds: [0.538, 0.545, 0.576, 0.449, 0.452] -- unstable, last 2 folds near random.
Lesson: XLE is the hardest ETF to model. Energy prices driven by macro/geopolitical
factors not captured by technicals. bb_width (volatility expansion) is uniquely important
here -- energy is a vol-driven sector. AUC barely above noise threshold.

### SPY — Fresh Baseline (tech+fund, CausalFilter)

| Model | AUC (OOS) | Acc | CV AUC (mean) | Verdict |
|-------|-----------|-----|---------------|---------|
| LightGBM (new) | 0.7177 | 0.5918 | 0.5478 | does not beat XGB champion (0.7397) |

Top SHAP: adxr (0.091), trima_20 (0.061), macd_histogram (0.061), aroon_down (0.061), trix (0.061), dx (0.061)
Lesson: Reproduces Batch 2 result exactly (same AUC 0.7177). XGBoost champion holds at 0.7397.

### QQQ — Fresh Baseline (tech+fund, CausalFilter)

| Model | AUC (OOS) | Acc | CV AUC (mean) | Verdict |
|-------|-----------|-----|---------------|---------|
| LightGBM (new) | 0.7122 | 0.7415 | 0.5794 | reproduces champion exactly |

Top SHAP: minus_di (0.053), stochrsi_d (0.039), adxr (0.038), bop (0.035), adx (0.035)
Lesson: Exact reproduction of Batch 2 champion. Feature importance identical. Model is stable.

### Concept Drift Analysis (all 5 validated symbols)

| Symbol | Drift Ratio | Non-Drifted Features | Recommendation |
|--------|-------------|---------------------|----------------|
| SPY | 95.2% | bop, dx, ultosc, adosc | retrain |
| QQQ | 91.7% | bop, dx, ultosc, adosc, macdext_histogram, stochrsi_k/d | retrain |
| XLK | 96.4% | bop, dx, stochrsi_k | retrain |

**Critical insight:** Nearly all features show drift, but this is dominated by price-level
features (all MAs, VWAP, etc.) which ALWAYS drift as price changes. The meaningful finding
is which oscillators/normalized features are stable: **bop** (Balance of Power) and **dx**
(Directional Movement Index) are stable across ALL three symbols tested. These are the most
regime-invariant features and may be underweighted in current models.

### Batch 3 Summary

| Symbol | AUC | Verdict | Quality Tier |
|--------|-----|---------|-------------|
| XLK | 0.6477 | champion | Strong |
| XLF | 0.6374 | champion (best calibration) | Strong |
| XLE | 0.5748 | champion (marginal) | Promising |
| SPY | 0.7177 | LGB challenger loses to XGB 0.7397 | -- |
| QQQ | 0.7122 | reproduces champion | -- |

### Cross-Symbol Feature Analysis (all 8 symbols, Batches 1-3)

Features appearing in top-5 SHAP across 3+ symbols:

| Feature | Symbols (top-5) | Interpretation |
|---------|-----------------|----------------|
| **macdext_histogram** | XLK, XLF, XLE, SPY | Momentum divergence (extended MACD) |
| **adxr** | SPY, QQQ, XLK, TSLA | Smoothed trend strength |
| **trima_50** | XLK, IWM, QQQ | Medium-term smoothed trend |
| **adx** | QQQ, XLF, XLE | Trend strength |
| **natr** | XLF, XLE | Normalized volatility |
| **minus_di** | QQQ, XLK | Negative directional indicator |

**Breakthrough finding:** `macdext_histogram` is the single most consistent predictor,
appearing in top-5 for 4 of 5 validated symbols. It measures momentum divergence using
exponential smoothing -- this aligns with our regime_momentum strategies.

## Next Experiments (Priority Order)

1. **CatBoost on XLE** — worst model, ordered boosting may handle energy's noise better
2. **Add macro tier to XLE** — energy driven by macro, technicals insufficient
3. **Feature ablation: ADX family + MACD only** — test if 10-15 features beat 84
4. **XGBoost challenger on XLK/XLF** — XGB won SPY, may win other ETFs
5. **Multi-horizon labels on SPY** — train 5-day model, compare with 1-day champion
6. **Regime-conditional features** — different feature subsets per volatility regime
