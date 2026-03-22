# ML Model Registry

Tracks all trained ML models in the system. Updated during /workshop and /reflect sessions.

## Champions (active models used for inference)

| Symbol | Type | AUC (OOS) | Acc (OOS) | CV AUC | IS-OOS Gap | Features | Samples | Trained | Status |
|--------|------|-----------|-----------|--------|------------|----------|---------|---------|--------|
| SPY | XGBoost | 0.7397 | 0.6216 | 0.6790 | 0.0607 | 84 | 737 | 2026-03-20 | champion |
| QQQ | LightGBM | 0.7122 | 0.7415 | 0.5794 | 0.1329 | 84 | 737 | 2026-03-21 | champion |
| XLK | LightGBM | 0.6477 | 0.7347 | 0.5510 | 0.0967 | 84 | 737 | 2026-03-21 | champion (new) |
| XLF | LightGBM | 0.6374 | 0.6054 | 0.6229 | 0.0145 | 84 | 737 | 2026-03-21 | champion (new, best calibration) |
| XLE | LightGBM | 0.5748 | 0.5170 | 0.5118 | 0.0630 | 84 | 737 | 2026-03-21 | champion (new, marginal) |
| NVDA | LightGBM | 0.6485 | 0.7500 | 0.5525 | 0.0961 | 84 | 737 | 2026-03-20 | champion |
| AAPL | LightGBM | 0.6033 | 0.5782 | 0.4850 | 0.1183 | 84 | 737 | 2026-03-20 | champion |
| TSLA | LightGBM | 0.5803 | 0.7365 | 0.5200 | 0.0603 | 84 | 737 | 2026-03-20 | champion |
| IWM | — | — | — | — | — | — | — | — | rejected (both models < 0.50) |

## Model Quality Tiers

- **Strong (AUC > 0.65):** SPY (0.74), QQQ (0.71), NVDA (0.65), XLK (0.65), XLF (0.64)
- **Acceptable (0.55-0.65):** AAPL (0.60), TSLA (0.58), XLE (0.57)
- **Rejected (< 0.52):** IWM — both LGB and XGB failed

## Champion Change Log

| Date | Symbol | Old Champion | New Champion | AUC Delta | Reason |
|------|--------|-------------|-------------|-----------|--------|
| 2026-03-21 | QQQ | XGBoost 0.6765 | LightGBM 0.7122 | +0.036 | Challenger model with diversified features beat incumbent |

## Key Observations

1. **LightGBM now dominates 4 of 5 symbols** — QQQ flipped from XGBoost to LightGBM in Batch 2.
2. SPY remains the only XGBoost symbol. Its regularization handles collinear MA features well.
3. XGBoost consistently fails on AAPL and TSLA (AUC < 0.50) — reproduced across 2 independent training runs.
4. QQQ LightGBM has high IS-OOS gap (0.133) — needs CausalFilter to improve calibration.
5. Feature ablation (42 of 84) hurt SPY by 0.039 AUC — full feature set is not redundant for XGBoost.
6. No fundamentals loaded — all 84 features are technical only.
7. Label imbalance ~33-41% positive rate across symbols. Class-weighted training used.

## Champion Change Log (continued)

| Date | Symbol | Old Champion | New Champion | AUC | Reason |
|------|--------|-------------|-------------|-----|--------|
| 2026-03-21 | XLK | — (new) | LightGBM | 0.6477 | Batch 3 initial training |
| 2026-03-21 | XLF | — (new) | LightGBM | 0.6374 | Batch 3 initial training |
| 2026-03-21 | XLE | — (new) | LightGBM | 0.5748 | Batch 3 initial training |

## Experiment Coverage (23 experiments total)

- Batch 1 (2026-03-20): 12 experiments — 2 model types x 6 symbols, established initial champions
- Batch 2 (2026-03-21): 6 experiments — 5 challengers + 1 feature ablation, promoted QQQ LGB
- Batch 3 (2026-03-21): 5 experiments — 3 new symbols (XLK/XLF/XLE) + 2 reproductions (SPY/QQQ)

## Configuration

- Lookback: 756 bars (~3 years daily)
- Feature tiers: technical only (fundamentals not cached)
- Label method: EventLabeler (ATR-based TP/SL)
- CV: PurgedKFoldCV with 1% embargo (5 folds)
- Models saved to: models/{SYMBOL}_latest.joblib

## Monitoring Alerts

- **QQQ LGB IS-OOS gap 0.133** — if next retrain shows AUC regression, revert to XGBoost 0.6765
- **AAPL LGB IS-OOS gap 0.118** — borderline, needs CausalFilter experiment
- **NVDA LGB IS-OOS gap 0.096** — acceptable but watch for drift
