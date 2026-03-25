# XLF Research & Trading Memory

## Strategies

| ID | Name | OOS Sharpe | Overfit Ratio | Status |
|----|------|-----------|--------------|--------|
| regime_momentum_v1 | Regime Momentum | 0.617 | 0.67 | validated — moderate |

## ML Models

| Date | Model | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-21 | LightGBM | 0.6374 | 0.014 | champion (best calibration) |

## Lessons (XLF-specific)

1. Financials — sector-specific regime behavior.
2. Best IS-OOS calibration (gap 0.014) — most stable model across all symbols.
