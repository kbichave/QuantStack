# SPY Research & Trading Memory

> Per-ticker memory file. Updated by research and trading loops.

## Evidence Map

_Last updated: 2026-03-20_

| Category | Key Findings | Tier | Direction |
|----------|-------------|------|-----------|
| Regime | Bear trend, ADX=49 (strong), range_position=12% | tier_4 | bearish |
| Technicals | RSI=30 (oversold), below all MAs | tier_1 | mixed (bearish trend, oversold) |
| Volatility | EGARCH persistence=1.175 (explosive), forecast vol 14.3% ann | tier_2 | high vol |
| VRP | VRP = +11.45 pct pts (IV 25% vs RV 13.55%) — extremely elevated | tier_2 | favors premium selling |

## Strategies

| ID | Name | Type | OOS Sharpe | Overfit Ratio | Status |
|----|------|------|-----------|--------------|--------|
| regime_momentum_v1 | Regime Momentum | position | 0.819 | 0.93 | validated |

## ML Models

| Date | Model | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-21 | XGBoost | 0.7397 | 0.061 | champion |

**Breakthrough feature**: `macdext_histogram` is top SHAP in 4/5 ETF models.

## Lessons (SPY-specific)

1. **Regime-following works**: Hold long trending_up, short trending_down, flat ranging. OR-logic noise reduction prevents whipsaw.
2. **Post-2010 data only**: Pre-GFC data poisons walk-forward. Structural break at QE onset. Use 2010+ for validation.
3. **VRP elevated in bear**: VRP +11.45 pct pts favors premium selling, but EGARCH persistence >1.0 means vol expansion risk.
4. **EGARCH vol regime**: Persistence 1.175 = explosive. Volatility expanding, not mean-reverting.
